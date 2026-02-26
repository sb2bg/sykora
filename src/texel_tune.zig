/// sykora-texel: native Zig Texel tuner for Sykora's HCE evaluation.
///
/// Replaces the Python texel_tune.py by calling evaluateWhite() directly
/// in-process, eliminating all subprocess/IPC overhead.
///
/// Algorithm: coordinate descent over all EvalParams scalar/array fields.
/// Each epoch tries +delta and -delta for every individual parameter;
/// if MSE decreases the change is kept. When a full pass produces no
/// improvement the step size is halved. Tuning stops when delta < 1 or
/// max_epochs is reached.
///
/// Dataset format (one position per line, pipe-separated):
///   <FEN> | <cp_score> | <result>
/// where result is 1.0 (white wins), 0.5 (draw), 0.0 (black wins).
const std = @import("std");
const Board = @import("bitboard.zig").Board;
const eval = @import("evaluation.zig");
const EvalParams = eval.EvalParams;

const stdout_file = std.fs.File.stdout();
const stderr_file = std.fs.File.stderr();

fn print(comptime fmt: []const u8, args: anytype) void {
    var buf: [4096]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch return;
    stdout_file.writeAll(s) catch {};
}

fn eprint(comptime fmt: []const u8, args: anytype) void {
    var buf: [4096]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch return;
    stderr_file.writeAll(s) catch {};
}

// ─────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ─────────────────────────────────────────────────────────────────────

const Args = struct {
    dataset: ?[]const u8 = null,
    output: []const u8 = "tune_params.txt",
    params: ?[]const u8 = null,
    positions: usize = 200_000,
    epochs: usize = 50,
    delta: i32 = 5,
    batch_size: usize = 10_000,
    K: ?f64 = null,
    seed: u64 = 1,
    no_bounds: bool = false,
    scalar_bound_mult: f64 = 3.0,
    array_bound_mult: f64 = 1.5,
    array_bound_min: i32 = 24,
    cp_target_weight: f64 = 0.35,
    sf_k: f64 = 1.0,
    min_improvement: f64 = 1e-6,
    full_eval_interval: usize = 1,
};

fn parseArgs(allocator: std.mem.Allocator) !Args {
    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);

    var args = Args{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (eql(arg, "--dataset")) {
            const val = requireOptionValue(argv, &i, "--dataset");
            args.dataset = try allocator.dupe(u8, val);
        } else if (eql(arg, "--output")) {
            const val = requireOptionValue(argv, &i, "--output");
            args.output = try allocator.dupe(u8, val);
        } else if (eql(arg, "--params")) {
            const val = requireOptionValue(argv, &i, "--params");
            args.params = try allocator.dupe(u8, val);
        } else if (eql(arg, "--positions")) {
            const val = requireOptionValue(argv, &i, "--positions");
            args.positions = try std.fmt.parseInt(usize, val, 10);
        } else if (eql(arg, "--epochs")) {
            const val = requireOptionValue(argv, &i, "--epochs");
            args.epochs = try std.fmt.parseInt(usize, val, 10);
        } else if (eql(arg, "--delta")) {
            const val = requireOptionValue(argv, &i, "--delta");
            args.delta = try std.fmt.parseInt(i32, val, 10);
        } else if (eql(arg, "--batch-size")) {
            const val = requireOptionValue(argv, &i, "--batch-size");
            args.batch_size = try std.fmt.parseInt(usize, val, 10);
        } else if (eql(arg, "--K")) {
            const val = requireOptionValue(argv, &i, "--K");
            args.K = try std.fmt.parseFloat(f64, val);
        } else if (eql(arg, "--seed")) {
            const val = requireOptionValue(argv, &i, "--seed");
            args.seed = try std.fmt.parseInt(u64, val, 10);
        } else if (eql(arg, "--no-bounds")) {
            args.no_bounds = true;
        } else if (eql(arg, "--scalar-bound-mult")) {
            const val = requireOptionValue(argv, &i, "--scalar-bound-mult");
            args.scalar_bound_mult = try std.fmt.parseFloat(f64, val);
        } else if (eql(arg, "--array-bound-mult")) {
            const val = requireOptionValue(argv, &i, "--array-bound-mult");
            args.array_bound_mult = try std.fmt.parseFloat(f64, val);
        } else if (eql(arg, "--array-bound-min")) {
            const val = requireOptionValue(argv, &i, "--array-bound-min");
            args.array_bound_min = try std.fmt.parseInt(i32, val, 10);
        } else if (eql(arg, "--cp-target-weight")) {
            const val = requireOptionValue(argv, &i, "--cp-target-weight");
            args.cp_target_weight = try std.fmt.parseFloat(f64, val);
        } else if (eql(arg, "--sf-k")) {
            const val = requireOptionValue(argv, &i, "--sf-k");
            args.sf_k = try std.fmt.parseFloat(f64, val);
        } else if (eql(arg, "--min-improvement")) {
            const val = requireOptionValue(argv, &i, "--min-improvement");
            args.min_improvement = try std.fmt.parseFloat(f64, val);
        } else if (eql(arg, "--full-eval-interval")) {
            const val = requireOptionValue(argv, &i, "--full-eval-interval");
            args.full_eval_interval = try std.fmt.parseInt(usize, val, 10);
        } else if (eql(arg, "--help") or eql(arg, "-h")) {
            printUsage();
            std.process.exit(0);
        } else {
            eprint("Unknown argument: {s}\n", .{arg});
            printUsage();
            std.process.exit(1);
        }
    }
    return args;
}

fn requireOptionValue(argv: []const []const u8, i: *usize, opt_name: []const u8) []const u8 {
    if (i.* + 1 >= argv.len) {
        eprint("Missing value for option: {s}\n", .{opt_name});
        printUsage();
        std.process.exit(1);
    }
    i.* += 1;
    return argv[i.*];
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn printUsage() void {
    stderr_file.writeAll(
        \\Usage: sykora-texel [options]
        \\
        \\Options:
        \\  --dataset <path>          Dataset file (required)
        \\  --output <path>           Output params file (default: tune_params.txt)
        \\  --params <path>           Starting params file (uses defaults if omitted)
        \\  --positions <N>           Max positions to load (default: 200000)
        \\  --epochs <N>              Max epochs (default: 50)
        \\  --delta <N>               Initial step size (default: 5)
        \\  --batch-size <N>          Sample size per epoch (default: 10000)
        \\  --K <float>               Sigmoid K (auto-tuned if omitted)
        \\  --seed <N>                Random seed (default: 1)
        \\  --no-bounds               Disable parameter bounds
        \\  --scalar-bound-mult <f>   Bound span multiplier (default: 3.0)
        \\  --array-bound-mult <f>    Array bound multiplier (default: 1.5)
        \\  --array-bound-min <N>     Array minimum bound span (default: 24)
        \\  --cp-target-weight <f>    Blend weight for cp target in [0,1] (default: 0.35)
        \\  --sf-k <f>                Sigmoid K for cp -> prob conversion (default: 1.0)
        \\  --min-improvement <f>     Min MSE improvement threshold (default: 1e-6)
        \\  --full-eval-interval <N>  Full-dataset eval every N epochs (default: 1)
        \\  --help                    Show this help
        \\
        \\Duplicate FEN rows are merged by averaging labels before tuning.
        \\Dataset rows with invalid FEN are treated as hard errors.
        \\
    ) catch {};
}

// ─────────────────────────────────────────────────────────────────────
// Dataset
// ─────────────────────────────────────────────────────────────────────

const Dataset = struct {
    boards: []Board,
    targets: []f64,
    count: usize,

    fn deinit(self: *Dataset, allocator: std.mem.Allocator) void {
        allocator.free(self.boards);
        allocator.free(self.targets);
        self.* = undefined;
    }
};

const AggregatedEntry = struct {
    board: Board,
    sum_result: f64,
    sum_cp_prob: f64,
    count: usize,
};

fn loadDataset(path: []const u8, max_positions: usize, args: *const Args, allocator: std.mem.Allocator) !Dataset {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var by_fen = std.StringHashMap(AggregatedEntry).init(allocator);
    defer {
        var it_free = by_fen.iterator();
        while (it_free.next()) |entry| allocator.free(entry.key_ptr.*);
        by_fen.deinit();
    }

    var file_buf: [64 * 1024]u8 = undefined;
    var reader = file.reader(&file_buf);

    var raw_rows: usize = 0;
    var skipped_bad_rows: usize = 0;
    var skipped_bad_cp: usize = 0;
    var skipped_bad_results: usize = 0;
    var merged_duplicates: usize = 0;
    var cp_nonzero_rows: usize = 0;
    var line_no: usize = 0;

    while (by_fen.count() < max_positions) {
        const maybe_line = reader.interface.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => {
                eprint("Error: dataset line too long near line {d}.\n", .{line_no + 1});
                return error.DatasetLineTooLong;
            },
            else => return err,
        };
        const line = maybe_line orelse break;

        line_no += 1;
        const trimmed = std.mem.trim(u8, line, " \r\t");
        if (trimmed.len == 0 or trimmed[0] == '#') continue;
        raw_rows += 1;

        // Parse "FEN | cp_score | result"
        var parts = std.mem.splitSequence(u8, trimmed, "|");
        const fen_part = parts.next() orelse {
            skipped_bad_rows += 1;
            continue;
        };
        const cp_part = parts.next() orelse {
            skipped_bad_rows += 1;
            continue;
        };
        const result_part = parts.next() orelse {
            skipped_bad_rows += 1;
            continue;
        };

        const fen = std.mem.trim(u8, fen_part, " \t");
        const cp_str = std.mem.trim(u8, cp_part, " \t");
        const result_str = std.mem.trim(u8, result_part, " \t\r");

        var cp: f64 = std.fmt.parseFloat(f64, cp_str) catch {
            skipped_bad_cp += 1;
            continue;
        };
        if (cp > 5000.0) cp = 5000.0;
        if (cp < -5000.0) cp = -5000.0;
        if (@abs(cp) >= 1.0) cp_nonzero_rows += 1;

        const result = std.fmt.parseFloat(f64, result_str) catch {
            skipped_bad_results += 1;
            continue;
        };

        // Validate result is 0.0, 0.5, or 1.0
        if (result != 0.0 and result != 0.5 and result != 1.0) {
            skipped_bad_results += 1;
            continue;
        }

        if (by_fen.getPtr(fen)) |entry| {
            entry.sum_result += result;
            entry.sum_cp_prob += sigmoid(cp, args.sf_k);
            entry.count += 1;
            merged_duplicates += 1;
            continue;
        }

        const board = Board.fromFen(fen) catch {
            eprint("Error: invalid FEN at line {d}.\n", .{line_no});
            return error.InvalidDatasetFen;
        };

        const fen_key = try allocator.dupe(u8, fen);
        errdefer allocator.free(fen_key);
        try by_fen.put(fen_key, .{
            .board = board,
            .sum_result = result,
            .sum_cp_prob = sigmoid(cp, args.sf_k),
            .count = 1,
        });
    }

    var effective_cp_weight = args.cp_target_weight;
    if (effective_cp_weight > 0.0 and cp_nonzero_rows == 0) {
        print("Warning: cp_target_weight > 0 but cp scores are all zero; falling back to result-only targets.\n", .{});
        effective_cp_weight = 0.0;
    }

    const unique = by_fen.count();
    const boards = try allocator.alloc(Board, unique);
    errdefer allocator.free(boards);
    const targets = try allocator.alloc(f64, unique);
    errdefer allocator.free(targets);

    var idx: usize = 0;
    var it = by_fen.iterator();
    while (it.next()) |entry| {
        const n_samples = @as(f64, @floatFromInt(entry.value_ptr.count));
        const mean_result = entry.value_ptr.sum_result / n_samples;
        const mean_cp_prob = entry.value_ptr.sum_cp_prob / n_samples;
        const blended_target = (1.0 - effective_cp_weight) * mean_result + effective_cp_weight * mean_cp_prob;
        boards[idx] = entry.value_ptr.board;
        targets[idx] = blended_target;
        idx += 1;
    }

    print("Loaded {d} unique positions from {s} ({d} raw rows)\n", .{ unique, path, raw_rows });
    if (merged_duplicates > 0)
        print("  merged {d} duplicate rows by averaging labels\n", .{merged_duplicates});
    if (skipped_bad_rows > 0)
        print("  skipped {d} malformed rows\n", .{skipped_bad_rows});
    if (skipped_bad_cp > 0)
        print("  skipped {d} rows with invalid cp scores\n", .{skipped_bad_cp});
    if (skipped_bad_results > 0)
        print("  skipped {d} rows with invalid result labels\n", .{skipped_bad_results});
    print("  target blend: result={d:.2}, cp={d:.2}, sf_k={d:.3}\n", .{ 1.0 - effective_cp_weight, effective_cp_weight, args.sf_k });

    return .{ .boards = boards, .targets = targets, .count = boards.len };
}

// ─────────────────────────────────────────────────────────────────────
// Sigmoid & MSE
// ─────────────────────────────────────────────────────────────────────

fn sigmoid(cp: f64, K: f64) f64 {
    return 1.0 / (1.0 + std.math.pow(f64, 10.0, -K * cp / 400.0));
}

fn computeMse(boards: []Board, targets: []const f64, count: usize, K: f64) f64 {
    var sum: f64 = 0.0;
    for (0..count) |i| {
        var b = boards[i];
        const score: f64 = @floatFromInt(eval.evaluateWhite(&b));
        const pred = sigmoid(score, K);
        const err = pred - targets[i];
        sum += err * err;
    }
    return sum / @as(f64, @floatFromInt(count));
}

fn computeMseSubset(
    boards: []Board,
    targets: []const f64,
    indices: []const usize,
    K: f64,
) f64 {
    var sum: f64 = 0.0;
    for (indices) |idx| {
        var b = boards[idx];
        const score: f64 = @floatFromInt(eval.evaluateWhite(&b));
        const pred = sigmoid(score, K);
        const err = pred - targets[idx];
        sum += err * err;
    }
    return sum / @as(f64, @floatFromInt(indices.len));
}

// ─────────────────────────────────────────────────────────────────────
// K auto-tuning
// ─────────────────────────────────────────────────────────────────────

fn tuneK(boards: []Board, targets: []const f64, count: usize) f64 {
    print("Auto-tuning K...", .{});

    // Pre-evaluate all positions once
    const scores = std.heap.page_allocator.alloc(f64, count) catch {
        print(" allocation failed, using K=1.0\n", .{});
        return 1.0;
    };
    defer std.heap.page_allocator.free(scores);

    for (0..count) |i| {
        var b = boards[i];
        scores[i] = @floatFromInt(eval.evaluateWhite(&b));
    }

    var best_K: f64 = 1.0;
    var best_mse: f64 = std.math.inf(f64);
    const steps: usize = 20;
    for (0..steps + 1) |step| {
        const K: f64 = 0.5 + @as(f64, @floatFromInt(step)) * (2.0 - 0.5) / @as(f64, @floatFromInt(steps));
        var sum: f64 = 0.0;
        for (0..count) |i| {
            const pred = sigmoid(scores[i], K);
            const err = pred - targets[i];
            sum += err * err;
        }
        const mse = sum / @as(f64, @floatFromInt(count));
        if (mse < best_mse) {
            best_mse = mse;
            best_K = K;
        }
    }

    print(" K={d:.4}  (MSE={d:.6})\n", .{ best_K, best_mse });
    return best_K;
}

// ─────────────────────────────────────────────────────────────────────
// Parameter bounds
// ─────────────────────────────────────────────────────────────────────

const Bounds = struct { lo: i32, hi: i32 };

/// Get bounds for a scalar parameter given its field name and initial value.
fn scalarBounds(comptime name: []const u8, base: i32, mult: f64) Bounds {
    // Piece value bounds (hard-coded)
    if (comptime eql(name, "pawn_value")) return .{ .lo = 50, .hi = 200 };
    if (comptime eql(name, "knight_value")) return .{ .lo = 180, .hi = 650 };
    if (comptime eql(name, "bishop_value")) return .{ .lo = 180, .hi = 650 };
    if (comptime eql(name, "rook_value")) return .{ .lo = 300, .hi = 1100 };
    if (comptime eql(name, "queen_value")) return .{ .lo = 500, .hi = 1800 };

    if (comptime eql(name, "endgame_phase_threshold")) return .{ .lo = 0, .hi = 256 };

    if (comptime endsWith(name, "_penalty") or endsWith(name, "_bonus")) {
        const abs_base = if (base < 0) -base else base;
        const from_mult: i32 = @intFromFloat(@as(f64, @floatFromInt(abs_base)) * mult);
        const hi = @max(20, @max(from_mult, abs_base + 30));
        return .{ .lo = 0, .hi = hi };
    }

    if (comptime endsWith(name, "_threshold")) {
        const abs_base = if (base < 0) -base else base;
        const span: i32 = @max(16, @as(i32, @intFromFloat(@as(f64, @floatFromInt(abs_base)) * mult)));
        return .{ .lo = @max(0, base - span), .hi = @min(512, base + span) };
    }

    // Default: symmetric range around base
    const abs_base = if (base < 0) -base else base;
    const span: i32 = @max(20, @as(i32, @intFromFloat(@as(f64, @floatFromInt(abs_base)) * mult)));
    return .{ .lo = base - span, .hi = base + span };
}

fn arrayBounds(comptime name: []const u8, base: i32, mult: f64, min_span: i32) Bounds {
    const abs_base = if (base < 0) -base else base;
    const from_mult: i32 = @as(i32, @intFromFloat(@as(f64, @floatFromInt(abs_base)) * mult));
    const span = @max(min_span, @max(8, from_mult));

    if (comptime eql(name, "passed_pawn_bonus")) {
        return .{ .lo = 0, .hi = @max(220, base + span) };
    }

    if (comptime endsWith(name, "_table")) {
        return .{ .lo = clamp(base - span, -200, 200), .hi = clamp(base + span, -200, 200) };
    }

    if (comptime endsWith(name, "_mobility")) {
        return .{ .lo = clamp(base - span, -120, 120), .hi = clamp(base + span, -120, 120) };
    }

    if (comptime endsWith(name, "_bonus") or endsWith(name, "_penalty")) {
        return .{ .lo = 0, .hi = @max(60, base + span) };
    }

    return .{ .lo = base - span, .hi = base + span };
}

fn endsWith(comptime s: []const u8, comptime suffix: []const u8) bool {
    if (s.len < suffix.len) return false;
    return comptime std.mem.eql(u8, s[s.len - suffix.len ..], suffix);
}

fn clamp(val: i32, lo: i32, hi: i32) i32 {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

// ─────────────────────────────────────────────────────────────────────
// Parameter access via comptime field iteration
// ─────────────────────────────────────────────────────────────────────

/// A flattened parameter index: (field_index, array_element_index).
/// For scalar fields, elem_idx is 0 and ignored.
const ParamId = struct {
    field_idx: u32,
    elem_idx: u32,
    is_array: bool,
};

/// Count all tunable parameter slots at comptime.
fn countParams() usize {
    comptime {
        var total: usize = 0;
        for (std.meta.fields(EvalParams)) |field| {
            switch (@typeInfo(field.type)) {
                .int => total += 1,
                .array => |arr| total += arr.len,
                else => {},
            }
        }
        return total;
    }
}

const PARAM_COUNT = countParams();

/// Build the flat list of all tunable parameter slots at comptime.
fn buildParamList() [PARAM_COUNT]ParamId {
    comptime {
        var list: [PARAM_COUNT]ParamId = undefined;
        var idx: usize = 0;
        for (std.meta.fields(EvalParams), 0..) |field, fi| {
            switch (@typeInfo(field.type)) {
                .int => {
                    list[idx] = .{ .field_idx = @intCast(fi), .elem_idx = 0, .is_array = false };
                    idx += 1;
                },
                .array => |arr| {
                    for (0..arr.len) |ei| {
                        list[idx] = .{ .field_idx = @intCast(fi), .elem_idx = @intCast(ei), .is_array = true };
                        idx += 1;
                    }
                },
                else => {},
            }
        }
        return list;
    }
}

const PARAM_LIST = buildParamList();

/// Get the current value of a parameter slot from g_params.
fn getParam(pid: ParamId) i32 {
    inline for (std.meta.fields(EvalParams), 0..) |field, fi| {
        if (fi == pid.field_idx) {
            const val = @field(eval.g_params, field.name);
            switch (@typeInfo(field.type)) {
                .int => return val,
                .array => return val[pid.elem_idx],
                else => unreachable,
            }
        }
    }
    unreachable;
}

/// Set the value of a parameter slot in g_params.
fn setParam(pid: ParamId, value: i32) void {
    inline for (std.meta.fields(EvalParams), 0..) |field, fi| {
        if (fi == pid.field_idx) {
            switch (@typeInfo(field.type)) {
                .int => @field(eval.g_params, field.name) = value,
                .array => @field(eval.g_params, field.name)[pid.elem_idx] = value,
                else => unreachable,
            }
            return;
        }
    }
}

/// Get scalar bounds for a parameter slot (only meaningful for scalar fields).
fn getParamBounds(pid: ParamId, scalar_mult: f64, array_mult: f64, array_min: i32) ?Bounds {
    inline for (std.meta.fields(EvalParams), 0..) |field, fi| {
        if (fi == pid.field_idx) {
            switch (@typeInfo(field.type)) {
                .int => return scalarBounds(field.name, @field(eval.g_params, field.name), scalar_mult),
                .array => {
                    const arr = @field(eval.g_params, field.name);
                    return arrayBounds(field.name, arr[pid.elem_idx], array_mult, array_min);
                },
                else => return null,
            }
        }
    }
    return null;
}

// ─────────────────────────────────────────────────────────────────────
// Random sampling
// ─────────────────────────────────────────────────────────────────────

/// Fisher-Yates partial shuffle to select `k` random indices from [0, n).
fn sampleIndices(indices: []usize, n: usize, k: usize, rng: std.Random) void {
    // Fill with 0..n-1
    for (0..n) |i| indices[i] = i;
    // Partial shuffle: only need first k elements
    const limit = @min(k, n);
    for (0..limit) |i| {
        const j = i + rng.uintLessThan(usize, n - i);
        const tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

fn snapshotParams(out: []i32) void {
    for (PARAM_LIST, 0..) |pid, i| out[i] = getParam(pid);
}

fn restoreParams(snapshot: []const i32) void {
    for (PARAM_LIST, 0..) |pid, i| setParam(pid, snapshot[i]);
}

// ─────────────────────────────────────────────────────────────────────
// Coordinate descent
// ─────────────────────────────────────────────────────────────────────

fn coordinateDescent(
    ds: *const Dataset,
    args: *const Args,
    K: f64,
    allocator: std.mem.Allocator,
) !void {
    const n = ds.count;
    var delta = args.delta;
    const use_bounds = !args.no_bounds;
    const scalar_mult = args.scalar_bound_mult;
    const array_mult = args.array_bound_mult;
    const array_min = args.array_bound_min;

    // Pre-compute parameter bounds from initial param values
    var bounds_cache: [PARAM_LIST.len]?Bounds = undefined;
    if (use_bounds) {
        for (PARAM_LIST, 0..) |pid, i| {
            bounds_cache[i] = getParamBounds(pid, scalar_mult, array_mult, array_min);
        }
    } else {
        for (0..PARAM_LIST.len) |i| bounds_cache[i] = null;
    }

    // Allocate index array for sampling
    const index_buf = try allocator.alloc(usize, n);
    defer allocator.free(index_buf);
    const epoch_snapshot = try allocator.alloc(i32, PARAM_LIST.len);
    defer allocator.free(epoch_snapshot);

    var rng = std.Random.DefaultPrng.init(args.seed);
    const random = rng.random();

    // Evaluate baseline on full dataset
    print("Evaluating baseline...", .{});
    const baseline_mse = computeMse(ds.boards, ds.targets, n, K);
    print(" MSE={d:.6}\n", .{baseline_mse});
    var current_full_mse = baseline_mse;
    var best_full_mse = baseline_mse;

    var epoch: usize = 1;
    while (epoch <= args.epochs) : (epoch += 1) {
        if (delta < 1) {
            print("Delta < 1, stopping after epoch {d}.\n", .{epoch - 1});
            break;
        }

        snapshotParams(epoch_snapshot);

        // Sample subset for this epoch
        const use_subset = args.batch_size < n;
        const sample_size = if (use_subset) args.batch_size else n;

        var sample_indices: []usize = undefined;
        if (use_subset) {
            sampleIndices(index_buf, n, sample_size, random);
            sample_indices = index_buf[0..sample_size];
        }

        // Baseline MSE on sample
        var current_sample_mse: f64 = undefined;
        if (use_subset) {
            current_sample_mse = computeMseSubset(ds.boards, ds.targets, sample_indices, K);
        } else {
            current_sample_mse = computeMse(ds.boards, ds.targets, n, K);
        }

        const epoch_start = std.time.nanoTimestamp();
        var improved: usize = 0;
        const total = PARAM_LIST.len;

        for (PARAM_LIST, 0..) |pid, param_i| {
            const orig = getParam(pid);
            const b = bounds_cache[param_i];

            // Try +delta
            const plus_candidate = if (b) |bv| clamp(orig + delta, bv.lo, bv.hi) else orig + delta;
            var mse_plus: f64 = std.math.inf(f64);
            if (plus_candidate != orig) {
                setParam(pid, plus_candidate);
                mse_plus = if (use_subset)
                    computeMseSubset(ds.boards, ds.targets, sample_indices, K)
                else
                    computeMse(ds.boards, ds.targets, n, K);
            }

            if (mse_plus + args.min_improvement < current_sample_mse) {
                current_sample_mse = mse_plus;
                improved += 1;
                continue; // keep +delta
            }

            // Reset before -delta
            setParam(pid, orig);

            // Try -delta
            const minus_candidate = if (b) |bv| clamp(orig - delta, bv.lo, bv.hi) else orig - delta;
            var mse_minus: f64 = std.math.inf(f64);
            if (minus_candidate != orig) {
                setParam(pid, minus_candidate);
                mse_minus = if (use_subset)
                    computeMseSubset(ds.boards, ds.targets, sample_indices, K)
                else
                    computeMse(ds.boards, ds.targets, n, K);
            }

            if (mse_minus + args.min_improvement < current_sample_mse) {
                current_sample_mse = mse_minus;
                improved += 1;
                continue; // keep -delta
            }

            // Neither helped — restore
            setParam(pid, orig);

            if ((param_i + 1) % 50 == 0) {
                const pct = @as(f64, @floatFromInt(param_i + 1)) * 100.0 / @as(f64, @floatFromInt(total));
                print("  [{d:5.1}%] MSE={d:.6} delta={d} improved={d}\r", .{
                    pct, current_sample_mse, delta, improved,
                });
            }
        }

        // Full-dataset eval at interval
        const do_full_eval = (args.full_eval_interval <= 1) or (epoch % args.full_eval_interval == 0);
        var rolled_back = false;
        if (do_full_eval) {
            const full_mse = computeMse(ds.boards, ds.targets, n, K);
            if (full_mse > best_full_mse + args.min_improvement) {
                restoreParams(epoch_snapshot);
                current_full_mse = best_full_mse;
                rolled_back = true;
            } else {
                current_full_mse = full_mse;
                if (full_mse + args.min_improvement < best_full_mse) {
                    best_full_mse = full_mse;
                }
            }
        }

        const elapsed_ns = std.time.nanoTimestamp() - epoch_start;
        const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;

        if (do_full_eval) {
            print("\nEpoch {d:3} | delta={d:3} | improved={d:4} | sample_MSE={d:.6} | full_MSE={d:.6} | {d:.1}s\n", .{
                epoch, delta, improved, current_sample_mse, current_full_mse, elapsed_s,
            });
            if (rolled_back) {
                print("  Full MSE regressed -> epoch reverted to previous best parameters.\n", .{});
            }
        } else {
            print("\nEpoch {d:3} | delta={d:3} | improved={d:4} | sample_MSE={d:.6} | full_MSE=skipped | {d:.1}s\n", .{
                epoch, delta, improved, current_sample_mse, elapsed_s,
            });
        }

        // Save params after each epoch
        try eval.saveParams(args.output, allocator);
        print("  Params saved to {s}\n", .{args.output});

        if (improved == 0 or rolled_back) {
            delta = @divTrunc(delta, 2);
            if (rolled_back) {
                print("  Rollback triggered -> delta halved to {d}\n", .{delta});
            } else {
                print("  No improvement -> delta halved to {d}\n", .{delta});
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try parseArgs(allocator);

    if (args.dataset == null) {
        eprint("Error: --dataset is required\n", .{});
        printUsage();
        std.process.exit(1);
    }
    if (args.batch_size == 0) {
        eprint("Error: --batch-size must be > 0\n", .{});
        std.process.exit(1);
    }
    if (args.full_eval_interval == 0) {
        eprint("Error: --full-eval-interval must be > 0\n", .{});
        std.process.exit(1);
    }
    if (args.scalar_bound_mult <= 0.0) {
        eprint("Error: --scalar-bound-mult must be > 0\n", .{});
        std.process.exit(1);
    }
    if (args.array_bound_mult <= 0.0) {
        eprint("Error: --array-bound-mult must be > 0\n", .{});
        std.process.exit(1);
    }
    if (args.array_bound_min <= 0) {
        eprint("Error: --array-bound-min must be > 0\n", .{});
        std.process.exit(1);
    }
    if (args.cp_target_weight < 0.0 or args.cp_target_weight > 1.0) {
        eprint("Error: --cp-target-weight must be in [0,1]\n", .{});
        std.process.exit(1);
    }
    if (args.sf_k <= 0.0) {
        eprint("Error: --sf-k must be > 0\n", .{});
        std.process.exit(1);
    }
    if (args.min_improvement < 0.0) {
        eprint("Error: --min-improvement must be >= 0\n", .{});
        std.process.exit(1);
    }

    // Load starting params
    if (args.params) |p| {
        try eval.loadParams(p, allocator);
        print("Loaded starting params from {s}\n", .{p});
    } else {
        print("No --params given; using compiled-in defaults.\n", .{});
    }

    // Load dataset
    var ds = try loadDataset(args.dataset.?, args.positions, &args, allocator);
    defer ds.deinit(allocator);
    if (ds.count == 0) {
        eprint("Error: dataset is empty.\n", .{});
        std.process.exit(1);
    }

    // Tune K
    const K = args.K orelse tuneK(ds.boards, ds.targets, ds.count);
    if (args.K != null) {
        print("Using provided K={d:.4}\n", .{K});
    }

    // Run coordinate descent
    try coordinateDescent(&ds, &args, K, allocator);

    try eval.saveParams(args.output, allocator);
    print("\nTuning complete. Final params saved to {s}\n", .{args.output});
    print("Apply to engine with:\n  python utils/tuning/apply_params.py {s}\n", .{args.output});
}
