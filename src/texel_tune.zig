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
            i += 1;
            args.dataset = try allocator.dupe(u8, argv[i]);
        } else if (eql(arg, "--output")) {
            i += 1;
            args.output = try allocator.dupe(u8, argv[i]);
        } else if (eql(arg, "--params")) {
            i += 1;
            args.params = try allocator.dupe(u8, argv[i]);
        } else if (eql(arg, "--positions")) {
            i += 1;
            args.positions = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (eql(arg, "--epochs")) {
            i += 1;
            args.epochs = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (eql(arg, "--delta")) {
            i += 1;
            args.delta = try std.fmt.parseInt(i32, argv[i], 10);
        } else if (eql(arg, "--batch-size")) {
            i += 1;
            args.batch_size = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (eql(arg, "--K")) {
            i += 1;
            args.K = try std.fmt.parseFloat(f64, argv[i]);
        } else if (eql(arg, "--seed")) {
            i += 1;
            args.seed = try std.fmt.parseInt(u64, argv[i], 10);
        } else if (eql(arg, "--no-bounds")) {
            args.no_bounds = true;
        } else if (eql(arg, "--scalar-bound-mult")) {
            i += 1;
            args.scalar_bound_mult = try std.fmt.parseFloat(f64, argv[i]);
        } else if (eql(arg, "--min-improvement")) {
            i += 1;
            args.min_improvement = try std.fmt.parseFloat(f64, argv[i]);
        } else if (eql(arg, "--full-eval-interval")) {
            i += 1;
            args.full_eval_interval = try std.fmt.parseInt(usize, argv[i], 10);
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
        \\  --min-improvement <f>     Min MSE improvement threshold (default: 1e-6)
        \\  --full-eval-interval <N>  Full-dataset eval every N epochs (default: 1)
        \\  --help                    Show this help
        \\
    ) catch {};
}

// ─────────────────────────────────────────────────────────────────────
// Dataset
// ─────────────────────────────────────────────────────────────────────

const Dataset = struct {
    boards: []Board,
    results: []f64,
    count: usize,
};

fn loadDataset(path: []const u8, max_positions: usize, allocator: std.mem.Allocator) !Dataset {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 2 * 1024 * 1024 * 1024); // 2 GB max
    defer allocator.free(content);

    // First pass: count lines to pre-allocate
    var line_count: usize = 0;
    {
        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \r\t");
            if (trimmed.len == 0 or trimmed[0] == '#') continue;
            line_count += 1;
            if (line_count >= max_positions) break;
        }
    }

    const cap = @min(line_count, max_positions);
    const boards = try allocator.alloc(Board, cap);
    const results = try allocator.alloc(f64, cap);
    var count: usize = 0;
    var skipped_bad_rows: usize = 0;
    var skipped_bad_results: usize = 0;
    var skipped_bad_fens: usize = 0;

    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        if (count >= max_positions) break;
        const trimmed = std.mem.trim(u8, line, " \r\t");
        if (trimmed.len == 0 or trimmed[0] == '#') continue;

        // Parse "FEN | cp_score | result"
        var parts = std.mem.splitSequence(u8, trimmed, "|");
        const fen_part = parts.next() orelse {
            skipped_bad_rows += 1;
            continue;
        };
        _ = parts.next() orelse {
            skipped_bad_rows += 1;
            continue;
        }; // skip cp_score
        const result_part = parts.next() orelse {
            skipped_bad_rows += 1;
            continue;
        };

        const fen = std.mem.trim(u8, fen_part, " \t");
        const result_str = std.mem.trim(u8, result_part, " \t\r");

        const result = std.fmt.parseFloat(f64, result_str) catch {
            skipped_bad_results += 1;
            continue;
        };

        // Validate result is 0.0, 0.5, or 1.0
        if (result != 0.0 and result != 0.5 and result != 1.0) {
            skipped_bad_results += 1;
            continue;
        }

        const board = Board.fromFen(fen) catch {
            skipped_bad_fens += 1;
            continue;
        };

        boards[count] = board;
        results[count] = result;
        count += 1;
    }

    print("Loaded {d} positions from {s}\n", .{ count, path });
    if (skipped_bad_rows > 0)
        print("  skipped {d} malformed rows\n", .{skipped_bad_rows});
    if (skipped_bad_results > 0)
        print("  skipped {d} rows with invalid result labels\n", .{skipped_bad_results});
    if (skipped_bad_fens > 0)
        print("  skipped {d} rows with invalid FENs\n", .{skipped_bad_fens});

    return .{ .boards = boards, .results = results, .count = count };
}

// ─────────────────────────────────────────────────────────────────────
// Sigmoid & MSE
// ─────────────────────────────────────────────────────────────────────

fn sigmoid(cp: f64, K: f64) f64 {
    return 1.0 / (1.0 + std.math.pow(f64, 10.0, -K * cp / 400.0));
}

fn computeMse(boards: []Board, results: []const f64, count: usize, K: f64) f64 {
    var sum: f64 = 0.0;
    for (0..count) |i| {
        var b = boards[i];
        const score: f64 = @floatFromInt(eval.evaluateWhite(&b));
        const pred = sigmoid(score, K);
        const err = pred - results[i];
        sum += err * err;
    }
    return sum / @as(f64, @floatFromInt(count));
}

fn computeMseSubset(
    boards: []Board,
    results: []const f64,
    indices: []const usize,
    K: f64,
) f64 {
    var sum: f64 = 0.0;
    for (indices) |idx| {
        var b = boards[idx];
        const score: f64 = @floatFromInt(eval.evaluateWhite(&b));
        const pred = sigmoid(score, K);
        const err = pred - results[idx];
        sum += err * err;
    }
    return sum / @as(f64, @floatFromInt(indices.len));
}

// ─────────────────────────────────────────────────────────────────────
// K auto-tuning
// ─────────────────────────────────────────────────────────────────────

fn tuneK(boards: []Board, results: []const f64, count: usize) f64 {
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
            const err = pred - results[i];
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
fn getParamBounds(pid: ParamId, mult: f64) ?Bounds {
    if (pid.is_array) return null; // Arrays have no bounds in the Python tuner
    inline for (std.meta.fields(EvalParams), 0..) |field, fi| {
        if (fi == pid.field_idx) {
            switch (@typeInfo(field.type)) {
                .int => return scalarBounds(field.name, @field(eval.g_params, field.name), mult),
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
    const mult = args.scalar_bound_mult;

    // Pre-compute scalar bounds from initial param values
    var bounds_cache: [PARAM_LIST.len]?Bounds = undefined;
    if (use_bounds) {
        for (PARAM_LIST, 0..) |pid, i| {
            bounds_cache[i] = getParamBounds(pid, mult);
        }
    } else {
        for (0..PARAM_LIST.len) |i| bounds_cache[i] = null;
    }

    // Allocate index array for sampling
    const index_buf = try allocator.alloc(usize, n);
    defer allocator.free(index_buf);

    var rng = std.Random.DefaultPrng.init(args.seed);
    const random = rng.random();

    // Evaluate baseline on full dataset
    print("Evaluating baseline...", .{});
    const baseline_mse = computeMse(ds.boards, ds.results, n, K);
    print(" MSE={d:.6}\n", .{baseline_mse});
    var current_full_mse = baseline_mse;

    var epoch: usize = 1;
    while (epoch <= args.epochs) : (epoch += 1) {
        if (delta < 1) {
            print("Delta < 1, stopping after epoch {d}.\n", .{epoch - 1});
            break;
        }

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
            current_sample_mse = computeMseSubset(ds.boards, ds.results, sample_indices, K);
        } else {
            current_sample_mse = computeMse(ds.boards, ds.results, n, K);
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
                    computeMseSubset(ds.boards, ds.results, sample_indices, K)
                else
                    computeMse(ds.boards, ds.results, n, K);
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
                    computeMseSubset(ds.boards, ds.results, sample_indices, K)
                else
                    computeMse(ds.boards, ds.results, n, K);
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
        if (do_full_eval) {
            current_full_mse = computeMse(ds.boards, ds.results, n, K);
        }

        const elapsed_ns = std.time.nanoTimestamp() - epoch_start;
        const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;

        if (do_full_eval) {
            print("\nEpoch {d:3} | delta={d:3} | improved={d:4} | sample_MSE={d:.6} | full_MSE={d:.6} | {d:.1}s\n", .{
                epoch, delta, improved, current_sample_mse, current_full_mse, elapsed_s,
            });
        } else {
            print("\nEpoch {d:3} | delta={d:3} | improved={d:4} | sample_MSE={d:.6} | full_MSE=skipped | {d:.1}s\n", .{
                epoch, delta, improved, current_sample_mse, elapsed_s,
            });
        }

        // Save params after each epoch
        try eval.saveParams(args.output, allocator);
        print("  Params saved to {s}\n", .{args.output});

        if (improved == 0) {
            delta = @divTrunc(delta, 2);
            print("  No improvement -> delta halved to {d}\n", .{delta});
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

    // Load starting params
    if (args.params) |p| {
        try eval.loadParams(p, allocator);
        print("Loaded starting params from {s}\n", .{p});
    } else {
        print("No --params given; using compiled-in defaults.\n", .{});
    }

    // Load dataset
    const ds = try loadDataset(args.dataset.?, args.positions, allocator);
    if (ds.count == 0) {
        eprint("Error: dataset is empty.\n", .{});
        std.process.exit(1);
    }

    // Tune K
    const K = args.K orelse tuneK(ds.boards, ds.results, ds.count);
    if (args.K != null) {
        print("Using provided K={d:.4}\n", .{K});
    }

    // Run coordinate descent
    try coordinateDescent(&ds, &args, K, allocator);

    try eval.saveParams(args.output, allocator);
    print("\nTuning complete. Final params saved to {s}\n", .{args.output});
    print("Apply to engine with:\n  python utils/tuning/apply_params.py {s}\n", .{args.output});
}
