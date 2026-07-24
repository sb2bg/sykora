const std = @import("std");
const nnue = @import("nnue.zig");
const Board = @import("bitboard.zig").Board;

pub const NnueCheckError = error{NnueCheckError};

const Options = struct {
    net_path: ?[]const u8 = null,
    fens_path: ?[]const u8 = null,
    verify_incremental: bool = false,
};

fn eprint(comptime fmt: []const u8, args: anytype) void {
    var buf: [4096]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch return;
    std.fs.File.stderr().writeAll(s) catch {};
}

fn parseArgs(args: []const []const u8) !Options {
    var opts = Options{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--net")) {
            i += 1;
            if (i >= args.len) return error.NnueCheckError;
            opts.net_path = args[i];
        } else if (std.mem.eql(u8, arg, "--fens")) {
            i += 1;
            if (i >= args.len) return error.NnueCheckError;
            opts.fens_path = args[i];
        } else if (std.mem.eql(u8, arg, "--verify-incremental")) {
            opts.verify_incremental = true;
        } else {
            eprint("nnuecheck: unknown argument: {s}\n", .{arg});
            return error.NnueCheckError;
        }
    }
    if (opts.fens_path == null) {
        eprint("nnuecheck: --fens FILE is required\n", .{});
        return error.NnueCheckError;
    }
    return opts;
}

fn accumulatorsMatchWide(
    narrow: *const nnue.AccumulatorPair,
    wide: *const nnue.AccumulatorPairWide,
    hidden_size: usize,
) bool {
    for (narrow.white[0..hidden_size], wide.white[0..hidden_size]) |n, w| {
        if (@as(i32, n) != w) return false;
    }
    for (narrow.black[0..hidden_size], wide.black[0..hidden_size]) |n, w| {
        if (@as(i32, n) != w) return false;
    }
    return true;
}

fn threatAccumulatorsMatch(
    left: *const nnue.ThreatAccumulatorPair,
    right: *const nnue.ThreatAccumulatorPair,
    hidden_size: usize,
) bool {
    return std.mem.eql(i16, left.white.values[0..hidden_size], right.white.values[0..hidden_size]) and
        std.mem.eql(i16, left.black.values[0..hidden_size], right.black.values[0..hidden_size]) and
        std.mem.eql(u64, &left.color_sets, &right.color_sets) and
        std.mem.eql(u64, &left.kind_sets, &right.kind_sets) and
        std.mem.eql(u8, &left.piece_map, &right.piece_map);
}

/// Verify, for every legal move from `b`, that the incremental i16 update
/// matches a full i16 recompute AND stays bit-identical to the i32 reference
/// backend (both accumulator contents and final evals).
fn verifyIncrementalMoves(net: *const nnue.Network, b: *Board) bool {
    const root_acc = nnue.initAccumulators(net, b);
    const root_wide = nnue.initAccumulatorsWide(net, b);
    var root_threats: nnue.ThreatAccumulatorPair = undefined;
    const has_threats = net.architecture == .pairwise_mlp_threats;
    if (has_threats) root_threats = nnue.initThreatAccumulators(net, b);
    var moves = @import("bitboard.zig").MoveList.init();
    b.generateLegalMoves(&moves) catch return false;
    const hidden_size: usize = @intCast(net.ft_hidden_size);

    if (!accumulatorsMatchWide(&root_acc, &root_wide, hidden_size)) return false;
    if (has_threats and
        nnue.evaluateFromCachedAccumulators(net, &root_acc, &root_threats, b) !=
            nnue.evaluateFromAccumulators(net, &root_acc, b))
    {
        return false;
    }

    for (moves.slice()) |move| {
        const undo = b.makeMoveWithUndoUnchecked(move);
        var incremental: nnue.AccumulatorPair = undefined;
        var incremental_wide: nnue.AccumulatorPairWide = undefined;
        nnue.updateAccumulators(
            net,
            b,
            &root_acc,
            &incremental,
            move.from(),
            move.to(),
            undo.moved_piece,
            undo.mover_color,
            undo.captured_piece,
            undo.captured_square,
            move.promotion(),
            undo.castle_rook_from != null,
            undo.castle_rook_from,
            undo.castle_rook_to,
            null,
        );
        nnue.updateAccumulators(
            net,
            b,
            &root_wide,
            &incremental_wide,
            move.from(),
            move.to(),
            undo.moved_piece,
            undo.mover_color,
            undo.captured_piece,
            undo.captured_square,
            move.promotion(),
            undo.castle_rook_from != null,
            undo.castle_rook_from,
            undo.castle_rook_to,
            null,
        );
        const full = nnue.initAccumulators(net, b);
        var matches = std.mem.eql(i16, incremental.white[0..hidden_size], full.white[0..hidden_size]) and
            std.mem.eql(i16, incremental.black[0..hidden_size], full.black[0..hidden_size]) and
            accumulatorsMatchWide(&incremental, &incremental_wide, hidden_size) and
            nnue.evaluateFromAccumulators(net, &incremental, b) == nnue.evaluateFromAccumulators(net, &full, b) and
            nnue.evaluateFromAccumulators(net, &incremental, b) == nnue.evaluateFromAccumulators(net, &incremental_wide, b);
        if (has_threats) {
            var incremental_threats: nnue.ThreatAccumulatorPair = undefined;
            nnue.updateThreatAccumulators(net, b, &root_threats, &incremental_threats);
            const full_threats = nnue.initThreatAccumulators(net, b);
            matches = matches and
                threatAccumulatorsMatch(&incremental_threats, &full_threats, hidden_size) and
                nnue.evaluateFromCachedAccumulators(net, &incremental, &incremental_threats, b) ==
                    nnue.evaluateFromAccumulators(net, &incremental, b);
        }
        b.unmakeMoveUnchecked(move, undo);
        if (!matches) return false;
    }
    return true;
}

/// Emit reference NNUE evals for a FEN suite through the non-incremental
/// `evaluate` path. Output is one `eval<TAB>fen` line per input FEN.
/// Usage: sykora nnuecheck [--net FILE] --fens FILE
pub fn run(args: []const []const u8, allocator: std.mem.Allocator) NnueCheckError!void {
    const opts = parseArgs(args) catch return error.NnueCheckError;

    var net = blk: {
        if (opts.net_path) |path| {
            break :blk nnue.Network.loadFromFile(allocator, path) catch |err| {
                eprint("nnuecheck: failed to load net {s}: {s}\n", .{ path, @errorName(err) });
                return error.NnueCheckError;
            };
        } else {
            break :blk nnue.Network.loadFromBytes(allocator, nnue.EMBEDDED_NET) catch |err| {
                eprint("nnuecheck: failed to load embedded net: {s}\n", .{@errorName(err)});
                return error.NnueCheckError;
            };
        }
    };
    defer net.deinit();

    const fens_path = opts.fens_path.?;
    const file = std.fs.cwd().openFile(fens_path, .{}) catch |err| {
        eprint("nnuecheck: failed to open {s}: {s}\n", .{ fens_path, @errorName(err) });
        return error.NnueCheckError;
    };
    defer file.close();
    const data = file.readToEndAlloc(allocator, 64 * 1024 * 1024) catch return error.NnueCheckError;
    defer allocator.free(data);

    var out_buf: [8192]u8 = undefined;
    var bw = std.fs.File.stdout().writer(&out_buf);
    const out = &bw.interface;

    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0) continue;

        var board = Board.fromFen(line) catch {
            eprint("nnuecheck: invalid FEN, skipping: {s}\n", .{line});
            continue;
        };
        if (opts.verify_incremental and !verifyIncrementalMoves(&net, &board)) {
            eprint("nnuecheck: incremental/full mismatch: {s}\n", .{line});
            return error.NnueCheckError;
        }
        const eval = nnue.evaluate(&net, &board);
        out.print("{d}\t{s}\n", .{ eval, line }) catch return error.NnueCheckError;
    }

    out.flush() catch return error.NnueCheckError;
}
