const std = @import("std");
const Board = @import("bitboard.zig").Board;
const nnue = @import("nnue.zig");
const search = @import("search.zig");

pub const BenchError = error{BenchFailed};

const BENCH_DEPTH: u64 = 10;
const BENCH_HASH_MB: usize = 16;

const POSITIONS = [_][]const u8{
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bq1rk1/pppn1ppp/2pbpn2/3p4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 9",
    "2rq1rk1/1b2bppp/p2ppn2/1pn5/3NP3/1BN1B3/PPQ2PPP/2RR2K1 w - - 2 16",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/8/2p5/2P2kp1/p1KP4/1p3PpP/1P4P1/8 w - - 8 66",
};

/// Deterministic fixed-depth benchmark used by OpenBench for build identity
/// and worker speed normalization. The final line intentionally matches the
/// node/NPS patterns parsed by OpenBench's Client/bench.py.
pub fn run(allocator: std.mem.Allocator) BenchError!void {
    var network = nnue.Network.loadFromBytes(allocator, nnue.EMBEDDED_NET) catch return error.BenchFailed;
    defer network.deinit();

    var tt = search.TranspositionTable.init(allocator, BENCH_HASH_MB) catch return error.BenchFailed;
    defer tt.deinit();

    var stop_flag = std.atomic.Value(bool).init(false);
    const start = std.time.Instant.now() catch return error.BenchFailed;
    var total_nodes: u64 = 0;

    for (POSITIONS) |fen| {
        tt.clear();
        stop_flag.store(false, .seq_cst);

        var board = Board.fromFen(fen) catch return error.BenchFailed;
        var engine = search.SearchEngine.init(
            &board,
            allocator,
            &stop_flag,
            &tt,
            true,
            &network,
            100,
            100,
        ) catch return error.BenchFailed;
        defer engine.deinit();

        const result = engine.search(.{ .depth = BENCH_DEPTH }) catch return error.BenchFailed;
        total_nodes += @intCast(result.nodes);
    }

    const elapsed_ns = (std.time.Instant.now() catch return error.BenchFailed).since(start);
    const safe_elapsed_ns = @max(elapsed_ns, 1);
    const nps_wide = (@as(u128, total_nodes) * std.time.ns_per_s) / safe_elapsed_ns;
    const nps: u64 = @intCast(@min(nps_wide, std.math.maxInt(u64)));

    var stdout_buffer: [256]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    stdout_writer.interface.print("{d} nodes {d} nps\n", .{ total_nodes, nps }) catch return error.BenchFailed;
    stdout_writer.interface.flush() catch return error.BenchFailed;
}

test "OpenBench position suite is valid" {
    for (POSITIONS) |fen| {
        _ = try Board.fromFen(fen);
    }
}
