const std = @import("std");
const board_mod = @import("bitboard.zig");
const Board = board_mod.Board;
const Move = board_mod.Move;
const MoveList = board_mod.MoveList;
const piece = @import("piece.zig");
const eval = @import("evaluation.zig");
const search_mod = @import("search.zig");
const SearchEngine = search_mod.SearchEngine;
const SearchOptions = search_mod.SearchOptions;
const TranspositionTable = search_mod.TranspositionTable;

pub const Options = struct {
    output: []const u8,
    games: u32 = 10000,
    depth: u32 = 8,
    random_plies: u32 = 10,
    min_ply: u32 = 16,
    max_ply: u32 = 400,
    seed: u64 = 1,
    sample_pct: u32 = 25,
    report_interval: u32 = 100,
};

/// BulletFormat binary record: 32 bytes, matches DirectSequentialDataLoader.
///
/// Offset  Field     Type      Bytes
/// ------  --------  --------  -----
/// 0-7     occ       u64       8      Occupancy bitboard
/// 8-23    pcs       [16]u8    16     Piece nibbles (2 per byte)
/// 24-25   score     i16       2      Eval in cp, white-relative
/// 26      result    u8        1      0=Black win, 1=Draw, 2=White win
/// 27      ksq       u8        1      STM king square
/// 28      opp_ksq   u8        1      Opponent king square
/// 29-31   extra     [3]u8     3      Reserved (zero)
const BulletRecord = extern struct {
    occ: u64,
    pcs: [16]u8,
    score: i16,
    result: u8,
    ksq: u8,
    opp_ksq: u8,
    extra: [3]u8 = .{ 0, 0, 0 },
};
comptime {
    std.debug.assert(@sizeOf(BulletRecord) == 32);
}

/// Piece nibble encoding: 0bCPPP
///   C=0 white, C=1 black
///   PPP: 0=pawn 1=knight 2=bishop 3=rook 4=queen 5=king
fn pieceNibble(color: piece.Color, pt: piece.Type) u4 {
    const ppp: u4 = switch (pt) {
        .pawn => 0,
        .knight => 1,
        .bishop => 2,
        .rook => 3,
        .queen => 4,
        .king => 5,
    };
    const c: u4 = if (color == .black) 8 else 0;
    return c | ppp;
}

fn boardToBulletRecord(b: *Board, score_white_cp: i16) BulletRecord {
    const state = b.board;
    var rec: BulletRecord = .{
        .occ = state.occupied(),
        .pcs = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        .score = score_white_cp,
        .result = 1, // placeholder, set later
        .ksq = 0,
        .opp_ksq = 0,
    };

    // Walk occupied squares LSB→MSB, pack nibbles
    var occ = rec.occ;
    var idx: u8 = 0;
    while (occ != 0) {
        const sq: u8 = @intCast(@ctz(occ));
        occ &= occ - 1;

        // Determine which piece is on this square
        var nibble: u4 = 0;
        var found = false;
        inline for ([_]piece.Color{ .white, .black }) |color| {
            if (!found and (state.getColorBitboard(color) & (@as(u64, 1) << @intCast(sq))) != 0) {
                inline for ([_]piece.Type{ .pawn, .knight, .bishop, .rook, .queen, .king }) |pt| {
                    if (!found and (state.getKindBitboard(pt) & (@as(u64, 1) << @intCast(sq))) != 0) {
                        nibble = pieceNibble(color, pt);
                        found = true;
                    }
                }
            }
        }

        // Pack: even index → lower nibble, odd index → upper nibble
        const byte_idx = idx / 2;
        if (idx % 2 == 0) {
            rec.pcs[byte_idx] = nibble;
        } else {
            rec.pcs[byte_idx] |= @as(u8, nibble) << 4;
        }
        idx += 1;
    }

    // King squares — STM and opponent
    const stm = state.move;
    const nstm: piece.Color = if (stm == .white) .black else .white;
    const stm_king_bb = state.getColorBitboard(stm) & state.getKindBitboard(.king);
    const nstm_king_bb = state.getColorBitboard(nstm) & state.getKindBitboard(.king);
    rec.ksq = @intCast(@ctz(stm_king_bb));
    rec.opp_ksq = @intCast(@ctz(nstm_king_bb));

    return rec;
}

/// Game result encoding for BulletFormat.
const RESULT_BLACK_WIN: u8 = 0;
const RESULT_DRAW: u8 = 1;
const RESULT_WHITE_WIN: u8 = 2;

fn eprint(comptime fmt: []const u8, args: anytype) void {
    var buf: [4096]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch return;
    std.fs.File.stderr().writeAll(s) catch {};
}

pub fn run(opts: Options, allocator: std.mem.Allocator) !void {

    // Open output file
    const file = try std.fs.cwd().createFile(opts.output, .{});
    defer file.close();
    var buf_storage: [8192]u8 = undefined;
    var bw = file.writer(&buf_storage);

    // Init PRNG
    var rng = std.Random.DefaultPrng.init(opts.seed);
    const random = rng.random();

    // Init TT (16MB, shared across games)
    var tt = try TranspositionTable.init(allocator, 16);
    defer tt.deinit();

    var stop_flag = std.atomic.Value(bool).init(false);
    var total_positions: u64 = 0;

    eprint("gensfen: starting {d} games, depth={d}, random_plies={d}, sample_pct={d}%\n", .{
        opts.games, opts.depth, opts.random_plies, opts.sample_pct,
    });

    var game_num: u32 = 0;
    while (game_num < opts.games) : (game_num += 1) {
        var b = Board.startpos();
        var game_hashes: [512]u64 = undefined;
        var hash_count: usize = 0;

        // Buffer positions until game result is known
        var positions = std.ArrayList(BulletRecord).empty;
        defer positions.deinit(allocator);

        // Random opening phase
        var random_ply: u32 = 0;
        while (random_ply < opts.random_plies) : (random_ply += 1) {
            var legal_moves = MoveList.init();
            b.generateLegalMoves(&legal_moves) catch break;
            if (legal_moves.count == 0) break;

            const move = legal_moves.moves[random.intRangeLessThan(usize, 0, legal_moves.count)];
            _ = b.makeMoveWithUndoUnchecked(move);

            if (hash_count < game_hashes.len) {
                game_hashes[hash_count] = b.zobrist_hasher.zobrist_hash;
                hash_count += 1;
            }
        }

        // Self-play with search
        var ply: u32 = random_ply;
        var game_result: ?u8 = null;
        var adj_count: u32 = 0; // consecutive plies with |score| > 3000
        const ADJ_THRESHOLD: i32 = 3000;
        const ADJ_COUNT_NEEDED: u32 = 5;

        while (game_result == null) {
            var legal_moves = MoveList.init();
            b.generateLegalMoves(&legal_moves) catch {
                game_result = RESULT_DRAW;
                break;
            };

            if (legal_moves.count == 0) {
                // Checkmate or stalemate
                const stm = b.board.move;
                if (b.isInCheck(stm)) {
                    // Checkmate — side to move lost
                    game_result = if (stm == .white) RESULT_BLACK_WIN else RESULT_WHITE_WIN;
                } else {
                    game_result = RESULT_DRAW;
                }
                break;
            }

            // 50-move rule
            if (b.board.halfmove_clock >= 100) {
                game_result = RESULT_DRAW;
                break;
            }

            // Repetition detection
            if (hash_count >= 4) {
                const current_hash = b.zobrist_hasher.zobrist_hash;
                var rep_count: u32 = 0;
                var i: usize = 0;
                while (i < hash_count) : (i += 1) {
                    if (game_hashes[i] == current_hash) rep_count += 1;
                }
                if (rep_count >= 2) {
                    game_result = RESULT_DRAW;
                    break;
                }
            }

            // Search
            stop_flag.store(false, .seq_cst);
            var search_board = b;
            var engine = SearchEngine.init(&search_board, allocator, &stop_flag, &tt, true, null, 100, 100);
            engine.uci_output = null;
            if (hash_count > 0) {
                engine.setGameHistory(game_hashes[0..hash_count]);
            }

            const search_result = engine.search(.{
                .depth = opts.depth,
            }) catch {
                game_result = RESULT_DRAW;
                break;
            };

            const score_stm = search_result.score; // stm-relative

            // Score adjudication
            if (score_stm > ADJ_THRESHOLD or score_stm < -ADJ_THRESHOLD) {
                adj_count += 1;
            } else {
                adj_count = 0;
            }

            if (adj_count >= ADJ_COUNT_NEEDED) {
                // Adjudicate: side with positive eval wins
                const stm = b.board.move;
                if (score_stm > 0) {
                    game_result = if (stm == .white) RESULT_WHITE_WIN else RESULT_BLACK_WIN;
                } else {
                    game_result = if (stm == .white) RESULT_BLACK_WIN else RESULT_WHITE_WIN;
                }
                break;
            }

            // Convert score to white-relative for recording
            const score_white: i16 = blk: {
                const clamped = std.math.clamp(score_stm, -32000, 32000);
                if (b.board.move == .white) {
                    break :blk @intCast(clamped);
                } else {
                    break :blk @intCast(-clamped);
                }
            };

            // Maybe record position
            if (ply >= opts.min_ply and ply <= opts.max_ply) {
                const stm = b.board.move;
                if (!b.isInCheck(stm)) {
                    if (random.intRangeLessThan(u32, 0, 100) < opts.sample_pct) {
                        try positions.append(allocator, boardToBulletRecord(&b, score_white));
                    }
                }
            }

            // Make the best move
            _ = b.makeMoveWithUndoUnchecked(search_result.best_move);
            if (hash_count < game_hashes.len) {
                game_hashes[hash_count] = b.zobrist_hasher.zobrist_hash;
                hash_count += 1;
            }
            ply += 1;

            // Safety: max ply
            if (ply >= opts.max_ply) {
                game_result = RESULT_DRAW;
                break;
            }
        }

        // Write all buffered positions with final game result
        const result = game_result orelse RESULT_DRAW;
        for (positions.items) |*rec| {
            rec.result = result;
            const bytes: *const [32]u8 = @ptrCast(rec);
            try bw.interface.writeAll(bytes);
        }
        total_positions += positions.items.len;

        // Progress report
        if (opts.report_interval > 0 and (game_num + 1) % opts.report_interval == 0) {
            eprint("gensfen: game {d}/{d}, {d} positions written\n", .{
                game_num + 1, opts.games, total_positions,
            });
        }

        // Clear TT every 1000 games to avoid staleness
        if ((game_num + 1) % 1000 == 0) {
            tt.clear();
        }
    }

    bw.interface.flush() catch {};

    eprint("gensfen: done. {d} games, {d} positions written to {s}\n", .{
        opts.games, total_positions, opts.output,
    });
}

/// Parse gensfen CLI arguments into Options.
pub fn parseArgs(args: []const []const u8) !Options {
    var opts = Options{ .output = "data/train.data" };

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--output") or std.mem.eql(u8, arg, "-o")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.output = args[i];
        } else if (std.mem.eql(u8, arg, "--games")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.games = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidArgument;
        } else if (std.mem.eql(u8, arg, "--depth")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.depth = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidArgument;
        } else if (std.mem.eql(u8, arg, "--random-plies")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.random_plies = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidArgument;
        } else if (std.mem.eql(u8, arg, "--min-ply")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.min_ply = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidArgument;
        } else if (std.mem.eql(u8, arg, "--max-ply")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.max_ply = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidArgument;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.seed = std.fmt.parseInt(u64, args[i], 10) catch return error.InvalidArgument;
        } else if (std.mem.eql(u8, arg, "--sample-pct")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.sample_pct = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidArgument;
        } else if (std.mem.eql(u8, arg, "--report-interval")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.report_interval = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidArgument;
        } else {
            eprint("gensfen: unknown argument: {s}\n", .{arg});
            return error.InvalidArgument;
        }
    }

    return opts;
}
