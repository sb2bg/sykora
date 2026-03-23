const std = @import("std");
const board = @import("bitboard.zig");
const Board = board.Board;
const BitBoard = board.BitBoard;
const piece = @import("piece.zig");

// Non-tunable constants used by search / move ordering
pub const MATE_SCORE: i32 = 30000;
pub const MATE_BOUND: i32 = 29000;
pub const KING_VALUE: i32 = 20000; // used by move_picker for SEE

// Backward-compatible material aliases for search.zig / move_picker.zig.
// These match EvalParams defaults and are also updated by apply_params.py.
pub const PAWN_VALUE: i32 = 77;
pub const KNIGHT_VALUE: i32 = 316;
pub const BISHOP_VALUE: i32 = 331;
pub const ROOK_VALUE: i32 = 505;
pub const QUEEN_VALUE: i32 = 1043;

// ──────────────────────────────────────────────
// EvalParams – every tunable evaluation constant
// ──────────────────────────────────────────────
pub const EvalParams = struct {
    // Material values (centipawns)
    pawn_value: i32 = 77,
    knight_value: i32 = 316,
    bishop_value: i32 = 331,
    rook_value: i32 = 505,
    queen_value: i32 = 1043,

    // Scalar evaluation weights
    rook_open_file_bonus: i32 = 19,
    rook_semi_open_file_bonus: i32 = 15,
    rook_on_seventh_bonus: i32 = 26,
    connected_rooks_bonus: i32 = 7,
    isolated_pawn_penalty: i32 = 9,
    backward_pawn_penalty: i32 = 2,
    doubled_pawn_penalty: i32 = 9,
    bishop_pair_bonus: i32 = 23,
    knight_outpost_bonus: i32 = 15,
    bishop_outpost_bonus: i32 = 12,
    tempo_bonus: i32 = 10,
    king_pawn_shield_bonus: i32 = 11,
    king_open_file_penalty: i32 = 23,
    king_center_middlegame_penalty: i32 = 24,
    king_castled_bonus: i32 = 13,
    king_early_walk_penalty: i32 = 20,
    pawn_chain_bonus: i32 = 8,
    protected_passed_pawn_bonus: i32 = 9,
    connected_passed_pawn_bonus: i32 = 19,
    safe_pawn_advance_bonus: i32 = 1,
    mop_up_center_bonus: i32 = 1,
    mop_up_corner_bonus: i32 = 20,
    mop_up_king_proximity_bonus: i32 = 0,
    castling_rights_kingside_bonus: i32 = 43,
    castling_rights_queenside_bonus: i32 = 20,
    pawn_storm_advance_bonus: i32 = 0,
    pawn_storm_near_king_bonus: i32 = 4,
    king_activity_center_bonus: i32 = 1,
    king_activity_mobility_bonus: i32 = 0,
    endgame_phase_threshold: i32 = 158,

    // Piece-Square Tables (index 0 = a1, 63 = h8, white perspective)
    pawn_table: [64]i32 = .{
           0,    0,    0,    0,    0,    0,    0,    0,
          16,    9,   -2,  -29,  -27,   17,   20,    0,
          13,    2,   -1,   -4,    7,    7,   -3,   -4,
          14,   13,    7,   10,   16,    9,    6,   -1,
          24,    9,    4,   -4,   14,   17,    9,   10,
          40,   33,   22,    3,    6,   15,   22,   17,
          88,   78,   47,   38,   31,   48,    1,   45,
           0,    0,    0,    0,    0,    0,    0,    0,
    },
    knight_table: [64]i32 = .{
         -50,  -34,  -16,  -19,  -22,  -17,  -18,  -55,
         -47,  -38,   -2,    0,    2,   -7,    6,  -21,
         -31,  -11,   -5,    8,   19,    4,    1,  -28,
         -13,    4,   18,    7,   19,   16,  -16,    8,
         -13,    5,   17,   23,   14,   31,   24,   17,
          -7,    7,   18,   24,   39,   31,   10,  -26,
          -7,  -16,   13,   17,   15,   15,    0,  -41,
         -88,  -27,  -10,  -30,  -20,  -35,  -40,  -60,
    },
    bishop_table: [64]i32 = .{
         -25,  -15,  -10,   -7,  -13,  -11,  -17,  -15,
          -2,   -4,    5,  -10,   -5,    3,   13,  -21,
          -4,    1,    2,    1,   -1,    8,    4,   -4,
         -21,   -5,    1,   15,   17,   -5,    4,   -7,
         -12,    4,    2,   13,    5,    9,    8,   -3,
          -9,    4,   21,    4,   17,   26,   25,    6,
         -16,    8,   -5,  -13,   17,   -8,   -6,  -36,
         -15,   -5,    0,  -22,  -20,  -12,   -5,  -20,
    },
    rook_table: [64]i32 = .{
         -10,   -9,    0,   -4,   -1,    0,    7,  -11,
         -18,  -18,   -3,   -4,  -11,   -4,  -15,  -32,
         -23,  -19,   -7,  -11,   -4,   -7,    2,   -5,
          -3,    9,    5,   -2,  -10,    0,    2,    0,
           3,    1,   12,   16,   13,   22,    6,   -1,
          20,   27,   23,   24,   15,   29,   32,    9,
           3,   -1,    5,    9,    8,   19,   10,   17,
          24,   41,   29,   25,    9,   15,   24,   15,
    },
    queen_table: [64]i32 = .{
         -23,  -25,   -9,   -1,  -19,  -34,  -30,  -20,
         -38,  -12,    1,   -5,    1,   -5,   -7,  -33,
         -11,  -13,    2,   -2,    7,   10,    9,  -26,
         -20,   -9,    3,   15,   17,   12,   14,    0,
         -20,   -4,    8,   20,   35,   13,   20,   11,
         -14,    2,   12,   14,   36,   51,   42,   36,
          -8,    1,    4,   22,   34,   25,   29,   23,
         -33,   -6,    9,   13,   15,   12,  -20,    8,
    },
    king_middlegame_table: [64]i32 = .{
          25,   43,   34,   -9,  -18,  -16,   18,   21,
          15,   25,    0,    8,   -7,   -6,   35,   49,
         -10,  -20,  -15,  -30,  -10,  -25,  -30,  -11,
         -20,  -30,  -30,  -40,  -54,  -30,  -30,  -20,
         -30,  -40,  -40,  -50,  -50,  -40,  -40,  -30,
         -30,  -40,  -40,  -50,  -50,  -40,  -40,  -30,
         -30,  -40,  -40,  -50,  -50,  -40,  -40,  -30,
         -30,  -40,  -40,  -50,  -50,  -40,  -40,  -30,
    },
    king_endgame_table: [64]i32 = .{
         -45,  -43,  -22,  -35,  -35,  -17,  -39,  -52,
         -58,  -21,  -11,   -6,   -4,  -13,  -24,  -44,
         -30,   -5,    0,   14,   10,    3,  -12,  -30,
         -35,    0,   39,   49,   21,   13,   -9,  -23,
         -25,    0,   60,   76,   40,   35,   22,  -32,
         -30,   19,   33,   49,   40,   35,   -5,  -30,
         -30,  -20,   -5,    0,    0,  -10,  -20,  -30,
         -50,  -40,  -30,  -20,  -20,  -30,  -40,  -50,
    },

    // Passed pawn bonus by rank (rank 0–7, white perspective)
    passed_pawn_bonus: [8]i32 = .{
           0,    3,    3,   10,   37,   76,  103,    0,
    },

    // Mobility tables
    knight_mobility: [9]i32 = .{
         -17,   -1,    1,    9,   14,   21,   24,   27,
          23,
    },
    bishop_mobility: [14]i32 = .{
         -22,  -17,   -9,    2,    8,   16,   20,   23,
          25,   26,   26,   28,   36,   43,
    },
    rook_mobility: [15]i32 = .{
         -12,   -4,   -3,    1,    1,    8,   13,   15,
          18,   21,   22,   25,   24,   29,   20,
    },
};

/// Global mutable params – defaults at startup, overridden by loadParams().
pub var g_params: EvalParams = .{};

// ──────────────────────────────────────────────
// Params serialisation (for Texel tuner)
// ──────────────────────────────────────────────

/// Apply one key-value line into g_params.  Silently ignores unknown keys.
fn setParam(key: []const u8, value_str: []const u8) !void {
    inline for (std.meta.fields(EvalParams)) |field| {
        if (std.mem.eql(u8, key, field.name)) {
            switch (@typeInfo(field.type)) {
                .int => {
                    @field(g_params, field.name) = try std.fmt.parseInt(
                        i32,
                        std.mem.trim(u8, value_str, " \r\t"),
                        10,
                    );
                },
                .array => |arr_info| {
                    var arr: field.type = @field(g_params, field.name);
                    var it = std.mem.splitScalar(u8, value_str, ' ');
                    var i: usize = 0;
                    while (it.next()) |token| {
                        const t = std.mem.trim(u8, token, " \r\t");
                        if (t.len == 0) continue;
                        if (i >= arr_info.len) break;
                        arr[i] = try std.fmt.parseInt(i32, t, 10);
                        i += 1;
                    }
                    @field(g_params, field.name) = arr;
                },
                else => {},
            }
            return;
        }
    }
}

/// Load a params file (key-value, one per line) and override g_params.
/// Format: "pawn_value 100" or "pawn_table 0 0 0 ..." (64 space-separated ints)
pub fn loadParams(path: []const u8, allocator: std.mem.Allocator) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 4 * 1024 * 1024);
    defer allocator.free(content);

    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \r\t");
        if (trimmed.len == 0 or trimmed[0] == '#') continue;
        const space_idx = std.mem.indexOfScalar(u8, trimmed, ' ') orelse continue;
        const key = trimmed[0..space_idx];
        const value_str = trimmed[space_idx + 1 ..];
        setParam(key, value_str) catch {};
    }
}

/// Serialize g_params to a flat key-value text file.
pub fn saveParams(path: []const u8, allocator: std.mem.Allocator) !void {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    const writer = buf.writer(allocator);

    inline for (std.meta.fields(EvalParams)) |field| {
        const value = @field(g_params, field.name);
        switch (@typeInfo(field.type)) {
            .int => {
                try writer.print("{s} {d}\n", .{ field.name, value });
            },
            .array => |arr_info| {
                try writer.print("{s}", .{field.name});
                for (0..arr_info.len) |i| {
                    try writer.print(" {d}", .{value[i]});
                }
                try writer.print("\n", .{});
            },
            else => {},
        }
    }

    const out = try std.fs.cwd().createFile(path, .{});
    defer out.close();
    try out.writeAll(buf.items);
}

// ──────────────────────────────────────────────
// Internal helpers (unchanged logic, using g_params)
// ──────────────────────────────────────────────

/// Mirror a square index for black pieces
inline fn mirrorSquare(square: u8) u8 {
    return square ^ 56;
}

/// Get piece-square table value for a given piece at a square
fn getPieceSquareValue(piece_type: piece.Type, color: piece.Color, square: u8, is_endgame: bool) i32 {
    const sq = if (color == .white) square else mirrorSquare(square);

    return switch (piece_type) {
        .pawn => g_params.pawn_table[sq],
        .knight => g_params.knight_table[sq],
        .bishop => g_params.bishop_table[sq],
        .rook => g_params.rook_table[sq],
        .queen => g_params.queen_table[sq],
        .king => if (is_endgame) g_params.king_endgame_table[sq] else g_params.king_middlegame_table[sq],
    };
}

/// Get material value for a piece (uses g_params so tuner sees consistent values)
pub fn getPieceValue(piece_type: piece.Type) i32 {
    return switch (piece_type) {
        .pawn => g_params.pawn_value,
        .knight => g_params.knight_value,
        .bishop => g_params.bishop_value,
        .rook => g_params.rook_value,
        .queen => g_params.queen_value,
        .king => 0,
    };
}

/// Count total material for a color
fn countMaterial(b: BitBoard, color: piece.Color) i32 {
    const color_bb = b.getColorBitboard(color);
    var material: i32 = 0;

    material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.pawn)))) * g_params.pawn_value;
    material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.knight)))) * g_params.knight_value;
    material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.bishop)))) * g_params.bishop_value;
    material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.rook)))) * g_params.rook_value;
    material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.queen)))) * g_params.queen_value;

    return material;
}

/// Determine if we're in the endgame phase
fn isEndgame(b: BitBoard) bool {
    return getGamePhase(b) >= g_params.endgame_phase_threshold;
}

/// Calculate game phase (0 = opening, 256 = endgame)
fn getGamePhase(b: BitBoard) i32 {
    const knight_phase: i32 = 1;
    const bishop_phase: i32 = 1;
    const rook_phase: i32 = 2;
    const queen_phase: i32 = 4;
    const total_phase: i32 = 4 * knight_phase + 4 * bishop_phase + 4 * rook_phase + 2 * queen_phase;

    var phase: i32 = total_phase;
    phase -= @as(i32, @intCast(@popCount(b.getKindBitboard(.knight)))) * knight_phase;
    phase -= @as(i32, @intCast(@popCount(b.getKindBitboard(.bishop)))) * bishop_phase;
    phase -= @as(i32, @intCast(@popCount(b.getKindBitboard(.rook)))) * rook_phase;
    phase -= @as(i32, @intCast(@popCount(b.getKindBitboard(.queen)))) * queen_phase;

    phase = @max(0, @min(phase, total_phase));
    return @divTrunc(phase * 256 + @divTrunc(total_phase, 2), total_phase);
}

/// Evaluate pawn structure
fn evaluatePawnStructure(b: BitBoard, color: piece.Color) i32 {
    const pawns = b.getColorBitboard(color) & b.getKindBitboard(.pawn);
    const opponent_pawns = b.getColorBitboard(if (color == .white) .black else .white) & b.getKindBitboard(.pawn);
    var score: i32 = 0;

    const file_masks = [8]u64{
        0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
        0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
    };

    // Penalize doubled pawns
    for (0..8) |file| {
        const pawns_on_file = @popCount(pawns & file_masks[file]);
        if (pawns_on_file > 1) {
            score -= @as(i32, @intCast(pawns_on_file - 1)) * g_params.doubled_pawn_penalty;
        }
    }

    var pawn_bb = pawns;
    while (pawn_bb != 0) {
        const sq: u8 = @intCast(@ctz(pawn_bb));
        pawn_bb &= pawn_bb - 1;

        const file = sq % 8;
        const rank = sq / 8;

        var adjacent_files: u64 = 0;
        if (file > 0) adjacent_files |= file_masks[file - 1];
        if (file < 7) adjacent_files |= file_masks[file + 1];

        // Isolated pawn
        if ((pawns & adjacent_files) == 0) {
            score -= g_params.isolated_pawn_penalty;
        }

        // Pawn chain
        const pawn_attacks = board.getPawnAttacks(@intCast(sq), if (color == .white) .black else .white);
        if ((pawns & pawn_attacks) != 0) {
            score += g_params.pawn_chain_bonus;
        }

        // Mask squares ahead of this pawn
        var blocking_mask: u64 = file_masks[file];
        if (file > 0) blocking_mask |= file_masks[file - 1];
        if (file < 7) blocking_mask |= file_masks[file + 1];

        const ahead_shift_white: u6 = if (rank >= 7) 63 else @intCast((rank + 1) * 8);
        const ahead_shift_black: u6 = if (rank == 0) 63 else @intCast((8 - rank) * 8);
        const ahead_mask: u64 = if (color == .white)
            blocking_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) << ahead_shift_white)
        else
            blocking_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) >> ahead_shift_black);

        // Backward pawn
        const behind_shift_white: u6 = if (rank == 0) 63 else @intCast((8 - rank) * 8);
        const behind_shift_black: u6 = @intCast(rank * 8);
        const behind_mask: u64 = if (color == .white)
            adjacent_files & (@as(u64, 0xFFFFFFFFFFFFFFFF) >> behind_shift_white)
        else
            adjacent_files & (@as(u64, 0xFFFFFFFFFFFFFFFF) << behind_shift_black);

        if ((pawns & behind_mask) == 0 and (pawns & adjacent_files) == 0) {
            const front_sq: u6 = if (color == .white) @intCast(sq + 8) else @intCast(sq -| 8);
            const attackers_squares = board.getPawnAttacks(front_sq, color);
            if ((opponent_pawns & attackers_squares) != 0) {
                score -= g_params.backward_pawn_penalty;
            }
        }

        // Passed pawn
        if ((opponent_pawns & ahead_mask) == 0) {
            const advancement: usize = if (color == .white) rank else 7 - rank;
            score += g_params.passed_pawn_bonus[advancement];

            const our_pawn_attacks = board.getPawnAttacks(@intCast(sq), if (color == .white) .black else .white);
            if ((pawns & our_pawn_attacks) != 0) {
                score += g_params.protected_passed_pawn_bonus;
            }

            // Connected passed pawn
            const adj_passed_mask = adjacent_files & pawns;
            if (adj_passed_mask != 0) {
                var adj_bb = adj_passed_mask;
                while (adj_bb != 0) {
                    const adj_sq: u8 = @intCast(@ctz(adj_bb));
                    adj_bb &= adj_bb - 1;
                    const adj_file = adj_sq % 8;
                    const adj_rank = adj_sq / 8;

                    var adj_blocking: u64 = file_masks[adj_file];
                    if (adj_file > 0) adj_blocking |= file_masks[adj_file - 1];
                    if (adj_file < 7) adj_blocking |= file_masks[adj_file + 1];

                    const adj_ahead_shift_white: u6 = if (adj_rank >= 7) 63 else @intCast((adj_rank + 1) * 8);
                    const adj_ahead_shift_black: u6 = if (adj_rank == 0) 63 else @intCast((8 - adj_rank) * 8);
                    const adj_ahead: u64 = if (color == .white)
                        adj_blocking & (@as(u64, 0xFFFFFFFFFFFFFFFF) << adj_ahead_shift_white)
                    else
                        adj_blocking & (@as(u64, 0xFFFFFFFFFFFFFFFF) >> adj_ahead_shift_black);

                    if ((opponent_pawns & adj_ahead) == 0) {
                        score += @divTrunc(g_params.connected_passed_pawn_bonus, 2);
                        break;
                    }
                }
            }
        }

        // Safe pawn advance bonus
        if ((color == .white and rank < 7) or (color == .black and rank > 0)) {
            const front_sq_safe: u6 = if (color == .white) @intCast(sq + 8) else @intCast(sq - 8);
            const attackers_squares = board.getPawnAttacks(front_sq_safe, color);
            if ((opponent_pawns & attackers_squares) == 0) {
                score += g_params.safe_pawn_advance_bonus;
            }
        }
    }

    return score;
}

/// Evaluate king safety
fn evaluateKingSafety(b: BitBoard, color: piece.Color, is_endgame_phase: bool) i32 {
    if (is_endgame_phase) return 0;

    const king_bb = b.getColorBitboard(color) & b.getKindBitboard(.king);
    if (king_bb == 0) return 0;

    const king_sq: u8 = @intCast(@ctz(king_bb));
    const king_file = king_sq % 8;
    const king_rank = king_sq / 8;

    var score: i32 = 0;
    const pawns = b.getColorBitboard(color) & b.getKindBitboard(.pawn);
    const all_pawns = b.getKindBitboard(.pawn);

    const file_masks = [8]u64{
        0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
        0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
    };

    const shield_ranks = if (color == .white) [_]u8{ king_rank + 1, king_rank + 2 } else [_]u8{ king_rank -| 1, king_rank -| 2 };

    for (shield_ranks) |rank| {
        if (rank > 7) continue;
        for (0..3) |i| {
            const file = king_file +% i -% 1;
            if (file > 7) continue;
            const sq = rank * 8 + file;
            if ((pawns & (@as(u64, 1) << @intCast(sq))) != 0) {
                score += g_params.king_pawn_shield_bonus;
            }
        }
    }

    for (0..3) |i| {
        const file = king_file +% i -% 1;
        if (file > 7) continue;
        const file_mask = file_masks[file];
        if ((all_pawns & file_mask) == 0) {
            score -= g_params.king_open_file_penalty;
        } else if ((pawns & file_mask) == 0) {
            score -= @divTrunc(g_params.king_open_file_penalty, 2);
        }
    }

    if (king_file >= 2 and king_file <= 5) {
        score -= g_params.king_center_middlegame_penalty;
    }

    const is_castled = if (color == .white)
        (king_sq == 6 or king_sq == 2)
    else
        (king_sq == 62 or king_sq == 58);

    const home_sq: u8 = if (color == .white) 4 else 60;
    const king_has_moved = king_sq != home_sq;

    if (is_castled) {
        score += g_params.king_castled_bonus;
    } else if (king_has_moved) {
        score -= g_params.king_early_walk_penalty;
        const advanced_rank = if (color == .white) king_rank >= 2 else king_rank <= 5;
        if (advanced_rank) {
            score -= @divTrunc(g_params.king_early_walk_penalty, 2);
        }
    }

    return score;
}

/// Evaluate king activity in endgames
fn evaluateKingActivity(b: BitBoard, color: piece.Color, is_endgame_phase: bool) i32 {
    if (!is_endgame_phase) return 0;

    const king_bb = b.getColorBitboard(color) & b.getKindBitboard(.king);
    if (king_bb == 0) return 0;

    const king_sq: u8 = @intCast(@ctz(king_bb));
    const king_file: i32 = @intCast(king_sq % 8);
    const king_rank: i32 = @intCast(king_sq / 8);

    const d1: i32 = @intCast(@abs(king_file - 3) + @abs(king_rank - 3));
    const d2: i32 = @intCast(@abs(king_file - 4) + @abs(king_rank - 3));
    const d3: i32 = @intCast(@abs(king_file - 3) + @abs(king_rank - 4));
    const d4: i32 = @intCast(@abs(king_file - 4) + @abs(king_rank - 4));
    const center_dist: i32 = @min(@min(d1, d2), @min(d3, d4));

    var score: i32 = @max(0, 6 - center_dist) * g_params.king_activity_center_bonus;

    const friendly = b.getColorBitboard(color);
    const attacks = board.getKingAttacks(@intCast(king_sq)) & ~friendly;
    score += @as(i32, @intCast(@popCount(attacks))) * g_params.king_activity_mobility_bonus;

    return score;
}

/// Evaluate wing pawn storms
fn evaluatePawnStorm(b: BitBoard, color: piece.Color) i32 {
    const enemy = if (color == .white) piece.Color.black else piece.Color.white;
    const enemy_king_bb = b.getColorBitboard(enemy) & b.getKindBitboard(.king);
    if (enemy_king_bb == 0) return 0;

    const enemy_king_sq: u8 = @intCast(@ctz(enemy_king_bb));
    const enemy_king_file: i32 = @intCast(enemy_king_sq % 8);
    const enemy_king_rank: i32 = @intCast(enemy_king_sq / 8);

    if (enemy_king_file >= 3 and enemy_king_file <= 4) return 0;

    const own_king_bb = b.getColorBitboard(color) & b.getKindBitboard(.king);
    const own_king_file: i32 = if (own_king_bb != 0)
        @intCast((@as(u8, @intCast(@ctz(own_king_bb)))) % 8)
    else
        4;

    const attacking_kingside = enemy_king_file >= 5;
    const same_side_kings = (attacking_kingside and own_king_file >= 5) or
        (!attacking_kingside and own_king_file <= 2);

    const pawns = b.getColorBitboard(color) & b.getKindBitboard(.pawn);
    var pawn_bb = pawns;
    var score: i32 = 0;

    while (pawn_bb != 0) {
        const sq: u8 = @intCast(@ctz(pawn_bb));
        pawn_bb &= pawn_bb - 1;

        const file: i32 = @intCast(sq % 8);
        const rank: i32 = @intCast(sq / 8);
        const advancement = if (color == .white) rank - 1 else 6 - rank;
        if (advancement <= 0) continue;

        if (attacking_kingside) {
            if (file < 4) continue;
        } else {
            if (file > 3) continue;
        }

        score += advancement * g_params.pawn_storm_advance_bonus;

        const file_dist: i32 = @intCast(@abs(file - enemy_king_file));
        const rank_dist: i32 = @intCast(@abs(rank - enemy_king_rank));
        if (file_dist <= 2 and rank_dist <= 3) {
            const proximity_bonus = g_params.pawn_storm_near_king_bonus - (file_dist * 2 + rank_dist);
            if (proximity_bonus > 0) {
                score += proximity_bonus;
            }
        }
    }

    if (same_side_kings) {
        score = @divTrunc(score, 2);
    }

    return score;
}

/// Evaluate piece mobility
fn evaluateMobility(b: BitBoard, color: piece.Color) i32 {
    var score: i32 = 0;
    const occupied = b.getColorBitboard(.white) | b.getColorBitboard(.black);
    const friendly = b.getColorBitboard(color);
    const enemy_pawns = b.getColorBitboard(if (color == .white) .black else .white) & b.getKindBitboard(.pawn);

    var enemy_pawn_attacks: u64 = 0;
    var pawn_bb = enemy_pawns;
    while (pawn_bb != 0) {
        const sq: u6 = @intCast(@ctz(pawn_bb));
        pawn_bb &= pawn_bb - 1;
        enemy_pawn_attacks |= board.getPawnAttacks(sq, if (color == .white) .black else .white);
    }

    var knights = friendly & b.getKindBitboard(.knight);
    while (knights != 0) {
        const sq: u6 = @intCast(@ctz(knights));
        knights &= knights - 1;
        const attacks = board.getKnightAttacks(sq) & ~friendly & ~enemy_pawn_attacks;
        const mobility = @min(@popCount(attacks), 8);
        score += g_params.knight_mobility[mobility];
    }

    var bishops = friendly & b.getKindBitboard(.bishop);
    while (bishops != 0) {
        const sq: u6 = @intCast(@ctz(bishops));
        bishops &= bishops - 1;
        const attacks = board.getBishopAttacks(sq, occupied) & ~friendly;
        const mobility = @min(@popCount(attacks), 13);
        score += g_params.bishop_mobility[mobility];
    }

    var rooks = friendly & b.getKindBitboard(.rook);
    while (rooks != 0) {
        const sq: u6 = @intCast(@ctz(rooks));
        rooks &= rooks - 1;
        const attacks = board.getRookAttacks(sq, occupied) & ~friendly;
        const mobility = @min(@popCount(attacks), 14);
        score += g_params.rook_mobility[mobility];
    }

    return score;
}

/// Evaluate rook positioning
fn evaluateRooks(b: BitBoard, color: piece.Color) i32 {
    var score: i32 = 0;
    const rooks = b.getColorBitboard(color) & b.getKindBitboard(.rook);
    const friendly_pawns = b.getColorBitboard(color) & b.getKindBitboard(.pawn);
    const enemy_pawns = b.getColorBitboard(if (color == .white) .black else .white) & b.getKindBitboard(.pawn);
    const occupied = b.getColorBitboard(.white) | b.getColorBitboard(.black);

    const file_masks = [8]u64{
        0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
        0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
    };

    var rook_bb = rooks;
    while (rook_bb != 0) {
        const sq: u6 = @intCast(@ctz(rook_bb));
        rook_bb &= rook_bb - 1;

        const file = sq % 8;
        const rank = sq / 8;
        const file_mask = file_masks[file];

        if ((friendly_pawns & file_mask) == 0) {
            if ((enemy_pawns & file_mask) == 0) {
                score += g_params.rook_open_file_bonus;
            } else {
                score += g_params.rook_semi_open_file_bonus;
            }
        }

        const seventh_rank: u8 = if (color == .white) 6 else 1;
        if (rank == seventh_rank) {
            score += g_params.rook_on_seventh_bonus;
        }

        const rook_attacks = board.getRookAttacks(sq, occupied);
        if ((rook_attacks & rooks) != 0) {
            score += @divTrunc(g_params.connected_rooks_bonus, 2);
        }
    }

    return score;
}

/// Evaluate knight/bishop outposts
fn evaluateOutposts(b: BitBoard, color: piece.Color) i32 {
    var score: i32 = 0;
    const knights = b.getColorBitboard(color) & b.getKindBitboard(.knight);
    const bishops = b.getColorBitboard(color) & b.getKindBitboard(.bishop);
    const friendly_pawns = b.getColorBitboard(color) & b.getKindBitboard(.pawn);
    const enemy_pawns = b.getColorBitboard(if (color == .white) .black else .white) & b.getKindBitboard(.pawn);

    const file_masks = [8]u64{
        0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
        0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
    };

    var piece_bb = knights;
    while (piece_bb != 0) {
        const sq: u6 = @intCast(@ctz(piece_bb));
        piece_bb &= piece_bb - 1;

        const file = sq % 8;
        const rank = sq / 8;

        const in_outpost_zone = if (color == .white) rank >= 3 and rank <= 5 else rank >= 2 and rank <= 4;
        if (!in_outpost_zone) continue;

        const pawn_attacks = board.getPawnAttacks(sq, if (color == .white) .black else .white);
        if ((friendly_pawns & pawn_attacks) == 0) continue;

        var attack_mask: u64 = 0;
        if (file > 0) attack_mask |= file_masks[file - 1];
        if (file < 7) attack_mask |= file_masks[file + 1];

        const knight_ahead_shift_white: u6 = if (rank >= 7) 63 else @intCast((rank + 1) * 8);
        const knight_ahead_shift_black: u6 = if (rank == 0) 63 else @intCast((8 - rank) * 8);
        const ahead_mask: u64 = if (color == .white)
            attack_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) << knight_ahead_shift_white)
        else
            attack_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) >> knight_ahead_shift_black);

        if ((enemy_pawns & ahead_mask) == 0) {
            score += g_params.knight_outpost_bonus;
        }
    }

    piece_bb = bishops;
    while (piece_bb != 0) {
        const sq: u6 = @intCast(@ctz(piece_bb));
        piece_bb &= piece_bb - 1;

        const file = sq % 8;
        const rank = sq / 8;

        const in_outpost_zone = if (color == .white) rank >= 3 and rank <= 5 else rank >= 2 and rank <= 4;
        if (!in_outpost_zone) continue;

        const pawn_attacks = board.getPawnAttacks(sq, if (color == .white) .black else .white);
        if ((friendly_pawns & pawn_attacks) == 0) continue;

        var attack_mask: u64 = 0;
        if (file > 0) attack_mask |= file_masks[file - 1];
        if (file < 7) attack_mask |= file_masks[file + 1];

        const bishop_ahead_shift_white: u6 = if (rank >= 7) 63 else @intCast((rank + 1) * 8);
        const bishop_ahead_shift_black: u6 = if (rank == 0) 63 else @intCast((8 - rank) * 8);
        const bishop_ahead_mask: u64 = if (color == .white)
            attack_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) << bishop_ahead_shift_white)
        else
            attack_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) >> bishop_ahead_shift_black);

        if ((enemy_pawns & bishop_ahead_mask) == 0) {
            score += g_params.bishop_outpost_bonus;
        }
    }

    return score;
}

/// Endgame mop-up heuristics
fn evaluateMopUp(b: BitBoard, winning_color: piece.Color) i32 {
    const losing_color = if (winning_color == .white) piece.Color.black else piece.Color.white;

    const winning_king_bb = b.getColorBitboard(winning_color) & b.getKindBitboard(.king);
    const losing_king_bb = b.getColorBitboard(losing_color) & b.getKindBitboard(.king);

    if (winning_king_bb == 0 or losing_king_bb == 0) return 0;

    const winning_king_sq: u8 = @intCast(@ctz(winning_king_bb));
    const losing_king_sq: u8 = @intCast(@ctz(losing_king_bb));

    var score: i32 = 0;

    const losing_file: i32 = @intCast(losing_king_sq % 8);
    const losing_rank: i32 = @intCast(losing_king_sq / 8);
    const center_dist_file = @max(3 - losing_file, losing_file - 4);
    const center_dist_rank = @max(3 - losing_rank, losing_rank - 4);
    score += (center_dist_file + center_dist_rank) * g_params.mop_up_center_bonus;

    const corner_dist = @min(
        @min(losing_file + losing_rank, losing_file + (7 - losing_rank)),
        @min((7 - losing_file) + losing_rank, (7 - losing_file) + (7 - losing_rank)),
    );
    score += @divTrunc((7 - corner_dist) * g_params.mop_up_corner_bonus, 7);

    const winning_file: i32 = @intCast(winning_king_sq % 8);
    const winning_rank: i32 = @intCast(winning_king_sq / 8);
    const king_dist: i32 = @intCast(@abs(winning_file - losing_file) + @abs(winning_rank - losing_rank));
    score += (14 - king_dist) * g_params.mop_up_king_proximity_bonus;

    return score;
}

// ──────────────────────────────────────────────
// Public evaluation API
// ──────────────────────────────────────────────

/// Main evaluation function.
/// Returns score from the perspective of the side to move.
pub fn evaluate(b: *Board) i32 {
    const board_state = b.board;
    const side_to_move = board_state.move;

    const is_endgame_phase = isEndgame(board_state);
    const phase = getGamePhase(board_state);

    const white_material = countMaterial(board_state, .white);
    const black_material = countMaterial(board_state, .black);
    var score = white_material - black_material;

    // Tapered PST evaluation
    var mg_pst_score: i32 = 0;
    var eg_pst_score: i32 = 0;

    for (0..64) |sq| {
        const square: u8 = @intCast(sq);

        if (board_state.getPieceAt(square, .white)) |piece_type| {
            if (piece_type == .king) {
                mg_pst_score += getPieceSquareValue(piece_type, .white, square, false);
                eg_pst_score += getPieceSquareValue(piece_type, .white, square, true);
            } else {
                const pst_value = getPieceSquareValue(piece_type, .white, square, false);
                mg_pst_score += pst_value;
                eg_pst_score += pst_value;
            }
        }

        if (board_state.getPieceAt(square, .black)) |piece_type| {
            if (piece_type == .king) {
                mg_pst_score -= getPieceSquareValue(piece_type, .black, square, false);
                eg_pst_score -= getPieceSquareValue(piece_type, .black, square, true);
            } else {
                const pst_value = getPieceSquareValue(piece_type, .black, square, false);
                mg_pst_score -= pst_value;
                eg_pst_score -= pst_value;
            }
        }
    }

    score += @divTrunc((mg_pst_score * (256 - phase)) + (eg_pst_score * phase), 256);

    score += evaluatePawnStructure(board_state, .white);
    score -= evaluatePawnStructure(board_state, .black);

    const white_king_safety = evaluateKingSafety(board_state, .white, is_endgame_phase);
    const black_king_safety = evaluateKingSafety(board_state, .black, is_endgame_phase);
    score += @divTrunc((white_king_safety - black_king_safety) * (256 - phase), 256);

    score += evaluateMobility(board_state, .white);
    score -= evaluateMobility(board_state, .black);

    score += evaluateRooks(board_state, .white);
    score -= evaluateRooks(board_state, .black);

    score += evaluateOutposts(board_state, .white);
    score -= evaluateOutposts(board_state, .black);

    const pawn_storm_score = evaluatePawnStorm(board_state, .white) - evaluatePawnStorm(board_state, .black);
    score += @divTrunc(pawn_storm_score * (256 - phase), 256);

    const white_bishops = @popCount(board_state.getColorBitboard(.white) & board_state.getKindBitboard(.bishop));
    const black_bishops = @popCount(board_state.getColorBitboard(.black) & board_state.getKindBitboard(.bishop));
    if (white_bishops >= 2) score += g_params.bishop_pair_bonus;
    if (black_bishops >= 2) score -= g_params.bishop_pair_bonus;

    if (is_endgame_phase) {
        const material_diff = white_material - black_material;
        const mop_up_threshold: i32 = g_params.rook_value;

        if (material_diff >= mop_up_threshold) {
            score += evaluateMopUp(board_state, .white);
        } else if (material_diff <= -mop_up_threshold) {
            score -= evaluateMopUp(board_state, .black);
        }
    }

    score += evaluateKingActivity(board_state, .white, is_endgame_phase);
    score -= evaluateKingActivity(board_state, .black, is_endgame_phase);

    var castling_score: i32 = 0;
    if (board_state.castle_rights.white_kingside) castling_score += g_params.castling_rights_kingside_bonus;
    if (board_state.castle_rights.white_queenside) castling_score += g_params.castling_rights_queenside_bonus;
    if (board_state.castle_rights.black_kingside) castling_score -= g_params.castling_rights_kingside_bonus;
    if (board_state.castle_rights.black_queenside) castling_score -= g_params.castling_rights_queenside_bonus;
    score += @divTrunc(castling_score * (256 - phase), 256);

    score += if (side_to_move == .white) g_params.tempo_bonus else -g_params.tempo_bonus;

    return if (side_to_move == .white) score else -score;
}

/// Evaluate from white's perspective (for Texel tuner).
pub fn evaluateWhite(b: *Board) i32 {
    const score = evaluate(b);
    return if (b.board.move == .white) score else -score;
}

/// Convert a mate score to account for distance to mate
pub fn mateIn(ply: u32) i32 {
    return MATE_SCORE - @as(i32, @intCast(ply));
}

/// Check if a score represents a mate
pub fn isMateScore(score: i32) bool {
    return @abs(score) >= MATE_BOUND;
}
