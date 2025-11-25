const std = @import("std");
const board = @import("bitboard.zig");
const Board = board.Board;
const BitBoard = board.BitBoard;
const piece = @import("piece.zig");

// Material values in centipawns (tuned values)
pub const PAWN_VALUE: i32 = 100;
pub const KNIGHT_VALUE: i32 = 320;
pub const BISHOP_VALUE: i32 = 330;
pub const ROOK_VALUE: i32 = 500;
pub const QUEEN_VALUE: i32 = 950;
pub const KING_VALUE: i32 = 20000; // Not actually used in evaluation

pub const MATE_SCORE: i32 = 30000;
pub const MATE_BOUND: i32 = 29000; // Scores above this are mate scores

// Evaluation weights
const MOBILITY_WEIGHT: i32 = 4;
const ROOK_OPEN_FILE_BONUS: i32 = 25;
const ROOK_SEMI_OPEN_FILE_BONUS: i32 = 12;
const ROOK_ON_SEVENTH_BONUS: i32 = 20;
const CONNECTED_ROOKS_BONUS: i32 = 10;
const ISOLATED_PAWN_PENALTY: i32 = 15;
const BACKWARD_PAWN_PENALTY: i32 = 10;
const DOUBLED_PAWN_PENALTY: i32 = 15;
const BISHOP_PAIR_BONUS: i32 = 45;
const KNIGHT_OUTPOST_BONUS: i32 = 25;
const BISHOP_OUTPOST_BONUS: i32 = 15;
const TEMPO_BONUS: i32 = 10;
const KING_PAWN_SHIELD_BONUS: i32 = 12;
const KING_OPEN_FILE_PENALTY: i32 = 25;
const PAWN_CHAIN_BONUS: i32 = 5;

// Piece-Square Tables (from white's perspective)
// Values are in centipawns, will be mirrored for black

const PAWN_TABLE = [64]i32{
    0,  0,  0,   0,   0,   0,   0,  0,
    50, 50, 50,  50,  50,  50,  50, 50,
    10, 10, 20,  30,  30,  20,  10, 10,
    5,  5,  10,  25,  25,  10,  5,  5,
    0,  0,  0,   20,  20,  0,   0,  0,
    5,  -5, -10, 0,   0,   -10, -5, 5,
    5,  10, 10,  -20, -20, 10,  10, 5,
    0,  0,  0,   0,   0,   0,   0,  0,
};

const KNIGHT_TABLE = [64]i32{
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0,   0,   0,   0,   -20, -40,
    -30, 0,   10,  15,  15,  10,  0,   -30,
    -30, 5,   15,  20,  20,  15,  5,   -30,
    -30, 0,   15,  20,  20,  15,  0,   -30,
    -30, 5,   10,  15,  15,  10,  5,   -30,
    -40, -20, 0,   5,   5,   0,   -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
};

const BISHOP_TABLE = [64]i32{
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0,   0,   0,   0,   0,   0,   -10,
    -10, 0,   5,   10,  10,  5,   0,   -10,
    -10, 5,   5,   10,  10,  5,   5,   -10,
    -10, 0,   10,  10,  10,  10,  0,   -10,
    -10, 10,  10,  10,  10,  10,  10,  -10,
    -10, 5,   0,   0,   0,   0,   5,   -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
};

const ROOK_TABLE = [64]i32{
    0,  0,  0,  0,  0,  0,  0,  0,
    5,  10, 10, 10, 10, 10, 10, 5,
    -5, 0,  0,  0,  0,  0,  0,  -5,
    -5, 0,  0,  0,  0,  0,  0,  -5,
    -5, 0,  0,  0,  0,  0,  0,  -5,
    -5, 0,  0,  0,  0,  0,  0,  -5,
    -5, 0,  0,  0,  0,  0,  0,  -5,
    0,  0,  0,  5,  5,  0,  0,  0,
};

const QUEEN_TABLE = [64]i32{
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0,   0,   0,  0,  0,   0,   -10,
    -10, 0,   5,   5,  5,  5,   0,   -10,
    -5,  0,   5,   5,  5,  5,   0,   -5,
    0,   0,   5,   5,  5,  5,   0,   -5,
    -10, 5,   5,   5,  5,  5,   0,   -10,
    -10, 0,   5,   0,  0,  0,   0,   -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
};

const KING_MIDDLEGAME_TABLE = [64]i32{
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20,  20,  0,   0,   0,   0,   20,  20,
    20,  30,  10,  0,   0,   10,  30,  20,
};

const KING_ENDGAME_TABLE = [64]i32{
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0,   0,   -10, -20, -30,
    -30, -10, 20,  30,  30,  20,  -10, -30,
    -30, -10, 30,  40,  40,  30,  -10, -30,
    -30, -10, 30,  40,  40,  30,  -10, -30,
    -30, -10, 20,  30,  30,  20,  -10, -30,
    -30, -30, 0,   0,   0,   0,   -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
};

// Passed pawn bonus by rank (from white's perspective, rank 2-7)
const PASSED_PAWN_BONUS = [8]i32{ 0, 10, 15, 25, 40, 65, 100, 0 };

// Knight mobility bonus by number of attacks
const KNIGHT_MOBILITY = [9]i32{ -30, -15, -5, 0, 5, 10, 15, 18, 20 };

// Bishop mobility bonus
const BISHOP_MOBILITY = [14]i32{ -25, -15, -5, 0, 5, 10, 15, 18, 20, 22, 24, 25, 26, 27 };

// Rook mobility bonus
const ROOK_MOBILITY = [15]i32{ -20, -10, -5, 0, 3, 6, 9, 12, 15, 17, 19, 20, 21, 22, 23 };

/// Mirror a square index for black pieces
inline fn mirrorSquare(square: u8) u8 {
    return square ^ 56; // XOR with 56 flips rank
}

/// Get piece-square table value for a given piece at a square
fn getPieceSquareValue(piece_type: piece.Type, color: piece.Color, square: u8, is_endgame: bool) i32 {
    const sq = if (color == .white) square else mirrorSquare(square);

    return switch (piece_type) {
        .pawn => PAWN_TABLE[sq],
        .knight => KNIGHT_TABLE[sq],
        .bishop => BISHOP_TABLE[sq],
        .rook => ROOK_TABLE[sq],
        .queen => QUEEN_TABLE[sq],
        .king => if (is_endgame) KING_ENDGAME_TABLE[sq] else KING_MIDDLEGAME_TABLE[sq],
    };
}

/// Get material value for a piece
pub fn getPieceValue(piece_type: piece.Type) i32 {
    return switch (piece_type) {
        .pawn => PAWN_VALUE,
        .knight => KNIGHT_VALUE,
        .bishop => BISHOP_VALUE,
        .rook => ROOK_VALUE,
        .queen => QUEEN_VALUE,
        .king => 0, // King value not counted in material
    };
}

/// Count total material for a color
fn countMaterial(b: BitBoard, color: piece.Color) i32 {
    const color_bb = b.getColorBitboard(color);
    var material: i32 = 0;

    material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.pawn)))) * PAWN_VALUE;
    material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.knight)))) * KNIGHT_VALUE;
    material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.bishop)))) * BISHOP_VALUE;
    material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.rook)))) * ROOK_VALUE;
    material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.queen)))) * QUEEN_VALUE;

    return material;
}

/// Determine if we're in the endgame phase
/// Endgame is defined as: both sides have no queens, or every side which has a queen has additionally no other pieces or one minorpiece maximum
fn isEndgame(b: BitBoard) bool {
    const white_queens = @popCount(b.getColorBitboard(.white) & b.getKindBitboard(.queen));
    const black_queens = @popCount(b.getColorBitboard(.black) & b.getKindBitboard(.queen));

    // No queens on board
    if (white_queens == 0 and black_queens == 0) {
        return true;
    }

    // Check if material is low (simple heuristic: total non-pawn pieces <= 6)
    const white_pieces = @popCount(b.getColorBitboard(.white) & ~b.getKindBitboard(.pawn) & ~b.getKindBitboard(.king));
    const black_pieces = @popCount(b.getColorBitboard(.black) & ~b.getKindBitboard(.pawn) & ~b.getKindBitboard(.king));

    return (white_pieces + black_pieces) <= 6;
}

/// Calculate game phase (0 = endgame, 256 = opening)
/// Used for tapered evaluation
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

    // Ensure phase is in valid range
    phase = @max(0, @min(phase, total_phase));
    return @divTrunc(phase * 256 + @divTrunc(total_phase, 2), total_phase);
}

/// Evaluate pawn structure with improved metrics
fn evaluatePawnStructure(b: BitBoard, color: piece.Color) i32 {
    const pawns = b.getColorBitboard(color) & b.getKindBitboard(.pawn);
    const opponent_pawns = b.getColorBitboard(if (color == .white) .black else .white) & b.getKindBitboard(.pawn);
    var score: i32 = 0;

    // File masks
    const file_masks = [8]u64{
        0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
        0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
    };

    // Penalize doubled pawns
    for (0..8) |file| {
        const pawns_on_file = @popCount(pawns & file_masks[file]);
        if (pawns_on_file > 1) {
            score -= @as(i32, @intCast(pawns_on_file - 1)) * DOUBLED_PAWN_PENALTY;
        }
    }

    var pawn_bb = pawns;
    while (pawn_bb != 0) {
        const sq: u8 = @intCast(@ctz(pawn_bb));
        pawn_bb &= pawn_bb - 1;

        const file = sq % 8;
        const rank = sq / 8;

        // Get adjacent file masks
        var adjacent_files: u64 = 0;
        if (file > 0) adjacent_files |= file_masks[file - 1];
        if (file < 7) adjacent_files |= file_masks[file + 1];

        // Check for isolated pawns (no friendly pawns on adjacent files)
        if ((pawns & adjacent_files) == 0) {
            score -= ISOLATED_PAWN_PENALTY;
        }

        // Check for pawn chains (pawn defended by another pawn)
        const pawn_attacks = board.getPawnAttacks(@intCast(sq), if (color == .white) .black else .white);
        if ((pawns & pawn_attacks) != 0) {
            score += PAWN_CHAIN_BONUS;
        }

        // Create mask for files that would block this pawn
        var blocking_mask: u64 = file_masks[file];
        if (file > 0) blocking_mask |= file_masks[file - 1];
        if (file < 7) blocking_mask |= file_masks[file + 1];

        // Mask for squares ahead of this pawn
        const ahead_mask: u64 = if (color == .white)
            blocking_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) << @intCast((rank + 1) * 8))
        else
            blocking_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) >> @intCast((7 - rank + 1) * 8));

        // Check for backward pawns
        const behind_mask: u64 = if (color == .white)
            adjacent_files & (@as(u64, 0xFFFFFFFFFFFFFFFF) >> @intCast((8 - rank) * 8))
        else
            adjacent_files & (@as(u64, 0xFFFFFFFFFFFFFFFF) << @intCast(rank * 8));

        // Backward pawn: no friendly pawns behind on adjacent files that could defend it
        if ((pawns & behind_mask) == 0 and (pawns & adjacent_files) == 0) {
            // And it can't safely advance (opponent pawn attacks the square in front)
            const front_sq: u6 = if (color == .white) @intCast(sq + 8) else @intCast(sq -| 8);
            const front_attacks = board.getPawnAttacks(front_sq, if (color == .white) .black else .white);
            if ((opponent_pawns & front_attacks) != 0) {
                score -= BACKWARD_PAWN_PENALTY;
            }
        }

        // If no opponent pawns blocking, it's passed
        if ((opponent_pawns & ahead_mask) == 0) {
            const advancement: usize = if (color == .white) rank else 7 - rank;
            score += PASSED_PAWN_BONUS[advancement];
        }
    }

    return score;
}

/// Evaluate king safety
fn evaluateKingSafety(b: BitBoard, color: piece.Color, is_endgame_phase: bool) i32 {
    // In endgame, king safety is less important
    if (is_endgame_phase) return 0;

    const king_bb = b.getColorBitboard(color) & b.getKindBitboard(.king);
    if (king_bb == 0) return 0;

    const king_sq: u8 = @intCast(@ctz(king_bb));
    const king_file = king_sq % 8;
    const king_rank = king_sq / 8;

    var score: i32 = 0;
    const pawns = b.getColorBitboard(color) & b.getKindBitboard(.pawn);
    const all_pawns = b.getKindBitboard(.pawn);

    // File masks for pawn shield check
    const file_masks = [8]u64{
        0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
        0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
    };

    // Check pawn shield (pawns in front of king)
    const shield_ranks = if (color == .white) [_]u8{ king_rank + 1, king_rank + 2 } else [_]u8{ king_rank -| 1, king_rank -| 2 };

    for (shield_ranks) |rank| {
        if (rank > 7) continue;

        // Check pawns on king's file and adjacent files
        for (0..3) |i| {
            const file = king_file +% i -% 1; // king_file - 1, king_file, king_file + 1
            if (file > 7) continue;

            const sq = rank * 8 + file;
            if ((pawns & (@as(u64, 1) << @intCast(sq))) != 0) {
                score += KING_PAWN_SHIELD_BONUS;
            }
        }
    }

    // Penalize open/semi-open files near king
    for (0..3) |i| {
        const file = king_file +% i -% 1;
        if (file > 7) continue;

        const file_mask = file_masks[file];
        if ((all_pawns & file_mask) == 0) {
            // Open file near king
            score -= KING_OPEN_FILE_PENALTY;
        } else if ((pawns & file_mask) == 0) {
            // Semi-open file (no friendly pawns)
            score -= KING_OPEN_FILE_PENALTY / 2;
        }
    }

    // Penalize king in center in middlegame
    if (king_file >= 2 and king_file <= 5) {
        score -= 20;
    }

    return score;
}

/// Evaluate piece mobility
fn evaluateMobility(b: BitBoard, color: piece.Color) i32 {
    var score: i32 = 0;
    const occupied = b.getColorBitboard(.white) | b.getColorBitboard(.black);
    const friendly = b.getColorBitboard(color);
    const enemy_pawns = b.getColorBitboard(if (color == .white) .black else .white) & b.getKindBitboard(.pawn);

    // Calculate pawn attack masks
    var enemy_pawn_attacks: u64 = 0;
    var pawn_bb = enemy_pawns;
    while (pawn_bb != 0) {
        const sq: u6 = @intCast(@ctz(pawn_bb));
        pawn_bb &= pawn_bb - 1;
        enemy_pawn_attacks |= board.getPawnAttacks(sq, if (color == .white) .black else .white);
    }

    // Knight mobility
    var knights = friendly & b.getKindBitboard(.knight);
    while (knights != 0) {
        const sq: u6 = @intCast(@ctz(knights));
        knights &= knights - 1;
        const attacks = board.getKnightAttacks(sq) & ~friendly & ~enemy_pawn_attacks;
        const mobility = @min(@popCount(attacks), 8);
        score += KNIGHT_MOBILITY[mobility];
    }

    // Bishop mobility
    var bishops = friendly & b.getKindBitboard(.bishop);
    while (bishops != 0) {
        const sq: u6 = @intCast(@ctz(bishops));
        bishops &= bishops - 1;
        const attacks = board.getBishopAttacks(sq, occupied) & ~friendly;
        const mobility = @min(@popCount(attacks), 13);
        score += BISHOP_MOBILITY[mobility];
    }

    // Rook mobility
    var rooks = friendly & b.getKindBitboard(.rook);
    while (rooks != 0) {
        const sq: u6 = @intCast(@ctz(rooks));
        rooks &= rooks - 1;
        const attacks = board.getRookAttacks(sq, occupied) & ~friendly;
        const mobility = @min(@popCount(attacks), 14);
        score += ROOK_MOBILITY[mobility];
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

        // Open file bonus
        if ((friendly_pawns & file_mask) == 0) {
            if ((enemy_pawns & file_mask) == 0) {
                score += ROOK_OPEN_FILE_BONUS;
            } else {
                score += ROOK_SEMI_OPEN_FILE_BONUS;
            }
        }

        // Rook on 7th rank bonus
        const seventh_rank: u8 = if (color == .white) 6 else 1;
        if (rank == seventh_rank) {
            score += ROOK_ON_SEVENTH_BONUS;
        }

        // Connected rooks bonus
        const rook_attacks = board.getRookAttacks(sq, occupied);
        if ((rook_attacks & rooks) != 0) {
            score += CONNECTED_ROOKS_BONUS / 2; // Divide by 2 since both rooks get the bonus
        }
    }

    return score;
}

/// Evaluate knight outposts
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

    // Outpost squares: squares that can't be attacked by enemy pawns
    var piece_bb = knights;
    while (piece_bb != 0) {
        const sq: u6 = @intCast(@ctz(piece_bb));
        piece_bb &= piece_bb - 1;

        const file = sq % 8;
        const rank = sq / 8;

        // Check if in enemy territory (ranks 4-6 for white, 1-3 for black)
        const in_outpost_zone = if (color == .white) rank >= 3 and rank <= 5 else rank >= 2 and rank <= 4;
        if (!in_outpost_zone) continue;

        // Check if defended by pawn
        const pawn_attacks = board.getPawnAttacks(sq, if (color == .white) .black else .white);
        if ((friendly_pawns & pawn_attacks) == 0) continue;

        // Check if can't be attacked by enemy pawns
        var attack_mask: u64 = 0;
        if (file > 0) attack_mask |= file_masks[file - 1];
        if (file < 7) attack_mask |= file_masks[file + 1];

        // Mask squares in front of the knight
        const ahead_mask: u64 = if (color == .white)
            attack_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) << @intCast((rank + 1) * 8))
        else
            attack_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) >> @intCast((7 - rank + 1) * 8));

        if ((enemy_pawns & ahead_mask) == 0) {
            score += KNIGHT_OUTPOST_BONUS;
        }
    }

    // Similar for bishops (with smaller bonus)
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

        const ahead_mask: u64 = if (color == .white)
            attack_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) << @intCast((rank + 1) * 8))
        else
            attack_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) >> @intCast((7 - rank + 1) * 8));

        if ((enemy_pawns & ahead_mask) == 0) {
            score += BISHOP_OUTPOST_BONUS;
        }
    }

    return score;
}

/// Main evaluation function
/// Returns score from the perspective of the side to move
pub fn evaluate(b: *Board) i32 {
    const board_state = b.board;
    const side_to_move = board_state.move;

    const is_endgame_phase = isEndgame(board_state);
    const phase = getGamePhase(board_state);

    // Material evaluation
    const white_material = countMaterial(board_state, .white);
    const black_material = countMaterial(board_state, .black);
    var score = white_material - black_material;

    // Piece-square tables (with tapered eval for king)
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

    // Tapered evaluation for PST
    score += @divTrunc((mg_pst_score * (256 - phase)) + (eg_pst_score * phase), 256);

    // Pawn structure
    score += evaluatePawnStructure(board_state, .white);
    score -= evaluatePawnStructure(board_state, .black);

    // King safety (reduced in endgame through phase)
    const white_king_safety = evaluateKingSafety(board_state, .white, is_endgame_phase);
    const black_king_safety = evaluateKingSafety(board_state, .black, is_endgame_phase);
    score += @divTrunc((white_king_safety - black_king_safety) * (256 - phase), 256);

    // Piece mobility
    score += evaluateMobility(board_state, .white);
    score -= evaluateMobility(board_state, .black);

    // Rook evaluation
    score += evaluateRooks(board_state, .white);
    score -= evaluateRooks(board_state, .black);

    // Outpost evaluation
    score += evaluateOutposts(board_state, .white);
    score -= evaluateOutposts(board_state, .black);

    // Bishop pair bonus (more valuable in open positions/endgames)
    const white_bishops = @popCount(board_state.getColorBitboard(.white) & board_state.getKindBitboard(.bishop));
    const black_bishops = @popCount(board_state.getColorBitboard(.black) & board_state.getKindBitboard(.bishop));
    if (white_bishops >= 2) score += BISHOP_PAIR_BONUS;
    if (black_bishops >= 2) score -= BISHOP_PAIR_BONUS;

    // Tempo bonus - small bonus for side to move
    score += if (side_to_move == .white) TEMPO_BONUS else -TEMPO_BONUS;

    // Return score from side to move's perspective
    return if (side_to_move == .white) score else -score;
}

/// Convert a mate score to account for distance to mate
pub fn mateIn(ply: u32) i32 {
    return MATE_SCORE - @as(i32, @intCast(ply));
}

/// Check if a score represents a mate
pub fn isMateScore(score: i32) bool {
    return @abs(score) >= MATE_BOUND;
}
