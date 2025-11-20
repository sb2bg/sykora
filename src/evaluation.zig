const std = @import("std");
const board = @import("bitboard.zig");
const Board = board.Board;
const BitBoard = board.BitBoard;
const piece = @import("piece.zig");

// Material values in centipawns
pub const PAWN_VALUE: i32 = 100;
pub const KNIGHT_VALUE: i32 = 320;
pub const BISHOP_VALUE: i32 = 330;
pub const ROOK_VALUE: i32 = 500;
pub const QUEEN_VALUE: i32 = 900;
pub const KING_VALUE: i32 = 20000; // Not actually used in evaluation

pub const MATE_SCORE: i32 = 30000;
pub const MATE_BOUND: i32 = 29000; // Scores above this are mate scores

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

/// Evaluate pawn structure
fn evaluatePawnStructure(b: BitBoard, color: piece.Color) i32 {
    const pawns = b.getColorBitboard(color) & b.getKindBitboard(.pawn);
    var score: i32 = 0;

    // Penalize doubled pawns
    for (0..8) |file| {
        const file_mask: u64 = @as(u64, 0x0101010101010101) << @intCast(file);
        const pawns_on_file = @popCount(pawns & file_mask);
        if (pawns_on_file > 1) {
            score -= @as(i32, @intCast(pawns_on_file - 1)) * 20;
        }
    }

    // Check for passed pawns (simplified - just check if no enemy pawns on file or adjacent files ahead)
    const opponent_pawns = b.getColorBitboard(if (color == .white) .black else .white) & b.getKindBitboard(.pawn);
    var pawn_bb = pawns;
    while (pawn_bb != 0) {
        const sq: u8 = @intCast(@ctz(pawn_bb));
        pawn_bb &= pawn_bb - 1;

        const file = sq % 8;
        const rank = sq / 8;

        // Create mask for files that would block this pawn
        var blocking_mask: u64 = @as(u64, 0x0101010101010101) << @intCast(file); // Same file
        if (file > 0) blocking_mask |= @as(u64, 0x0101010101010101) << @intCast(file - 1); // Left file
        if (file < 7) blocking_mask |= @as(u64, 0x0101010101010101) << @intCast(file + 1); // Right file

        // Mask for squares ahead of this pawn
        const ahead_mask: u64 = if (color == .white)
            blocking_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) << @intCast((rank + 1) * 8))
        else
            blocking_mask & (@as(u64, 0xFFFFFFFFFFFFFFFF) >> @intCast((7 - rank + 1) * 8));

        // If no opponent pawns blocking, it's passed
        if ((opponent_pawns & ahead_mask) == 0) {
            const advancement = if (color == .white) rank else 7 - rank;
            score += @as(i32, @intCast(advancement)) * 15;
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

    // Check pawn shield (pawns in front of king)
    const pawns = b.getColorBitboard(color) & b.getKindBitboard(.pawn);
    const shield_ranks = if (color == .white) [_]u8{ king_rank + 1, king_rank + 2 } else [_]u8{ king_rank -| 1, king_rank -| 2 };

    for (shield_ranks) |rank| {
        if (rank > 7) continue;

        // Check pawns on king's file and adjacent files
        for (0..3) |i| {
            const file = king_file +% i -% 1; // king_file - 1, king_file, king_file + 1
            if (file > 7) continue;

            const sq = rank * 8 + file;
            if ((pawns & (@as(u64, 1) << @intCast(sq))) != 0) {
                score += 10;
            }
        }
    }

    // Penalize king in center in middlegame
    if (king_file >= 2 and king_file <= 5) {
        score -= 20;
    }

    return score;
}

/// Main evaluation function
/// Returns score from the perspective of the side to move
pub fn evaluate(b: *Board) i32 {
    const board_state = b.board;
    const side_to_move = board_state.move;

    // Check for checkmate/stalemate
    // (In actual search, this should be detected before calling evaluate)

    const is_endgame_phase = isEndgame(board_state);

    // Material evaluation
    const white_material = countMaterial(board_state, .white);
    const black_material = countMaterial(board_state, .black);
    var score = white_material - black_material;

    // Piece-square tables
    for (0..64) |sq| {
        const square: u8 = @intCast(sq);

        if (board_state.getPieceAt(square, .white)) |piece_type| {
            score += getPieceSquareValue(piece_type, .white, square, is_endgame_phase);
        }

        if (board_state.getPieceAt(square, .black)) |piece_type| {
            score -= getPieceSquareValue(piece_type, .black, square, is_endgame_phase);
        }
    }

    // Pawn structure
    score += evaluatePawnStructure(board_state, .white);
    score -= evaluatePawnStructure(board_state, .black);

    // King safety
    score += evaluateKingSafety(board_state, .white, is_endgame_phase);
    score -= evaluateKingSafety(board_state, .black, is_endgame_phase);

    // Bishop pair bonus
    const white_bishops = @popCount(board_state.getColorBitboard(.white) & board_state.getKindBitboard(.bishop));
    const black_bishops = @popCount(board_state.getColorBitboard(.black) & board_state.getKindBitboard(.bishop));
    if (white_bishops >= 2) score += 30;
    if (black_bishops >= 2) score -= 30;

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

test "initial position evaluation" {
    var b = Board.startpos();
    const score = evaluate(&b);
    try std.testing.expectEqual(@as(i32, 0), score);
}

test "material imbalance" {
    // Remove a white pawn
    var b = Board.startpos();
    b.board.clearSquare(8); // a2 pawn
    const score = evaluate(&b);
    // Black should be winning (negative score for white to move? No, evaluate returns side-to-move perspective)
    // White to move. White is down a pawn. Score should be negative.
    // PAWN_VALUE is 100. Plus some PST difference maybe.
    // a2 pawn is on rank 2. PST value is 50.
    // So white loses 100 + 50 = 150 material/positional value.
    // Score should be around -150.
    try std.testing.expect(score < -50);
}
