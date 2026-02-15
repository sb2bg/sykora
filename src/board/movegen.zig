const pieceInfo = @import("../piece.zig");
const attacks = @import("attacks.zig");
const legality = @import("legality.zig");

const RANK_1_MASK: u64 = 0x00000000000000FF;
const RANK_8_MASK: u64 = 0xFF00000000000000;
const NOT_A_FILE: u64 = 0xFEFEFEFEFEFEFEFE;
const NOT_H_FILE: u64 = 0x7F7F7F7F7F7F7F7F;

inline fn getKnightAttacks(square: u6) u64 {
    return attacks.getKnightAttacks(square);
}

inline fn getKingAttacks(square: u6) u64 {
    return attacks.getKingAttacks(square);
}

inline fn getBishopAttacks(square: u6, occupied: u64) u64 {
    return attacks.getBishopAttacks(square, occupied);
}

inline fn getRookAttacks(square: u6, occupied: u64) u64 {
    return attacks.getRookAttacks(square, occupied);
}

inline fn appendMove(moves: anytype, from: anytype, to: anytype, promo: ?pieceInfo.Type) void {
    const MoveT = @TypeOf(moves.*.moves[0]);
    moves.append(MoveT.init(@intCast(from), @intCast(to), promo));
}

pub fn generatePseudoLegalMoves(self: anytype, moves: anytype) !void {
    const color = self.board.move;
    const our_pieces = self.board.getColorBitboard(color);
    const opponent_pieces = self.board.getColorBitboard(if (color == .white) .black else .white);
    const occupied = self.board.occupied();

    // Generate pawn moves
    try generatePawnMoves(self, moves, color, our_pieces, opponent_pieces, occupied);

    // Generate knight moves
    try generateKnightMoves(self, moves, color, our_pieces);

    // Generate bishop moves
    try generateBishopMoves(self, moves, color, our_pieces, occupied);

    // Generate rook moves
    try generateRookMoves(self, moves, color, our_pieces, occupied);

    // Generate queen moves
    try generateQueenMoves(self, moves, color, our_pieces, occupied);

    // Generate king moves
    try generateKingMoves(self, moves, color, our_pieces, opponent_pieces, occupied);
}

/// Generate only capture moves (for quiescence search)
/// This is more efficient than generating all moves and filtering
pub fn generateCaptures(self: anytype, moves: anytype) void {
    const color = self.board.move;
    const our_pieces = self.board.getColorBitboard(color);
    const opponent_pieces = self.board.getColorBitboard(if (color == .white) .black else .white);
    const occupied = self.board.occupied();

    // Generate pawn captures (including promotions and en passant)
    generatePawnCaptures(self, moves, color, our_pieces, opponent_pieces);

    // Generate knight captures
    generateKnightCaptures(self, moves, our_pieces, opponent_pieces);

    // Generate bishop captures
    generateBishopCaptures(self, moves, our_pieces, opponent_pieces, occupied);

    // Generate rook captures
    generateRookCaptures(self, moves, our_pieces, opponent_pieces, occupied);

    // Generate queen captures
    generateQueenCaptures(self, moves, our_pieces, opponent_pieces, occupied);

    // Generate king captures
    generateKingCaptures(self, moves, our_pieces, opponent_pieces);
}

/// Generate legal captures only
pub fn generateLegalCaptures(self: anytype, moves: anytype) void {
    var pseudo_captures = @TypeOf(moves.*).init();
    generateCaptures(self, &pseudo_captures);

    const color = self.board.move;

    for (pseudo_captures.slice()) |move| {
        // Save state
        const old_board = self.board;

        // Make move
        self.applyMoveUncheckedForLegality(move);

        // Check legality
        const legal = !self.isInCheck(color);

        // Restore state
        self.board = old_board;

        if (legal) {
            moves.append(move);
        }
    }
}

/// Generate only quiet (non-capture, non-promotion) moves
pub fn generateQuietMoves(self: anytype, moves: anytype) void {
    const color = self.board.move;
    const our_pieces = self.board.getColorBitboard(color);
    const opponent_pieces = self.board.getColorBitboard(if (color == .white) .black else .white);
    const occupied = self.board.occupied();

    // Generate pawn quiet moves (single and double pushes, no promotions)
    generatePawnQuietMoves(self, moves, color, our_pieces, occupied);

    // Generate knight quiet moves
    generateKnightQuietMoves(self, moves, our_pieces, occupied);

    // Generate bishop quiet moves
    generateBishopQuietMoves(self, moves, our_pieces, occupied);

    // Generate rook quiet moves
    generateRookQuietMoves(self, moves, our_pieces, occupied);

    // Generate queen quiet moves
    generateQueenQuietMoves(self, moves, our_pieces, occupied);

    // Generate king quiet moves (including castling)
    generateKingQuietMoves(self, moves, color, our_pieces, opponent_pieces, occupied);
}

/// Generate legal quiet moves only
pub fn generateLegalQuietMoves(self: anytype, moves: anytype) void {
    var pseudo_quiets = @TypeOf(moves.*).init();
    generateQuietMoves(self, &pseudo_quiets);

    const color = self.board.move;

    for (pseudo_quiets.slice()) |move| {
        // Save state
        const old_board = self.board;

        // Make move
        self.applyMoveUncheckedForLegality(move);

        // Check legality
        const legal = !self.isInCheck(color);

        // Restore state
        self.board = old_board;

        if (legal) {
            moves.append(move);
        }
    }
}

// ========== Capture move generators ==========

fn generatePawnCaptures(self: anytype, moves: anytype, color: pieceInfo.Color, our_pieces: u64, opponent_pieces: u64) void {
    const pawns = our_pieces & self.board.getKindBitboard(.pawn);

    if (color == .white) {
        const promo_rank_mask: u64 = RANK_8_MASK;

        // Left captures (not A-file)
        const left_captures = ((pawns & NOT_A_FILE) << 7) & opponent_pieces;
        const left_captures_no_promo = left_captures & ~promo_rank_mask;
        const left_captures_promo = left_captures & promo_rank_mask;

        var bb = left_captures_no_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to - 7, to, null);
        }

        bb = left_captures_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            const from: u6 = to - 7;
            const promo_types = [_]pieceInfo.Type{ .queen, .knight, .rook, .bishop };
            inline for (promo_types) |promo_type| {
                appendMove(moves, from, to, promo_type);
            }
        }

        // Right captures (not H-file)
        const right_captures = ((pawns & NOT_H_FILE) << 9) & opponent_pieces;
        const right_captures_no_promo = right_captures & ~promo_rank_mask;
        const right_captures_promo = right_captures & promo_rank_mask;

        bb = right_captures_no_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to - 9, to, null);
        }

        bb = right_captures_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            const from: u6 = to - 9;
            const promo_types = [_]pieceInfo.Type{ .queen, .knight, .rook, .bishop };
            inline for (promo_types) |promo_type| {
                appendMove(moves, from, to, promo_type);
            }
        }

        // Push promotions (captures to 8th rank are already handled, but we need non-capture promotions)
        const push_promo = ((pawns << 8) & ~(our_pieces | opponent_pieces)) & promo_rank_mask;
        bb = push_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            const from: u6 = to - 8;
            const promo_types = [_]pieceInfo.Type{ .queen, .knight, .rook, .bishop };
            inline for (promo_types) |promo_type| {
                appendMove(moves, from, to, promo_type);
            }
        }

        // En passant
        if (self.board.en_passant_square) |ep_sq| {
            const ep_bb: u64 = @as(u64, 1) << @intCast(ep_sq);
            const ep_left = ((pawns & NOT_A_FILE) << 7) & ep_bb;
            const ep_right = ((pawns & NOT_H_FILE) << 9) & ep_bb;

            if (ep_left != 0) {
                appendMove(moves, ep_sq - 7, ep_sq, null);
            }
            if (ep_right != 0) {
                appendMove(moves, ep_sq - 9, ep_sq, null);
            }
        }
    } else {
        const promo_rank_mask: u64 = RANK_1_MASK;

        // Left captures (not H-file for black)
        const left_captures = ((pawns & NOT_H_FILE) >> 7) & opponent_pieces;
        const left_captures_no_promo = left_captures & ~promo_rank_mask;
        const left_captures_promo = left_captures & promo_rank_mask;

        var bb = left_captures_no_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to + 7, to, null);
        }

        bb = left_captures_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            const from: u6 = to + 7;
            const promo_types = [_]pieceInfo.Type{ .queen, .knight, .rook, .bishop };
            inline for (promo_types) |promo_type| {
                appendMove(moves, from, to, promo_type);
            }
        }

        // Right captures (not A-file for black)
        const right_captures = ((pawns & NOT_A_FILE) >> 9) & opponent_pieces;
        const right_captures_no_promo = right_captures & ~promo_rank_mask;
        const right_captures_promo = right_captures & promo_rank_mask;

        bb = right_captures_no_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to + 9, to, null);
        }

        bb = right_captures_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            const from: u6 = to + 9;
            const promo_types = [_]pieceInfo.Type{ .queen, .knight, .rook, .bishop };
            inline for (promo_types) |promo_type| {
                appendMove(moves, from, to, promo_type);
            }
        }

        // Push promotions
        const push_promo = ((pawns >> 8) & ~(our_pieces | opponent_pieces)) & promo_rank_mask;
        bb = push_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            const from: u6 = to + 8;
            const promo_types = [_]pieceInfo.Type{ .queen, .knight, .rook, .bishop };
            inline for (promo_types) |promo_type| {
                appendMove(moves, from, to, promo_type);
            }
        }

        // En passant
        if (self.board.en_passant_square) |ep_sq| {
            const ep_bb: u64 = @as(u64, 1) << @intCast(ep_sq);
            const ep_left = ((pawns & NOT_H_FILE) >> 7) & ep_bb;
            const ep_right = ((pawns & NOT_A_FILE) >> 9) & ep_bb;

            if (ep_left != 0) {
                appendMove(moves, ep_sq + 7, ep_sq, null);
            }
            if (ep_right != 0) {
                appendMove(moves, ep_sq + 9, ep_sq, null);
            }
        }
    }
}

fn generateKnightCaptures(self: anytype, moves: anytype, our_pieces: u64, opponent_pieces: u64) void {
    const knights = our_pieces & self.board.getKindBitboard(.knight);
    var knight_bb = knights;

    while (knight_bb != 0) {
        const from: u6 = @intCast(@ctz(knight_bb));
        knight_bb &= knight_bb - 1;

        var attack_bb = getKnightAttacks(from) & opponent_pieces;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateBishopCaptures(self: anytype, moves: anytype, our_pieces: u64, opponent_pieces: u64, occupied: u64) void {
    const bishops = our_pieces & self.board.getKindBitboard(.bishop);
    var bishop_bb = bishops;

    while (bishop_bb != 0) {
        const from: u6 = @intCast(@ctz(bishop_bb));
        bishop_bb &= bishop_bb - 1;

        var attack_bb = getBishopAttacks(from, occupied) & opponent_pieces;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateRookCaptures(self: anytype, moves: anytype, our_pieces: u64, opponent_pieces: u64, occupied: u64) void {
    const rooks = our_pieces & self.board.getKindBitboard(.rook);
    var rook_bb = rooks;

    while (rook_bb != 0) {
        const from: u6 = @intCast(@ctz(rook_bb));
        rook_bb &= rook_bb - 1;

        var attack_bb = getRookAttacks(from, occupied) & opponent_pieces;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateQueenCaptures(self: anytype, moves: anytype, our_pieces: u64, opponent_pieces: u64, occupied: u64) void {
    const queens = our_pieces & self.board.getKindBitboard(.queen);
    var queen_bb = queens;

    while (queen_bb != 0) {
        const from: u6 = @intCast(@ctz(queen_bb));
        queen_bb &= queen_bb - 1;

        var attack_bb = (getBishopAttacks(from, occupied) | getRookAttacks(from, occupied)) & opponent_pieces;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateKingCaptures(self: anytype, moves: anytype, our_pieces: u64, opponent_pieces: u64) void {
    const kings = our_pieces & self.board.getKindBitboard(.king);
    if (kings == 0) return;

    const from: u6 = @intCast(@ctz(kings));
    var attack_bb = getKingAttacks(from) & opponent_pieces;

    while (attack_bb != 0) {
        const to: u6 = @intCast(@ctz(attack_bb));
        attack_bb &= attack_bb - 1;
        appendMove(moves, from, to, null);
    }
}

// ========== Quiet move generators ==========

fn generatePawnQuietMoves(self: anytype, moves: anytype, color: pieceInfo.Color, our_pieces: u64, occupied: u64) void {
    const pawns = our_pieces & self.board.getKindBitboard(.pawn);
    const empty = ~occupied;

    if (color == .white) {
        // Single pushes (excluding promotions - those are tactical)
        const push_one = (pawns << 8) & empty;
        const push_one_no_promo = push_one & ~RANK_8_MASK;

        var bb = push_one_no_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to - 8, to, null);
        }

        // Double pushes
        const rank_3_mask: u64 = 0x0000000000FF0000;
        const push_two = ((push_one & rank_3_mask) << 8) & empty;
        bb = push_two;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to - 16, to, null);
        }
    } else {
        // Single pushes (excluding promotions)
        const push_one = (pawns >> 8) & empty;
        const push_one_no_promo = push_one & ~RANK_1_MASK;

        var bb = push_one_no_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to + 8, to, null);
        }

        // Double pushes
        const rank_6_mask: u64 = 0x0000FF0000000000;
        const push_two = ((push_one & rank_6_mask) >> 8) & empty;
        bb = push_two;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to + 16, to, null);
        }
    }
}

fn generateKnightQuietMoves(self: anytype, moves: anytype, our_pieces: u64, occupied: u64) void {
    const knights = our_pieces & self.board.getKindBitboard(.knight);
    const empty = ~occupied;
    var knight_bb = knights;

    while (knight_bb != 0) {
        const from: u6 = @intCast(@ctz(knight_bb));
        knight_bb &= knight_bb - 1;

        var attack_bb = getKnightAttacks(from) & empty;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateBishopQuietMoves(self: anytype, moves: anytype, our_pieces: u64, occupied: u64) void {
    const bishops = our_pieces & self.board.getKindBitboard(.bishop);
    const empty = ~occupied;
    var bishop_bb = bishops;

    while (bishop_bb != 0) {
        const from: u6 = @intCast(@ctz(bishop_bb));
        bishop_bb &= bishop_bb - 1;

        var attack_bb = getBishopAttacks(from, occupied) & empty;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateRookQuietMoves(self: anytype, moves: anytype, our_pieces: u64, occupied: u64) void {
    const rooks = our_pieces & self.board.getKindBitboard(.rook);
    const empty = ~occupied;
    var rook_bb = rooks;

    while (rook_bb != 0) {
        const from: u6 = @intCast(@ctz(rook_bb));
        rook_bb &= rook_bb - 1;

        var attack_bb = getRookAttacks(from, occupied) & empty;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateQueenQuietMoves(self: anytype, moves: anytype, our_pieces: u64, occupied: u64) void {
    const queens = our_pieces & self.board.getKindBitboard(.queen);
    const empty = ~occupied;
    var queen_bb = queens;

    while (queen_bb != 0) {
        const from: u6 = @intCast(@ctz(queen_bb));
        queen_bb &= queen_bb - 1;

        var attack_bb = (getBishopAttacks(from, occupied) | getRookAttacks(from, occupied)) & empty;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateKingQuietMoves(self: anytype, moves: anytype, color: pieceInfo.Color, our_pieces: u64, _: u64, occupied: u64) void {
    const kings = our_pieces & self.board.getKindBitboard(.king);
    if (kings == 0) return;

    const from: u6 = @intCast(@ctz(kings));
    const empty = ~occupied;

    // Regular king moves to empty squares
    var attack_bb = getKingAttacks(from) & empty;

    while (attack_bb != 0) {
        const to: u6 = @intCast(@ctz(attack_bb));
        attack_bb &= attack_bb - 1;
        appendMove(moves, from, to, null);
    }

    // Castling
    const opponent = if (color == .white) pieceInfo.Color.black else pieceInfo.Color.white;

    if (color == .white) {
        if (self.board.castle_rights.white_kingside) {
            if ((occupied & ((@as(u64, 1) << 5) | (@as(u64, 1) << 6))) == 0 and
                !legality.isSquareAttackedBy(self, 4, opponent) and
                !legality.isSquareAttackedBy(self, 5, opponent))
            {
                appendMove(moves, 4, 6, null);
            }
        }
        if (self.board.castle_rights.white_queenside) {
            if ((occupied & ((@as(u64, 1) << 1) | (@as(u64, 1) << 2) | (@as(u64, 1) << 3))) == 0 and
                !legality.isSquareAttackedBy(self, 4, opponent) and
                !legality.isSquareAttackedBy(self, 3, opponent))
            {
                appendMove(moves, 4, 2, null);
            }
        }
    } else {
        if (self.board.castle_rights.black_kingside) {
            if ((occupied & ((@as(u64, 1) << 61) | (@as(u64, 1) << 62))) == 0 and
                !legality.isSquareAttackedBy(self, 60, opponent) and
                !legality.isSquareAttackedBy(self, 61, opponent))
            {
                appendMove(moves, 60, 62, null);
            }
        }
        if (self.board.castle_rights.black_queenside) {
            if ((occupied & ((@as(u64, 1) << 57) | (@as(u64, 1) << 58) | (@as(u64, 1) << 59))) == 0 and
                !legality.isSquareAttackedBy(self, 60, opponent) and
                !legality.isSquareAttackedBy(self, 59, opponent))
            {
                appendMove(moves, 60, 58, null);
            }
        }
    }
}

fn generatePawnMoves(self: anytype, moves: anytype, color: pieceInfo.Color, our_pieces: u64, opponent_pieces: u64, occupied: u64) !void {
    const pawns = our_pieces & self.board.getKindBitboard(.pawn);
    const empty = ~occupied;

    if (color == .white) {
        // White pawns move up the board (towards rank 8)
        const promo_rank_mask: u64 = RANK_8_MASK;

        // Single pushes
        const push_one = (pawns << 8) & empty;
        const push_one_no_promo = push_one & ~promo_rank_mask;
        const push_one_promo = push_one & promo_rank_mask;

        // Generate single push moves (non-promotion)
        var bb = push_one_no_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to - 8, to, null);
        }

        // Generate promotion moves from single pushes
        bb = push_one_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            const from: u6 = to - 8;
            // Unroll promotion generation for better performance
            const promo_types = [_]pieceInfo.Type{ .queen, .knight, .rook, .bishop };
            inline for (promo_types) |promo_type| {
                appendMove(moves, from, to, promo_type);
            }
        }

        // Double pushes (only from rank 2, so pushed pawns are on rank 3)
        const rank_3_mask: u64 = 0x0000000000FF0000;
        const push_two = ((push_one & rank_3_mask) << 8) & empty;
        bb = push_two;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to - 16, to, null);
        }

        // Left captures (not A-file)
        const left_captures = ((pawns & NOT_A_FILE) << 7) & opponent_pieces;
        const left_captures_no_promo = left_captures & ~promo_rank_mask;
        const left_captures_promo = left_captures & promo_rank_mask;

        bb = left_captures_no_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to - 7, to, null);
        }

        bb = left_captures_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            const from: u6 = to - 7;
            const promo_types = [_]pieceInfo.Type{ .queen, .knight, .rook, .bishop };
            inline for (promo_types) |promo_type| {
                appendMove(moves, from, to, promo_type);
            }
        }

        // Right captures (not H-file)
        const right_captures = ((pawns & NOT_H_FILE) << 9) & opponent_pieces;
        const right_captures_no_promo = right_captures & ~promo_rank_mask;
        const right_captures_promo = right_captures & promo_rank_mask;

        bb = right_captures_no_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to - 9, to, null);
        }

        bb = right_captures_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            const from: u6 = to - 9;
            const promo_types = [_]pieceInfo.Type{ .queen, .knight, .rook, .bishop };
            inline for (promo_types) |promo_type| {
                appendMove(moves, from, to, promo_type);
            }
        }

        // En passant
        if (self.board.en_passant_square) |ep_sq| {
            const ep_bb: u64 = @as(u64, 1) << @intCast(ep_sq);
            const ep_left = ((pawns & NOT_A_FILE) << 7) & ep_bb;
            const ep_right = ((pawns & NOT_H_FILE) << 9) & ep_bb;

            if (ep_left != 0) {
                appendMove(moves, ep_sq - 7, ep_sq, null);
            }
            if (ep_right != 0) {
                appendMove(moves, ep_sq - 9, ep_sq, null);
            }
        }
    } else {
        // Black pawns move down the board (towards rank 1)
        const promo_rank_mask: u64 = RANK_1_MASK;

        // Single pushes
        const push_one = (pawns >> 8) & empty;
        const push_one_no_promo = push_one & ~promo_rank_mask;
        const push_one_promo = push_one & promo_rank_mask;

        // Generate single push moves (non-promotion)
        var bb = push_one_no_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to + 8, to, null);
        }

        // Generate promotion moves from single pushes
        bb = push_one_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            const from: u6 = to + 8;
            const promo_types = [_]pieceInfo.Type{ .queen, .knight, .rook, .bishop };
            inline for (promo_types) |promo_type| {
                appendMove(moves, from, to, promo_type);
            }
        }

        // Double pushes (only from rank 7, so pushed pawns are on rank 6)
        const rank_6_mask: u64 = 0x0000FF0000000000;
        const push_two = ((push_one & rank_6_mask) >> 8) & empty;
        bb = push_two;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to + 16, to, null);
        }

        // Left captures (not H-file for black, moving down-left)
        const left_captures = ((pawns & NOT_H_FILE) >> 7) & opponent_pieces;
        const left_captures_no_promo = left_captures & ~promo_rank_mask;
        const left_captures_promo = left_captures & promo_rank_mask;

        bb = left_captures_no_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to + 7, to, null);
        }

        bb = left_captures_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            const from: u6 = to + 7;
            const promo_types = [_]pieceInfo.Type{ .queen, .knight, .rook, .bishop };
            inline for (promo_types) |promo_type| {
                appendMove(moves, from, to, promo_type);
            }
        }

        // Right captures (not A-file for black, moving down-right)
        const right_captures = ((pawns & NOT_A_FILE) >> 9) & opponent_pieces;
        const right_captures_no_promo = right_captures & ~promo_rank_mask;
        const right_captures_promo = right_captures & promo_rank_mask;

        bb = right_captures_no_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            appendMove(moves, to + 9, to, null);
        }

        bb = right_captures_promo;
        while (bb != 0) {
            const to: u6 = @intCast(@ctz(bb));
            bb &= bb - 1;
            const from: u6 = to + 9;
            const promo_types = [_]pieceInfo.Type{ .queen, .knight, .rook, .bishop };
            inline for (promo_types) |promo_type| {
                appendMove(moves, from, to, promo_type);
            }
        }

        // En passant
        if (self.board.en_passant_square) |ep_sq| {
            const ep_bb: u64 = @as(u64, 1) << @intCast(ep_sq);
            const ep_left = ((pawns & NOT_H_FILE) >> 7) & ep_bb;
            const ep_right = ((pawns & NOT_A_FILE) >> 9) & ep_bb;

            if (ep_left != 0) {
                appendMove(moves, ep_sq + 7, ep_sq, null);
            }
            if (ep_right != 0) {
                appendMove(moves, ep_sq + 9, ep_sq, null);
            }
        }
    }
}

fn generateKnightMoves(self: anytype, moves: anytype, _: pieceInfo.Color, our_pieces: u64) !void {
    const knights = our_pieces & self.board.getKindBitboard(.knight);
    var knight_bb = knights;

    while (knight_bb != 0) {
        const from: u6 = @intCast(@ctz(knight_bb));
        knight_bb &= knight_bb - 1;

        var attack_bb = getKnightAttacks(from) & ~our_pieces;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateBishopMoves(self: anytype, moves: anytype, _: pieceInfo.Color, our_pieces: u64, occupied: u64) !void {
    const bishops = our_pieces & self.board.getKindBitboard(.bishop);
    var bishop_bb = bishops;

    while (bishop_bb != 0) {
        const from: u6 = @intCast(@ctz(bishop_bb));
        bishop_bb &= bishop_bb - 1;

        var attack_bb = getBishopAttacks(from, occupied) & ~our_pieces;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateRookMoves(self: anytype, moves: anytype, _: pieceInfo.Color, our_pieces: u64, occupied: u64) !void {
    const rooks = our_pieces & self.board.getKindBitboard(.rook);
    var rook_bb = rooks;

    while (rook_bb != 0) {
        const from: u6 = @intCast(@ctz(rook_bb));
        rook_bb &= rook_bb - 1;

        var attack_bb = getRookAttacks(from, occupied) & ~our_pieces;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateQueenMoves(self: anytype, moves: anytype, _: pieceInfo.Color, our_pieces: u64, occupied: u64) !void {
    const queens = our_pieces & self.board.getKindBitboard(.queen);
    var queen_bb = queens;

    while (queen_bb != 0) {
        const from: u6 = @intCast(@ctz(queen_bb));
        queen_bb &= queen_bb - 1;

        var attack_bb = (getBishopAttacks(from, occupied) | getRookAttacks(from, occupied)) & ~our_pieces;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateKingMoves(self: anytype, moves: anytype, color: pieceInfo.Color, our_pieces: u64, _: u64, occupied: u64) !void {
    const kings = our_pieces & self.board.getKindBitboard(.king);
    if (kings == 0) return;

    const from: u6 = @intCast(@ctz(kings));

    // Regular king moves
    var attack_bb = getKingAttacks(from) & ~our_pieces;

    while (attack_bb != 0) {
        const to: u6 = @intCast(@ctz(attack_bb));
        attack_bb &= attack_bb - 1;
        appendMove(moves, from, to, null);
    }

    // Castling - only add if not currently in check and path is clear
    // Check attacks will be verified during move legality testing
    const opponent = if (color == .white) pieceInfo.Color.black else pieceInfo.Color.white;

    if (color == .white) {
        // White kingside
        if (self.board.castle_rights.white_kingside) {
            if ((occupied & ((@as(u64, 1) << 5) | (@as(u64, 1) << 6))) == 0 and
                !legality.isSquareAttackedBy(self, 4, opponent) and
                !legality.isSquareAttackedBy(self, 5, opponent))
            {
                appendMove(moves, 4, 6, null);
            }
        }
        // White queenside
        if (self.board.castle_rights.white_queenside) {
            if ((occupied & ((@as(u64, 1) << 1) | (@as(u64, 1) << 2) | (@as(u64, 1) << 3))) == 0 and
                !legality.isSquareAttackedBy(self, 4, opponent) and
                !legality.isSquareAttackedBy(self, 3, opponent))
            {
                appendMove(moves, 4, 2, null);
            }
        }
    } else {
        // Black kingside
        if (self.board.castle_rights.black_kingside) {
            if ((occupied & ((@as(u64, 1) << 61) | (@as(u64, 1) << 62))) == 0 and
                !legality.isSquareAttackedBy(self, 60, opponent) and
                !legality.isSquareAttackedBy(self, 61, opponent))
            {
                appendMove(moves, 60, 62, null);
            }
        }
        // Black queenside
        if (self.board.castle_rights.black_queenside) {
            if ((occupied & ((@as(u64, 1) << 57) | (@as(u64, 1) << 58) | (@as(u64, 1) << 59))) == 0 and
                !legality.isSquareAttackedBy(self, 60, opponent) and
                !legality.isSquareAttackedBy(self, 59, opponent))
            {
                appendMove(moves, 60, 58, null);
            }
        }
    }
}
