const pieceInfo = @import("../piece.zig");
const attacks = @import("attacks.zig");

inline fn getKnightAttacks(square: u6) u64 {
    return attacks.getKnightAttacks(square);
}

inline fn getKingAttacks(square: u6) u64 {
    return attacks.getKingAttacks(square);
}

inline fn getPawnAttacks(square: u6, pawn_color: pieceInfo.Color) u64 {
    return attacks.getPawnAttacks(square, pawn_color);
}

inline fn getRookAttacks(square: u6, occupied: u64) u64 {
    return attacks.getRookAttacks(square, occupied);
}

inline fn getBishopAttacks(square: u6, occupied: u64) u64 {
    return attacks.getBishopAttacks(square, occupied);
}

pub fn countCheckingPieces(self: anytype, king_color: pieceInfo.Color) u32 {
    const king_bb = self.board.getColorBitboard(king_color) & self.board.getKindBitboard(.king);
    if (king_bb == 0) return 0;

    const king_square: u6 = @intCast(@ctz(king_bb));
    const attacker_color = if (king_color == .white) pieceInfo.Color.black else pieceInfo.Color.white;
    const attacker_bb = self.board.getColorBitboard(attacker_color);
    const occupied = self.board.occupied();

    var count: u32 = 0;

    const pawn_attacks = getPawnAttacks(king_square, king_color);
    if ((pawn_attacks & attacker_bb & self.board.getKindBitboard(.pawn)) != 0) {
        count += @popCount(pawn_attacks & attacker_bb & self.board.getKindBitboard(.pawn));
    }

    const knight_attacks = getKnightAttacks(king_square);
    if ((knight_attacks & attacker_bb & self.board.getKindBitboard(.knight)) != 0) {
        count += @popCount(knight_attacks & attacker_bb & self.board.getKindBitboard(.knight));
    }

    const bishop_attacks = getBishopAttacks(king_square, occupied);
    if ((bishop_attacks & attacker_bb & (self.board.getKindBitboard(.bishop) | self.board.getKindBitboard(.queen))) != 0) {
        count += @popCount(bishop_attacks & attacker_bb & (self.board.getKindBitboard(.bishop) | self.board.getKindBitboard(.queen)));
    }

    const rook_attacks = getRookAttacks(king_square, occupied);
    if ((rook_attacks & attacker_bb & (self.board.getKindBitboard(.rook) | self.board.getKindBitboard(.queen))) != 0) {
        count += @popCount(rook_attacks & attacker_bb & (self.board.getKindBitboard(.rook) | self.board.getKindBitboard(.queen)));
    }

    return count;
}

pub fn isDirectCheck(self: anytype, move: anytype, moving_color: pieceInfo.Color, opponent_color: pieceInfo.Color) bool {
    const king_bb = self.board.getColorBitboard(opponent_color) & self.board.getKindBitboard(.king);
    if (king_bb == 0) return false;

    const king_square: u6 = @intCast(@ctz(king_bb));
    const piece_type = self.board.getPieceAt(move.to(), moving_color) orelse return false;
    const occupied = self.board.occupied();
    const move_to: u6 = @intCast(move.to());

    return switch (piece_type) {
        .pawn => blk: {
            const pawn_attacks = getPawnAttacks(king_square, opponent_color);
            break :blk (pawn_attacks & (@as(u64, 1) << move_to)) != 0;
        },
        .knight => blk: {
            break :blk (getKnightAttacks(king_square) & (@as(u64, 1) << move_to)) != 0;
        },
        .bishop => blk: {
            break :blk (getBishopAttacks(king_square, occupied) & (@as(u64, 1) << move_to)) != 0;
        },
        .rook => blk: {
            break :blk (getRookAttacks(king_square, occupied) & (@as(u64, 1) << move_to)) != 0;
        },
        .queen => blk: {
            const queen_attacks = getBishopAttacks(king_square, occupied) | getRookAttacks(king_square, occupied);
            break :blk (queen_attacks & (@as(u64, 1) << move_to)) != 0;
        },
        .king => false,
    };
}

pub fn isInCheck(self: anytype, color: pieceInfo.Color) bool {
    const king_bb = self.board.getColorBitboard(color) & self.board.getKindBitboard(.king);
    if (king_bb == 0) return false;

    const king_square: u6 = @intCast(@ctz(king_bb));
    const opponent_color = if (color == .white) pieceInfo.Color.black else pieceInfo.Color.white;

    return isSquareAttackedBy(self, king_square, opponent_color);
}

pub fn isSquareAttackedBy(self: anytype, square: u6, attacker_color: pieceInfo.Color) bool {
    const attacker_bb = self.board.getColorBitboard(attacker_color);

    const knights = attacker_bb & self.board.getKindBitboard(.knight);
    if ((getKnightAttacks(square) & knights) != 0) {
        return true;
    }

    const kings = attacker_bb & self.board.getKindBitboard(.king);
    if ((getKingAttacks(square) & kings) != 0) {
        return true;
    }

    const pawns = attacker_bb & self.board.getKindBitboard(.pawn);
    if (pawns != 0) {
        const defender_color = if (attacker_color == .white) pieceInfo.Color.black else pieceInfo.Color.white;
        const pawn_attacks = getPawnAttacks(square, defender_color);
        if ((pawn_attacks & pawns) != 0) {
            return true;
        }
    }

    const queens = attacker_bb & self.board.getKindBitboard(.queen);
    const bishops_queens = (attacker_bb & self.board.getKindBitboard(.bishop)) | queens;
    const rooks_queens = (attacker_bb & self.board.getKindBitboard(.rook)) | queens;

    if (bishops_queens == 0 and rooks_queens == 0) {
        return false;
    }

    const occupied = self.board.occupied();

    if (bishops_queens != 0) {
        if ((getBishopAttacks(square, occupied) & bishops_queens) != 0) {
            return true;
        }
    }

    if (rooks_queens != 0) {
        if ((getRookAttacks(square, occupied) & rooks_queens) != 0) {
            return true;
        }
    }

    return false;
}
