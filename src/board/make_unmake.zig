const std = @import("std");
const pieceInfo = @import("../piece.zig");

pub fn applyMoveUnchecked(self: anytype, move: anytype) void {
    const color = self.board.move;
    const from_sq = move.from();
    const to_sq = move.to();
    const piece_type = self.board.getPieceAt(from_sq, color) orelse return;
    const opponent_color = if (color == .white) pieceInfo.Color.black else pieceInfo.Color.white;

    const captured_piece = self.board.getPieceAt(to_sq, opponent_color);
    const is_en_passant_capture = piece_type == .pawn and self.board.en_passant_square == to_sq and captured_piece == null;
    const is_capture = captured_piece != null or is_en_passant_capture;

    self.board.clearSquare(to_sq);

    if (is_en_passant_capture) {
        const ep_capture_square = if (color == .white) to_sq - 8 else to_sq + 8;
        self.board.clearSquare(ep_capture_square);
    }

    if (piece_type == .king) {
        const from_file = from_sq % 8;
        const to_file = to_sq % 8;

        if (from_file == 4 and to_file == 6) {
            const rook_from = from_sq + 3;
            const rook_to = from_sq + 1;
            self.board.clearSquare(rook_from);
            self.board.setPieceAt(rook_to, color, .rook);
        } else if (from_file == 4 and to_file == 2) {
            const rook_from = from_sq - 4;
            const rook_to = from_sq - 1;
            self.board.clearSquare(rook_from);
            self.board.setPieceAt(rook_to, color, .rook);
        }
    }

    self.board.clearSquare(from_sq);
    const final_piece = if (move.promotion()) |promo| promo else piece_type;
    self.board.setPieceAt(to_sq, color, final_piece);

    self.board.en_passant_square = null;
    if (piece_type == .pawn) {
        const from_rank = from_sq / 8;
        const to_rank = to_sq / 8;
        if (@as(i16, to_rank) - @as(i16, from_rank) == 2 or @as(i16, to_rank) - @as(i16, from_rank) == -2) {
            self.board.en_passant_square = if (color == .white) from_sq + 8 else from_sq - 8;
        }
    }

    if (piece_type == .king) {
        if (color == .white) {
            self.board.castle_rights.white_kingside = false;
            self.board.castle_rights.white_queenside = false;
        } else {
            self.board.castle_rights.black_kingside = false;
            self.board.castle_rights.black_queenside = false;
        }
    } else if (piece_type == .rook) {
        if (color == .white) {
            if (from_sq == 0) self.board.castle_rights.white_queenside = false;
            if (from_sq == 7) self.board.castle_rights.white_kingside = false;
        } else {
            if (from_sq == 56) self.board.castle_rights.black_queenside = false;
            if (from_sq == 63) self.board.castle_rights.black_kingside = false;
        }
    }

    if (captured_piece) |captured| {
        if (captured == .rook) {
            if (opponent_color == .white) {
                if (to_sq == 0) self.board.castle_rights.white_queenside = false;
                if (to_sq == 7) self.board.castle_rights.white_kingside = false;
            } else {
                if (to_sq == 56) self.board.castle_rights.black_queenside = false;
                if (to_sq == 63) self.board.castle_rights.black_kingside = false;
            }
        }
    }

    if (piece_type == .pawn or is_capture) {
        self.board.halfmove_clock = 0;
    } else if (self.board.halfmove_clock < std.math.maxInt(u8)) {
        self.board.halfmove_clock += 1;
    }

    if (color == .black and self.board.fullmove_number < std.math.maxInt(u16)) {
        self.board.fullmove_number += 1;
    }

    self.board.move = opponent_color;
}

pub fn applyMoveUncheckedForLegality(self: anytype, move: anytype) void {
    const color = self.board.move;
    const from_sq = move.from();
    const to_sq = move.to();
    const piece_type = self.board.getPieceAt(from_sq, color) orelse return;
    const opponent_color = if (color == .white) pieceInfo.Color.black else pieceInfo.Color.white;

    self.board.clearSquare(to_sq);

    if (piece_type == .pawn and self.board.en_passant_square == to_sq) {
        const ep_capture_square = if (color == .white) to_sq - 8 else to_sq + 8;
        self.board.clearSquare(ep_capture_square);
    }

    if (piece_type == .king) {
        const from_file = from_sq % 8;
        const to_file = to_sq % 8;
        if (from_file == 4 and to_file == 6) {
            const rook_from = from_sq + 3;
            const rook_to = from_sq + 1;
            self.board.clearSquare(rook_from);
            self.board.setPieceAt(rook_to, color, .rook);
        } else if (from_file == 4 and to_file == 2) {
            const rook_from = from_sq - 4;
            const rook_to = from_sq - 1;
            self.board.clearSquare(rook_from);
            self.board.setPieceAt(rook_to, color, .rook);
        }
    }

    self.board.clearSquare(from_sq);
    const final_piece = if (move.promotion()) |promo| promo else piece_type;
    self.board.setPieceAt(to_sq, color, final_piece);

    self.board.move = opponent_color;
}
