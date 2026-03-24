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
        if (self.getCastlingInfo(color, from_sq, to_sq, false)) |castle| {
            self.board.clearSquare(castle.rook_from);
            self.board.setPieceAt(castle.rook_to, color, .rook);
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
        self.board.setKingsideCastleRight(color, false, null);
        self.board.setQueensideCastleRight(color, false, null);
    } else if (piece_type == .rook) {
        if (self.board.getQueensideCastleRookSquare(color)) |rook_sq| {
            if (from_sq == rook_sq) self.board.setQueensideCastleRight(color, false, null);
        }
        if (self.board.getKingsideCastleRookSquare(color)) |rook_sq| {
            if (from_sq == rook_sq) self.board.setKingsideCastleRight(color, false, null);
        }
    }

    if (captured_piece) |captured| {
        if (captured == .rook) {
            if (self.board.getQueensideCastleRookSquare(opponent_color)) |rook_sq| {
                if (to_sq == rook_sq) self.board.setQueensideCastleRight(opponent_color, false, null);
            }
            if (self.board.getKingsideCastleRookSquare(opponent_color)) |rook_sq| {
                if (to_sq == rook_sq) self.board.setKingsideCastleRight(opponent_color, false, null);
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
        if (self.getCastlingInfo(color, from_sq, to_sq, false)) |castle| {
            self.board.clearSquare(castle.rook_from);
            self.board.setPieceAt(castle.rook_to, color, .rook);
        }
    }

    self.board.clearSquare(from_sq);
    const final_piece = if (move.promotion()) |promo| promo else piece_type;
    self.board.setPieceAt(to_sq, color, final_piece);

    self.board.move = opponent_color;
}
