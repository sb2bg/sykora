const std = @import("std");
const UciError = @import("uci_error.zig").UciError;
const piece = @import("piece.zig");
const BitBoard = @import("bitboard.zig").BitBoard;

pub const FenParser = struct {
    const Self = @This();

    fn enableCastleRightFromRookSquare(board: *BitBoard, color: piece.Color, rook_sq: u8) UciError!void {
        if (board.getPieceAt(rook_sq, color) != .rook) return error.InvalidFen;

        const king_sq = board.getKingSquare(color) orelse return error.InvalidFen;
        const rook_file = rook_sq % 8;
        const king_file = king_sq % 8;

        if (rook_file > king_file) {
            board.setKingsideCastleRight(color, true, rook_sq);
        } else if (rook_file < king_file) {
            board.setQueensideCastleRight(color, true, rook_sq);
        } else {
            return error.InvalidFen;
        }
    }

    pub fn parse(fen: []const u8) UciError!BitBoard {
        var board = BitBoard.init();
        var parser = std.mem.tokenizeAny(u8, fen, " ");

        // Parse piece placement
        const piece_placement = parser.next() orelse return error.InvalidFen;
        var rank: u8 = 7; // Start from rank 8 (top)
        var file: u8 = 0;

        for (piece_placement) |c| {
            if (c == '/') {
                rank -= 1;
                file = 0;
                continue;
            }

            if (std.ascii.isDigit(c)) {
                file += c - '0';
                continue;
            }

            const pieceType = switch (std.ascii.toLower(c)) {
                'p' => piece.Type.pawn,
                'n' => piece.Type.knight,
                'b' => piece.Type.bishop,
                'r' => piece.Type.rook,
                'q' => piece.Type.queen,
                'k' => piece.Type.king,
                else => return error.InvalidFen,
            };

            const color = if (std.ascii.isUpper(c)) piece.Color.white else piece.Color.black;
            const index = rank * 8 + file;
            board.color_sets[@intFromEnum(color)] |= @as(u64, 1) << @intCast(index);
            board.kind_sets[@intFromEnum(pieceType)] |= @as(u64, 1) << @intCast(index);
            file += 1;
        }

        const white_king_sq = board.getKingSquare(.white) orelse return error.InvalidFen;
        const black_king_sq = board.getKingSquare(.black) orelse return error.InvalidFen;
        board.setKingHomeSquare(.white, white_king_sq);
        board.setKingHomeSquare(.black, black_king_sq);

        // Parse active color
        const active_color_option = parser.next();

        if (active_color_option) |active_color| {
            board.move = if (active_color[0] == 'w') piece.Color.white else piece.Color.black;
        }

        // Parse castling availability
        const castling_option = parser.next();

        if (castling_option) |castling| {
            if (castling[0] != '-') {
                for (castling) |c| {
                    switch (c) {
                        'K' => try enableCastleRightFromRookSquare(&board, .white, board.findKingsideCastlingRook(.white) orelse return error.InvalidFen),
                        'Q' => try enableCastleRightFromRookSquare(&board, .white, board.findQueensideCastlingRook(.white) orelse return error.InvalidFen),
                        'k' => try enableCastleRightFromRookSquare(&board, .black, board.findKingsideCastlingRook(.black) orelse return error.InvalidFen),
                        'q' => try enableCastleRightFromRookSquare(&board, .black, board.findQueensideCastlingRook(.black) orelse return error.InvalidFen),
                        'A'...'H' => try enableCastleRightFromRookSquare(&board, .white, c - 'A'),
                        'a'...'h' => try enableCastleRightFromRookSquare(&board, .black, 56 + (c - 'a')),
                        else => return error.InvalidFen,
                    }
                }
            }
        }

        // Parse en passant target square
        const en_passant_option = parser.next();

        if (en_passant_option) |en_passant| {
            if (en_passant[0] != '-') {
                if (en_passant.len != 2) return error.InvalidFen;
                const ep_file = en_passant[0];
                const ep_rank = en_passant[1];
                if (ep_file < 'a' or ep_file > 'h' or ep_rank < '1' or ep_rank > '8') {
                    return error.InvalidFen;
                }
                board.en_passant_square = (ep_rank - '1') * 8 + (ep_file - 'a');
            }
        }

        // Parse halfmove clock
        const halfmove_clock_option = parser.next();

        if (halfmove_clock_option) |halfmove_clock| {
            board.halfmove_clock = std.fmt.parseInt(u8, halfmove_clock, 10) catch return UciError.InvalidFen;
        }

        // Parse fullmove number
        const fullmove_number_option = parser.next();

        if (fullmove_number_option) |fullmove_number| {
            board.fullmove_number = std.fmt.parseInt(u16, fullmove_number, 10) catch return UciError.InvalidFen;
        }

        return board;
    }
};
