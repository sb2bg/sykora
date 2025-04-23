const std = @import("std");
const UciError = @import("uci_error.zig").UciError;
const piece = @import("piece.zig");
const BitBoard = @import("bitboard.zig").BitBoard;

pub const FenParser = struct {
    const Self = @This();

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
                        'K' => board.white_kingside_castle = true,
                        'Q' => board.white_queenside_castle = true,
                        'k' => board.black_kingside_castle = true,
                        'q' => board.black_queenside_castle = true,
                        else => {},
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
            board.halfmove_clock = std.fmt.parseInt(u8, halfmove_clock, 10) catch {
                return error.InvalidFen;
            };
        }

        // Parse fullmove number
        const fullmove_number_option = parser.next();

        if (fullmove_number_option) |fullmove_number| {
            board.fullmove_number = std.fmt.parseInt(u16, fullmove_number, 10) catch {
                return error.InvalidFen;
            };
        }

        return board;
    }
};
