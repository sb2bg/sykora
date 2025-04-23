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
        const piecePlacement = parser.next() orelse return error.InvalidFen;
        var rank: u8 = 7; // Start from rank 8 (top)
        var file: u8 = 0;

        for (piecePlacement) |c| {
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
        const activeColor = parser.next() orelse return error.InvalidFen;
        board.move = if (activeColor[0] == 'w') piece.Color.white else piece.Color.black;

        // Parse castling availability
        const castling = parser.next() orelse return error.InvalidFen;
        if (castling[0] != '-') {
            for (castling) |c| {
                switch (c) {
                    'K' => board.white_kingside_castle = true,
                    'Q' => board.white_queenside_castle = true,
                    'k' => board.black_kingside_castle = true,
                    'q' => board.black_queenside_castle = true,
                    else => return error.InvalidFen,
                }
            }
        } else {
            board.white_kingside_castle = false;
            board.white_queenside_castle = false;
            board.black_kingside_castle = false;
            board.black_queenside_castle = false;
        }

        // Parse en passant target square
        const enPassant = parser.next() orelse return error.InvalidFen;
        if (enPassant[0] != '-') {
            if (enPassant.len != 2) return error.InvalidFen;
            const epFile = enPassant[0];
            const epRank = enPassant[1];
            if (epFile < 'a' or epFile > 'h' or epRank < '1' or epRank > '8') {
                return error.InvalidFen;
            }
            board.en_passant_square = (epRank - '1') * 8 + (epFile - 'a');
        }

        // Parse halfmove clock
        const halfmoveClock = parser.next() orelse return error.InvalidFen;
        board.halfmove_clock = std.fmt.parseInt(u8, halfmoveClock, 10) catch {
            return error.InvalidFen;
        };

        // Parse fullmove number
        const fullmoveNumber = parser.next() orelse return error.InvalidFen;
        board.fullmove_number = std.fmt.parseInt(u16, fullmoveNumber, 10) catch {
            return error.InvalidFen;
        };

        return board;
    }
};
