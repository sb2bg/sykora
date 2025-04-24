const std = @import("std");
const UciError = @import("uci_error.zig").UciError;
const pieceInfo = @import("piece.zig");
const fen = @import("fen.zig");

pub const Move = struct {
    const Self = @This();
    from: u8,
    to: u8,
    promotion: ?pieceInfo.Type = null,

    pub fn init(from: u8, to: u8, promotion: ?pieceInfo.Type) Self {
        return Self{ .from = from, .to = to, .promotion = promotion };
    }

    pub fn fromString(str: []const u8) UciError!Self {
        if (str.len < 4 or str.len > 5) return error.InvalidArgument;

        const promotion_str = if (str.len == 5) std.ascii.toLower(str[4]) else null;
        const from_file = std.ascii.toLower(str[0]);
        const from_rank = str[1];
        const to_file = std.ascii.toLower(str[2]);
        const to_rank = str[3];

        if (from_file < 'a' or from_file > 'h' or from_rank < '1' or from_rank > '8' or
            to_file < 'a' or to_file > 'h' or to_rank < '1' or to_rank > '8')
        {
            return error.InvalidArgument;
        }

        const promotion = if (promotion_str) |p| pieceInfo.Type.fromChar(p) else null;
        const from_index = Board.rankFileToIndex(from_rank, from_file);
        const to_index = Board.rankFileToIndex(to_rank, to_file);

        return Self.init(from_index, to_index, promotion);
    }

    pub fn format(
        self: Self,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        const from_file = 'a' + (self.from % 8);
        const from_rank = '1' + (self.from / 8);
        const to_file = 'a' + (self.to % 8);
        const to_rank = '1' + (self.to / 8);

        try writer.print("{c}{c}{c}{c}", .{ from_file, from_rank, to_file, to_rank });

        if (self.promotion) |p| {
            try writer.print("{c}", .{std.ascii.toLower(p.getName())});
        }
    }
};

pub const Board = struct {
    const Self = @This();
    board: BitBoard,

    pub fn init() Self {
        return Self{ .board = .{} };
    }

    pub fn startpos() Self {
        return fromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") catch unreachable;
    }

    pub fn fromFen(fen_str: []const u8) UciError!Self {
        return Self{ .board = try fen.FenParser.parse(fen_str) };
    }

    pub fn makeMove(self: *Self, move: []const u8) UciError!void {
        if (move.len != 4) return error.InvalidArgument;

        const from_file = std.ascii.toLower(move[0]);
        const from_rank = move[1];
        const to_file = std.ascii.toLower(move[2]);
        const to_rank = move[3];

        if (from_file < 'a' or from_file > 'h' or from_rank < '1' or from_rank > '8' or
            to_file < 'a' or to_file > 'h' or to_rank < '1' or to_rank > '8')
        {
            return error.InvalidArgument;
        }

        const from_index = rankFileToIndex(from_rank, from_file);
        const to_index = rankFileToIndex(to_rank, to_file);

        const color = self.getTurn();
        const piece_type = self.board.getPieceAt(from_index, color) orelse return error.InvalidMove;

        // TODO: make sure the move is valid

        if (piece_type == .pawn) {
            if (self.board.en_passant_square) |en_passant_square| {
                if (en_passant_square == to_index) {
                    self.board.clearSquare(en_passant_square);
                }
            }
        }

        self.board.clearSquare(from_index);
        self.board.setPieceAt(to_index, color, piece_type);

        self.board.white_to_move = !self.board.white_to_move;
        self.board.fullmove_number += if (color == .black) 1 else 0;
    }

    fn rankFileToIndex(rank: u8, file: u8) u8 {
        return 8 * (rank - '1') + (file - 'a');
    }

    fn getTurn(self: Self) pieceInfo.Color {
        return if (self.board.white_to_move) pieceInfo.Color.white else pieceInfo.Color.black;
    }

    pub fn format(
        self: Self,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.writeAll("  +---+---+---+---+---+---+---+---+\n");

        var rank: i32 = 7;
        while (rank >= 0) : (rank -= 1) {
            try writer.print("{d} ", .{rank + 1});
            for (0..8) |file| {
                const index: u8 = @intCast(@as(u32, @intCast(rank)) * 8 + file);
                const white_piece = self.board.getPieceAt(index, .white);
                const black_piece = self.board.getPieceAt(index, .black);

                try writer.writeAll("| ");
                if (white_piece) |p| {
                    try writer.print("{c}", .{p.getName()});
                } else if (black_piece) |p| {
                    const name = p.getName();
                    try writer.print("{c}", .{std.ascii.toLower(name)});
                } else {
                    try writer.writeAll(" ");
                }
                try writer.writeAll(" ");
            }
            try writer.writeAll("|\n");
            try writer.writeAll("  +---+---+---+---+---+---+---+---+\n");
        }

        try writer.writeAll("    a   b   c   d   e   f   g   h\n");
    }

    pub fn getFenString(self: Self) UciError![]u8 {
        return getFenStringGenericError(self) catch {
            return error.IOError;
        };
    }

    fn getFenStringGenericError(self: Self) ![]u8 {
        var buffer: [100]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buffer);
        const writer = stream.writer();

        for (0..8) |rank| {
            const actual_rank = 7 - rank;
            var empty_count: u8 = 0;

            for (0..8) |file| {
                const index = actual_rank * 8 + file;
                const white_piece = self.board.getPieceAt(@intCast(index), .white);
                const black_piece = self.board.getPieceAt(@intCast(index), .black);

                if (white_piece) |p| {
                    if (empty_count > 0) {
                        try writer.print("{}", .{empty_count});
                        empty_count = 0;
                    }
                    try writer.writeByte(p.getName());
                } else if (black_piece) |p| {
                    if (empty_count > 0) {
                        try writer.print("{}", .{empty_count});
                        empty_count = 0;
                    }
                    try writer.writeByte(std.ascii.toLower(p.getName()));
                } else {
                    empty_count += 1;
                }
            }

            if (empty_count > 0) {
                try writer.print("{}", .{empty_count});
            }

            if (rank != 7) {
                try writer.writeByte('/');
            }
        }

        // Turn
        const turn: u8 = if (self.board.white_to_move) 'w' else 'b';
        try writer.writeByte(' ');
        try writer.writeByte(turn);

        // Castling rights
        try writer.writeByte(' ');
        var any_castle = false;
        if (self.board.white_kingside_castle) {
            try writer.writeByte('K');
            any_castle = true;
        }
        if (self.board.white_queenside_castle) {
            try writer.writeByte('Q');
            any_castle = true;
        }
        if (self.board.black_kingside_castle) {
            try writer.writeByte('k');
            any_castle = true;
        }
        if (self.board.black_queenside_castle) {
            try writer.writeByte('q');
            any_castle = true;
        }
        if (!any_castle) try writer.writeByte('-');

        // En passant
        try writer.writeByte(' ');
        if (self.board.en_passant_square) |sq| {
            const file = sq % 8 + 'a';
            const rank = sq / 8 + '1';
            try writer.writeByte(file);
            try writer.writeByte(rank);
        } else {
            try writer.writeByte('-');
        }

        // Halfmove and fullmove
        try writer.print(" {} {}", .{ self.board.halfmove_clock, self.board.fullmove_number });
        return buffer[0..stream.pos];
    }
};

pub const BitBoard = struct {
    const Self = @This();

    color_sets: [2]u64 = undefined,
    kind_sets: [6]u64 = undefined,

    move: pieceInfo.Color = pieceInfo.Color.white,

    white_kingside_castle: bool = false,
    white_queenside_castle: bool = false,
    black_kingside_castle: bool = false,
    black_queenside_castle: bool = false,

    en_passant_square: ?u8 = null,
    halfmove_clock: u8 = 0,
    fullmove_number: u16 = 1,
    white_to_move: bool = true,

    pub fn init() Self {
        return Self{};
    }

    fn getColorBitboard(self: Self, color: pieceInfo.Color) u64 {
        return self.color_sets[@intFromEnum(color)];
    }

    fn getKindBitboard(self: Self, kind: pieceInfo.Type) u64 {
        return self.kind_sets[@intFromEnum(kind)];
    }

    fn clearSquare(self: *Self, index: u8) void {
        const mask = @as(u64, 1) << @intCast(index);

        for (0..2) |i| {
            self.color_sets[i] &= ~mask;
        }
        for (0..6) |i| {
            self.kind_sets[i] &= ~mask;
        }
    }

    fn setPieceAt(self: *Self, index: u8, color: pieceInfo.Color, kind: pieceInfo.Type) void {
        const mask = @as(u64, 1) << @intCast(index);
        self.color_sets[@intFromEnum(color)] |= mask;
        self.kind_sets[@intFromEnum(kind)] |= mask;
    }

    fn getPieceAt(self: Self, index: u8, color: pieceInfo.Color) ?pieceInfo.Type {
        const color_mask = @as(u64, 1) << @intCast(index);
        if ((self.color_sets[@intFromEnum(color)] & color_mask) == 0)
            return null;

        for (self.kind_sets, 0..) |kind_mask, i| {
            if ((kind_mask & color_mask) != 0)
                return @enumFromInt(i);
        }

        return null;
    }
};
