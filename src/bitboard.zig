const std = @import("std");
const UciError = @import("uci_error.zig").UciError;
const piece = @import("piece.zig");
const fen = @import("fen.zig");

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
        if (move.len < 4) return error.InvalidArgument;

        const fromFile = std.ascii.toLower(move[0]);
        const fromRank = move[1];
        const toFile = std.ascii.toLower(move[2]);
        const toRank = move[3];

        if (fromFile < 'a' or fromFile > 'h' or fromRank < '1' or fromRank > '8' or
            toFile < 'a' or toFile > 'h' or toRank < '1' or toRank > '8')
        {
            return error.InvalidArgument;
        }

        const fromIndex = rankFileToIndex(fromRank, fromFile);
        const toIndex = rankFileToIndex(toRank, toFile);

        const color = self.getTurn();
        const piece_type = self.board.getPieceAt(fromIndex, color) orelse return error.InvalidMove;

        self.board.clearSquare(fromIndex);
        self.board.setPieceAt(toIndex, color, piece_type);

        self.board.white_to_move = !self.board.white_to_move;
        self.board.fullmove_number += if (color == .black) 1 else 0;
    }

    fn rankFileToIndex(rank: u8, file: u8) u8 {
        return 8 * (rank - '1') + (file - 'a');
    }

    fn getTurn(self: Self) piece.Color {
        return if (self.board.white_to_move) piece.Color.white else piece.Color.black;
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
};

pub const BitBoard = struct {
    const Self = @This();

    color_sets: [2]u64,
    kind_sets: [6]u64,

    move: piece.Color = piece.Color.white,

    white_kingside_castle: bool = true,
    white_queenside_castle: bool = true,
    black_kingside_castle: bool = true,
    black_queenside_castle: bool = true,

    en_passant_square: ?u8 = null,
    halfmove_clock: u8 = 0,
    fullmove_number: u16 = 1,
    white_to_move: bool = true,

    pub fn init() Self {
        return Self{
            .color_sets = undefined,
            .kind_sets = undefined,
        };
    }

    fn getColorBitboard(self: Self, color: piece.Color) u64 {
        return self.color_sets[@intFromEnum(color)];
    }

    fn getKindBitboard(self: Self, kind: piece.Type) u64 {
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

    fn setPieceAt(self: *Self, index: u8, color: piece.Color, kind: piece.Type) void {
        const mask = @as(u64, 1) << @intCast(index);
        self.color_sets[@intFromEnum(color)] |= mask;
        self.kind_sets[@intFromEnum(kind)] |= mask;
    }

    fn getPieceAt(self: Self, index: u8, color: piece.Color) ?piece.Type {
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
