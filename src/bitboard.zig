const std = @import("std");
const UciError = @import("uci_error.zig").UciError;
const pieceInfo = @import("piece.zig");
const fen = @import("fen.zig");
const zobrist = @import("zobrist.zig");
const ZobristHasher = zobrist.ZobristHasher;

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
        if (self.from == 0 and self.to == 0 and self.promotion == null) {
            // null move
            try writer.writeAll("0000");
            return;
        }

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
    zobrist_hasher: ZobristHasher = ZobristHasher.init(),

    pub fn init() Self {
        const board = BitBoard.init();
        var self = Self{ .board = board };
        self.zobrist_hasher.hash(self.board);
        return self;
    }

    pub fn startpos() Self {
        return fromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") catch unreachable;
    }

    pub fn fromFen(fen_str: []const u8) UciError!Self {
        const board = try fen.FenParser.parse(fen_str);
        var self = Self{ .board = board };
        self.zobrist_hasher.hash(self.board);
        return self;
    }

    pub fn makeStrMove(self: *Self, move: []const u8) UciError!void {
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
        const is_capture = self.board.getPieceAt(to_index, color) != null;

        const prev_castle = self.board.castle_rights;
        const prev_ep = self.board.en_passant_square;
        const captured = self.board.getPieceAt(to_index, if (color == .white) .black else .white);

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

        self.board.move = if (self.board.whiteToMove()) pieceInfo.Color.black else pieceInfo.Color.white;
        self.board.fullmove_number += if (color == .black) 1 else 0;

        if (piece_type == .pawn or is_capture) {
            self.board.halfmove_clock = 0;
        } else {
            self.board.halfmove_clock += 1;
        }

        self.zobrist_hasher.updateHash(
            from_index,
            to_index,
            piece_type,
            color,
            captured,
            prev_castle,
            self.board.castle_rights,
            prev_ep,
            self.board.en_passant_square,
        );
    }

    pub fn makeMove(self: *Self, move: Move) UciError!void {
        _ = move;
        _ = self;
        return error.Unimplemented;
    }

    inline fn rankFileToIndex(rank: u8, file: u8) u8 {
        return rankFileToSquare(rank - '1', file - 'a');
    }

    inline fn rankFileToSquare(rank: u8, file: u8) u8 {
        return rank * 8 + file;
    }

    inline fn getTurn(self: Self) pieceInfo.Color {
        return self.board.move;
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

    pub fn getFenString(self: Self, allocator: std.mem.Allocator) UciError![]u8 {
        return getFenStringGenericError(self, allocator) catch UciError.IOError;
    }

    fn getFenStringGenericError(self: Self, allocator: std.mem.Allocator) ![]u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        defer buffer.deinit();
        const writer = buffer.writer();

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
        const turn: u8 = if (self.board.move == .white) 'w' else 'b';
        try writer.writeByte(' ');
        try writer.writeByte(turn);

        // Castling rights
        try writer.writeByte(' ');
        var any_castle = false;
        if (self.board.castle_rights.white_kingside) {
            try writer.writeByte('K');
            any_castle = true;
        }
        if (self.board.castle_rights.white_queenside) {
            try writer.writeByte('Q');
            any_castle = true;
        }
        if (self.board.castle_rights.black_kingside) {
            try writer.writeByte('k');
            any_castle = true;
        }
        if (self.board.castle_rights.black_queenside) {
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
        return buffer.toOwnedSlice();
    }
};

pub const CastleRights = struct {
    white_kingside: bool = false,
    white_queenside: bool = false,
    black_kingside: bool = false,
    black_queenside: bool = false,
};

pub const BitBoard = struct {
    const Self = @This();

    color_sets: [2]u64 = undefined,
    kind_sets: [6]u64 = undefined,

    move: pieceInfo.Color = pieceInfo.Color.white,

    castle_rights: CastleRights = CastleRights{},

    en_passant_square: ?u8 = null,
    halfmove_clock: u8 = 0,
    fullmove_number: u16 = 1,

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

    pub fn getPieceAt(self: Self, index: u8, color: pieceInfo.Color) ?pieceInfo.Type {
        const color_mask = @as(u64, 1) << @intCast(index);
        if ((self.color_sets[@intFromEnum(color)] & color_mask) == 0)
            return null;

        for (self.kind_sets, 0..) |kind_mask, i| {
            if ((kind_mask & color_mask) != 0)
                return @enumFromInt(i);
        }

        return null;
    }

    /// Check if there are any pawns of the given color adjacent to the specified square.
    /// This is used for determining if an en passant capture is possible.
    ///
    /// Parameters:
    ///   - board: The current board position
    ///   - ep_sq: The en passant square to check
    ///   - color: The color of the pawns to look for
    ///
    /// Returns: true if there are adjacent pawns that could make an en passant capture
    pub fn hasAdjacentPawn(self: Self, ep_sq: u8, color: pieceInfo.Color) bool {
        const file = ep_sq % 8;

        var result = false;

        if (file > 0) {
            const left = ep_sq - 1;
            if (self.getPieceAt(left, color)) |pt| {
                if (pt == .pawn) result = true;
            }
        }

        if (file < 7) {
            const right = ep_sq + 1;
            if (self.getPieceAt(right, color)) |pt| {
                if (pt == .pawn) result = true;
            }
        }

        return result;
    }

    pub fn whiteToMove(self: Self) bool {
        return self.move == pieceInfo.Color.white;
    }

    inline fn occupied(self: Self) u64 {
        return self.color_sets[0] | self.color_sets[1];
    }

    inline fn empty(self: Self) u64 {
        return ~self.occupied();
    }
};
