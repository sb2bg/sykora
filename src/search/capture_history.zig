const std = @import("std");
const board = @import("../bitboard.zig");
const piece = @import("../piece.zig");

/// Compile-time gate used to prepare isolated OpenBench SPRT revisions.
pub const ENABLED = true;
const PIECE_TYPE_COUNT: usize = 6;
const MAX_CELL: i32 = 16_384;
const MAX_BONUS: i32 = 400;
pub const MAX_SCORE: i32 = MAX_CELL;

pub const Key = struct {
    color: piece.Color,
    moving_piece: piece.Type,
    to_sq: u8,
    captured_piece: piece.Type,
};

/// Capture-cutoff history indexed by side, moving piece, destination, and
/// captured piece. The existing tactical score still decides whether a
/// capture belongs to the good or bad stage; history only learns ordering
/// inside those stages.
pub const CaptureHistory = struct {
    scores: [2][PIECE_TYPE_COUNT][64][PIECE_TYPE_COUNT]i16,

    pub fn init() CaptureHistory {
        var self: CaptureHistory = undefined;
        @memset(std.mem.asBytes(&self), 0);
        return self;
    }

    pub fn keyFor(b: *const board.BitBoard, move: board.Move) ?Key {
        const color = b.move;
        const opponent = if (color == .white) piece.Color.black else piece.Color.white;
        const board_piece = b.getPieceAt(move.from(), color) orelse return null;
        const captured_piece = b.getPieceAt(move.to(), opponent) orelse blk: {
            const is_en_passant = board_piece == .pawn and b.en_passant_square == move.to();
            if (!is_en_passant) return null;
            break :blk piece.Type.pawn;
        };

        return .{
            .color = color,
            .moving_piece = move.promotion() orelse board_piece,
            .to_sq = move.to(),
            .captured_piece = captured_piece,
        };
    }

    pub fn get(self: *const CaptureHistory, key: Key) i32 {
        const cell = &self.scores[@intFromEnum(key.color)][@intFromEnum(key.moving_piece)][key.to_sq][@intFromEnum(key.captured_piece)];
        return cell.*;
    }

    pub fn score(self: *const CaptureHistory, b: *const board.BitBoard, move: board.Move) i32 {
        const key = keyFor(b, move) orelse return 0;
        return self.get(key);
    }

    pub fn reward(self: *CaptureHistory, key: Key, depth: u32) void {
        updateCell(self.cellFor(key), bonusForDepth(depth));
    }

    pub fn penalize(self: *CaptureHistory, key: Key, depth: u32) void {
        updateCell(self.cellFor(key), -bonusForDepth(depth));
    }

    pub fn age(self: *CaptureHistory) void {
        for (&self.scores) |*color_scores| {
            for (color_scores) |*piece_scores| {
                for (piece_scores) |*to_scores| {
                    for (to_scores) |*cell| {
                        cell.* = @intCast(@divTrunc(cell.*, 2));
                    }
                }
            }
        }
    }

    fn cellFor(self: *CaptureHistory, key: Key) *i16 {
        return &self.scores[@intFromEnum(key.color)][@intFromEnum(key.moving_piece)][key.to_sq][@intFromEnum(key.captured_piece)];
    }

    fn bonusForDepth(depth: u32) i32 {
        const bounded_depth = @min(depth, 64);
        return @intCast(@min(bounded_depth * bounded_depth, MAX_BONUS));
    }

    fn updateCell(cell: *i16, bonus: i32) void {
        if (bonus == 0) return;
        const current: i32 = cell.*;
        const gravity = @divTrunc(current * @as(i32, @intCast(@abs(bonus))), MAX_CELL);
        const next = std.math.clamp(current + bonus - gravity, -MAX_CELL, MAX_CELL);
        cell.* = @intCast(next);
    }
};

test "capture history rewards and penalizes independent keys" {
    var history = CaptureHistory.init();
    const white_capture = Key{
        .color = .white,
        .moving_piece = .knight,
        .to_sq = 28,
        .captured_piece = .pawn,
    };
    const black_capture = Key{
        .color = .black,
        .moving_piece = .knight,
        .to_sq = 28,
        .captured_piece = .pawn,
    };

    history.reward(white_capture, 8);
    try std.testing.expect(history.get(white_capture) > 0);
    try std.testing.expectEqual(@as(i32, 0), history.get(black_capture));

    for (0..8) |_| history.penalize(white_capture, 12);
    try std.testing.expect(history.get(white_capture) < 0);
}

test "capture history recognizes en passant victims" {
    const position = try board.Board.fromFen("8/8/8/3pP3/8/8/8/K6k w - d6 0 1");
    const move = board.Move.init(36, 43, null);
    const key = CaptureHistory.keyFor(&position.board, move).?;

    try std.testing.expectEqual(piece.Type.pawn, key.moving_piece);
    try std.testing.expectEqual(piece.Type.pawn, key.captured_piece);
    try std.testing.expectEqual(@as(u8, 43), key.to_sq);
}

test "capture history uses the promoted piece and remains bounded" {
    var history = CaptureHistory.init();
    const position = try board.Board.fromFen("6kr/6P1/8/8/8/8/8/K7 w - - 0 1");
    const move = board.Move.init(54, 63, .queen);
    const key = CaptureHistory.keyFor(&position.board, move).?;
    try std.testing.expectEqual(piece.Type.queen, key.moving_piece);
    try std.testing.expectEqual(piece.Type.rook, key.captured_piece);

    for (0..256) |_| history.reward(key, 64);
    const saturated = history.get(key);
    try std.testing.expect(saturated > 0);
    try std.testing.expect(saturated <= MAX_SCORE);

    history.age();
    try std.testing.expect(@abs(history.get(key)) <= @abs(saturated));
}
