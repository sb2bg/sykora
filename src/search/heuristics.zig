const board = @import("../bitboard.zig");
const Move = board.Move;
const piece = @import("../piece.zig");

pub const MAX_PLY = 64;
pub const MAX_KILLER_MOVES = 2;

pub const KillerMoves = struct {
    moves: [MAX_PLY][MAX_KILLER_MOVES]Move,

    pub fn init() KillerMoves {
        return KillerMoves{
            .moves = [_][MAX_KILLER_MOVES]Move{[_]Move{Move.init(0, 0, null)} ** MAX_KILLER_MOVES} ** MAX_PLY,
        };
    }

    pub fn add(self: *KillerMoves, move: Move, ply: u32) void {
        if (ply >= MAX_PLY) return;

        if (self.moves[ply][0].from() == move.from() and
            self.moves[ply][0].to() == move.to())
        {
            return;
        }

        self.moves[ply][1] = self.moves[ply][0];
        self.moves[ply][0] = move;
    }

    pub fn isKiller(self: *KillerMoves, move: Move, ply: u32) bool {
        if (ply >= MAX_PLY) return false;

        for (self.moves[ply]) |killer| {
            if (killer.from() == move.from() and killer.to() == move.to()) {
                return true;
            }
        }
        return false;
    }
};

pub const CounterMoveTable = struct {
    moves: [64][64]Move,

    pub fn init() CounterMoveTable {
        return CounterMoveTable{
            .moves = [_][64]Move{[_]Move{Move.init(0, 0, null)} ** 64} ** 64,
        };
    }

    pub fn update(self: *CounterMoveTable, previous_move: Move, counter_move: Move) void {
        self.moves[previous_move.from()][previous_move.to()] = counter_move;
    }

    pub fn get(self: *const CounterMoveTable, previous_move: Move) ?Move {
        const move = self.moves[previous_move.from()][previous_move.to()];
        if (move.from() == 0 and move.to() == 0) return null;
        return move;
    }

    pub fn clear(self: *CounterMoveTable) void {
        self.moves = [_][64]Move{[_]Move{Move.init(0, 0, null)} ** 64} ** 64;
    }
};

pub const HistoryTable = struct {
    scores: [2][64][64]i32,

    pub fn init() HistoryTable {
        return HistoryTable{
            .scores = [_][64][64]i32{[_][64]i32{[_]i32{0} ** 64} ** 64} ** 2,
        };
    }

    pub fn update(self: *HistoryTable, move: Move, depth: u32, color: piece.Color) void {
        const c: usize = @intFromEnum(color);
        const bonus = @as(i32, @intCast(@min(depth * depth, 400)));
        const current = self.scores[c][move.from()][move.to()];
        const abs_current: i32 = @intCast(@abs(current));
        const adjusted_bonus = bonus - @divTrunc(bonus * abs_current, 16384);
        self.scores[c][move.from()][move.to()] += adjusted_bonus;
        self.scores[c][move.from()][move.to()] = @max(-16384, @min(16384, self.scores[c][move.from()][move.to()]));
    }

    pub fn penalize(self: *HistoryTable, move: Move, depth: u32, color: piece.Color) void {
        const c: usize = @intFromEnum(color);
        const penalty = @as(i32, @intCast(@min(depth * depth, 400)));
        const current = self.scores[c][move.from()][move.to()];
        const abs_current: i32 = @intCast(@abs(current));
        const adjusted_penalty = penalty - @divTrunc(penalty * abs_current, 16384);
        self.scores[c][move.from()][move.to()] -= adjusted_penalty;
        self.scores[c][move.from()][move.to()] = @max(-16384, @min(16384, self.scores[c][move.from()][move.to()]));
    }

    pub fn get(self: *const HistoryTable, move: Move) i32 {
        return self.scores[0][move.from()][move.to()] + self.scores[1][move.from()][move.to()];
    }

    pub fn getForColor(self: *const HistoryTable, move: Move, color: piece.Color) i32 {
        const c: usize = @intFromEnum(color);
        return self.scores[c][move.from()][move.to()];
    }

    pub fn clear(self: *HistoryTable) void {
        self.scores = [_][64][64]i32{[_][64]i32{[_]i32{0} ** 64} ** 64} ** 2;
    }

    pub fn age(self: *HistoryTable) void {
        for (0..2) |c| {
            for (0..64) |from| {
                for (0..64) |to| {
                    self.scores[c][from][to] = @divTrunc(self.scores[c][from][to], 2);
                }
            }
        }
    }
};
