const std = @import("std");
const board = @import("bitboard.zig");
const Board = board.Board;
const Move = board.Move;
const MoveList = board.MoveList;
const piece = @import("piece.zig");
const UciError = @import("uci_error.zig").UciError;
const eval = @import("evaluation.zig");
const zobrist = @import("zobrist.zig");

/// Staged move picker for efficient move ordering
/// Generates moves lazily in stages: TT move -> Good captures -> Killers -> Quiet moves -> Bad captures
pub const MovePicker = struct {
    const Self = @This();

    pub const Stage = enum {
        tt_move,
        generate_captures,
        good_captures,
        generate_killers,
        killers,
        generate_quiets,
        quiets,
        bad_captures,
        done,
    };

    // Move scoring constants
    const TT_MOVE_SCORE: i32 = 2_000_000;
    const GOOD_CAPTURE_BASE: i32 = 1_000_000;
    const KILLER_SCORE: i32 = 900_000;
    const BAD_CAPTURE_SCORE: i32 = -100_000;

    board_ptr: *Board,
    stage: Stage,
    tt_move: ?Move,
    killer_moves: *const [MAX_KILLER_MOVES]Move,
    history: *const HistoryTable,
    ply: u32,

    // Move lists for each category
    captures: MoveList,
    quiets: MoveList,
    bad_captures: MoveList,

    // Scores for sorting
    capture_scores: [256]i32,
    quiet_scores: [256]i32,

    // Current indices
    capture_idx: usize,
    quiet_idx: usize,
    bad_capture_idx: usize,
    killer_idx: usize,

    pub fn init(
        board_ptr: *Board,
        tt_move: ?Move,
        killer_moves: *const [MAX_KILLER_MOVES]Move,
        history: *const HistoryTable,
        ply: u32,
    ) Self {
        return Self{
            .board_ptr = board_ptr,
            .stage = .tt_move,
            .tt_move = tt_move,
            .killer_moves = killer_moves,
            .history = history,
            .ply = ply,
            .captures = MoveList.init(),
            .quiets = MoveList.init(),
            .bad_captures = MoveList.init(),
            .capture_scores = undefined,
            .quiet_scores = undefined,
            .capture_idx = 0,
            .quiet_idx = 0,
            .bad_capture_idx = 0,
            .killer_idx = 0,
        };
    }

    /// Get the next move, or null if no more moves
    pub fn next(self: *Self) ?Move {
        while (true) {
            switch (self.stage) {
                .tt_move => {
                    self.stage = .generate_captures;
                    if (self.tt_move) |tt| {
                        // Validate TT move is pseudo-legal (basic check)
                        if (self.isPseudoLegal(tt)) {
                            return tt;
                        }
                    }
                },
                .generate_captures => {
                    self.board_ptr.generateLegalCaptures(&self.captures);
                    self.scoreCaptures();
                    self.stage = .good_captures;
                },
                .good_captures => {
                    while (self.capture_idx < self.captures.count) {
                        const idx = self.selectBestCapture();
                        const move = self.captures.moves[idx];
                        const score = self.capture_scores[idx];

                        // Swap to front and increment
                        if (idx != self.capture_idx) {
                            self.captures.moves[idx] = self.captures.moves[self.capture_idx];
                            self.capture_scores[idx] = self.capture_scores[self.capture_idx];
                            self.captures.moves[self.capture_idx] = move;
                            self.capture_scores[self.capture_idx] = score;
                        }
                        self.capture_idx += 1;

                        // Skip if same as TT move
                        if (self.isTTMove(move)) continue;

                        // Good capture (positive SEE or equal trade)
                        if (score >= GOOD_CAPTURE_BASE) {
                            return move;
                        } else {
                            // Bad capture - save for later
                            self.bad_captures.append(move);
                        }
                    }
                    self.stage = .generate_killers;
                },
                .generate_killers => {
                    self.stage = .killers;
                },
                .killers => {
                    while (self.killer_idx < MAX_KILLER_MOVES) {
                        const killer = self.killer_moves[self.killer_idx];
                        self.killer_idx += 1;

                        // Skip null/invalid killers
                        if (killer.from() == 0 and killer.to() == 0) continue;

                        // Skip if same as TT move
                        if (self.isTTMove(killer)) continue;

                        // Skip if it's a capture (already generated)
                        if (self.isCapture(killer)) continue;

                        // Verify it's pseudo-legal and legal
                        if (self.isPseudoLegalQuiet(killer) and self.isLegal(killer)) {
                            return killer;
                        }
                    }
                    self.stage = .generate_quiets;
                },
                .generate_quiets => {
                    self.board_ptr.generateLegalQuietMoves(&self.quiets);
                    self.scoreQuiets();
                    self.stage = .quiets;
                },
                .quiets => {
                    while (self.quiet_idx < self.quiets.count) {
                        const idx = self.selectBestQuiet();
                        const move = self.quiets.moves[idx];
                        const score = self.quiet_scores[idx];

                        // Swap to front and increment
                        if (idx != self.quiet_idx) {
                            self.quiets.moves[idx] = self.quiets.moves[self.quiet_idx];
                            self.quiet_scores[idx] = self.quiet_scores[self.quiet_idx];
                            self.quiets.moves[self.quiet_idx] = move;
                            self.quiet_scores[self.quiet_idx] = score;
                        }
                        self.quiet_idx += 1;

                        // Skip if same as TT move or killer
                        if (self.isTTMove(move)) continue;
                        if (self.isKiller(move)) continue;

                        return move;
                    }
                    self.stage = .bad_captures;
                },
                .bad_captures => {
                    if (self.bad_capture_idx < self.bad_captures.count) {
                        const move = self.bad_captures.moves[self.bad_capture_idx];
                        self.bad_capture_idx += 1;
                        return move;
                    }
                    self.stage = .done;
                },
                .done => return null,
            }
        }
    }

    fn scoreCaptures(self: *Self) void {
        const opponent_color = if (self.board_ptr.board.move == .white) piece.Color.black else piece.Color.white;

        for (self.captures.slice(), 0..) |move, i| {
            var score: i32 = GOOD_CAPTURE_BASE;

            // MVV-LVA scoring
            if (self.board_ptr.board.getPieceAt(move.to(), opponent_color)) |victim| {
                const victim_value = eval.getPieceValue(victim);
                const attacker_value = if (self.board_ptr.board.getPieceAt(move.from(), self.board_ptr.board.move)) |att|
                    eval.getPieceValue(att)
                else
                    0;

                // MVV-LVA: victim value * 10 - attacker value
                // Higher victim value = better, lower attacker value = better
                score += victim_value * 10 - attacker_value;

                // If victim < attacker, it might be a bad capture
                // Simple heuristic: if attacker significantly more valuable, mark as potentially bad
                if (attacker_value > victim_value + 50) {
                    score = BAD_CAPTURE_SCORE + victim_value * 10 - attacker_value;
                }
            }

            // Promotion bonus
            if (move.promotion()) |promo| {
                score += eval.getPieceValue(promo);
            }

            self.capture_scores[i] = score;
        }
    }

    fn scoreQuiets(self: *Self) void {
        const occupied = self.board_ptr.board.occupied();
        const friendly = self.board_ptr.board.getColorBitboard(self.board_ptr.board.move);

        for (self.quiets.slice(), 0..) |move, i| {
            var score = self.history.get(move);

            // Add mobility bonus - moves that increase piece mobility
            if (self.board_ptr.board.getPieceAt(move.from(), self.board_ptr.board.move)) |piece_type| {
                var mobility_bonus: i32 = 0;
                const to_sq: u8 = move.to();

                switch (piece_type) {
                    .knight => {
                        const attacks = board.getKnightAttacks(@intCast(to_sq)) & ~friendly;
                        mobility_bonus = @as(i32, @intCast(@popCount(attacks))) * 3;
                    },
                    .bishop => {
                        const attacks = board.getBishopAttacks(@intCast(to_sq), occupied) & ~friendly;
                        mobility_bonus = @as(i32, @intCast(@popCount(attacks))) * 2;
                    },
                    .rook => {
                        const attacks = board.getRookAttacks(@intCast(to_sq), occupied) & ~friendly;
                        mobility_bonus = @as(i32, @intCast(@popCount(attacks))) * 2;
                    },
                    .queen => {
                        const attacks = (board.getBishopAttacks(@intCast(to_sq), occupied) | board.getRookAttacks(@intCast(to_sq), occupied)) & ~friendly;
                        mobility_bonus = @as(i32, @intCast(@popCount(attacks)));
                    },
                    else => {},
                }
                score += mobility_bonus;
            }

            self.quiet_scores[i] = score;
        }
    }

    fn selectBestCapture(self: *Self) usize {
        var best_idx = self.capture_idx;
        var best_score = self.capture_scores[self.capture_idx];

        for (self.capture_idx + 1..self.captures.count) |i| {
            if (self.capture_scores[i] > best_score) {
                best_score = self.capture_scores[i];
                best_idx = i;
            }
        }

        return best_idx;
    }

    fn selectBestQuiet(self: *Self) usize {
        var best_idx = self.quiet_idx;
        var best_score = self.quiet_scores[self.quiet_idx];

        for (self.quiet_idx + 1..self.quiets.count) |i| {
            if (self.quiet_scores[i] > best_score) {
                best_score = self.quiet_scores[i];
                best_idx = i;
            }
        }

        return best_idx;
    }

    fn isTTMove(self: *Self, move: Move) bool {
        if (self.tt_move) |tt| {
            return tt.from() == move.from() and tt.to() == move.to() and Move.eqlPromotion(tt.promotion(), move.promotion());
        }
        return false;
    }

    fn isKiller(self: *Self, move: Move) bool {
        for (self.killer_moves) |killer| {
            if (killer.from() == move.from() and killer.to() == move.to()) {
                return true;
            }
        }
        return false;
    }

    fn isCapture(self: *Self, move: Move) bool {
        const opponent_color = if (self.board_ptr.board.move == .white) piece.Color.black else piece.Color.white;
        return self.board_ptr.board.getPieceAt(move.to(), opponent_color) != null or
            (self.board_ptr.board.en_passant_square == move.to() and
                self.board_ptr.board.getPieceAt(move.from(), self.board_ptr.board.move) == .pawn);
    }

    fn isPseudoLegal(self: *Self, move: Move) bool {
        const color = self.board_ptr.board.move;
        const from_sq = move.from();

        // Check if we have a piece on the from square
        const piece_type = self.board_ptr.board.getPieceAt(from_sq, color) orelse return false;

        // Basic validation - piece exists and destination isn't our own piece
        const our_pieces = self.board_ptr.board.getColorBitboard(color);
        const to_mask = @as(u64, 1) << @intCast(move.to());
        if ((our_pieces & to_mask) != 0) return false;

        // For promotions, verify it's a pawn on the right rank
        if (move.promotion() != null) {
            if (piece_type != .pawn) return false;
            const from_rank = from_sq / 8;
            if (color == .white and from_rank != 6) return false;
            if (color == .black and from_rank != 1) return false;
        }

        return true;
    }

    fn isPseudoLegalQuiet(self: *Self, move: Move) bool {
        const color = self.board_ptr.board.move;
        const from_sq = move.from();

        // Check if we have a piece on the from square
        const piece_type = self.board_ptr.board.getPieceAt(from_sq, color) orelse return false;

        // Destination must be empty (quiet move)
        const occupied = self.board_ptr.board.occupied();
        const to_mask = @as(u64, 1) << @intCast(move.to());
        if ((occupied & to_mask) != 0) return false;

        // Not a promotion (killers shouldn't be promotions in quiet context)
        if (move.promotion() != null) return false;

        _ = piece_type;
        return true;
    }

    fn isLegal(self: *Self, move: Move) bool {
        const color = self.board_ptr.board.move;

        // Save state
        const old_board = self.board_ptr.board;
        const old_hash = self.board_ptr.zobrist_hasher.zobrist_hash;

        // Make move
        self.board_ptr.applyMoveUnchecked(move);

        // Check legality
        const legal = !self.board_ptr.isInCheck(color);

        // Restore state
        self.board_ptr.board = old_board;
        self.board_ptr.zobrist_hasher.zobrist_hash = old_hash;

        return legal;
    }
};

pub const SearchOptions = struct {
    infinite: bool = false,
    move_time: ?u64 = null,
    wtime: ?u64 = null,
    btime: ?u64 = null,
    winc: ?u64 = null,
    binc: ?u64 = null,
    depth: ?u64 = null,
};

pub const SearchResult = struct {
    best_move: Move,
    score: i32,
    nodes: usize,
    time_ms: i64,
    depth: u32,
};

const MAX_PLY = 64;
const MAX_KILLER_MOVES = 2;

const INF: i32 = 32000;
const DRAW_SCORE: i32 = 0;

// Adjust mate score for TT storage (store relative to root, not current ply)
fn scoreToTT(score: i32, ply: u32) i32 {
    if (score >= eval.MATE_BOUND) {
        return score + @as(i32, @intCast(ply));
    } else if (score <= -eval.MATE_BOUND) {
        return score - @as(i32, @intCast(ply));
    }
    return score;
}

// Adjust mate score when retrieving from TT (convert back to ply-relative)
fn scoreFromTT(score: i32, ply: u32) i32 {
    if (score >= eval.MATE_BOUND) {
        return score - @as(i32, @intCast(ply));
    } else if (score <= -eval.MATE_BOUND) {
        return score + @as(i32, @intCast(ply));
    }
    return score;
}

// Late Move Reduction parameters
const LMR_MIN_DEPTH: u32 = 3;
const LMR_FULL_DEPTH_MOVES: u32 = 4;
const LMR_REDUCTION_LIMIT: u32 = 3;

// Null move pruning parameters
const NULL_MOVE_MIN_DEPTH: u32 = 3;
const NULL_MOVE_REDUCTION: u32 = 3;
const NULL_MOVE_VERIFICATION_DEPTH: u32 = 8; // Verify at deeper depths

// Futility pruning parameters
const FUTILITY_MARGIN: i32 = 200;
const FUTILITY_MARGIN_MULTIPLIER: i32 = 120;

// Razoring parameters
const RAZOR_MARGIN: i32 = 400;

// Transposition table entry
const TTEntryBound = enum(u8) {
    exact,
    lower_bound,
    upper_bound,
};

const TTEntry = struct {
    hash: u64,
    depth: u8,
    score: i32,
    bound: TTEntryBound,
    best_move: Move,
    age: u8,

    fn init() TTEntry {
        return TTEntry{
            .hash = 0,
            .depth = 0,
            .score = 0,
            .bound = .exact,
            .best_move = Move.init(0, 0, null),
            .age = 0,
        };
    }
};

// Transposition table
const TranspositionTable = struct {
    const Self = @This();

    entries: []TTEntry,
    size: usize,
    current_age: u8,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator, size_mb: usize) !Self {
        const entry_size = @sizeOf(TTEntry);
        const num_entries = (size_mb * 1024 * 1024) / entry_size;
        const entries = try allocator.alloc(TTEntry, num_entries);

        for (entries) |*entry| {
            entry.* = TTEntry.init();
        }

        return Self{
            .entries = entries,
            .size = num_entries,
            .current_age = 0,
            .allocator = allocator,
        };
    }

    fn deinit(self: *Self) void {
        self.allocator.free(self.entries);
    }

    fn clear(self: *Self) void {
        for (self.entries) |*entry| {
            entry.* = TTEntry.init();
        }
        self.current_age = 0;
    }

    fn nextAge(self: *Self) void {
        self.current_age +%= 1;
    }

    fn index(self: *Self, hash: u64) usize {
        return @as(usize, @intCast(hash % @as(u64, @intCast(self.size))));
    }

    fn probe(self: *Self, hash: u64) ?*TTEntry {
        const idx = self.index(hash);
        if (self.entries[idx].hash == hash) {
            return &self.entries[idx];
        }
        return null;
    }

    fn store(self: *Self, hash: u64, depth: u8, score: i32, bound: TTEntryBound, best_move: Move) void {
        const idx = self.index(hash);
        const entry = &self.entries[idx];

        // Replace if: empty, same position, deeper search, or older entry
        if (entry.hash == 0 or
            entry.hash == hash or
            depth >= entry.depth or
            entry.age != self.current_age)
        {
            entry.hash = hash;
            entry.depth = depth;
            entry.score = score;
            entry.bound = bound;
            entry.best_move = best_move;
            entry.age = self.current_age;
        }
    }
};

// Killer moves tracker
const KillerMoves = struct {
    moves: [MAX_PLY][MAX_KILLER_MOVES]Move,

    fn init() KillerMoves {
        return KillerMoves{
            .moves = [_][MAX_KILLER_MOVES]Move{[_]Move{Move.init(0, 0, null)} ** MAX_KILLER_MOVES} ** MAX_PLY,
        };
    }

    fn add(self: *KillerMoves, move: Move, ply: u32) void {
        if (ply >= MAX_PLY) return;

        // Don't add if already first killer
        if (self.moves[ply][0].from() == move.from() and
            self.moves[ply][0].to() == move.to())
        {
            return;
        }

        // Shift down and add new killer
        self.moves[ply][1] = self.moves[ply][0];
        self.moves[ply][0] = move;
    }

    fn isKiller(self: *KillerMoves, move: Move, ply: u32) bool {
        if (ply >= MAX_PLY) return false;

        for (self.moves[ply]) |killer| {
            if (killer.from() == move.from() and killer.to() == move.to()) {
                return true;
            }
        }
        return false;
    }
};

// Counter moves - moves that refute the opponent's last move
const CounterMoveTable = struct {
    moves: [64][64]Move,

    fn init() CounterMoveTable {
        return CounterMoveTable{
            .moves = [_][64]Move{[_]Move{Move.init(0, 0, null)} ** 64} ** 64,
        };
    }

    fn update(self: *CounterMoveTable, previous_move: Move, counter_move: Move) void {
        self.moves[previous_move.from()][previous_move.to()] = counter_move;
    }

    fn get(self: *const CounterMoveTable, previous_move: Move) ?Move {
        const move = self.moves[previous_move.from()][previous_move.to()];
        if (move.from() == 0 and move.to() == 0) return null;
        return move;
    }

    fn clear(self: *CounterMoveTable) void {
        self.moves = [_][64]Move{[_]Move{Move.init(0, 0, null)} ** 64} ** 64;
    }
};

// History heuristic with gravity/aging
const HistoryTable = struct {
    scores: [2][64][64]i32, // [color][from][to]

    fn init() HistoryTable {
        return HistoryTable{
            .scores = [_][64][64]i32{[_][64]i32{[_]i32{0} ** 64} ** 64} ** 2,
        };
    }

    fn update(self: *HistoryTable, move: Move, depth: u32, color: piece.Color) void {
        const c: usize = @intFromEnum(color);
        const bonus = @as(i32, @intCast(@min(depth * depth, 400)));
        const current = self.scores[c][move.from()][move.to()];
        // History gravity formula: bonus - bonus * |current| / max_history
        const abs_current: i32 = @intCast(@abs(current));
        const adjusted_bonus = bonus - @divTrunc(bonus * abs_current, 16384);
        self.scores[c][move.from()][move.to()] += adjusted_bonus;

        // Cap at reasonable value
        self.scores[c][move.from()][move.to()] = @max(-16384, @min(16384, self.scores[c][move.from()][move.to()]));
    }

    fn penalize(self: *HistoryTable, move: Move, depth: u32, color: piece.Color) void {
        const c: usize = @intFromEnum(color);
        const penalty = @as(i32, @intCast(@min(depth * depth, 400)));
        const current = self.scores[c][move.from()][move.to()];
        const abs_current: i32 = @intCast(@abs(current));
        const adjusted_penalty = penalty - @divTrunc(penalty * abs_current, 16384);
        self.scores[c][move.from()][move.to()] -= adjusted_penalty;
        self.scores[c][move.from()][move.to()] = @max(-16384, @min(16384, self.scores[c][move.from()][move.to()]));
    }

    fn get(self: *const HistoryTable, move: Move) i32 {
        // Return combined score (sum both colors for simplicity)
        return self.scores[0][move.from()][move.to()] + self.scores[1][move.from()][move.to()];
    }

    fn getForColor(self: *const HistoryTable, move: Move, color: piece.Color) i32 {
        const c: usize = @intFromEnum(color);
        return self.scores[c][move.from()][move.to()];
    }

    fn clear(self: *HistoryTable) void {
        self.scores = [_][64][64]i32{[_][64]i32{[_]i32{0} ** 64} ** 64} ** 2;
    }

    fn age(self: *HistoryTable) void {
        // Reduce all history scores by half between searches
        for (0..2) |c| {
            for (0..64) |from| {
                for (0..64) |to| {
                    self.scores[c][from][to] = @divTrunc(self.scores[c][from][to], 2);
                }
            }
        }
    }
};

pub const SearchEngine = struct {
    const Self = @This();

    board: *Board,
    allocator: std.mem.Allocator,
    stop_search: *std.atomic.Value(bool),
    uci_writer: ?std.io.AnyWriter,

    // Search state
    tt: TranspositionTable,
    killer_moves: KillerMoves,
    history: HistoryTable,
    counter_moves: CounterMoveTable,
    nodes_searched: usize,
    seldepth: u32,
    previous_move: ?Move,
    // Position history for repetition detection during search
    position_history: [512]u64,
    history_count: usize,
    // Game history (positions before search started) - for proper repetition detection
    game_history: [512]u64,
    game_history_count: usize,

    pub fn init(
        board_ptr: *Board,
        allocator: std.mem.Allocator,
        stop_search: *std.atomic.Value(bool),
    ) !Self {
        const tt = try TranspositionTable.init(allocator, 64); // 64MB TT

        return Self{
            .board = board_ptr,
            .allocator = allocator,
            .stop_search = stop_search,
            .uci_writer = null,
            .tt = tt,
            .killer_moves = KillerMoves.init(),
            .history = HistoryTable.init(),
            .counter_moves = CounterMoveTable.init(),
            .nodes_searched = 0,
            .seldepth = 0,
            .previous_move = null,
            .position_history = undefined,
            .history_count = 0,
            .game_history = undefined,
            .game_history_count = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.tt.deinit();
    }

    /// Set the game history from prior positions (for repetition detection)
    /// Call this before search with all positions from the game so far
    pub fn setGameHistory(self: *Self, positions: []const u64) void {
        const count = @min(positions.len, 512);
        for (0..count) |i| {
            self.game_history[i] = positions[i];
        }
        self.game_history_count = count;
    }

    /// Add current position to game history
    pub fn addPositionToHistory(self: *Self, hash: u64) void {
        if (self.game_history_count < 512) {
            self.game_history[self.game_history_count] = hash;
            self.game_history_count += 1;
        }
    }

    /// Run a search and return the best move
    pub fn search(self: *Self, options: SearchOptions) !SearchResult {
        const start_time = std.time.milliTimestamp();

        // Reset search state
        self.nodes_searched = 0;
        self.seldepth = 0;
        self.killer_moves = KillerMoves.init();
        self.history.age(); // Age history instead of clearing
        self.counter_moves.clear();
        self.tt.nextAge();
        self.previous_move = null;
        // Initialize position history
        self.position_history[0] = self.board.zobrist_hasher.zobrist_hash;
        self.history_count = 1;

        // Calculate time limit
        const time_limit = self.calculateTimeLimit(options);

        // Generate legal moves
        var legal_moves = MoveList.init();
        try self.board.generateLegalMoves(&legal_moves);

        if (legal_moves.count == 0) {
            return SearchResult{
                .best_move = Move.init(0, 0, null),
                .score = 0,
                .nodes = 0,
                .time_ms = 0,
                .depth = 0,
            };
        }

        // If only one legal move, return it immediately
        if (legal_moves.count == 1) {
            return SearchResult{
                .best_move = legal_moves.moves[0],
                .score = 0,
                .nodes = 1,
                .time_ms = std.time.milliTimestamp() - start_time,
                .depth = 0,
            };
        }

        var best_move = legal_moves.moves[0];
        var best_score: i32 = -INF;
        const max_depth: u32 = if (options.depth) |d| @intCast(d) else 64;

        // Iterative deepening
        var depth: u32 = 1;
        while (depth <= max_depth) : (depth += 1) {
            if (self.stop_search.load(.seq_cst)) break;

            const iter_start = std.time.milliTimestamp();
            const score = try self.alphaBeta(-INF, INF, depth, 0, true);

            // Skip incomplete iterations (stopped mid-search)
            if (self.stop_search.load(.seq_cst)) break;

            // Get best move from TT - this should always succeed after a complete search
            if (self.tt.probe(self.board.zobrist_hasher.zobrist_hash)) |entry| {
                // Only update if we found a valid move
                if (entry.best_move.from() != 0 or entry.best_move.to() != 0) {
                    best_move = entry.best_move;
                    best_score = score;
                }
            }

            const iter_time = std.time.milliTimestamp() - iter_start;
            const total_time = std.time.milliTimestamp() - start_time;

            // UCI info output at every depth
            if (self.uci_writer) |writer| {
                const nps = if (total_time > 0) (self.nodes_searched * 1000) / @as(usize, @intCast(total_time)) else 0;
                writer.print("info depth {d} seldepth {d} score cp {d} nodes {d} time {d} nps {d} pv {s}\n", .{
                    depth,
                    self.seldepth,
                    score,
                    self.nodes_searched,
                    total_time,
                    nps,
                    best_move,
                }) catch {};
            }

            // Check time limit
            if (time_limit) |limit| {
                // Stop if we've used most of our time or if next iteration unlikely to finish
                if (total_time >= limit or total_time + (iter_time * 2) >= limit) {
                    break;
                }
            }

            // Stop if we found a mate
            if (eval.isMateScore(score)) {
                break;
            }
        }

        const elapsed = std.time.milliTimestamp() - start_time;

        return SearchResult{
            .best_move = best_move,
            .score = best_score,
            .nodes = self.nodes_searched,
            .time_ms = elapsed,
            .depth = depth - 1,
        };
    }

    fn calculateTimeLimit(self: *Self, options: SearchOptions) ?u64 {
        if (options.infinite) {
            return null;
        } else if (options.move_time) |move_time| {
            return move_time;
        }

        // Determine which color we are and get our time/increment
        const our_time: ?u64 = if (self.board.board.move == piece.Color.white)
            options.wtime
        else
            options.btime;

        const our_increment: u64 = if (self.board.board.move == piece.Color.white)
            options.winc orelse 0
        else
            options.binc orelse 0;

        if (our_time == null) {
            return null;
        }

        const time_remaining = our_time.?;

        // Calculate moves played and estimate moves remaining
        const moves_played = self.board.board.fullmove_number;
        const estimated_moves_remaining = @max(20, 40 - @min(moves_played, 40));

        // Base time allocation: divide remaining time by estimated moves
        // Use a slightly larger divisor for safety margin
        const divisor = estimated_moves_remaining + 5;
        const base_time = time_remaining / divisor;

        // Add most of the increment since we get it back after the move
        const time_budget = base_time + (our_increment * 9 / 10);

        // Apply min/max bounds
        const min_time: u64 = 100; // Minimum 100ms
        var max_time = @min(time_remaining / 5, 30000); // Max 20% of remaining or 30s

        // If low on time, be more conservative
        if (time_remaining < 10000) {
            max_time = @min(time_remaining * 15 / 100, max_time);
        } else if (time_remaining < 5000) {
            max_time = @min(time_remaining / 10, max_time);
        }

        const time_for_move = @max(min_time, @min(time_budget, max_time));

        return time_for_move;
    }

    /// Alpha-beta search (negamax variant) with various pruning techniques
    fn alphaBeta(self: *Self, alpha_in: i32, beta: i32, depth: u32, ply: u32, do_null: bool) !i32 {
        // Quiescence at depth 0
        if (depth == 0) {
            return self.quiescence(alpha_in, beta, ply);
        }

        if (self.stop_search.load(.seq_cst)) {
            return 0;
        }

        self.seldepth = @max(self.seldepth, ply);

        // Check for draw by repetition
        if (ply > 0 and self.isRepetition()) {
            return DRAW_SCORE;
        }

        // Check for draw by 50 move rule
        if (self.board.board.halfmove_clock >= 100) {
            return DRAW_SCORE;
        }

        const in_check = self.board.isInCheck(self.board.board.move);
        var alpha = alpha_in;
        var beta_adj = beta;

        // Mate distance pruning - don't search for mates we can't reach
        alpha = @max(alpha, -eval.mateIn(ply));
        beta_adj = @min(beta_adj, eval.mateIn(ply + 1));
        if (alpha >= beta_adj) {
            return alpha;
        }

        const original_alpha = alpha;

        // Check extension - extend search when in check
        var search_depth = depth;
        if (in_check and depth < 20) {
            search_depth += 1;
        }

        const is_pv_node = (beta_adj - alpha) > 1;

        // Probe transposition table
        var tt_move: ?Move = null;
        if (self.tt.probe(self.board.zobrist_hasher.zobrist_hash)) |entry| {
            tt_move = entry.best_move;

            if (entry.depth >= search_depth) {
                const tt_score = scoreFromTT(entry.score, ply);

                if (!is_pv_node) {
                    switch (entry.bound) {
                        .exact => return tt_score,
                        .lower_bound => alpha = @max(alpha, tt_score),
                        .upper_bound => beta_adj = @min(beta_adj, tt_score),
                    }

                    if (alpha >= beta_adj) {
                        return tt_score;
                    }
                }
            }
        }

        // Static evaluation for pruning decisions
        const static_eval = if (!in_check) eval.evaluate(self.board) else -INF;

        // Razoring - if we're far behind, drop into quiescence
        if (!is_pv_node and !in_check and depth <= 2 and static_eval + RAZOR_MARGIN < alpha) {
            const razor_score = try self.quiescence(alpha, beta_adj, ply);
            if (razor_score < alpha) {
                return razor_score;
            }
        }

        // Null move pruning - try giving opponent a free move
        if (do_null and !is_pv_node and !in_check and depth >= NULL_MOVE_MIN_DEPTH and static_eval >= beta_adj) {
            // Don't do null move if we're in a pawn endgame or have very little material
            const our_non_pawn_material = countMaterial(self.board.board, self.board.board.move) -
                @as(i32, @intCast(@popCount(self.board.board.getColorBitboard(self.board.board.move) & self.board.board.getKindBitboard(.pawn)))) * eval.PAWN_VALUE;

            if (our_non_pawn_material > eval.BISHOP_VALUE) {
                // Save state
                const old_board = self.board.board;
                const old_hash = self.board.zobrist_hasher.zobrist_hash;

                // Make null move (just flip side to move)
                self.board.board.move = if (self.board.board.move == .white) .black else .white;
                self.board.zobrist_hasher.zobrist_hash ^= zobrist.RandomTurn;

                // Clear en passant
                if (self.board.board.en_passant_square) |ep_sq| {
                    self.board.zobrist_hasher.zobrist_hash ^= zobrist.RandomEnPassant[ep_sq % 8];
                    self.board.board.en_passant_square = null;
                }

                // Search with reduced depth - adaptive reduction based on depth and eval margin
                var reduction: u32 = NULL_MOVE_REDUCTION;
                if (depth > 6) reduction += 1;
                if (static_eval - beta_adj > 200) reduction += 1; // More reduction if we're way ahead
                reduction = @min(reduction, depth - 1);

                const null_score = -try self.alphaBeta(-beta_adj, -beta_adj + 1, depth -| reduction, ply + 1, false);

                // Restore state
                self.board.board = old_board;
                self.board.zobrist_hasher.zobrist_hash = old_hash;

                // Beta cutoff from null move
                if (null_score >= beta_adj) {
                    // Don't return mate scores from null move
                    if (eval.isMateScore(null_score)) {
                        return beta_adj;
                    }

                    // Verification search at high depths to avoid zugzwang
                    if (depth >= NULL_MOVE_VERIFICATION_DEPTH) {
                        const verify_score = try self.alphaBeta(beta_adj - 1, beta_adj, depth - reduction, ply, false);
                        if (verify_score >= beta_adj) {
                            return null_score;
                        }
                    } else {
                        return null_score;
                    }
                }
            }
        }

        // Futility pruning - if we're way behind and near the leaf, prune quiet moves
        var futile = false;
        if (!is_pv_node and !in_check and depth <= 3) {
            const futility_margin = FUTILITY_MARGIN + FUTILITY_MARGIN_MULTIPLIER * @as(i32, @intCast(depth));
            if (static_eval + futility_margin < alpha) {
                futile = true;
            }
        }

        // Use staged move picker for efficient move ordering
        const killers = if (ply < MAX_PLY) &self.killer_moves.moves[ply] else &[_]Move{Move.init(0, 0, null)} ** MAX_KILLER_MOVES;
        var move_picker = MovePicker.init(self.board, tt_move, killers, &self.history, ply);

        var best_move: ?Move = null;
        var best_score: i32 = -INF;
        var moves_searched: u32 = 0;
        var quiets_tried: [64]Move = undefined;
        var quiets_tried_count: usize = 0;
        const color = self.board.board.move;

        while (move_picker.next()) |move| {
            // Save state
            const old_board = self.board.board;
            const old_hash = self.board.zobrist_hasher.zobrist_hash;
            const old_previous = self.previous_move;

            const is_capture = old_board.getPieceAt(move.to(), if (old_board.move == .white) .black else .white) != null;
            const is_promotion = move.promotion() != null;

            // Futility pruning - skip quiet moves if futile (but not if move gives check)
            if (futile and !is_capture and !is_promotion and moves_searched > 0) {
                // Make the move temporarily to check if it gives check
                self.board.makeMoveUnchecked(move);
                const gives_check = self.board.isInCheck(self.board.board.move);
                self.board.board = old_board;
                self.board.zobrist_hasher.zobrist_hash = old_hash;

                if (!gives_check and !self.killer_moves.isKiller(move, ply)) {
                    continue;
                }
            }

            // Make move
            self.board.makeMoveUnchecked(move);
            self.previous_move = move;
            self.nodes_searched += 1;

            // Add position to history for repetition detection
            const old_hist_count = self.history_count;
            if (self.history_count < 512) {
                self.position_history[self.history_count] = self.board.zobrist_hasher.zobrist_hash;
                self.history_count += 1;
            }

            var score: i32 = undefined;

            // Late Move Reductions (LMR)
            if (moves_searched >= LMR_FULL_DEPTH_MOVES and
                depth >= LMR_MIN_DEPTH and
                !in_check and
                !is_capture and
                !is_promotion and
                !self.board.isInCheck(self.board.board.move))
            {
                // Calculate reduction based on depth and moves searched
                var reduction: u32 = 1;
                if (moves_searched > 6) reduction += 1;
                if (depth > 6) reduction += 1;
                if (!is_pv_node) reduction += 1;
                reduction = @min(reduction, LMR_REDUCTION_LIMIT);
                reduction = @min(reduction, depth - 1);

                // Search with reduced depth
                score = -try self.alphaBeta(-alpha - 1, -alpha, depth - 1 - reduction, ply + 1, true);

                // If LMR found something good, re-search at full depth
                if (score > alpha) {
                    score = -try self.alphaBeta(-alpha - 1, -alpha, depth - 1, ply + 1, true);
                }
            } else {
                // PVS - search with null window first if not first move
                if (moves_searched > 0) {
                    score = -try self.alphaBeta(-alpha - 1, -alpha, depth - 1, ply + 1, true);
                } else {
                    // First move always searched with full window
                    score = alpha + 1; // Force full search
                }
            }

            // Full window search if PVS failed high or first move
            if (score > alpha) {
                score = -try self.alphaBeta(-beta, -alpha, depth - 1, ply + 1, true);
            }

            // Unmake move
            self.board.board = old_board;
            self.board.zobrist_hasher.zobrist_hash = old_hash;
            self.previous_move = old_previous;
            self.history_count = old_hist_count; // Restore history count

            moves_searched += 1;

            // Track quiet moves for history updates
            if (!is_capture and !is_promotion and quiets_tried_count < 64) {
                quiets_tried[quiets_tried_count] = move;
                quiets_tried_count += 1;
            }

            if (score > best_score) {
                best_score = score;
                best_move = move;
            }

            alpha = @max(alpha, score);

            // Beta cutoff
            if (alpha >= beta) {
                // Update heuristics for non-captures
                if (!is_capture and !is_promotion) {
                    self.killer_moves.add(move, ply);
                    self.history.update(move, search_depth, color);

                    // Update counter move
                    if (old_previous) |prev| {
                        self.counter_moves.update(prev, move);
                    }

                    // Penalize other quiets that were tried before the cutoff
                    for (quiets_tried[0..quiets_tried_count]) |quiet| {
                        if (quiet.from() != move.from() or quiet.to() != move.to()) {
                            self.history.penalize(quiet, search_depth, color);
                        }
                    }
                }
                break;
            }
        }

        // Checkmate/stalemate detection
        if (moves_searched == 0) {
            if (in_check) {
                return -eval.mateIn(ply); // Checkmate
            } else {
                return DRAW_SCORE; // Stalemate
            }
        }

        // Store in transposition table (adjust mate scores for storage)
        const bound: TTEntryBound = if (best_score <= original_alpha)
            .upper_bound
        else if (best_score >= beta)
            .lower_bound
        else
            .exact;

        const score_to_store = scoreToTT(best_score, ply);
        self.tt.store(self.board.zobrist_hasher.zobrist_hash, @intCast(search_depth), score_to_store, bound, best_move orelse Move.init(0, 0, null));

        return best_score;
    }

    /// Check if current position is a repetition
    fn isRepetition(self: *Self) bool {
        const current_hash = self.board.zobrist_hasher.zobrist_hash;
        var count: u32 = 0;

        // First check game history (positions before search started)
        // Check every other position (same side to move) going backwards from most recent
        if (self.game_history_count > 0) {
            var i: usize = self.game_history_count - 1;
            var moves_checked: usize = 0;
            const max_check = @min(self.game_history_count, self.board.board.halfmove_clock + 1);

            while (moves_checked < max_check) {
                if (i % 2 == (self.game_history_count - 1) % 2) { // Same side to move
                    if (self.game_history[i] == current_hash) {
                        count += 1;
                        if (count >= 1) return true; // One prior occurrence = draw
                    }
                }
                if (i == 0) break;
                i -= 1;
                moves_checked += 1;
            }
        }

        // Then check search history (positions during current search)
        // Only need to check positions since last irreversible move
        const halfmove = @as(usize, @intCast(self.board.board.halfmove_clock));
        const start = if (self.history_count > halfmove)
            self.history_count - halfmove
        else
            0;

        // Check every other position (same side to move)
        var i = start;
        while (i + 2 < self.history_count) : (i += 2) {
            if (self.position_history[i] == current_hash) {
                return true; // Found repetition in search
            }
        }
        return false;
    }

    fn countMaterial(b: board.BitBoard, color: piece.Color) i32 {
        const color_bb = b.getColorBitboard(color);
        var material: i32 = 0;

        material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.pawn)))) * eval.PAWN_VALUE;
        material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.knight)))) * eval.KNIGHT_VALUE;
        material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.bishop)))) * eval.BISHOP_VALUE;
        material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.rook)))) * eval.ROOK_VALUE;
        material += @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.queen)))) * eval.QUEEN_VALUE;

        return material;
    }

    /// Quiescence search - search only tactical moves to avoid horizon effect
    fn quiescence(self: *Self, alpha_in: i32, beta: i32, ply: u32) !i32 {
        if (self.stop_search.load(.seq_cst)) {
            return 0;
        }

        self.nodes_searched += 1;
        self.seldepth = @max(self.seldepth, ply);

        // Check if we're in check - if so, we must search all evasions (not just captures)
        const in_check = self.board.isInCheck(self.board.board.move);

        var alpha = alpha_in;

        // Stand pat - but only when not in check
        var stand_pat: i32 = -INF;
        if (!in_check) {
            stand_pat = eval.evaluate(self.board);

            if (stand_pat >= beta) {
                return beta;
            }
            if (alpha < stand_pat) {
                alpha = stand_pat;
            }
        }

        // Generate moves - all moves if in check, only captures otherwise
        var moves = MoveList.init();
        if (in_check) {
            // When in check, we need to search all legal moves to find an escape
            try self.board.generateLegalMoves(&moves);
        } else {
            self.board.generateLegalCaptures(&moves);
        }

        // Check for checkmate/stalemate when in check with no moves
        if (moves.count == 0) {
            if (in_check) {
                return -eval.mateIn(ply); // Checkmate
            }
            return stand_pat; // No captures available, return stand pat
        }

        // Order captures by MVV-LVA
        self.orderCaptures(&moves);

        const opponent_color = if (self.board.board.move == .white) piece.Color.black else piece.Color.white;

        for (moves.slice()) |move| {
            // Delta pruning - skip captures that can't possibly raise alpha (skip when in check)
            if (!in_check) {
                const captured_value = if (self.board.board.getPieceAt(move.to(), opponent_color)) |p|
                    eval.getPieceValue(p)
                else
                    0;

                // Add promotion value to delta
                const promo_value: i32 = if (move.promotion()) |promo|
                    eval.getPieceValue(promo) - eval.PAWN_VALUE
                else
                    0;

                // Delta pruning margin increased to 300 to avoid being too aggressive
                if (stand_pat + captured_value + promo_value + 300 < alpha) {
                    continue; // Futile capture
                }
            }

            // Save state
            const old_board = self.board.board;
            const old_hash = self.board.zobrist_hasher.zobrist_hash;

            // Make move
            self.board.makeMoveUnchecked(move);

            // Recursive search
            const score = -try self.quiescence(-beta, -alpha, ply + 1);

            // Unmake move
            self.board.board = old_board;
            self.board.zobrist_hasher.zobrist_hash = old_hash;

            if (score >= beta) {
                return beta;
            }
            if (score > alpha) {
                alpha = score;
            }
        }

        return alpha;
    }

    /// Generate only tactical moves (captures and promotions)
    fn generateTacticalMoves(self: *Self, moves: *MoveList) !void {
        var all_moves = MoveList.init();
        try self.board.generateLegalMoves(&all_moves);

        const opponent_color = if (self.board.board.move == .white) piece.Color.black else piece.Color.white;

        for (all_moves.slice()) |move| {
            // Include captures
            if (self.board.board.getPieceAt(move.to(), opponent_color) != null) {
                moves.append(move);
            }
            // Include promotions
            else if (move.promotion() != null) {
                moves.append(move);
            }
            // Include en passant
            else if (self.board.board.en_passant_square == move.to()) {
                const piece_type = self.board.board.getPieceAt(move.from(), self.board.board.move);
                if (piece_type == .pawn) {
                    moves.append(move);
                }
            }
        }
    }

    /// Order moves for better alpha-beta performance
    fn orderMoves(self: *Self, moves: *MoveList, tt_move: ?Move, ply: u32) void {
        const move_slice = moves.sliceMut();

        // Score each move
        var scores: [256]i32 = undefined;

        const opponent_color = if (self.board.board.move == .white) piece.Color.black else piece.Color.white;

        for (move_slice, 0..) |move, i| {
            var score: i32 = 0;

            // TT move gets highest priority
            if (tt_move) |tt| {
                if (tt.from() == move.from() and tt.to() == move.to() and Move.eqlPromotion(tt.promotion(), move.promotion())) {
                    scores[i] = 1000000;
                    continue;
                }
            }

            // Captures scored by MVV-LVA
            if (self.board.board.getPieceAt(move.to(), opponent_color)) |victim| {
                const victim_value = eval.getPieceValue(victim);
                const attacker_value = if (self.board.board.getPieceAt(move.from(), self.board.board.move)) |att|
                    eval.getPieceValue(att)
                else
                    0;

                score = victim_value * 100 - attacker_value + 10000;
            }
            // Promotions
            else if (move.promotion()) |promo| {
                score = eval.getPieceValue(promo) + 9000;
            }
            // Killer moves
            else if (self.killer_moves.isKiller(move, ply)) {
                score = 8000;
            }
            // History heuristic
            else {
                score = self.history.get(move);
            }

            scores[i] = score;
        }

        // Simple selection sort (good enough for small move lists)
        for (0..move_slice.len) |i| {
            var best_idx = i;
            for (i + 1..move_slice.len) |j| {
                if (scores[j] > scores[best_idx]) {
                    best_idx = j;
                }
            }

            if (best_idx != i) {
                const tmp_move = move_slice[i];
                move_slice[i] = move_slice[best_idx];
                move_slice[best_idx] = tmp_move;

                const tmp_score = scores[i];
                scores[i] = scores[best_idx];
                scores[best_idx] = tmp_score;
            }
        }
    }

    /// Order captures by MVV-LVA
    fn orderCaptures(self: *Self, moves: *MoveList) void {
        const move_slice = moves.sliceMut();
        var scores: [256]i32 = undefined;

        const opponent_color = if (self.board.board.move == .white) piece.Color.black else piece.Color.white;

        for (move_slice, 0..) |move, i| {
            var score: i32 = 0;

            if (self.board.board.getPieceAt(move.to(), opponent_color)) |victim| {
                const victim_value = eval.getPieceValue(victim);
                const attacker_value = if (self.board.board.getPieceAt(move.from(), self.board.board.move)) |att|
                    eval.getPieceValue(att)
                else
                    0;

                score = victim_value * 100 - attacker_value;
            }

            scores[i] = score;
        }

        // Selection sort
        for (0..move_slice.len) |i| {
            var best_idx = i;
            for (i + 1..move_slice.len) |j| {
                if (scores[j] > scores[best_idx]) {
                    best_idx = j;
                }
            }

            if (best_idx != i) {
                const tmp_move = move_slice[i];
                move_slice[i] = move_slice[best_idx];
                move_slice[best_idx] = tmp_move;

                const tmp_score = scores[i];
                scores[i] = scores[best_idx];
                scores[best_idx] = tmp_score;
            }
        }
    }
};
