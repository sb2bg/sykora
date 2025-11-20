const std = @import("std");
const board = @import("bitboard.zig");
const Board = board.Board;
const Move = board.Move;
const MoveList = board.MoveList;
const piece = @import("piece.zig");
const UciError = @import("uci_error.zig").UciError;
const eval = @import("evaluation.zig");

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

        // Replace if: empty, same position, or lower depth and older age
        if (entry.hash == 0 or entry.hash == hash or
            (entry.depth <= depth and entry.age != self.current_age))
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
        if (self.moves[ply][0].from == move.from and
            self.moves[ply][0].to == move.to)
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
            if (killer.from == move.from and killer.to == move.to) {
                return true;
            }
        }
        return false;
    }
};

// History heuristic
const HistoryTable = struct {
    scores: [64][64]i32,

    fn init() HistoryTable {
        return HistoryTable{
            .scores = [_][64]i32{[_]i32{0} ** 64} ** 64,
        };
    }

    fn update(self: *HistoryTable, move: Move, depth: u32) void {
        const bonus = @as(i32, @intCast(depth * depth));
        self.scores[move.from][move.to] += bonus;

        // Cap at a reasonable value to prevent overflow
        if (self.scores[move.from][move.to] > 10000) {
            self.scores[move.from][move.to] = 10000;
        }
    }

    fn get(self: *HistoryTable, move: Move) i32 {
        return self.scores[move.from][move.to];
    }

    fn clear(self: *HistoryTable) void {
        self.scores = [_][64]i32{[_]i32{0} ** 64} ** 64;
    }
};

pub const SearchEngine = struct {
    const Self = @This();

    board: *Board,
    allocator: std.mem.Allocator,
    stop_search: *std.atomic.Value(bool),
    info_callback: ?*const fn ([]const u8) void,

    // Search state
    tt: TranspositionTable,
    killer_moves: KillerMoves,
    history: HistoryTable,
    nodes_searched: usize,
    seldepth: u32,

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
            .info_callback = null,
            .tt = tt,
            .killer_moves = KillerMoves.init(),
            .history = HistoryTable.init(),
            .nodes_searched = 0,
            .seldepth = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.tt.deinit();
    }

    /// Run a search and return the best move
    pub fn search(self: *Self, options: SearchOptions) !SearchResult {
        const start_time = std.time.milliTimestamp();

        // Reset search state
        self.nodes_searched = 0;
        self.seldepth = 0;
        self.killer_moves = KillerMoves.init();
        self.history.clear();
        self.tt.nextAge();

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

            // Get best move from TT
            if (self.tt.probe(self.board.zobrist_hasher.zobrist_hash)) |entry| {
                if (entry.depth >= depth) {
                    best_move = entry.best_move;
                    best_score = score;
                }
            }

            const iter_time = std.time.milliTimestamp() - iter_start;
            const total_time = std.time.milliTimestamp() - start_time;

            // UCI info output
            if (self.info_callback) |callback| {
                var buf: [256]u8 = undefined;
                const info = std.fmt.bufPrint(&buf, "info depth {d} seldepth {d} score cp {d} nodes {d} time {d} nps {d} pv {s}\n", .{
                    depth,
                    self.seldepth,
                    score,
                    self.nodes_searched,
                    total_time,
                    if (total_time > 0) (self.nodes_searched * 1000) / @as(usize, @intCast(total_time)) else 0,
                    best_move,
                }) catch "";
                callback(info);
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

    /// Alpha-beta search (negamax variant)
    fn alphaBeta(self: *Self, alpha_in: i32, beta: i32, depth: u32, ply: u32, do_null: bool) !i32 {
        _ = do_null; // Reserved for null move pruning

        if (depth == 0) {
            return self.quiescence(alpha_in, beta, ply);
        }

        if (self.stop_search.load(.seq_cst)) {
            return 0;
        }

        self.seldepth = @max(self.seldepth, ply);

        // Check for draw by repetition (simplified - just check 50 move rule)
        if (self.board.board.halfmove_clock >= 100) {
            return DRAW_SCORE;
        }

        var alpha = alpha_in;
        const original_alpha = alpha;

        // Probe transposition table
        var tt_move: ?Move = null;
        if (self.tt.probe(self.board.zobrist_hasher.zobrist_hash)) |entry| {
            if (entry.depth >= depth) {
                tt_move = entry.best_move;

                switch (entry.bound) {
                    .exact => return entry.score,
                    .lower_bound => alpha = @max(alpha, entry.score),
                    .upper_bound => {},
                }

                if (alpha >= beta) {
                    return entry.score;
                }
            } else {
                tt_move = entry.best_move;
            }
        }

        // Generate moves
        var moves = MoveList.init();
        try self.board.generateLegalMoves(&moves);

        // Checkmate/stalemate detection
        if (moves.count == 0) {
            if (self.board.isInCheck(self.board.board.move)) {
                return -eval.mateIn(ply); // Checkmate
            } else {
                return DRAW_SCORE; // Stalemate
            }
        }

        // Order moves
        self.orderMoves(&moves, tt_move, ply);

        var best_move = moves.moves[0];
        var best_score: i32 = -INF;

        for (moves.slice()) |move| {
            // Save state
            const old_board = self.board.board;
            const old_hash = self.board.zobrist_hasher.zobrist_hash;

            // Make move
            self.board.makeMoveUnchecked(move);
            self.nodes_searched += 1;

            // Recursive search
            const score = -try self.alphaBeta(-beta, -alpha, depth - 1, ply + 1, true);

            // Unmake move
            self.board.board = old_board;
            self.board.zobrist_hasher.zobrist_hash = old_hash;

            if (score > best_score) {
                best_score = score;
                best_move = move;
            }

            alpha = @max(alpha, score);

            // Beta cutoff
            if (alpha >= beta) {
                // Store killer move for non-captures
                const is_capture = old_board.getPieceAt(move.to, if (old_board.move == .white) .black else .white) != null;
                if (!is_capture) {
                    self.killer_moves.add(move, ply);
                    self.history.update(move, depth);
                }
                break;
            }
        }

        // Store in transposition table
        const bound: TTEntryBound = if (best_score <= original_alpha)
            .upper_bound
        else if (best_score >= beta)
            .lower_bound
        else
            .exact;

        self.tt.store(self.board.zobrist_hasher.zobrist_hash, @intCast(depth), best_score, bound, best_move);

        return best_score;
    }

    /// Quiescence search - search only tactical moves to avoid horizon effect
    fn quiescence(self: *Self, alpha_in: i32, beta: i32, ply: u32) !i32 {
        if (self.stop_search.load(.seq_cst)) {
            return 0;
        }

        self.nodes_searched += 1;
        self.seldepth = @max(self.seldepth, ply);

        // Stand pat - assume we can at least maintain current position
        const stand_pat = eval.evaluate(self.board);

        var alpha = alpha_in;
        if (stand_pat >= beta) {
            return beta;
        }
        if (alpha < stand_pat) {
            alpha = stand_pat;
        }

        // Generate only captures and promotions
        var moves = MoveList.init();
        try self.generateTacticalMoves(&moves);

        if (moves.count == 0) {
            return stand_pat;
        }

        // Order captures by MVV-LVA
        self.orderCaptures(&moves);

        for (moves.slice()) |move| {
            // Delta pruning - skip captures that can't possibly raise alpha
            const captured_value = if (self.board.board.getPieceAt(move.to, if (self.board.board.move == .white) .black else .white)) |p|
                eval.getPieceValue(p)
            else
                0;

            if (stand_pat + captured_value + 200 < alpha) {
                continue; // Futile capture
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
            if (self.board.board.getPieceAt(move.to, opponent_color) != null) {
                moves.append(move);
            }
            // Include promotions
            else if (move.promotion != null) {
                moves.append(move);
            }
            // Include en passant
            else if (self.board.board.en_passant_square == move.to) {
                const piece_type = self.board.board.getPieceAt(move.from, self.board.board.move);
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
                if (tt.from == move.from and tt.to == move.to and tt.promotion == move.promotion) {
                    scores[i] = 1000000;
                    continue;
                }
            }

            // Captures scored by MVV-LVA
            if (self.board.board.getPieceAt(move.to, opponent_color)) |victim| {
                const victim_value = eval.getPieceValue(victim);
                const attacker_value = if (self.board.board.getPieceAt(move.from, self.board.board.move)) |att|
                    eval.getPieceValue(att)
                else
                    0;

                score = victim_value * 100 - attacker_value + 10000;
            }
            // Promotions
            else if (move.promotion) |promo| {
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

            if (self.board.board.getPieceAt(move.to, opponent_color)) |victim| {
                const victim_value = eval.getPieceValue(victim);
                const attacker_value = if (self.board.board.getPieceAt(move.from, self.board.board.move)) |att|
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
