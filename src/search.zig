const std = @import("std");
const board = @import("bitboard.zig");
const Board = board.Board;
const Move = board.Move;
const MoveList = board.MoveList;
const piece = @import("piece.zig");
const UciError = @import("uci_error.zig").UciError;
const eval = @import("evaluation.zig");
const zobrist = @import("zobrist.zig");
const nnue = @import("nnue.zig");
const move_picker_mod = @import("search/move_picker.zig");
const tt_mod = @import("search/tt.zig");
const heuristics = @import("search/heuristics.zig");
const capture_history_mod = @import("search/capture_history.zig");

const KillerMoves = heuristics.KillerMoves;
const CounterMoveTable = heuristics.CounterMoveTable;
const HistoryTable = heuristics.HistoryTable;
const ContinuationHistoryTable = heuristics.ContinuationHistoryTable;
const CaptureHistory = capture_history_mod.CaptureHistory;
const continuationKey = heuristics.continuationKey;
const INVALID_CONTINUATION_KEY = heuristics.INVALID_CONTINUATION_KEY;

pub const MovePicker = move_picker_mod.MovePicker;
pub const TTEntryBound = tt_mod.TTEntryBound;
pub const TTEntry = tt_mod.TTEntry;
pub const TranspositionTable = tt_mod.TranspositionTable;

inline fn oppositeColor(color: piece.Color) piece.Color {
    return move_picker_mod.oppositeColor(color);
}

inline fn seePieceValue(piece_type: piece.Type) i32 {
    return move_picker_mod.seePieceValue(piece_type);
}

inline fn staticExchangeEvalPosition(b: *const board.BitBoard, move: Move) i32 {
    return move_picker_mod.staticExchangeEvalPosition(b, move);
}

inline fn movesEqual(a: Move, b: Move) bool {
    return a.data == b.data;
}

pub const SearchOptions = struct {
    infinite: bool = false,
    move_time: ?u64 = null,
    wtime: ?u64 = null,
    btime: ?u64 = null,
    winc: ?u64 = null,
    binc: ?u64 = null,
    moves_to_go: ?u64 = null,
    depth: ?u64 = null,
    start_depth: u32 = 1,
    /// Absolute start of the GUI's `go` command, including worker setup.
    start_time: ?std.time.Instant = null,
    move_overhead: u64 = 30,
    /// Helpers obey the hard deadline but only the main worker stops at soft time.
    enforce_soft_limit: bool = true,
};

const TimeBudget = struct {
    soft_ms: u64,
    hard_ms: u64,
};

pub const SearchResult = struct {
    best_move: Move,
    score: i32,
    nodes: usize,
    time_ms: i64,
    depth: u32,
};

/// Smooth search-shape controls exposed through UCI for OpenBench tuning.
/// Percentage fields are fixed-point integers where 100 represents 1.00.
/// Defaults track the latest accepted SPSA candidate.
pub const SearchTuning = struct {
    lmr_scale_pct: i32 = 109,
    lmr_history_scale_pct: i32 = 101,
    lmp_move_scale_pct: i32 = 94,
    history_max_bonus: u32 = 380,
};

const MAX_PLY = 64;
const ACC_STACK_SIZE: usize = 128;
// Check extensions and an in-check quiescence tail can outlive MAX_PLY. Every
// search path must terminate before exhausting the NNUE accumulator stack.
const MAX_SEARCH_PLY: u32 = 120;
const MAX_KILLER_MOVES = 2;
const ROOT_MOVE_BUCKETS = 64 * 64;
const STATIC_EVAL_STACK_SIZE = MAX_PLY;
const EVAL_CACHE_SIZE = 16384; // Must be power-of-two for fast masking.
const EVAL_CACHE_EMPTY_KEY = std.math.maxInt(u64);
const SEE_CAPTURE_SCALE: i32 = 128;

const INF: i32 = 32000;
const DRAW_SCORE: i32 = 0;
const PAWN_ENDGAME_ROOT_EXTENSION: u32 = 1;
const STOP_CHECK_INTERVAL: u32 = 1024;

fn elapsedMs(start: std.time.Instant) i64 {
    const now = std.time.Instant.now() catch return 0;
    const ns = now.since(start);
    return @intCast(@divFloor(ns, std.time.ns_per_ms));
}

fn scaledFraction(value: u64, numerator: u64, denominator: u64) u64 {
    std.debug.assert(denominator > 0 and numerator <= denominator);
    return (value / denominator) * numerator +
        ((value % denominator) * numerator) / denominator;
}

fn calculateTimeBudgetFor(options: SearchOptions, side: piece.Color, fullmove_number: u16) ?TimeBudget {
    if (options.infinite) return null;

    if (options.move_time) |move_time| {
        // `go movetime` is already an explicit search budget. Move Overhead is
        // reserved for clock-managed games and must not skew fixed-time tests.
        return .{ .soft_ms = move_time, .hard_ms = move_time };
    }

    const remaining = if (side == .white) options.wtime else options.btime;
    const increment = if (side == .white) options.winc orelse 0 else options.binc orelse 0;
    const usable = (remaining orelse return null) -| options.move_overhead;
    if (usable == 0) return .{ .soft_ms = 0, .hard_ms = 0 };

    const explicit_moves_to_go = options.moves_to_go != null;
    const horizon: u64 = if (options.moves_to_go) |moves_to_go|
        std.math.clamp(moves_to_go, 1, 50)
    else
        @max(@as(u64, 20), 40 - @min(@as(u64, fullmove_number), 40));

    const divisor = horizon + if (explicit_moves_to_go) @as(u64, 1) else 5;
    const increment_share = scaledFraction(increment, 9, 10);
    var soft = (usable / divisor) +| increment_share;

    var max_soft: u64 = if (explicit_moves_to_go) blk: {
        if (horizon == 1) break :blk usable - usable / 5;
        const two_shares = (usable / (horizon + 1)) * 2 +
            ((usable % (horizon + 1)) * 2) / (horizon + 1);
        break :blk @min(usable / 2, two_shares);
    } else @min(usable / 5, 30000);

    if (!explicit_moves_to_go) {
        if (usable < 5000) {
            max_soft = @min(usable / 10, max_soft);
        } else if (usable < 10000) {
            max_soft = @min(scaledFraction(usable, 15, 100), max_soft);
        }
    }

    // Apply the minimum before final caps; the remaining-clock cap always wins.
    soft = @max(soft, 100);
    soft = @min(soft, max_soft);
    soft = @min(soft, usable);
    if (soft == 0) soft = 1;

    const expanded = std.math.mul(u64, soft, 4) catch std.math.maxInt(u64);
    var hard = @min(usable, expanded);
    hard = @min(hard, @max(usable / 2, soft));
    hard = @max(hard, soft);

    return .{ .soft_ms = soft, .hard_ms = hard };
}

inline fn rootMoveBucket(move: Move) usize {
    return @as(usize, move.from()) * 64 + @as(usize, move.to());
}

fn scalePermille(value: u64, factor: u64) u64 {
    const scaled = (@as(u128, value) * @as(u128, factor)) / 1000;
    return @intCast(@min(scaled, std.math.maxInt(u64)));
}

fn adaptiveSoftLimit(
    budget: TimeBudget,
    depth: u32,
    best_move_nodes: usize,
    iteration_nodes: usize,
    best_move_stability: u32,
    score_delta: ?u32,
) u64 {
    if (depth < 5 or iteration_nodes == 0) return budget.soft_ms;

    const node_share: u64 = @intCast(@min(
        (@as(u128, best_move_nodes) * 1000) / @as(u128, iteration_nodes),
        1000,
    ));
    const node_factor = std.math.clamp(((1480 - node_share) * 1680) / 1000, 500, 2000);
    const stable_plies = @min(best_move_stability, 14);
    const stability_factor = @max(@as(u64, 700), 1100 - @as(u64, stable_plies) * 30);

    var factor = (node_factor * stability_factor) / 1000;
    if (score_delta) |delta| {
        if (delta >= 25) factor = (factor * 1100) / 1000;
    }
    factor = std.math.clamp(factor, 650, 1800);

    return @min(scalePermille(budget.soft_ms, factor), budget.hard_ms);
}

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

inline fn isPurePawnEndgame(b: board.BitBoard) bool {
    const kings = b.getKindBitboard(.king);
    const pawns = b.getKindBitboard(.pawn);
    return b.occupied() == (kings | pawns);
}

inline fn lmpQuietMoveLimit(depth: u32, scale_pct: i32) u32 {
    const base: u32 = switch (depth) {
        0 => 0,
        1 => 5,
        2 => 10,
        3 => 16,
        else => std.math.maxInt(u32),
    };
    if (base == std.math.maxInt(u32)) return base;
    const scaled = @divTrunc(@as(i64, base) * @as(i64, scale_pct), 100);
    return @intCast(@max(scaled, 0));
}

inline fn addLmpQuietMoveBonus(limit: u32, bonus: u32) u32 {
    return std.math.add(u32, limit, bonus) catch std.math.maxInt(u32);
}

inline fn speculativeSacPenalty(see_score: i32, depth: u32) i32 {
    const loss = -see_score;
    if (loss < eval.PAWN_VALUE) return 0;

    // Root-only bias against dubious minor-piece-for-pawn sacs.
    // The HCE overvalues king safety destruction from open files/missing shield pawns,
    // making these sacs look better than they are.
    const capped_depth: i32 = @intCast(@min(depth, 12));
    return 60 + @divTrunc(loss - eval.PAWN_VALUE, 2) + capped_depth * 4;
}

// Pre-computed fixed-point LMR table. Keeping two decimal places lets the
// runtime scale option make useful sub-integer changes before final rounding.
const LMR_MIN_DEPTH: u32 = 3;
const LMR_FULL_DEPTH_MOVES: u32 = 4;
const LMR_TABLE_MAX_DEPTH = 64;
const LMR_TABLE_MAX_MOVES = 64;
const lmr_table_x100: [LMR_TABLE_MAX_DEPTH][LMR_TABLE_MAX_MOVES]i32 = blk: {
    @setEvalBranchQuota(10000);
    var table: [LMR_TABLE_MAX_DEPTH][LMR_TABLE_MAX_MOVES]i32 = undefined;
    for (0..LMR_TABLE_MAX_DEPTH) |d| {
        for (0..LMR_TABLE_MAX_MOVES) |m| {
            if (d == 0 or m == 0) {
                table[d][m] = 0;
            } else {
                const ln_d = @log(@as(f64, @floatFromInt(d)));
                const ln_m = @log(@as(f64, @floatFromInt(m)));
                const val = ln_d * ln_m * 50.0;
                table[d][m] = @intFromFloat(@max(val, 0.0));
            }
        }
    }
    break :blk table;
};

const LMP_MAX_DEPTH: u32 = 3;
const LMP_IMPROVING_BONUS: u32 = 2;

const NULL_MOVE_MIN_DEPTH: u32 = 3;
const NULL_MOVE_REDUCTION: u32 = 3;
const NULL_MOVE_DEEP_DEPTH: u32 = 6;
const NULL_MOVE_VERIFICATION_DEPTH: u32 = 8;
const NULL_MOVE_STATIC_EVAL_MARGIN: i32 = 80;
const SINGULAR_MIN_DEPTH: u32 = 8;
const SINGULAR_REDUCTION: u32 = 2;
const SINGULAR_MARGIN_BASE_CP: i32 = 30;
const SINGULAR_MARGIN_PER_PLY_CP: i32 = 25;
const SINGULAR_MULTICUT_MIN_BEATERS: u32 = 2;

// Futility pruning parameters
const FUTILITY_MARGIN: i32 = 200;
const FUTILITY_MARGIN_MULTIPLIER: i32 = 120;

// Reverse futility pruning parameters
const REVERSE_FUTILITY_MARGIN_PER_PLY: i32 = 100;

// Razoring parameters
const RAZOR_MARGIN: i32 = 500;
const QS_SEE_PRUNE_MARGIN_CP: i32 = -80;
const MAIN_SEE_PRUNE_MAX_DEPTH: u32 = 4;
const MAIN_SEE_PRUNE_MARGIN_PER_PLY_CP: i32 = 70;

pub const SearchEngine = struct {
    const Self = @This();

    const TtProbeResult = struct {
        tt_move: ?Move,
        cutoff: ?i32,
        tt_score: ?i32,
        tt_depth: u32,
        tt_bound: ?TTEntryBound,
    };

    const SingularResult = struct {
        extension: u32,
        cutoff: ?i32,
    };

    const StaticEvalContext = struct {
        static_eval: i32,
        improving: bool,
        futile: bool,
    };

    const AccumulatorUpdate = struct {
        from_sq: u8,
        to_sq: u8,
        moved_piece: piece.Type,
        moved_color: piece.Color,
        captured_piece: ?piece.Type,
        capture_sq: ?u8,
        promotion: ?piece.Type,
        rook_from: ?u8,
        rook_to: ?u8,
    };

    board: *Board,
    allocator: std.mem.Allocator,
    stop_search: *std.atomic.Value(bool),
    uci_output: ?std.fs.File,
    root_best_move: Move,
    root_move_nodes: [ROOT_MOVE_BUCKETS]usize,

    // Search state
    tt: *TranspositionTable,
    killer_moves: KillerMoves,
    history: HistoryTable,
    capture_history: CaptureHistory,
    counter_moves: CounterMoveTable,
    continuation_history: *ContinuationHistoryTable,
    nodes_searched: usize,
    seldepth: u32,
    previous_move: ?Move,
    continuation_keys: [MAX_PLY]u16,
    // Position history for repetition detection during search
    position_history: [512]u64,
    history_count: usize,
    // Game history (positions before search started) - for proper repetition detection
    game_history: [512]u64,
    game_history_count: usize,
    use_nnue: bool,
    nnue_net: ?*const nnue.Network,
    nnue_blend: i32,
    nnue_scale: i32,
    tuning: SearchTuning,
    // Incremental NNUE accumulator stack (heap-allocated when NNUE is active)
    acc_stack: ?[]nnue.AccumulatorPair,
    acc_refresh_cache: ?*nnue.AccumulatorRefreshCache,
    acc_ply: u32,
    acc_valid: [ACC_STACK_SIZE]bool,
    acc_updates: [ACC_STACK_SIZE]AccumulatorUpdate,
    eval_cache_keys: [EVAL_CACHE_SIZE]u64,
    eval_cache_values: [EVAL_CACHE_SIZE]i32,
    static_eval_stack: [STATIC_EVAL_STACK_SIZE]i32,
    search_start: std.time.Instant,
    hard_deadline_ns: ?u64,
    stop_check_countdown: u32,

    pub fn init(
        board_ptr: *Board,
        allocator: std.mem.Allocator,
        stop_search: *std.atomic.Value(bool),
        tt: *TranspositionTable,
        use_nnue: bool,
        nnue_net: ?*const nnue.Network,
        nnue_blend: i32,
        nnue_scale: i32,
    ) error{OutOfMemory}!Self {
        const continuation_history = try allocator.create(ContinuationHistoryTable);
        errdefer allocator.destroy(continuation_history);
        continuation_history.initInPlace();

        return Self{
            .board = board_ptr,
            .allocator = allocator,
            .stop_search = stop_search,
            .uci_output = null,
            .root_best_move = Move.init(0, 0, null),
            .root_move_nodes = [_]usize{0} ** ROOT_MOVE_BUCKETS,
            .tt = tt,
            .killer_moves = KillerMoves.init(),
            .history = HistoryTable.init(),
            .capture_history = CaptureHistory.init(),
            .counter_moves = CounterMoveTable.init(),
            .continuation_history = continuation_history,
            .nodes_searched = 0,
            .seldepth = 0,
            .previous_move = null,
            .continuation_keys = [_]u16{INVALID_CONTINUATION_KEY} ** MAX_PLY,
            .position_history = undefined,
            .history_count = 0,
            .game_history = undefined,
            .game_history_count = 0,
            .use_nnue = use_nnue,
            .nnue_net = nnue_net,
            .nnue_blend = nnue_blend,
            .nnue_scale = nnue_scale,
            .tuning = .{},
            .acc_stack = null,
            .acc_refresh_cache = null,
            .acc_ply = 0,
            .acc_valid = [_]bool{false} ** ACC_STACK_SIZE,
            .acc_updates = undefined,
            .eval_cache_keys = [_]u64{EVAL_CACHE_EMPTY_KEY} ** EVAL_CACHE_SIZE,
            .eval_cache_values = [_]i32{0} ** EVAL_CACHE_SIZE,
            .static_eval_stack = [_]i32{-INF} ** STATIC_EVAL_STACK_SIZE,
            .search_start = std.time.Instant.now() catch unreachable,
            .hard_deadline_ns = null,
            .stop_check_countdown = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.deinitAccumulatorStack();
        self.allocator.destroy(self.continuation_history);
    }

    inline fn evalCacheIndex(hash: u64) usize {
        return @intCast(hash & (EVAL_CACHE_SIZE - 1));
    }

    inline fn evalCacheGet(self: *Self, hash: u64) ?i32 {
        const idx = evalCacheIndex(hash);
        if (self.eval_cache_keys[idx] == hash) {
            return self.eval_cache_values[idx];
        }
        return null;
    }

    inline fn evalCachePut(self: *Self, hash: u64, value: i32) void {
        const idx = evalCacheIndex(hash);
        self.eval_cache_keys[idx] = hash;
        self.eval_cache_values[idx] = value;
    }

    /// Allocate and initialize the incremental accumulator stack for NNUE.
    fn initAccumulatorStack(self: *Self) !void {
        if (self.use_nnue and self.nnue_net != null) {
            const stack = self.allocator.alloc(nnue.AccumulatorPair, ACC_STACK_SIZE) catch return;
            self.acc_stack = stack;
            stack[0] = nnue.initAccumulators(self.nnue_net.?, self.board);
            if (self.allocator.create(nnue.AccumulatorRefreshCache)) |cache| {
                cache.initInPlace();
                nnue.seedAccumulatorRefreshCache(self.nnue_net.?, self.board, &stack[0], cache);
                self.acc_refresh_cache = cache;
            } else |_| {}
            self.acc_ply = 0;
            self.acc_valid = [_]bool{false} ** ACC_STACK_SIZE;
            self.acc_valid[0] = true;
        }
    }

    fn deinitAccumulatorStack(self: *Self) void {
        if (self.acc_refresh_cache) |cache| {
            self.allocator.destroy(cache);
            self.acc_refresh_cache = null;
        }
        if (self.acc_stack) |stack| {
            self.allocator.free(stack);
            self.acc_stack = null;
        }
    }

    fn materializeCurrentAccumulator(self: *Self) void {
        const stack = self.acc_stack orelse return;
        if (self.acc_valid[self.acc_ply]) return;

        std.debug.assert(self.acc_ply > 0);
        std.debug.assert(self.acc_valid[self.acc_ply - 1]);
        const update = self.acc_updates[self.acc_ply];
        nnue.updateAccumulators(
            self.nnue_net.?,
            self.board,
            &stack[self.acc_ply - 1],
            &stack[self.acc_ply],
            update.from_sq,
            update.to_sq,
            update.moved_piece,
            update.moved_color,
            update.captured_piece,
            update.capture_sq,
            update.promotion,
            update.rook_from != null,
            update.rook_from,
            update.rook_to,
            self.acc_refresh_cache,
        );
        self.acc_valid[self.acc_ply] = true;
    }

    /// A child accumulator depends on this position, so make it current before
    /// mutating the board. Nodes that cut off before this point avoid the work.
    inline fn prepareAccumulatorForMove(self: *Self) void {
        if (self.acc_stack != null and !self.acc_valid[self.acc_ply]) {
            self.materializeCurrentAccumulator();
        }
    }

    /// Record an incremental update after a move and defer applying it until
    /// evaluation or until a searched child needs this accumulator as a base.
    inline fn pushAccumulator(self: *Self, move: Move, undo: Board.Undo) void {
        if (self.acc_stack) |stack| {
            std.debug.assert(self.acc_ply + 1 < stack.len);
            self.acc_ply += 1;
            self.acc_updates[self.acc_ply] = .{
                .from_sq = move.from(),
                .to_sq = move.to(),
                .moved_piece = undo.moved_piece,
                .moved_color = undo.mover_color,
                .captured_piece = undo.captured_piece,
                .capture_sq = undo.captured_square,
                .promotion = move.promotion(),
                .rook_from = undo.castle_rook_from,
                .rook_to = undo.castle_rook_to,
            };
            self.acc_valid[self.acc_ply] = false;
        }
    }

    /// Restore accumulator state after unmaking a move.
    inline fn popAccumulator(self: *Self) void {
        if (self.acc_stack != null) {
            self.acc_ply -= 1;
        }
    }

    inline fn evaluatePosition(self: *Self) i32 {
        // Cache static evals only when NNUE is enabled, since NNUE eval is much costlier.
        const hash = self.board.zobrist_hasher.zobrist_hash;
        if (self.use_nnue) {
            if (self.evalCacheGet(hash)) |cached| {
                return cached;
            }
        }

        var score: i32 = undefined;
        if (self.use_nnue) {
            if (self.nnue_net) |net| {
                // Use incremental accumulators if available, otherwise full recompute
                const nn_raw = if (self.acc_stack) |stack| blk: {
                    self.materializeCurrentAccumulator();
                    break :blk nnue.evaluateFromAccumulators(
                        net,
                        &stack[self.acc_ply],
                        self.board,
                    );
                } else nnue.evaluate(net, self.board);
                const nn = @divTrunc(nn_raw * self.nnue_scale, 100);
                if (self.nnue_blend >= 100) {
                    score = nn;
                    self.evalCachePut(hash, score);
                    return score;
                }
                if (self.nnue_blend <= 0) {
                    score = eval.evaluate(self.board);
                    self.evalCachePut(hash, score);
                    return score;
                }

                const cl = eval.evaluate(self.board);
                score = @divTrunc(nn * self.nnue_blend + cl * (100 - self.nnue_blend), 100);
                self.evalCachePut(hash, score);
                return score;
            }
        }
        score = eval.evaluate(self.board);
        if (self.use_nnue) {
            self.evalCachePut(hash, score);
        }
        return score;
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

    /// Poll the shared stop flag every node and the absolute hard deadline
    /// periodically. Every worker receives the same start time and hard limit.
    inline fn checkShouldStop(self: *Self) bool {
        if (self.stop_search.load(.monotonic)) return true;

        if (self.stop_check_countdown > 0) {
            self.stop_check_countdown -= 1;
            return false;
        }
        self.stop_check_countdown = STOP_CHECK_INTERVAL - 1;

        if (self.hard_deadline_ns) |deadline_ns| {
            const now = std.time.Instant.now() catch return false;
            if (now.since(self.search_start) >= deadline_ns) {
                self.stop_search.store(true, .monotonic);
                return true;
            }
        }

        return false;
    }

    /// Run a search and return the best move
    pub fn search(self: *Self, options: SearchOptions) anyerror!SearchResult {
        const start_time = options.start_time orelse (std.time.Instant.now() catch unreachable);

        // Reset search state
        self.nodes_searched = 0;
        self.seldepth = 0;
        self.killer_moves = KillerMoves.init();
        self.history.age(); // Age history instead of clearing
        self.capture_history.age();
        self.counter_moves.clear();
        self.continuation_history.age();
        self.previous_move = null;
        self.continuation_keys = [_]u16{INVALID_CONTINUATION_KEY} ** MAX_PLY;
        self.static_eval_stack = [_]i32{-INF} ** STATIC_EVAL_STACK_SIZE;
        // The UCI layer normally includes the root as the final game-history
        // entry. Do not duplicate it: that shifts repetition parity and can
        // turn a single pre-root occurrence into a false draw.
        const root_hash = self.board.zobrist_hasher.zobrist_hash;
        if (self.game_history_count == 0 or self.game_history[self.game_history_count - 1] != root_hash) {
            self.position_history[0] = root_hash;
            self.history_count = 1;
        } else {
            self.history_count = 0;
        }

        // Initialize incremental NNUE accumulators at root position
        try self.initAccumulatorStack();
        defer self.deinitAccumulatorStack();

        const time_budget = calculateTimeBudgetFor(options, self.board.board.move, self.board.board.fullmove_number);
        self.search_start = start_time;
        self.hard_deadline_ns = if (time_budget) |budget|
            std.math.mul(u64, budget.hard_ms, std.time.ns_per_ms) catch std.math.maxInt(u64)
        else
            null;
        self.stop_check_countdown = 0;

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
                .time_ms = elapsedMs(start_time),
                .depth = 0,
            };
        }

        var best_move = legal_moves.moves[0];
        var best_score: i32 = -INF;
        self.root_best_move = best_move;
        const max_depth: u32 = if (options.depth) |d| @intCast(d) else 64;
        var completed_depth: u32 = 0;
        var previous_completed_move: ?Move = null;
        var previous_completed_score: ?i32 = null;
        var best_move_stability: u32 = 0;

        // Iterative deepening
        var depth: u32 = options.start_depth;
        while (depth <= max_depth) : (depth += 1) {
            if (self.stop_search.load(.monotonic)) break;

            const iter_start = std.time.Instant.now() catch unreachable;
            const iteration_nodes_before = self.nodes_searched;
            self.root_move_nodes = [_]usize{0} ** ROOT_MOVE_BUCKETS;

            // Aspiration windows - narrow search window around previous score
            var asp_alpha: i32 = -INF;
            var asp_beta: i32 = INF;
            var asp_delta: i32 = 25;

            if (depth >= 4 and !eval.isMateScore(best_score)) {
                asp_alpha = @max(best_score - asp_delta, -INF);
                asp_beta = @min(best_score + asp_delta, INF);
            }

            var score: i32 = -INF;

            // Aspiration window loop - re-search with wider window on fail
            while (true) {
                score = try self.alphaBeta(asp_alpha, asp_beta, depth, 0, true);

                if (self.stop_search.load(.monotonic)) break;

                if (score <= asp_alpha) {
                    // Fail low - widen alpha
                    asp_alpha = @max(asp_alpha - asp_delta, -INF);
                    asp_delta *= 2;
                } else if (score >= asp_beta) {
                    // Fail high - widen beta
                    asp_beta = @min(asp_beta + asp_delta, INF);
                    asp_delta *= 2;
                } else {
                    break; // Score within window
                }
            }

            // Skip incomplete iterations (stopped mid-search)
            if (self.stop_search.load(.monotonic)) break;

            best_move = self.root_best_move;
            best_score = score;
            completed_depth = depth;

            if (previous_completed_move) |previous_move| {
                if (movesEqual(best_move, previous_move)) {
                    best_move_stability +|= 1;
                } else {
                    best_move_stability = 0;
                }
            }
            const score_delta: ?u32 = if (previous_completed_score) |previous_score|
                @abs(score - previous_score)
            else
                null;
            previous_completed_move = best_move;
            previous_completed_score = score;

            const iter_time = elapsedMs(iter_start);
            const total_time = elapsedMs(start_time);
            const iteration_nodes = self.nodes_searched - iteration_nodes_before;

            // UCI info output at every depth
            if (self.uci_output != null) {
                const nps = if (total_time > 0) (self.nodes_searched * 1000) / @as(usize, @intCast(total_time)) else 0;
                self.writeUciLine("info depth {d} seldepth {d} score cp {d} nodes {d} time {d} nps {d} pv {f}", .{
                    depth,
                    self.seldepth,
                    score,
                    self.nodes_searched,
                    total_time,
                    nps,
                    best_move,
                });
            }

            // Only the main worker acts on soft time. All workers enforce hard time.
            if (options.enforce_soft_limit) {
                if (time_budget) |budget| {
                    const total_time_ms: u64 = @intCast(@max(total_time, 0));
                    const iter_time_ms: u64 = @intCast(@max(iter_time, 0));
                    const soft_limit = if (options.move_time == null)
                        adaptiveSoftLimit(
                            budget,
                            depth,
                            self.root_move_nodes[rootMoveBucket(best_move)],
                            iteration_nodes,
                            best_move_stability,
                            score_delta,
                        )
                    else
                        budget.soft_ms;
                    const should_stop = if (options.move_time != null)
                        total_time_ms >= soft_limit
                    else
                        total_time_ms >= soft_limit or total_time_ms +| (iter_time_ms *| 2) >= soft_limit;
                    if (should_stop) {
                        break;
                    }
                }
            }

            // Stop if we found a mate
            if (eval.isMateScore(score)) {
                break;
            }
        }

        const elapsed = elapsedMs(start_time);

        if (completed_depth == 0) {
            return SearchResult{
                .best_move = legal_moves.moves[0],
                .score = 0,
                .nodes = self.nodes_searched,
                .time_ms = elapsed,
                .depth = 0,
            };
        }

        return SearchResult{
            .best_move = best_move,
            .score = best_score,
            .nodes = self.nodes_searched,
            .time_ms = elapsed,
            .depth = completed_depth,
        };
    }

    inline fn continuationContext(self: *const Self, ply: u32) struct { prev: ?u16, prev2: ?u16 } {
        const prev = if (ply >= 1 and self.continuation_keys[ply - 1] != INVALID_CONTINUATION_KEY)
            self.continuation_keys[ply - 1]
        else
            null;
        const prev2 = if (ply >= 2 and self.continuation_keys[ply - 2] != INVALID_CONTINUATION_KEY)
            self.continuation_keys[ply - 2]
        else
            null;
        return .{ .prev = prev, .prev2 = prev2 };
    }

    inline fn moveContinuationKey(
        self: *const Self,
        move: Move,
        color: piece.Color,
        moving_piece: piece.Type,
    ) u16 {
        _ = self;
        return continuationKey(color, move.promotion() orelse moving_piece, move.to());
    }

    inline fn quietHeuristicScore(
        self: *const Self,
        move: Move,
        color: piece.Color,
        ply: u32,
    ) i32 {
        var score = self.history.getForColor(move, color);
        if (self.board.board.getPieceAt(move.from(), color)) |piece_type| {
            const ctx = self.continuationContext(ply);
            score += self.continuation_history.get(
                ctx.prev,
                ctx.prev2,
                continuationKey(color, piece_type, move.to()),
            );
        }
        return score;
    }

    fn probeTransposition(
        self: *Self,
        search_depth: u32,
        ply: u32,
        is_pv_node: bool,
        alpha: i32,
        beta_adj: i32,
    ) TtProbeResult {
        var tt_move: ?Move = null;
        if (self.tt.probe(self.board.zobrist_hasher.zobrist_hash)) |entry| {
            if (entry.best_move.from() != 0 or entry.best_move.to() != 0) {
                tt_move = entry.best_move;
            }

            if (!is_pv_node and entry.depth >= search_depth) {
                const tt_score = scoreFromTT(entry.score, ply);
                switch (entry.bound) {
                    .exact => return .{
                        .tt_move = tt_move,
                        .cutoff = tt_score,
                        .tt_score = tt_score,
                        .tt_depth = entry.depth,
                        .tt_bound = entry.bound,
                    },
                    .lower_bound => {
                        if (tt_score >= beta_adj) return .{
                            .tt_move = tt_move,
                            .cutoff = tt_score,
                            .tt_score = tt_score,
                            .tt_depth = entry.depth,
                            .tt_bound = entry.bound,
                        };
                    },
                    .upper_bound => {
                        if (tt_score <= alpha) return .{
                            .tt_move = tt_move,
                            .cutoff = tt_score,
                            .tt_score = tt_score,
                            .tt_depth = entry.depth,
                            .tt_bound = entry.bound,
                        };
                    },
                }
            }

            const tt_score = scoreFromTT(entry.score, ply);
            return .{
                .tt_move = tt_move,
                .cutoff = null,
                .tt_score = tt_score,
                .tt_depth = entry.depth,
                .tt_bound = entry.bound,
            };
        }

        return .{
            .tt_move = tt_move,
            .cutoff = null,
            .tt_score = null,
            .tt_depth = 0,
            .tt_bound = null,
        };
    }

    inline fn moveGivesCheck(self: *Self, move: Move) bool {
        const probe_undo = self.board.makeMoveWithUndoUnchecked(move);
        const gives_check = self.board.isInCheck(self.board.board.move);
        self.board.unmakeMoveUnchecked(move, probe_undo);
        return gives_check;
    }

    fn trySingularExtension(
        self: *Self,
        tt_probe: TtProbeResult,
        search_depth: u32,
        ply: u32,
        is_pv_node: bool,
        in_check: bool,
        beta_adj: i32,
    ) !SingularResult {
        if (is_pv_node or in_check or ply == 0) {
            return .{ .extension = 0, .cutoff = null };
        }

        const tt_move = tt_probe.tt_move orelse return .{ .extension = 0, .cutoff = null };
        const tt_score = tt_probe.tt_score orelse return .{ .extension = 0, .cutoff = null };
        const tt_bound = tt_probe.tt_bound orelse return .{ .extension = 0, .cutoff = null };
        if (tt_bound != .exact and tt_bound != .lower_bound) {
            return .{ .extension = 0, .cutoff = null };
        }
        if (tt_probe.tt_depth < search_depth or search_depth < SINGULAR_MIN_DEPTH or eval.isMateScore(tt_score)) {
            return .{ .extension = 0, .cutoff = null };
        }

        const singular_margin = SINGULAR_MARGIN_BASE_CP + SINGULAR_MARGIN_PER_PLY_CP * @as(i32, @intCast(search_depth));
        const singular_beta = @max(tt_score - singular_margin, -INF);
        if (search_depth <= SINGULAR_REDUCTION + 1) {
            return .{ .extension = 0, .cutoff = null };
        }
        const reduced_depth = search_depth - 1 - SINGULAR_REDUCTION;

        const killers = if (ply < MAX_PLY) &self.killer_moves.moves[ply] else &[_]Move{Move.init(0, 0, null)} ** MAX_KILLER_MOVES;
        const counter_move: ?Move = if (self.previous_move) |prev| self.counter_moves.get(prev) else null;
        const continuation_ctx = self.continuationContext(ply);
        var move_picker = MovePicker.init(
            self.board,
            tt_move,
            killers,
            counter_move,
            &self.history,
            &self.capture_history,
            self.continuation_history,
            continuation_ctx.prev,
            continuation_ctx.prev2,
            ply,
        );

        var beaters: u32 = 0;
        while (move_picker.next()) |move| {
            if (movesEqual(move, tt_move)) continue;

            const move_color = self.board.board.move;
            const moving_piece = self.board.board.getPieceAt(move.from(), move_color);
            const old_previous = self.previous_move;
            const old_hist_count = self.history_count;
            const old_continuation_key = if (ply < MAX_PLY) self.continuation_keys[ply] else INVALID_CONTINUATION_KEY;

            self.prepareAccumulatorForMove();
            const undo = self.board.makeMoveWithUndoUnchecked(move);
            self.pushAccumulator(move, undo);
            self.previous_move = move;
            if (ply < MAX_PLY and moving_piece != null) {
                self.continuation_keys[ply] = self.moveContinuationKey(move, move_color, moving_piece.?);
            }
            if (self.history_count < 512) {
                self.position_history[self.history_count] = self.board.zobrist_hasher.zobrist_hash;
                self.history_count += 1;
            }

            const score = -try self.alphaBeta(-singular_beta, -singular_beta + 1, reduced_depth, ply + 1, true);

            self.popAccumulator();
            self.board.unmakeMoveUnchecked(move, undo);
            self.previous_move = old_previous;
            self.history_count = old_hist_count;
            if (ply < MAX_PLY) {
                self.continuation_keys[ply] = old_continuation_key;
            }

            if (self.stop_search.load(.monotonic)) {
                return .{ .extension = 0, .cutoff = null };
            }

            if (score >= singular_beta) {
                beaters += 1;
                if (beaters >= SINGULAR_MULTICUT_MIN_BEATERS and tt_bound == .lower_bound and tt_score >= beta_adj) {
                    return .{ .extension = 0, .cutoff = beta_adj };
                }
            }
        }

        return .{
            .extension = if (beaters == 0) 1 else 0,
            .cutoff = null,
        };
    }

    fn computeStaticEvalContext(
        self: *Self,
        in_check: bool,
        ply: u32,
        search_depth: u32,
        is_pv_node: bool,
        alpha: i32,
        beta_adj: i32,
    ) StaticEvalContext {
        const static_eval = if (!in_check) self.evaluatePosition() else -INF;
        const improving = blk: {
            if (in_check or ply < 2 or ply >= STATIC_EVAL_STACK_SIZE) {
                break :blk false;
            }
            const prev_eval = self.static_eval_stack[ply - 2];
            break :blk prev_eval != -INF and static_eval > prev_eval;
        };
        if (ply < STATIC_EVAL_STACK_SIZE) {
            self.static_eval_stack[ply] = static_eval;
        }

        var futile = false;
        if (!is_pv_node and !in_check and search_depth <= 3) {
            var futility_margin = FUTILITY_MARGIN + FUTILITY_MARGIN_MULTIPLIER * @as(i32, @intCast(search_depth));
            if (improving) futility_margin += 80;
            if (static_eval + futility_margin < alpha) {
                futile = true;
            }
        }

        _ = beta_adj;
        return .{
            .static_eval = static_eval,
            .improving = improving,
            .futile = futile,
        };
    }

    inline fn tryReverseFutilityPrune(
        is_pv_node: bool,
        in_check: bool,
        search_depth: u32,
        static_eval: i32,
        beta_adj: i32,
    ) ?i32 {
        if (!is_pv_node and !in_check and search_depth <= 5) {
            const rfp_margin = REVERSE_FUTILITY_MARGIN_PER_PLY * @as(i32, @intCast(search_depth));
            if (static_eval - rfp_margin >= beta_adj) {
                return static_eval;
            }
        }
        return null;
    }

    fn tryRazoring(
        self: *Self,
        is_pv_node: bool,
        in_check: bool,
        search_depth: u32,
        static_eval: i32,
        alpha: i32,
        beta_adj: i32,
        ply: u32,
    ) !?i32 {
        if (!is_pv_node and !in_check and search_depth <= 2 and static_eval + RAZOR_MARGIN < alpha) {
            const razor_score = try self.quiescence(alpha, beta_adj, ply);
            if (razor_score < alpha) {
                return razor_score;
            }
        }
        return null;
    }

    fn tryNullMovePrune(
        self: *Self,
        do_null: bool,
        is_pv_node: bool,
        in_check: bool,
        search_depth: u32,
        static_eval: i32,
        beta_adj: i32,
        ply: u32,
    ) !?i32 {
        if (!(do_null and
            !is_pv_node and
            !in_check and
            search_depth >= NULL_MOVE_MIN_DEPTH and
            static_eval >= beta_adj + NULL_MOVE_STATIC_EVAL_MARGIN))
        {
            return null;
        }

        const stm = self.board.board.move;
        const our_non_pawn_material = nonPawnMaterial(self.board.board, stm);
        if (our_non_pawn_material <= eval.BISHOP_VALUE) return null;

        const old_board = self.board.board;
        const old_hash = self.board.zobrist_hasher.zobrist_hash;
        const old_move = self.board.board.move;
        const old_ep_file = Board.epFileForHash(self.board.board);

        self.board.board.move = if (self.board.board.move == .white) .black else .white;
        self.board.zobrist_hasher.zobrist_hash ^= zobrist.RandomTurn;

        if (old_ep_file) |f| {
            self.board.zobrist_hasher.zobrist_hash ^= zobrist.RandomEnPassant[f];
        }
        self.board.board.en_passant_square = null;

        if (self.board.board.halfmove_clock < std.math.maxInt(u8)) {
            self.board.board.halfmove_clock += 1;
        }
        if (old_move == .black and self.board.board.fullmove_number < std.math.maxInt(u16)) {
            self.board.board.fullmove_number += 1;
        }

        const old_continuation_key = if (ply < MAX_PLY) self.continuation_keys[ply] else INVALID_CONTINUATION_KEY;
        if (ply < MAX_PLY) {
            self.continuation_keys[ply] = INVALID_CONTINUATION_KEY;
        }

        var reduction: u32 = NULL_MOVE_REDUCTION;
        if (search_depth > NULL_MOVE_DEEP_DEPTH) reduction += 1;
        if (static_eval - beta_adj > 200) reduction += 1;
        reduction = @min(reduction, search_depth - 1);

        const null_score = -try self.alphaBeta(-beta_adj, -beta_adj + 1, search_depth -| reduction, ply + 1, false);

        self.board.board = old_board;
        self.board.zobrist_hasher.zobrist_hash = old_hash;
        if (ply < MAX_PLY) {
            self.continuation_keys[ply] = old_continuation_key;
        }

        if (self.stop_search.load(.monotonic)) return null;

        if (null_score < beta_adj) return null;
        if (eval.isMateScore(null_score)) return beta_adj;

        if (search_depth >= NULL_MOVE_VERIFICATION_DEPTH) {
            const verify_score = try self.alphaBeta(beta_adj - 1, beta_adj, search_depth - reduction, ply, false);
            if (self.stop_search.load(.monotonic)) return null;
            if (verify_score >= beta_adj) {
                return null_score;
            }
            return null;
        }

        return null_score;
    }

    fn updateCaptureHeuristicsOnBetaCutoff(
        self: *Self,
        move: Move,
        search_depth: u32,
        captures_tried: []const Move,
    ) void {
        if (!capture_history_mod.ENABLED) return;
        const best_key = CaptureHistory.keyFor(&self.board.board, move) orelse return;
        self.capture_history.reward(best_key, search_depth);

        for (captures_tried) |capture| {
            if (movesEqual(capture, move)) continue;
            const key = CaptureHistory.keyFor(&self.board.board, capture) orelse continue;
            self.capture_history.penalize(key, search_depth);
        }
    }

    fn updateQuietHeuristicsOnBetaCutoff(
        self: *Self,
        move: Move,
        search_depth: u32,
        ply: u32,
        color: piece.Color,
        old_previous: ?Move,
        quiets_tried: []const Move,
    ) void {
        self.killer_moves.add(move, ply);
        self.history.update(move, search_depth, color, self.tuning.history_max_bonus);

        if (self.board.board.getPieceAt(move.from(), color)) |piece_type| {
            const ctx = self.continuationContext(ply);
            const best_key = continuationKey(color, piece_type, move.to());
            self.continuation_history.update(ctx.prev, ctx.prev2, best_key, search_depth, self.tuning.history_max_bonus);

            for (quiets_tried) |quiet| {
                if ((quiet.from() != move.from() or quiet.to() != move.to()) and
                    self.board.board.getPieceAt(quiet.from(), color) != null)
                {
                    const quiet_piece = self.board.board.getPieceAt(quiet.from(), color).?;
                    const quiet_key = continuationKey(color, quiet_piece, quiet.to());
                    self.continuation_history.penalize(ctx.prev, ctx.prev2, quiet_key, search_depth, self.tuning.history_max_bonus);
                }
            }
        }

        if (old_previous) |prev| {
            self.counter_moves.update(prev, move);
        }

        for (quiets_tried) |quiet| {
            if (quiet.from() != move.from() or quiet.to() != move.to()) {
                self.history.penalize(quiet, search_depth, color, self.tuning.history_max_bonus);
            }
        }
    }

    fn storeAlphaBetaResult(
        self: *Self,
        best_score: i32,
        original_alpha: i32,
        beta_adj: i32,
        search_depth: u32,
        ply: u32,
        best_move: ?Move,
    ) void {
        if (self.stop_search.load(.monotonic)) return;
        const bound: TTEntryBound = if (best_score <= original_alpha)
            .upper_bound
        else if (best_score >= beta_adj)
            .lower_bound
        else
            .exact;
        const score_to_store = scoreToTT(best_score, ply);
        self.tt.store(
            self.board.zobrist_hasher.zobrist_hash,
            @intCast(search_depth),
            score_to_store,
            bound,
            best_move orelse Move.init(0, 0, null),
        );
    }

    fn storeQuiescenceResult(
        self: *Self,
        best_score: i32,
        original_alpha: i32,
        beta: i32,
        ply: u32,
        best_move: ?Move,
    ) void {
        if (self.stop_search.load(.monotonic)) return;
        const bound: TTEntryBound = if (best_score <= original_alpha)
            .upper_bound
        else if (best_score >= beta)
            .lower_bound
        else
            .exact;
        const score_to_store = scoreToTT(best_score, ply);
        self.tt.store(
            self.board.zobrist_hasher.zobrist_hash,
            0,
            score_to_store,
            bound,
            best_move orelse Move.init(0, 0, null),
        );
    }

    fn rule50Score(self: *Self, in_check: bool, ply: u32) !?i32 {
        if (self.board.board.halfmove_clock < 100) return null;
        if (!in_check) return DRAW_SCORE;

        // Checkmate ends the game before a rule-50 draw can be claimed.
        var evasions = MoveList.init();
        try self.board.generateLegalMoves(&evasions);
        return if (evasions.count == 0) -eval.mateIn(ply) else DRAW_SCORE;
    }

    /// Alpha-beta search (negamax variant) with various pruning techniques
    fn alphaBeta(self: *Self, alpha_in: i32, beta: i32, depth: u32, ply: u32, do_null: bool) anyerror!i32 {
        // Quiescence at depth 0
        if (depth == 0) {
            return self.quiescence(alpha_in, beta, ply);
        }

        if (self.checkShouldStop()) {
            return 0;
        }

        self.nodes_searched += 1;
        self.seldepth = @max(self.seldepth, ply);

        if (ply >= MAX_SEARCH_PLY) {
            return if (self.board.isInCheck(self.board.board.move))
                DRAW_SCORE
            else
                self.evaluatePosition();
        }

        const in_check = self.board.isInCheck(self.board.board.move);

        // Check for draw by repetition
        if (ply > 0 and self.isRepetition()) {
            return self.repetitionScore();
        }

        if (try self.rule50Score(in_check, ply)) |score| {
            return score;
        }

        var alpha = alpha_in;
        var beta_adj = beta;

        // Mate distance pruning - don't search for mates we can't reach
        alpha = @max(alpha, -eval.mateIn(ply));
        beta_adj = @min(beta_adj, eval.mateIn(ply + 1));
        if (alpha >= beta_adj) {
            return alpha;
        }

        const original_alpha = alpha;

        // Check extension - extend search when in check (with ply cap to prevent seldepth explosion)
        var search_depth = depth;
        if (in_check and ply < 2 * depth + 8) {
            search_depth += 1;
        }

        const is_pv_node = (beta_adj - alpha) > 1;

        const tt_probe = self.probeTransposition(search_depth, ply, is_pv_node, alpha, beta_adj);
        if (tt_probe.cutoff) |tt_score| {
            return tt_score;
        }
        const tt_move = tt_probe.tt_move;

        const singular = try self.trySingularExtension(
            tt_probe,
            search_depth,
            ply,
            is_pv_node,
            in_check,
            beta_adj,
        );
        if (self.stop_search.load(.monotonic)) return 0;
        if (singular.cutoff) |score| {
            return score;
        }
        const singular_extension = singular.extension;

        const eval_ctx = self.computeStaticEvalContext(in_check, ply, search_depth, is_pv_node, alpha, beta_adj);
        const static_eval = eval_ctx.static_eval;
        const improving = eval_ctx.improving;
        const futile = eval_ctx.futile;

        if (tryReverseFutilityPrune(is_pv_node, in_check, search_depth, static_eval, beta_adj)) |pruned| {
            return pruned;
        }

        const razor_score = try self.tryRazoring(is_pv_node, in_check, search_depth, static_eval, alpha, beta_adj, ply);
        if (self.stop_search.load(.monotonic)) return 0;
        if (razor_score) |score| {
            return score;
        }

        const null_score = try self.tryNullMovePrune(do_null, is_pv_node, in_check, search_depth, static_eval, beta_adj, ply);
        if (self.stop_search.load(.monotonic)) return 0;
        if (null_score) |score| return score;

        // Use staged move picker for efficient move ordering
        const killers = if (ply < MAX_PLY) &self.killer_moves.moves[ply] else &[_]Move{Move.init(0, 0, null)} ** MAX_KILLER_MOVES;
        const counter_move: ?Move = if (self.previous_move) |prev| self.counter_moves.get(prev) else null;
        const continuation_ctx = self.continuationContext(ply);
        var move_picker = MovePicker.init(
            self.board,
            tt_move,
            killers,
            counter_move,
            &self.history,
            &self.capture_history,
            self.continuation_history,
            continuation_ctx.prev,
            continuation_ctx.prev2,
            ply,
        );

        var best_move: ?Move = null;
        var best_score: i32 = -INF;
        var moves_searched: u32 = 0;
        var quiets_seen: u32 = 0;
        var quiets_tried: [64]Move = undefined;
        var quiets_tried_count: usize = 0;
        var captures_tried: [256]Move = undefined;
        var captures_tried_count: usize = 0;
        const color = self.board.board.move;
        const root_pawn_endgame = ply == 0 and search_depth >= 2 and isPurePawnEndgame(self.board.board);

        while (move_picker.next()) |move| {
            const old_previous = self.previous_move;
            const root_nodes_before = if (ply == 0) self.nodes_searched else 0;

            const move_color = self.board.board.move;
            const opponent_color = oppositeColor(move_color);
            const is_en_passant = self.board.board.en_passant_square == move.to() and
                self.board.board.getPieceAt(move.from(), move_color) == .pawn and
                self.board.board.getPieceAt(move.to(), opponent_color) == null;
            const is_capture = self.board.board.getPieceAt(move.to(), opponent_color) != null or is_en_passant;
            const is_promotion = move.promotion() != null;
            const moving_piece = self.board.board.getPieceAt(move.from(), move_color);
            const captured_piece: ?piece.Type = if (is_en_passant)
                .pawn
            else
                self.board.board.getPieceAt(move.to(), opponent_color);

            var speculative_sac_candidate = false;
            var speculative_sac_see: i32 = 0;
            if (ply == 0 and !in_check and is_capture and !is_promotion and moving_piece != null and captured_piece != null) {
                const attacker = moving_piece.?;
                const victim = captured_piece.?;
                const to_file = move.to() % 8;
                const non_central = to_file <= 2 or to_file >= 5;
                if ((attacker == .bishop or attacker == .knight) and victim == .pawn and non_central) {
                    speculative_sac_see = staticExchangeEvalPosition(&self.board.board, move);
                    speculative_sac_candidate = speculative_sac_see <= -eval.PAWN_VALUE;
                }
            }

            if (!is_capture and !is_promotion) {
                quiets_seen += 1;
                const improving_lmp_bonus: u32 = if (improving) LMP_IMPROVING_BONUS else 0;
                const quiet_lmp_limit = addLmpQuietMoveBonus(
                    lmpQuietMoveLimit(search_depth, self.tuning.lmp_move_scale_pct),
                    improving_lmp_bonus,
                );

                // Late Move Pruning (LMP): at shallow non-PV nodes, trim low-history late quiets.
                if (!is_pv_node and
                    !in_check and
                    search_depth <= LMP_MAX_DEPTH and
                    moves_searched > 0 and
                    quiets_seen > quiet_lmp_limit and
                    !self.killer_moves.isKiller(move, ply) and
                    self.quietHeuristicScore(move, color, ply) <= 0)
                {
                    continue;
                }
            }

            // SEE capture pruning - trim clearly losing captures at shallow non-PV nodes.
            if (!is_pv_node and
                !in_check and
                is_capture and
                !is_promotion and
                moves_searched > 0 and
                search_depth <= MAIN_SEE_PRUNE_MAX_DEPTH)
            {
                const is_tt_move = tt_move != null and movesEqual(move, tt_move.?);
                if (!is_tt_move) {
                    const see_score = staticExchangeEvalPosition(&self.board.board, move);
                    const see_margin = MAIN_SEE_PRUNE_MARGIN_PER_PLY_CP * @as(i32, @intCast(search_depth));
                    if (see_score < -see_margin and !self.moveGivesCheck(move)) {
                        continue;
                    }
                }
            }

            // Futility pruning - skip quiet moves if futile (but not if move gives check)
            if (futile and !is_capture and !is_promotion and moves_searched > 0) {
                if (!self.moveGivesCheck(move) and !self.killer_moves.isKiller(move, ply)) {
                    continue;
                }
            }

            // Make move
            self.prepareAccumulatorForMove();
            const undo = self.board.makeMoveWithUndoUnchecked(move);
            self.pushAccumulator(move, undo);
            self.previous_move = move;
            const old_continuation_key = if (ply < MAX_PLY) self.continuation_keys[ply] else INVALID_CONTINUATION_KEY;
            if (ply < MAX_PLY and moving_piece != null) {
                self.continuation_keys[ply] = self.moveContinuationKey(move, move_color, moving_piece.?);
            }
            const gives_check = self.board.isInCheck(self.board.board.move);

            // Add position to history for repetition detection
            const old_hist_count = self.history_count;
            if (self.history_count < 512) {
                self.position_history[self.history_count] = self.board.zobrist_hasher.zobrist_hash;
                self.history_count += 1;
            }
            const extension: u32 = if (ply == 0 and
                search_depth >= 2 and
                root_pawn_endgame)
                PAWN_ENDGAME_ROOT_EXTENSION
            else if (singular_extension > 0 and tt_move != null and movesEqual(move, tt_move.?))
                singular_extension
            else
                0;
            const next_depth = search_depth - 1 + extension;

            var score: i32 = undefined;

            // Late Move Reductions (LMR) — logarithmic formula with history modulation
            if (moves_searched >= LMR_FULL_DEPTH_MOVES and
                search_depth >= LMR_MIN_DEPTH and
                !in_check and
                !is_capture and
                !is_promotion and
                !gives_check)
            {
                // Base reduction from pre-computed log table
                const move_number = moves_searched + 1;
                const d_idx = @min(search_depth, LMR_TABLE_MAX_DEPTH - 1);
                const m_idx = @min(move_number, LMR_TABLE_MAX_MOVES - 1);
                var reduction: i32 = @intCast(@divTrunc(
                    @as(i64, lmr_table_x100[d_idx][m_idx]) * @as(i64, self.tuning.lmr_scale_pct),
                    10_000,
                ));

                // History modulation: good history reduces less, bad history reduces more
                const hist_score = self.quietHeuristicScore(move, color, ply);
                reduction -= @intCast(@divTrunc(
                    @as(i64, hist_score) * @as(i64, self.tuning.lmr_history_scale_pct),
                    819_200,
                ));

                // Killer and counter moves get reduced less
                const is_killer = self.killer_moves.isKiller(move, ply);
                var is_counter_move = false;
                if (counter_move) |cm| {
                    is_counter_move = cm.from() == move.from() and cm.to() == move.to();
                }
                if (is_killer) {
                    reduction -= 1;
                }
                if (is_counter_move) {
                    reduction -= 1;
                }
                if (improving) {
                    reduction -= 1;
                }

                // PV nodes get reduced less
                if (is_pv_node) {
                    reduction -= 1;
                }

                // Clamp reduction: at least 1, at most depth-2 (leave at least 1 ply)
                const r: u32 = @intCast(@max(1, @min(reduction, @as(i32, @intCast(next_depth)) - 1)));

                // Search with reduced depth
                score = -try self.alphaBeta(-alpha - 1, -alpha, next_depth - r, ply + 1, true);

                // If LMR found something good, re-search at full depth
                if (score > alpha and !self.stop_search.load(.monotonic)) {
                    score = -try self.alphaBeta(-alpha - 1, -alpha, next_depth, ply + 1, true);
                }
            } else {
                // PVS - search with null window first if not first move
                if (moves_searched > 0) {
                    score = -try self.alphaBeta(-alpha - 1, -alpha, next_depth, ply + 1, true);
                } else {
                    // First move always searched with full window
                    score = alpha + 1; // Force full search
                }
            }

            // Full window search if PVS failed high or first move
            if (score > alpha and !self.stop_search.load(.monotonic)) {
                score = -try self.alphaBeta(-beta, -alpha, next_depth, ply + 1, true);
            }

            // Unmake move
            self.popAccumulator();
            self.board.unmakeMoveUnchecked(move, undo);
            self.previous_move = old_previous;
            if (ply < MAX_PLY) {
                self.continuation_keys[ply] = old_continuation_key;
            }
            self.history_count = old_hist_count; // Restore history count

            // A stop sentinel is not a score. Restore all state, then leave
            // before it can affect the root move, heuristics, or TT.
            if (self.stop_search.load(.monotonic)) return 0;

            if (ply == 0) {
                self.root_move_nodes[rootMoveBucket(move)] +|= self.nodes_searched - root_nodes_before;
            }

            // Discourage speculative non-checking minor-piece sacs for pawns unless
            // search already proves concrete compensation.
            if (speculative_sac_candidate and !gives_check and !eval.isMateScore(score)) {
                score -= speculativeSacPenalty(speculative_sac_see, search_depth);
            }

            moves_searched += 1;

            // Track quiet moves for history updates
            if (!is_capture and !is_promotion and quiets_tried_count < 64) {
                quiets_tried[quiets_tried_count] = move;
                quiets_tried_count += 1;
            } else if (is_capture and captures_tried_count < captures_tried.len) {
                captures_tried[captures_tried_count] = move;
                captures_tried_count += 1;
            }

            if (score > best_score) {
                best_score = score;
                best_move = move;
                if (ply == 0) {
                    self.root_best_move = move;
                }
            }

            alpha = @max(alpha, score);

            // Beta cutoff (use beta_adj for consistency with bound determination)
            if (alpha >= beta_adj) {
                // Update heuristics for non-captures
                if (!is_capture and !is_promotion) {
                    self.updateQuietHeuristicsOnBetaCutoff(
                        move,
                        search_depth,
                        ply,
                        color,
                        old_previous,
                        quiets_tried[0..quiets_tried_count],
                    );
                } else if (is_capture) {
                    self.updateCaptureHeuristicsOnBetaCutoff(
                        move,
                        search_depth,
                        captures_tried[0..captures_tried_count],
                    );
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

        self.storeAlphaBetaResult(best_score, original_alpha, beta_adj, search_depth, ply, best_move);

        return best_score;
    }

    inline fn combinedHistoryCount(self: *Self) usize {
        return self.game_history_count + self.history_count;
    }

    inline fn hashAtCombinedIndex(self: *Self, idx: usize) u64 {
        if (idx < self.game_history_count) {
            return self.game_history[idx];
        }
        return self.position_history[idx - self.game_history_count];
    }

    /// Check for repetition draw.
    /// A repeat formed inside the search is immediately forceable. A position
    /// seen only before the root needs two prior occurrences, making this node
    /// the third occurrence rather than a false twofold draw.
    fn isRepetition(self: *Self) bool {
        const current_hash = self.board.zobrist_hasher.zobrist_hash;
        const total = self.combinedHistoryCount();
        if (total < 3) return false;

        const halfmove = @as(usize, @intCast(self.board.board.halfmove_clock));
        if (halfmove < 4) return false;
        const max_back = @min(halfmove, total - 1);

        var pre_root_matches: u32 = 0;
        var plies_back: usize = 2;
        while (plies_back <= max_back) : (plies_back += 2) {
            const idx = total - 1 - plies_back;
            if (self.hashAtCombinedIndex(idx) == current_hash) {
                if (idx >= self.game_history_count) return true;
                pre_root_matches += 1;
                if (pre_root_matches >= 2) return true;
            }
        }
        return false;
    }

    fn repetitionScore(self: *Self) i32 {
        _ = self;
        return DRAW_SCORE;
    }

    fn nonPawnMaterial(b: board.BitBoard, color: piece.Color) i32 {
        const color_bb = b.getColorBitboard(color);
        const pawn_count = @as(i32, @intCast(@popCount(color_bb & b.getKindBitboard(.pawn))));
        return countMaterial(b, color) - pawn_count * eval.PAWN_VALUE;
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
    fn quiescence(self: *Self, alpha_in: i32, beta: i32, ply: u32) anyerror!i32 {
        if (self.checkShouldStop()) {
            return 0;
        }

        self.nodes_searched += 1;
        self.seldepth = @max(self.seldepth, ply);

        // Check if we're in check - if so, we must search all evasions (not just captures)
        const in_check = self.board.isInCheck(self.board.board.move);

        if (ply >= MAX_SEARCH_PLY) {
            return if (in_check) DRAW_SCORE else self.evaluatePosition();
        }

        if (ply > 0 and self.isRepetition()) {
            return self.repetitionScore();
        }
        if (try self.rule50Score(in_check, ply)) |score| {
            return score;
        }

        var alpha = alpha_in;
        const original_alpha = alpha_in;
        const is_pv_node = (beta - alpha_in) > 1;
        var tt_move: ?Move = null;

        if (self.tt.probe(self.board.zobrist_hasher.zobrist_hash)) |entry| {
            if (entry.best_move.from() != 0 or entry.best_move.to() != 0) {
                tt_move = entry.best_move;
            }

            if (!is_pv_node) {
                const tt_score = scoreFromTT(entry.score, ply);
                switch (entry.bound) {
                    .exact => return tt_score,
                    .lower_bound => {
                        if (tt_score >= beta) return tt_score;
                    },
                    .upper_bound => {
                        if (tt_score <= alpha) return tt_score;
                    },
                }
            }
        }

        // Stand pat - but only when not in check
        var stand_pat: i32 = -INF;
        if (!in_check) {
            stand_pat = self.evaluatePosition();

            if (stand_pat >= beta) {
                self.storeQuiescenceResult(stand_pat, original_alpha, beta, ply, tt_move);
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
        if (tt_move) |tt| {
            for (moves.sliceMut(), 0..) |move, i| {
                if (move.from() == tt.from() and move.to() == tt.to() and move.promotion() == tt.promotion()) {
                    if (i != 0) {
                        const tmp = moves.moves[0];
                        moves.moves[0] = moves.moves[i];
                        moves.moves[i] = tmp;
                    }
                    break;
                }
            }
        }

        const move_color = self.board.board.move;
        const opponent_color = oppositeColor(move_color);
        var best_move: ?Move = null;

        for (moves.slice()) |move| {
            // Delta pruning - skip captures that can't possibly raise alpha (skip when in check)
            if (!in_check) {
                const is_en_passant = self.board.board.en_passant_square == move.to() and
                    self.board.board.getPieceAt(move.from(), move_color) == .pawn and
                    self.board.board.getPieceAt(move.to(), opponent_color) == null;
                const captured_value = if (is_en_passant)
                    eval.PAWN_VALUE
                else if (self.board.board.getPieceAt(move.to(), opponent_color)) |p|
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

                // SEE pruning - skip clearly losing non-promotion captures.
                if (move.promotion() == null) {
                    const is_tt_move = tt_move != null and movesEqual(move, tt_move.?);
                    if (!is_tt_move and
                        staticExchangeEvalPosition(&self.board.board, move) < QS_SEE_PRUNE_MARGIN_CP and
                        !self.moveGivesCheck(move))
                    {
                        continue;
                    }
                }
            }

            // Make move
            self.prepareAccumulatorForMove();
            const undo = self.board.makeMoveWithUndoUnchecked(move);
            self.pushAccumulator(move, undo);

            const old_hist_count = self.history_count;
            if (self.history_count < 512) {
                self.position_history[self.history_count] = self.board.zobrist_hasher.zobrist_hash;
                self.history_count += 1;
            }

            // Recursive search
            const score = -try self.quiescence(-beta, -alpha, ply + 1);

            // Unmake move
            self.popAccumulator();
            self.board.unmakeMoveUnchecked(move, undo);
            self.history_count = old_hist_count;

            if (self.stop_search.load(.monotonic)) return 0;

            if (score >= beta) {
                self.storeQuiescenceResult(score, original_alpha, beta, ply, move);
                return beta;
            }
            if (score > alpha) {
                alpha = score;
                best_move = move;
            }
        }

        self.storeQuiescenceResult(alpha, original_alpha, beta, ply, best_move);
        return alpha;
    }

    /// Order tactical moves with SEE first, MVV-LVA second.
    /// In quiescence check evasions, legal quiets may appear as well.
    fn orderCaptures(self: *Self, moves: *MoveList) void {
        const move_slice = moves.sliceMut();
        var scores: [256]i32 = undefined;

        const move_color = self.board.board.move;
        const opponent_color = oppositeColor(move_color);

        for (move_slice, 0..) |move, i| {
            var score = staticExchangeEvalPosition(&self.board.board, move) * SEE_CAPTURE_SCALE;
            if (capture_history_mod.ENABLED) {
                score += self.capture_history.score(&self.board.board, move);
            }

            if (self.board.board.en_passant_square == move.to() and
                self.board.board.getPieceAt(move.from(), move_color) == .pawn and
                self.board.board.getPieceAt(move.to(), opponent_color) == null)
            {
                score += eval.PAWN_VALUE * 10;
            } else if (self.board.board.getPieceAt(move.to(), opponent_color)) |victim| {
                const victim_value = eval.getPieceValue(victim);
                const attacker_value = if (self.board.board.getPieceAt(move.from(), self.board.board.move)) |att|
                    eval.getPieceValue(att)
                else
                    0;

                score += victim_value * 12 - attacker_value;
            } else if (move.promotion()) |promo| {
                score += seePieceValue(promo) * 4;
            } else {
                // In-check quiescence can include quiet evasions.
                score += self.history.getForColor(move, move_color);
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

    fn writeUciLine(self: *Self, comptime fmt: []const u8, args: anytype) void {
        const file = self.uci_output orelse return;
        var out_buf: [1024]u8 = undefined;
        const line = std.fmt.bufPrint(&out_buf, fmt, args) catch return;
        file.writeAll(line) catch return;
        file.writeAll("\n") catch return;
    }
};

test "time budget stays inside usable clock and preserves fixed movetime" {
    const fixed = calculateTimeBudgetFor(.{
        .move_time = 80,
        .move_overhead = 30,
    }, .white, 1).?;
    try std.testing.expectEqual(@as(u64, 80), fixed.soft_ms);
    try std.testing.expectEqual(@as(u64, 80), fixed.hard_ms);

    const low = calculateTimeBudgetFor(.{
        .wtime = 50,
        .move_overhead = 30,
    }, .white, 1).?;
    try std.testing.expect(low.soft_ms <= 20);
    try std.testing.expect(low.hard_ms <= 20);
    try std.testing.expect(low.soft_ms <= low.hard_ms);

    const exhausted = calculateTimeBudgetFor(.{
        .wtime = 20,
        .move_overhead = 30,
    }, .white, 1).?;
    try std.testing.expectEqual(@as(u64, 0), exhausted.soft_ms);
    try std.testing.expectEqual(@as(u64, 0), exhausted.hard_ms);

    const last_move = calculateTimeBudgetFor(.{
        .wtime = 10_000,
        .moves_to_go = 1,
        .move_overhead = 0,
    }, .white, 20).?;
    try std.testing.expect(last_move.soft_ms >= 4_000);
    try std.testing.expect(last_move.soft_ms <= last_move.hard_ms);
    try std.testing.expect(last_move.hard_ms <= 10_000);

    const huge = calculateTimeBudgetFor(.{
        .wtime = std.math.maxInt(u64),
        .winc = std.math.maxInt(u64),
        .move_overhead = 0,
    }, .white, 1).?;
    try std.testing.expect(huge.soft_ms <= huge.hard_ms);
}

test "adaptive soft limit spends more on unstable roots and less on stable roots" {
    const budget = TimeBudget{ .soft_ms = 100, .hard_ms = 400 };
    try std.testing.expectEqual(@as(u64, 100), adaptiveSoftLimit(budget, 4, 300, 1000, 0, 40));

    const unstable = adaptiveSoftLimit(budget, 8, 300, 1000, 0, 40);
    const stable = adaptiveSoftLimit(budget, 8, 900, 1000, 14, 0);
    try std.testing.expect(unstable > budget.soft_ms);
    try std.testing.expect(stable < budget.soft_ms);
    try std.testing.expect(unstable <= budget.hard_ms);
    try std.testing.expect(stable <= budget.hard_ms);

    const capped = adaptiveSoftLimit(.{ .soft_ms = 500, .hard_ms = 600 }, 8, 300, 1000, 0, 40);
    try std.testing.expectEqual(@as(u64, 600), capped);
}

test "repetition distinguishes duplicate root, pre-root twofold, and search cycles" {
    var test_board = Board.startpos();
    test_board.board.halfmove_clock = 8;
    var stop = std.atomic.Value(bool).init(false);
    var tt = try TranspositionTable.init(std.testing.allocator, 1);
    defer tt.deinit();
    var engine = try SearchEngine.init(&test_board, std.testing.allocator, &stop, &tt, false, null, 0, 100);
    defer engine.deinit();

    const hash = test_board.zobrist_hasher.zobrist_hash;

    engine.setGameHistory(&.{hash});
    _ = try engine.search(.{ .depth = 0 });
    try std.testing.expectEqual(@as(usize, 0), engine.history_count);

    engine.setGameHistory(&.{});
    _ = try engine.search(.{ .depth = 0 });
    try std.testing.expectEqual(@as(usize, 1), engine.history_count);

    engine.game_history[0] = hash;
    engine.game_history[1] = hash ^ 0x11;
    engine.game_history[2] = hash;
    engine.game_history_count = 3;
    engine.history_count = 0;
    try std.testing.expect(!engine.isRepetition());

    engine.game_history[0] = hash;
    engine.game_history[1] = hash ^ 0x11;
    engine.game_history[2] = hash;
    engine.game_history[3] = hash ^ 0x22;
    engine.game_history[4] = hash;
    engine.game_history_count = 5;
    try std.testing.expect(engine.isRepetition());

    engine.game_history[0] = hash ^ 0x33;
    engine.game_history_count = 1;
    engine.position_history[0] = hash;
    engine.position_history[1] = hash ^ 0x44;
    engine.position_history[2] = hash;
    engine.history_count = 3;
    try std.testing.expect(engine.isRepetition());
    try std.testing.expectEqual(DRAW_SCORE, try engine.quiescence(-INF, INF, 1));
}

test "rule-50 handling preserves checkmate precedence in search and qsearch" {
    var test_board = try Board.fromFen("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1");
    var stop = std.atomic.Value(bool).init(false);
    var tt = try TranspositionTable.init(std.testing.allocator, 1);
    defer tt.deinit();
    var engine = try SearchEngine.init(&test_board, std.testing.allocator, &stop, &tt, false, null, 0, 100);
    defer engine.deinit();

    const expected = -eval.mateIn(0);
    try std.testing.expectEqual(expected, try engine.alphaBeta(-INF, INF, 1, 0, true));
    try std.testing.expectEqual(expected, try engine.quiescence(-INF, INF, 0));
}

test "search ply ceiling leaves accumulator headroom" {
    try std.testing.expect(MAX_SEARCH_PLY < ACC_STACK_SIZE);
}
