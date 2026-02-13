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
    counter_move: ?Move,
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
        counter_move: ?Move,
        history: *const HistoryTable,
        ply: u32,
    ) Self {
        return Self{
            .board_ptr = board_ptr,
            .stage = .tt_move,
            .tt_move = tt_move,
            .killer_moves = killer_moves,
            .counter_move = counter_move,
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
                        // Validate TT move is fully legal (not just pseudo-legal)
                        if (self.isPseudoLegal(tt) and self.isLegal(tt)) {
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
            var score = self.history.getForColor(move, self.board_ptr.board.move);

            // Counter move bonus - prioritize moves that refute opponent's last move
            if (self.counter_move) |cm| {
                if (cm.from() == move.from() and cm.to() == move.to()) {
                    score += 100_000;
                }
            }

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
        return self.isPseudoLegalImpl(move, false);
    }

    fn isPseudoLegalQuiet(self: *Self, move: Move) bool {
        return self.isPseudoLegalImpl(move, true);
    }

    fn isPseudoLegalImpl(self: *Self, move: Move, quiet_only: bool) bool {
        const color = self.board_ptr.board.move;
        const opponent_color = if (color == .white) piece.Color.black else piece.Color.white;
        const from_sq = move.from();
        const to_sq = move.to();

        if (from_sq == to_sq) return false;

        // Check if we have a piece on the from square
        const piece_type = self.board_ptr.board.getPieceAt(from_sq, color) orelse return false;

        // Destination can't contain our own piece
        const our_pieces = self.board_ptr.board.getColorBitboard(color);
        const opponent_pieces = self.board_ptr.board.getColorBitboard(opponent_color);
        const occupied = self.board_ptr.board.occupied();
        const to_mask = @as(u64, 1) << @intCast(to_sq);
        if ((our_pieces & to_mask) != 0) return false;

        const promotion = move.promotion();
        if (promotion) |promo| {
            // Only proper promotion pieces are legal
            if (promo == .pawn or promo == .king) return false;
            if (piece_type != .pawn) return false;
        }

        const is_en_passant = piece_type == .pawn and
            self.board_ptr.board.en_passant_square == to_sq and
            (opponent_pieces & to_mask) == 0;
        const is_capture = (opponent_pieces & to_mask) != 0 or is_en_passant;

        if (quiet_only and (is_capture or promotion != null)) {
            return false;
        }

        return switch (piece_type) {
            .pawn => blk: {
                const from_file: i16 = @intCast(from_sq % 8);
                const to_file: i16 = @intCast(to_sq % 8);
                const from_rank: u8 = from_sq / 8;
                const to_rank: u8 = to_sq / 8;
                const file_delta = to_file - from_file;
                const rank_delta: i16 = @as(i16, @intCast(to_rank)) - @as(i16, @intCast(from_rank));

                const forward_delta: i16 = if (color == .white) 1 else -1;
                const start_rank: u8 = if (color == .white) 1 else 6;
                const promotion_rank: u8 = if (color == .white) 7 else 0;

                // Promotions are mandatory when reaching the last rank
                if ((to_rank == promotion_rank) != (promotion != null)) {
                    break :blk false;
                }

                // Forward pawn pushes
                if (file_delta == 0) {
                    if (is_capture) break :blk false;

                    // Single push
                    if (rank_delta == forward_delta) {
                        break :blk (occupied & to_mask) == 0;
                    }

                    // Double push from starting rank
                    if (rank_delta == forward_delta * 2 and from_rank == start_rank) {
                        const mid_sq: u8 = if (color == .white) from_sq + 8 else from_sq - 8;
                        const mid_mask = @as(u64, 1) << @intCast(mid_sq);
                        break :blk (occupied & mid_mask) == 0 and (occupied & to_mask) == 0;
                    }

                    break :blk false;
                }

                // Diagonal captures (including en passant)
                if (@abs(file_delta) == 1 and rank_delta == forward_delta) {
                    break :blk (opponent_pieces & to_mask) != 0 or self.board_ptr.board.en_passant_square == to_sq;
                }

                break :blk false;
            },
            .knight => {
                if (promotion != null) return false;
                return (board.getKnightAttacks(@intCast(from_sq)) & to_mask) != 0;
            },
            .bishop => {
                if (promotion != null) return false;
                return (board.getBishopAttacks(@intCast(from_sq), occupied) & to_mask) != 0;
            },
            .rook => {
                if (promotion != null) return false;
                return (board.getRookAttacks(@intCast(from_sq), occupied) & to_mask) != 0;
            },
            .queen => {
                if (promotion != null) return false;
                return (board.getQueenAttacks(@intCast(from_sq), occupied) & to_mask) != 0;
            },
            .king => blk: {
                if (promotion != null) break :blk false;

                // Regular king moves
                if ((board.getKingAttacks(@intCast(from_sq)) & to_mask) != 0) {
                    break :blk true;
                }

                // Castling (must be quiet by definition)
                if (is_capture) break :blk false;

                if (color == .white and from_sq == 4) {
                    if (to_sq == 6) {
                        if (!self.board_ptr.board.castle_rights.white_kingside) break :blk false;
                        if ((occupied & ((@as(u64, 1) << 5) | (@as(u64, 1) << 6))) != 0) break :blk false;
                        if (self.board_ptr.board.getPieceAt(7, .white) != .rook) break :blk false;
                        if (self.isSquareAttackedBy(4, opponent_color) or self.isSquareAttackedBy(5, opponent_color)) break :blk false;
                        break :blk true;
                    }
                    if (to_sq == 2) {
                        if (!self.board_ptr.board.castle_rights.white_queenside) break :blk false;
                        if ((occupied & ((@as(u64, 1) << 1) | (@as(u64, 1) << 2) | (@as(u64, 1) << 3))) != 0) break :blk false;
                        if (self.board_ptr.board.getPieceAt(0, .white) != .rook) break :blk false;
                        if (self.isSquareAttackedBy(4, opponent_color) or self.isSquareAttackedBy(3, opponent_color)) break :blk false;
                        break :blk true;
                    }
                } else if (color == .black and from_sq == 60) {
                    if (to_sq == 62) {
                        if (!self.board_ptr.board.castle_rights.black_kingside) break :blk false;
                        if ((occupied & ((@as(u64, 1) << 61) | (@as(u64, 1) << 62))) != 0) break :blk false;
                        if (self.board_ptr.board.getPieceAt(63, .black) != .rook) break :blk false;
                        if (self.isSquareAttackedBy(60, opponent_color) or self.isSquareAttackedBy(61, opponent_color)) break :blk false;
                        break :blk true;
                    }
                    if (to_sq == 58) {
                        if (!self.board_ptr.board.castle_rights.black_queenside) break :blk false;
                        if ((occupied & ((@as(u64, 1) << 57) | (@as(u64, 1) << 58) | (@as(u64, 1) << 59))) != 0) break :blk false;
                        if (self.board_ptr.board.getPieceAt(56, .black) != .rook) break :blk false;
                        if (self.isSquareAttackedBy(60, opponent_color) or self.isSquareAttackedBy(59, opponent_color)) break :blk false;
                        break :blk true;
                    }
                }

                break :blk false;
            },
        };
    }

    fn isSquareAttackedBy(self: *Self, square: u6, attacker_color: piece.Color) bool {
        const b = self.board_ptr.board;
        const attacker_bb = b.getColorBitboard(attacker_color);
        const occupied = b.occupied();

        // Knights
        if ((board.getKnightAttacks(square) & attacker_bb & b.getKindBitboard(.knight)) != 0) {
            return true;
        }

        // Kings
        if ((board.getKingAttacks(square) & attacker_bb & b.getKindBitboard(.king)) != 0) {
            return true;
        }

        // Pawns
        const defender_color = if (attacker_color == .white) piece.Color.black else piece.Color.white;
        if ((board.getPawnAttacks(square, defender_color) & attacker_bb & b.getKindBitboard(.pawn)) != 0) {
            return true;
        }

        // Diagonal sliders
        const bishops_queens = attacker_bb & (b.getKindBitboard(.bishop) | b.getKindBitboard(.queen));
        if (bishops_queens != 0 and (board.getBishopAttacks(square, occupied) & bishops_queens) != 0) {
            return true;
        }

        // Orthogonal sliders
        const rooks_queens = attacker_bb & (b.getKindBitboard(.rook) | b.getKindBitboard(.queen));
        if (rooks_queens != 0 and (board.getRookAttacks(square, occupied) & rooks_queens) != 0) {
            return true;
        }

        return false;
    }

    fn isLegal(self: *Self, move: Move) bool {
        const color = self.board_ptr.board.move;

        // Save state
        const old_board = self.board_ptr.board;

        // Make move
        self.board_ptr.applyMoveUncheckedForLegality(move);

        // Check legality
        const legal = !self.board_ptr.isInCheck(color);

        // Restore state
        self.board_ptr.board = old_board;

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
    moves_to_go: ?u64 = null,
    depth: ?u64 = null,
    start_depth: u32 = 1,
};

pub const SearchResult = struct {
    best_move: Move,
    score: i32,
    nodes: usize,
    time_ms: i64,
    depth: u32,
};

const TimeBudget = struct {
    soft_ms: u64,
    hard_ms: u64,
};

const MAX_PLY = 64;
const MAX_KILLER_MOVES = 2;
const EVAL_CACHE_SIZE = 16384; // Must be power-of-two for fast masking.
const EVAL_CACHE_EMPTY_KEY = std.math.maxInt(u64);

const INF: i32 = 32000;
const DRAW_SCORE: i32 = 0;
const REPETITION_BASE_CONTEMPT_CP: i32 = 20;
const REPETITION_SMALL_ADV_CP: i32 = 40;
const REPETITION_MEDIUM_ADV_CP: i32 = 80;
const REPETITION_LARGE_ADV_CP: i32 = 160;
const REPETITION_HUGE_ADV_CP: i32 = 260;
const REPETITION_ADV_EVAL_THRESHOLD_CP: i32 = 30;
const REPETITION_CYCLE_EVAL_THRESHOLD_CP: i32 = 80;
const REPETITION_CYCLE_PENALTY_CP: i32 = 200;
const PAWN_ENDGAME_ROOT_EXTENSION: u32 = 1;
const TIME_CHECK_NODE_INTERVAL_MASK: usize = 2047; // Check hard deadline every 2048 counted nodes.

fn elapsedMs(start: std.time.Instant) i64 {
    const now = std.time.Instant.now() catch return 0;
    const ns = now.since(start);
    return @intCast(@divFloor(ns, std.time.ns_per_ms));
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
pub const TTEntryBound = enum(u8) {
    exact,
    lower_bound,
    upper_bound,
};

pub const TTEntry = struct {
    hash: u64,
    depth: u8,
    score: i32,
    bound: TTEntryBound,
    best_move: Move,
    age: u8,

    pub fn init() TTEntry {
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
pub const TranspositionTable = struct {
    const Self = @This();

    entries: []TTEntry,
    size: usize,
    current_age: u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size_mb: usize) !Self {
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

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.entries);
    }

    pub fn clear(self: *Self) void {
        for (self.entries) |*entry| {
            entry.* = TTEntry.init();
        }
        self.current_age = 0;
    }

    pub fn nextAge(self: *Self) void {
        self.current_age +%= 1;
    }

    pub fn resize(self: *Self, new_size_mb: usize) !void {
        self.allocator.free(self.entries);
        const entry_size = @sizeOf(TTEntry);
        const num_entries = (new_size_mb * 1024 * 1024) / entry_size;
        const entries = try self.allocator.alloc(TTEntry, num_entries);
        for (entries) |*entry| {
            entry.* = TTEntry.init();
        }
        self.entries = entries;
        self.size = num_entries;
        self.current_age = 0;
    }

    pub fn index(self: *Self, hash: u64) usize {
        return @as(usize, @intCast(hash % @as(u64, @intCast(self.size))));
    }

    pub fn probe(self: *Self, hash: u64) ?*TTEntry {
        const idx = self.index(hash);
        if (self.entries[idx].hash == hash) {
            return &self.entries[idx];
        }
        return null;
    }

    pub fn store(self: *Self, hash: u64, depth: u8, score: i32, bound: TTEntryBound, best_move: Move) void {
        const idx = self.index(hash);
        const entry = &self.entries[idx];

        // Replacement strategy:
        // - Always replace empty slots
        // - Always replace old entries from previous searches
        // - For same position: only replace if depth >= existing depth
        // - For different position (collision): use depth-preferred replacement
        const replace = entry.hash == 0 or // empty
            entry.age != self.current_age or // old entry
            (entry.hash == hash and depth >= entry.depth) or // same position, equal or deeper
            (entry.hash != hash and depth > entry.depth); // collision, deeper search wins

        if (replace) {
            entry.hash = hash;
            entry.depth = depth;
            entry.score = score;
            entry.bound = bound;
            entry.best_move = best_move;
            entry.age = self.current_age;
        } else if (entry.hash == hash and best_move.from() != 0) {
            // For same position with shallower search, still update best_move if we found one
            // This helps preserve the PV even from shallower searches
            entry.best_move = best_move;
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
    uci_output: ?std.fs.File,
    root_best_move: Move,

    // Search state
    tt: *TranspositionTable,
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
    use_nnue: bool,
    nnue_net: ?*const nnue.Network,
    nnue_blend: i32,
    nnue_scale: i32,
    nnue_screlu: bool,
    eval_cache_keys: [EVAL_CACHE_SIZE]u64,
    eval_cache_values: [EVAL_CACHE_SIZE]i32,
    search_start_time: ?std.time.Instant,
    soft_time_limit_ms: ?u64,
    hard_time_limit_ms: ?u64,

    pub fn init(
        board_ptr: *Board,
        allocator: std.mem.Allocator,
        stop_search: *std.atomic.Value(bool),
        tt: *TranspositionTable,
        use_nnue: bool,
        nnue_net: ?*const nnue.Network,
        nnue_blend: i32,
        nnue_scale: i32,
        nnue_screlu: bool,
    ) Self {
        return Self{
            .board = board_ptr,
            .allocator = allocator,
            .stop_search = stop_search,
            .uci_output = null,
            .root_best_move = Move.init(0, 0, null),
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
            .use_nnue = use_nnue,
            .nnue_net = nnue_net,
            .nnue_blend = nnue_blend,
            .nnue_scale = nnue_scale,
            .nnue_screlu = nnue_screlu,
            .eval_cache_keys = [_]u64{EVAL_CACHE_EMPTY_KEY} ** EVAL_CACHE_SIZE,
            .eval_cache_values = [_]i32{0} ** EVAL_CACHE_SIZE,
            .search_start_time = null,
            .soft_time_limit_ms = null,
            .hard_time_limit_ms = null,
        };
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
                const nn_raw = nnue.evaluate(net, self.board, self.nnue_screlu);
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

    /// Run a search and return the best move
    pub fn search(self: *Self, options: SearchOptions) !SearchResult {
        const start_time = std.time.Instant.now() catch unreachable;

        // Reset search state
        self.nodes_searched = 0;
        self.seldepth = 0;
        self.killer_moves = KillerMoves.init();
        self.history.age(); // Age history instead of clearing
        self.counter_moves.clear();
        self.previous_move = null;
        // Initialize position history
        self.position_history[0] = self.board.zobrist_hasher.zobrist_hash;
        self.history_count = 1;

        // Calculate time budget (soft/hard limits) for clocked search.
        const time_budget = self.calculateTimeBudget(options);
        self.search_start_time = start_time;
        self.soft_time_limit_ms = if (time_budget) |b| b.soft_ms else null;
        self.hard_time_limit_ms = if (time_budget) |b| b.hard_ms else null;

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

        // Iterative deepening
        var depth: u32 = options.start_depth;
        while (depth <= max_depth) : (depth += 1) {
            if (self.stop_search.load(.seq_cst)) break;

            const iter_start = std.time.Instant.now() catch unreachable;

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

                if (self.stop_search.load(.seq_cst)) break;

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
            if (self.stop_search.load(.seq_cst)) break;

            best_move = self.root_best_move;
            best_score = score;

            const iter_time = elapsedMs(iter_start);
            const total_time = elapsedMs(start_time);

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

            // Check soft/hard time limits at iteration boundary.
            if (time_budget) |budget| {
                const total_time_ms: u64 = @intCast(@max(total_time, 0));
                const iter_time_ms: u64 = @intCast(@max(iter_time, 0));
                // Stop on soft deadline after a completed iteration.
                if (total_time_ms >= budget.soft_ms) {
                    break;
                }
                // Hard cap: do not start another iteration if it's unlikely to finish in time.
                if (total_time_ms >= budget.hard_ms or total_time_ms + (iter_time_ms * 2) >= budget.hard_ms) {
                    break;
                }
            }

            // Stop if we found a mate
            if (eval.isMateScore(score)) {
                break;
            }
        }

        const elapsed = elapsedMs(start_time);
        self.search_start_time = null;
        self.soft_time_limit_ms = null;
        self.hard_time_limit_ms = null;

        return SearchResult{
            .best_move = best_move,
            .score = best_score,
            .nodes = self.nodes_searched,
            .time_ms = elapsed,
            .depth = depth - 1,
        };
    }

    fn calculateTimeBudget(self: *Self, options: SearchOptions) ?TimeBudget {
        if (options.infinite) {
            return null;
        } else if (options.move_time) |move_time| {
            const hard = @max(@as(u64, 1), move_time);
            const soft = @max(@as(u64, 1), hard * 9 / 10);
            return .{ .soft_ms = soft, .hard_ms = hard };
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
        if (time_remaining == 0) {
            return .{ .soft_ms = 1, .hard_ms = 1 };
        }

        // Use GUI-provided movestogo when available; otherwise estimate by phase.
        var moves_to_go: u64 = options.moves_to_go orelse 0;
        if (moves_to_go == 0) {
            const fullmove_no: u64 = @intCast(self.board.board.fullmove_number);
            moves_to_go = if (fullmove_no < 20)
                30
            else if (fullmove_no < 40)
                24
            else
                18;
        }
        moves_to_go = @max(moves_to_go, 2);

        // Keep a small reserve for move transmission / scheduling jitter.
        var reserve_ms = @max(@as(u64, 10), @min(time_remaining / 20, @as(u64, 250)));
        if (time_remaining < 2000) {
            reserve_ms = @max(@as(u64, 5), time_remaining / 10);
        }
        if (reserve_ms + 5 >= time_remaining) {
            reserve_ms = time_remaining / 2;
        }
        var usable_ms = time_remaining -| reserve_ms;
        usable_ms = @max(usable_ms, 1);

        // Baseline: remaining / (moves_to_go + safety) + most of increment.
        const base_ms = usable_ms / (moves_to_go + 3);
        var soft_ms = base_ms + (our_increment * 3 / 4);

        // Fractional cap to avoid over-spending a move.
        var cap_divisor: u64 = 8; // 12.5% of usable time
        if (moves_to_go <= 10) cap_divisor = 5; // 20%
        if (moves_to_go <= 5) cap_divisor = 4; // 25%
        if (time_remaining < 5000) cap_divisor = @max(cap_divisor, @as(u64, 12)); // ~8.3%
        if (time_remaining < 2000) cap_divisor = @max(cap_divisor, @as(u64, 18)); // ~5.6%

        var soft_cap_ms = usable_ms / cap_divisor;
        soft_cap_ms = @max(soft_cap_ms, 10);
        soft_cap_ms = @min(soft_cap_ms, 30_000);

        const min_soft_ms: u64 = if (time_remaining < 1500) 5 else 20;
        if (soft_ms < min_soft_ms) soft_ms = min_soft_ms;
        if (soft_ms > soft_cap_ms) soft_ms = soft_cap_ms;
        if (soft_ms > usable_ms) soft_ms = usable_ms;

        // Hard deadline is above soft, but still bounded by usable time.
        const hard_extra_ms = @max(@as(u64, 25), soft_ms / 2);
        var hard_ms = soft_ms + hard_extra_ms;
        if (hard_ms > usable_ms) hard_ms = usable_ms;
        if (hard_ms < soft_ms) hard_ms = soft_ms;
        hard_ms = @max(hard_ms, 1);
        soft_ms = @max(@as(u64, 1), @min(soft_ms, hard_ms));

        return .{ .soft_ms = soft_ms, .hard_ms = hard_ms };
    }

    inline fn maybeStopOnHardDeadline(self: *Self) void {
        const hard_limit_ms = self.hard_time_limit_ms orelse return;
        if ((self.nodes_searched & TIME_CHECK_NODE_INTERVAL_MASK) != 0) {
            return;
        }
        const start_time = self.search_start_time orelse return;
        const elapsed = elapsedMs(start_time);
        if (elapsed <= 0) {
            return;
        }
        const elapsed_ms: u64 = @intCast(elapsed);
        if (elapsed_ms >= hard_limit_ms) {
            self.stop_search.store(true, .seq_cst);
        }
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
            return self.repetitionScore();
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
        if (in_check) {
            search_depth += 1;
        }

        const is_pv_node = (beta_adj - alpha) > 1;

        // Probe transposition table
        var tt_move: ?Move = null;
        if (self.tt.probe(self.board.zobrist_hasher.zobrist_hash)) |entry| {
            if (entry.best_move.from() != 0 or entry.best_move.to() != 0) {
                tt_move = entry.best_move;
            }

            // Use TT score for cutoffs (only in non-PV nodes with sufficient depth)
            if (!is_pv_node and entry.depth >= search_depth) {
                const tt_score = scoreFromTT(entry.score, ply);
                switch (entry.bound) {
                    .exact => return tt_score,
                    .lower_bound => {
                        if (tt_score >= beta_adj) return tt_score;
                    },
                    .upper_bound => {
                        if (tt_score <= alpha) return tt_score;
                    },
                }
            }
        }

        // Static evaluation for pruning decisions
        const static_eval = if (!in_check) self.evaluatePosition() else -INF;

        // Reverse futility pruning (static null move pruning)
        // If static eval is far above beta at shallow depths, prune immediately
        if (!is_pv_node and !in_check and depth <= 5) {
            const rfp_margin = 80 * @as(i32, @intCast(depth));
            if (static_eval - rfp_margin >= beta_adj) {
                return static_eval;
            }
        }

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
                const old_move = self.board.board.move;

                // Compute EP file contribution BEFORE flipping side (hash was built with old side)
                const old_ep_file = Board.epFileForHash(self.board.board);

                // Make null move (just flip side to move)
                self.board.board.move = if (self.board.board.move == .white) .black else .white;
                self.board.zobrist_hasher.zobrist_hash ^= zobrist.RandomTurn;

                // Clear en passant — only XOR if it was actually in the hash
                if (old_ep_file) |f| {
                    self.board.zobrist_hasher.zobrist_hash ^= zobrist.RandomEnPassant[f];
                }
                self.board.board.en_passant_square = null;

                // Null move is a quiet move for clock purposes.
                if (self.board.board.halfmove_clock < std.math.maxInt(u8)) {
                    self.board.board.halfmove_clock += 1;
                }
                if (old_move == .black and self.board.board.fullmove_number < std.math.maxInt(u16)) {
                    self.board.board.fullmove_number += 1;
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
        const counter_move: ?Move = if (self.previous_move) |prev| self.counter_moves.get(prev) else null;
        var move_picker = MovePicker.init(self.board, tt_move, killers, counter_move, &self.history, ply);

        var best_move: ?Move = null;
        var best_score: i32 = -INF;
        var moves_searched: u32 = 0;
        var quiets_tried: [64]Move = undefined;
        var quiets_tried_count: usize = 0;
        const color = self.board.board.move;
        const root_pawn_endgame = ply == 0 and search_depth >= 2 and isPurePawnEndgame(self.board.board);

        while (move_picker.next()) |move| {
            const old_previous = self.previous_move;

            const move_color = self.board.board.move;
            const opponent_color = if (move_color == .white) piece.Color.black else piece.Color.white;
            const is_capture = self.board.board.getPieceAt(move.to(), opponent_color) != null;
            const is_promotion = move.promotion() != null;

            // Futility pruning - skip quiet moves if futile (but not if move gives check)
            if (futile and !is_capture and !is_promotion and moves_searched > 0) {
                // Make the move temporarily to check if it gives check
                const probe_undo = self.board.makeMoveWithUndoUnchecked(move);
                const gives_check = self.board.isInCheck(self.board.board.move);
                self.board.unmakeMoveUnchecked(move, probe_undo);

                if (!gives_check and !self.killer_moves.isKiller(move, ply)) {
                    continue;
                }
            }

            // Make move
            const undo = self.board.makeMoveWithUndoUnchecked(move);
            self.previous_move = move;
            self.nodes_searched += 1;
            self.maybeStopOnHardDeadline();
            if (self.stop_search.load(.seq_cst)) {
                self.board.unmakeMoveUnchecked(move, undo);
                self.previous_move = old_previous;
                break;
            }

            // Add position to history for repetition detection
            const old_hist_count = self.history_count;
            if (self.history_count < 512) {
                self.position_history[self.history_count] = self.board.zobrist_hasher.zobrist_hash;
                self.history_count += 1;
            }
            const repetition_matches_after_move = self.repetitionMatchCount();
            const extension: u32 = if (ply == 0 and
                search_depth >= 2 and
                root_pawn_endgame)
                PAWN_ENDGAME_ROOT_EXTENSION
            else
                0;
            const next_depth = search_depth - 1 + extension;

            var score: i32 = undefined;

            // Late Move Reductions (LMR)
            if (moves_searched >= LMR_FULL_DEPTH_MOVES and
                search_depth >= LMR_MIN_DEPTH and
                !is_pv_node and
                !in_check and
                !is_capture and
                !is_promotion and
                !self.board.isInCheck(self.board.board.move))
            {
                // Calculate reduction based on depth and moves searched
                var reduction: u32 = 1;
                if (moves_searched > 6) reduction += 1;
                if (search_depth > 6) reduction += 1;
                if (!is_pv_node) reduction += 1;
                reduction = @min(reduction, LMR_REDUCTION_LIMIT);
                reduction = @min(reduction, search_depth - 1);

                // Search with reduced depth
                score = -try self.alphaBeta(-alpha - 1, -alpha, next_depth - reduction, ply + 1, true);

                // If LMR found something good, re-search at full depth
                if (score > alpha) {
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
            if (score > alpha) {
                score = -try self.alphaBeta(-beta, -alpha, next_depth, ply + 1, true);
            }

            // Unmake move
            self.board.unmakeMoveUnchecked(move, undo);
            self.previous_move = old_previous;
            self.history_count = old_hist_count; // Restore history count

            moves_searched += 1;

            // Avoid entering obvious twofold cycles when we're already clearly better.
            if (repetition_matches_after_move == 1 and !is_capture and !is_promotion) {
                if (static_eval >= REPETITION_CYCLE_EVAL_THRESHOLD_CP) {
                    score -= REPETITION_CYCLE_PENALTY_CP;
                } else if (static_eval <= -REPETITION_CYCLE_EVAL_THRESHOLD_CP) {
                    score += REPETITION_CYCLE_PENALTY_CP;
                }
            }

            // Track quiet moves for history updates
            if (!is_capture and !is_promotion and quiets_tried_count < 64) {
                quiets_tried[quiets_tried_count] = move;
                quiets_tried_count += 1;
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

        // Store in transposition table
        const bound: TTEntryBound = if (best_score <= original_alpha)
            .upper_bound
        else if (best_score >= beta_adj)
            .lower_bound
        else
            .exact;
        const score_to_store = scoreToTT(best_score, ply);
        self.tt.store(self.board.zobrist_hasher.zobrist_hash, @intCast(search_depth), score_to_store, bound, best_move orelse Move.init(0, 0, null));

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

    inline fn repetitionMatchCount(self: *Self) u32 {
        const current_hash = self.board.zobrist_hasher.zobrist_hash;
        const total = self.combinedHistoryCount();
        if (total < 3) return 0;

        // 50-move clock bounds how far back a repetition can exist.
        const halfmove = @as(usize, @intCast(self.board.board.halfmove_clock));
        const max_back = @min(halfmove, total - 1);

        var matches: u32 = 0;
        var plies_back: usize = 2; // same side to move only
        while (plies_back <= max_back) : (plies_back += 2) {
            const idx = total - 1 - plies_back;
            if (self.hashAtCombinedIndex(idx) == current_hash) {
                matches += 1;
            }
        }

        return matches;
    }

    /// Check if current position is a true threefold repetition.
    /// We require two prior matches of the current hash with the same side to move.
    fn isRepetition(self: *Self) bool {
        return self.repetitionMatchCount() >= 2;
    }

    /// Return a contempt-adjusted score for repetition draws.
    /// Positive means draw is attractive for side-to-move (typically when worse),
    /// negative means avoid draw when better.
    fn repetitionScore(self: *Self) i32 {
        const static_eval = self.evaluatePosition();
        const stm = self.board.board.move;
        const opp = if (stm == .white) piece.Color.black else piece.Color.white;

        const stm_material = countMaterial(self.board.board, stm);
        const opp_material = countMaterial(self.board.board, opp);
        const material_adv = stm_material - opp_material;

        var contempt: i32 = REPETITION_BASE_CONTEMPT_CP;
        const abs_eval = @abs(static_eval);
        const abs_material_adv = @abs(material_adv);

        if (abs_eval >= 80 or abs_material_adv >= eval.PAWN_VALUE) {
            contempt = REPETITION_SMALL_ADV_CP;
        }
        if (abs_eval >= 160 or abs_material_adv >= 2 * eval.PAWN_VALUE) {
            contempt = REPETITION_MEDIUM_ADV_CP;
        }
        if (abs_eval >= 280 or abs_material_adv >= eval.ROOK_VALUE) {
            contempt = REPETITION_LARGE_ADV_CP;
        }
        if (abs_eval >= 500 or abs_material_adv >= eval.QUEEN_VALUE) {
            contempt = REPETITION_HUGE_ADV_CP;
        }

        // Side to move appears better: avoid repetition draw.
        if (static_eval > REPETITION_ADV_EVAL_THRESHOLD_CP or material_adv > eval.PAWN_VALUE / 2) {
            return -contempt;
        }
        // Side to move appears worse: prefer repetition draw.
        if (static_eval < -REPETITION_ADV_EVAL_THRESHOLD_CP or material_adv < -(eval.PAWN_VALUE / 2)) {
            return contempt;
        }
        return DRAW_SCORE;
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
        self.maybeStopOnHardDeadline();
        if (self.stop_search.load(.seq_cst)) {
            return 0;
        }
        self.seldepth = @max(self.seldepth, ply);

        // Check if we're in check - if so, we must search all evasions (not just captures)
        const in_check = self.board.isInCheck(self.board.board.move);

        var alpha = alpha_in;

        // Stand pat - but only when not in check
        var stand_pat: i32 = -INF;
        if (!in_check) {
            stand_pat = self.evaluatePosition();

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

            // Make move
            const undo = self.board.makeMoveWithUndoUnchecked(move);

            // Recursive search
            const score = -try self.quiescence(-beta, -alpha, ply + 1);

            // Unmake move
            self.board.unmakeMoveUnchecked(move, undo);

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

    fn writeUciLine(self: *Self, comptime fmt: []const u8, args: anytype) void {
        const file = self.uci_output orelse return;
        var out_buf: [1024]u8 = undefined;
        const line = std.fmt.bufPrint(&out_buf, fmt, args) catch return;
        file.writeAll(line) catch return;
        file.writeAll("\n") catch return;
    }
};
