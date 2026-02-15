const board = @import("../bitboard.zig");
const Board = board.Board;
const Move = board.Move;
const MoveList = board.MoveList;
const piece = @import("../piece.zig");
const eval = @import("../evaluation.zig");
const heuristics = @import("heuristics.zig");
const HistoryTable = heuristics.HistoryTable;
const MAX_KILLER_MOVES = heuristics.MAX_KILLER_MOVES;

pub const SEE_CAPTURE_SCALE: i32 = 128;

pub inline fn oppositeColor(color: piece.Color) piece.Color {
    return if (color == .white) .black else .white;
}

pub inline fn seePieceValue(piece_type: piece.Type) i32 {
    return switch (piece_type) {
        .pawn => eval.PAWN_VALUE,
        .knight => eval.KNIGHT_VALUE,
        .bishop => eval.BISHOP_VALUE,
        .rook => eval.ROOK_VALUE,
        .queen => eval.QUEEN_VALUE,
        .king => eval.KING_VALUE,
    };
}

const SeeAttacker = struct {
    from_sq: u6,
    piece_type: piece.Type,
};

inline fn attackersToSquare(b: board.BitBoard, square: u6, occupied: u64) u64 {
    const white = b.getColorBitboard(.white) & occupied;
    const black = b.getColorBitboard(.black) & occupied;
    const pawns = b.getKindBitboard(.pawn) & occupied;
    const knights = b.getKindBitboard(.knight) & occupied;
    const bishops = b.getKindBitboard(.bishop) & occupied;
    const rooks = b.getKindBitboard(.rook) & occupied;
    const queens = b.getKindBitboard(.queen) & occupied;
    const kings = b.getKindBitboard(.king) & occupied;

    var attackers: u64 = 0;

    attackers |= board.getPawnAttacks(square, .black) & white & pawns;
    attackers |= board.getPawnAttacks(square, .white) & black & pawns;
    attackers |= board.getKnightAttacks(square) & knights;
    attackers |= board.getKingAttacks(square) & kings;

    const bishop_like = bishops | queens;
    if (bishop_like != 0) {
        attackers |= board.getBishopAttacks(square, occupied) & bishop_like;
    }

    const rook_like = rooks | queens;
    if (rook_like != 0) {
        attackers |= board.getRookAttacks(square, occupied) & rook_like;
    }

    return attackers;
}

fn leastValuableAttacker(b: board.BitBoard, attackers: u64, color: piece.Color) ?SeeAttacker {
    const own_attackers = attackers & b.getColorBitboard(color);
    if (own_attackers == 0) return null;

    const order = [_]piece.Type{ .pawn, .knight, .bishop, .rook, .queen, .king };
    inline for (order) |piece_type| {
        const set = own_attackers & b.getKindBitboard(piece_type);
        if (set != 0) {
            return SeeAttacker{
                .from_sq = @intCast(@ctz(set)),
                .piece_type = piece_type,
            };
        }
    }

    return null;
}

fn staticExchangeRec(
    b: board.BitBoard,
    occupied: u64,
    square: u6,
    side_to_move: piece.Color,
    captured_value: i32,
) i32 {
    const attackers = attackersToSquare(b, square, occupied);
    const next = leastValuableAttacker(b, attackers, side_to_move) orelse return 0;

    const from_mask = @as(u64, 1) << @intCast(next.from_sq);
    const next_occupied = occupied & ~from_mask;

    const gain = captured_value - staticExchangeRec(
        b,
        next_occupied,
        square,
        oppositeColor(side_to_move),
        seePieceValue(next.piece_type),
    );
    return @max(gain, 0);
}

pub fn staticExchangeEvalPosition(b: board.BitBoard, move: Move) i32 {
    const us = b.move;
    const them = oppositeColor(us);
    const from_sq = move.from();
    const to_sq = move.to();

    const moving_piece = b.getPieceAt(from_sq, us) orelse return 0;
    var captured_piece = b.getPieceAt(to_sq, them);
    const is_en_passant = moving_piece == .pawn and b.en_passant_square == to_sq and captured_piece == null;
    if (is_en_passant) {
        captured_piece = .pawn;
    }

    const promotion = move.promotion();
    if (captured_piece == null and promotion == null) {
        return 0;
    }

    var occupied = b.occupied();
    const from_mask = @as(u64, 1) << @intCast(from_sq);
    const to_mask = @as(u64, 1) << @intCast(to_sq);

    occupied &= ~from_mask;
    if (is_en_passant) {
        const ep_capture_sq: u8 = if (us == .white) to_sq - 8 else to_sq + 8;
        occupied &= ~(@as(u64, 1) << @intCast(ep_capture_sq));
    }
    occupied |= to_mask;

    const captured_value = if (captured_piece) |cp| seePieceValue(cp) else 0;
    const moved_value = if (promotion) |promo| seePieceValue(promo) else seePieceValue(moving_piece);
    const promotion_gain: i32 = if (promotion != null and moving_piece == .pawn)
        moved_value - seePieceValue(.pawn)
    else
        0;

    const reply_gain = staticExchangeRec(b, occupied, @intCast(to_sq), them, moved_value);
    return captured_value + promotion_gain - reply_gain;
}

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

    captures: MoveList,
    quiets: MoveList,
    bad_captures: MoveList,

    capture_scores: [256]i32,
    quiet_scores: [256]i32,

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

    pub fn next(self: *Self) ?Move {
        while (true) {
            switch (self.stage) {
                .tt_move => {
                    self.stage = .generate_captures;
                    if (self.tt_move) |tt| {
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

                        if (idx != self.capture_idx) {
                            self.captures.moves[idx] = self.captures.moves[self.capture_idx];
                            self.capture_scores[idx] = self.capture_scores[self.capture_idx];
                            self.captures.moves[self.capture_idx] = move;
                            self.capture_scores[self.capture_idx] = score;
                        }
                        self.capture_idx += 1;

                        if (self.isTTMove(move)) continue;

                        if (score >= GOOD_CAPTURE_BASE) {
                            return move;
                        } else {
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

                        if (killer.from() == 0 and killer.to() == 0) continue;
                        if (self.isTTMove(killer)) continue;
                        if (self.isCapture(killer)) continue;

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

                        if (idx != self.quiet_idx) {
                            self.quiets.moves[idx] = self.quiets.moves[self.quiet_idx];
                            self.quiet_scores[idx] = self.quiet_scores[self.quiet_idx];
                            self.quiets.moves[self.quiet_idx] = move;
                            self.quiet_scores[self.quiet_idx] = score;
                        }
                        self.quiet_idx += 1;

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
        const opponent_color = oppositeColor(self.board_ptr.board.move);

        for (self.captures.slice(), 0..) |move, i| {
            const see_score = staticExchangeEvalPosition(self.board_ptr.board, move);
            var score: i32 = if (see_score >= 0) GOOD_CAPTURE_BASE else BAD_CAPTURE_SCORE;
            score += see_score * SEE_CAPTURE_SCALE;

            if (self.board_ptr.board.getPieceAt(move.to(), opponent_color)) |victim| {
                const victim_value = eval.getPieceValue(victim);
                const attacker_value = if (self.board_ptr.board.getPieceAt(move.from(), self.board_ptr.board.move)) |att|
                    eval.getPieceValue(att)
                else
                    0;

                score += victim_value * 12 - attacker_value;
            }

            if (move.promotion()) |promo| {
                score += seePieceValue(promo);
            }

            self.capture_scores[i] = score;
        }
    }

    fn scoreQuiets(self: *Self) void {
        const occupied = self.board_ptr.board.occupied();
        const friendly = self.board_ptr.board.getColorBitboard(self.board_ptr.board.move);

        for (self.quiets.slice(), 0..) |move, i| {
            var score = self.history.getForColor(move, self.board_ptr.board.move);

            if (self.counter_move) |cm| {
                if (cm.from() == move.from() and cm.to() == move.to()) {
                    score += 100_000;
                }
            }

            if (self.board_ptr.board.getPieceAt(move.from(), self.board_ptr.board.move)) |piece_type| {
                var mobility_bonus: i32 = 0;
                const to_sq: u8 = move.to();
                const from_mask = @as(u64, 1) << @intCast(move.from());
                const to_mask = @as(u64, 1) << @intCast(move.to());
                const occupied_after = (occupied & ~from_mask) | to_mask;

                switch (piece_type) {
                    .knight => {
                        const attacks = board.getKnightAttacks(@intCast(to_sq)) & ~friendly;
                        mobility_bonus = @as(i32, @intCast(@popCount(attacks))) * 3;
                    },
                    .bishop => {
                        const attacks = board.getBishopAttacks(@intCast(to_sq), occupied_after) & ~friendly;
                        mobility_bonus = @as(i32, @intCast(@popCount(attacks))) * 2;
                    },
                    .rook => {
                        const attacks = board.getRookAttacks(@intCast(to_sq), occupied_after) & ~friendly;
                        mobility_bonus = @as(i32, @intCast(@popCount(attacks))) * 2;
                    },
                    .queen => {
                        const attacks = (board.getBishopAttacks(@intCast(to_sq), occupied_after) | board.getRookAttacks(@intCast(to_sq), occupied_after)) & ~friendly;
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

        const piece_type = self.board_ptr.board.getPieceAt(from_sq, color) orelse return false;

        const our_pieces = self.board_ptr.board.getColorBitboard(color);
        const opponent_pieces = self.board_ptr.board.getColorBitboard(opponent_color);
        const occupied = self.board_ptr.board.occupied();
        const to_mask = @as(u64, 1) << @intCast(to_sq);
        if ((our_pieces & to_mask) != 0) return false;

        const promotion = move.promotion();
        if (promotion) |promo| {
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

                if ((to_rank == promotion_rank) != (promotion != null)) {
                    break :blk false;
                }

                if (file_delta == 0) {
                    if (is_capture) break :blk false;

                    if (rank_delta == forward_delta) {
                        break :blk (occupied & to_mask) == 0;
                    }

                    if (rank_delta == forward_delta * 2 and from_rank == start_rank) {
                        const mid_sq: u8 = if (color == .white) from_sq + 8 else from_sq - 8;
                        const mid_mask = @as(u64, 1) << @intCast(mid_sq);
                        break :blk (occupied & mid_mask) == 0 and (occupied & to_mask) == 0;
                    }

                    break :blk false;
                }

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

                if ((board.getKingAttacks(@intCast(from_sq)) & to_mask) != 0) {
                    break :blk true;
                }

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

        if ((board.getKnightAttacks(square) & attacker_bb & b.getKindBitboard(.knight)) != 0) {
            return true;
        }

        if ((board.getKingAttacks(square) & attacker_bb & b.getKindBitboard(.king)) != 0) {
            return true;
        }

        const defender_color = if (attacker_color == .white) piece.Color.black else piece.Color.white;
        if ((board.getPawnAttacks(square, defender_color) & attacker_bb & b.getKindBitboard(.pawn)) != 0) {
            return true;
        }

        const bishops_queens = attacker_bb & (b.getKindBitboard(.bishop) | b.getKindBitboard(.queen));
        if (bishops_queens != 0 and (board.getBishopAttacks(square, occupied) & bishops_queens) != 0) {
            return true;
        }

        const rooks_queens = attacker_bb & (b.getKindBitboard(.rook) | b.getKindBitboard(.queen));
        if (rooks_queens != 0 and (board.getRookAttacks(square, occupied) & rooks_queens) != 0) {
            return true;
        }

        return false;
    }

    fn isLegal(self: *Self, move: Move) bool {
        const color = self.board_ptr.board.move;
        const old_board = self.board_ptr.board;

        self.board_ptr.applyMoveUncheckedForLegality(move);
        const legal = !self.board_ptr.isInCheck(color);
        self.board_ptr.board = old_board;

        return legal;
    }
};
