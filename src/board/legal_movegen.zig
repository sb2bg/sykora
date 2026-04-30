const piece = @import("../piece.zig");
const board = @import("../bitboard.zig");
const attacks = @import("attacks.zig");
const legal_context = @import("legal_context.zig");
const LegalContext = legal_context.LegalContext;

const RANK_1: u64 = 0x00000000000000FF;
const RANK_8: u64 = 0xFF00000000000000;
const RANK_3: u64 = 0x0000000000FF0000;
const RANK_6: u64 = 0x0000FF0000000000;
const NOT_A: u64 = 0xFEFEFEFEFEFEFEFE;
const NOT_H: u64 = 0x7F7F7F7F7F7F7F7F;

pub const Mode = enum { all, captures, quiets };

inline fn appendMove(moves: anytype, from: u6, to: u6, promo: ?piece.Type) void {
    const MoveT = @TypeOf(moves.*.moves[0]);
    moves.append(MoveT.init(@intCast(from), @intCast(to), promo));
}

inline fn pinMaskFor(ctx: *const LegalContext, from: u6) u64 {
    return if (ctx.isPinned(from)) attacks.line(ctx.king_sq, from) else ~@as(u64, 0);
}

inline fn oppositeColor(c: piece.Color) piece.Color {
    return if (c == .white) .black else .white;
}

/// Squares attacked by the side `them`, computed with `our_king_sq` removed
/// from occupancy so the king cannot slide along a slider's ray and think it's
/// safe.
fn enemyAttackMapXKing(b: board.BitBoard, them: piece.Color, our_king_sq: u6) u64 {
    const occ_xk = b.occupied() & ~(@as(u64, 1) << our_king_sq);
    const enemy = b.getColorBitboard(them);
    var atk: u64 = 0;

    var pawns = enemy & b.getKindBitboard(.pawn);
    while (pawns != 0) {
        const sq: u6 = @intCast(@ctz(pawns));
        pawns &= pawns - 1;
        atk |= attacks.getPawnAttacks(sq, them);
    }

    var knights = enemy & b.getKindBitboard(.knight);
    while (knights != 0) {
        const sq: u6 = @intCast(@ctz(knights));
        knights &= knights - 1;
        atk |= attacks.getKnightAttacks(sq);
    }

    const bishops_queens = enemy & (b.getKindBitboard(.bishop) | b.getKindBitboard(.queen));
    var bq = bishops_queens;
    while (bq != 0) {
        const sq: u6 = @intCast(@ctz(bq));
        bq &= bq - 1;
        atk |= attacks.getBishopAttacks(sq, occ_xk);
    }

    const rooks_queens = enemy & (b.getKindBitboard(.rook) | b.getKindBitboard(.queen));
    var rq = rooks_queens;
    while (rq != 0) {
        const sq: u6 = @intCast(@ctz(rq));
        rq &= rq - 1;
        atk |= attacks.getRookAttacks(sq, occ_xk);
    }

    const kings = enemy & b.getKindBitboard(.king);
    if (kings != 0) {
        const sq: u6 = @intCast(@ctz(kings));
        atk |= attacks.getKingAttacks(sq);
    }

    return atk;
}

inline fn modeTarget(mode: Mode, friendly_complement: u64, enemy: u64, empty: u64) u64 {
    return switch (mode) {
        .all => friendly_complement,
        .captures => enemy,
        .quiets => empty,
    };
}

fn generateKnightMovesFast(
    b: board.BitBoard,
    moves: anytype,
    ctx: *const LegalContext,
    target: u64,
) void {
    const us = b.move;
    const ours = b.getColorBitboard(us);
    // A pinned knight has no legal moves (every knight move leaves the pin line).
    var knights = ours & b.getKindBitboard(.knight) & ~ctx.pinned;
    while (knights != 0) {
        const from: u6 = @intCast(@ctz(knights));
        knights &= knights - 1;
        var atk = attacks.getKnightAttacks(from) & target & ctx.check_mask;
        while (atk != 0) {
            const to: u6 = @intCast(@ctz(atk));
            atk &= atk - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateBishopMovesFast(
    b: board.BitBoard,
    moves: anytype,
    ctx: *const LegalContext,
    target: u64,
) void {
    const us = b.move;
    const ours = b.getColorBitboard(us);
    const occ = b.occupied();
    var bishops = ours & b.getKindBitboard(.bishop);
    while (bishops != 0) {
        const from: u6 = @intCast(@ctz(bishops));
        bishops &= bishops - 1;
        var atk = attacks.getBishopAttacks(from, occ) & target & ctx.check_mask & pinMaskFor(ctx, from);
        while (atk != 0) {
            const to: u6 = @intCast(@ctz(atk));
            atk &= atk - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateRookMovesFast(
    b: board.BitBoard,
    moves: anytype,
    ctx: *const LegalContext,
    target: u64,
) void {
    const us = b.move;
    const ours = b.getColorBitboard(us);
    const occ = b.occupied();
    var rooks = ours & b.getKindBitboard(.rook);
    while (rooks != 0) {
        const from: u6 = @intCast(@ctz(rooks));
        rooks &= rooks - 1;
        var atk = attacks.getRookAttacks(from, occ) & target & ctx.check_mask & pinMaskFor(ctx, from);
        while (atk != 0) {
            const to: u6 = @intCast(@ctz(atk));
            atk &= atk - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateQueenMovesFast(
    b: board.BitBoard,
    moves: anytype,
    ctx: *const LegalContext,
    target: u64,
) void {
    const us = b.move;
    const ours = b.getColorBitboard(us);
    const occ = b.occupied();
    var queens = ours & b.getKindBitboard(.queen);
    while (queens != 0) {
        const from: u6 = @intCast(@ctz(queens));
        queens &= queens - 1;
        const queen_atk = attacks.getBishopAttacks(from, occ) | attacks.getRookAttacks(from, occ);
        var atk = queen_atk & target & ctx.check_mask & pinMaskFor(ctx, from);
        while (atk != 0) {
            const to: u6 = @intCast(@ctz(atk));
            atk &= atk - 1;
            appendMove(moves, from, to, null);
        }
    }
}

fn generateKingMovesFast(
    b: board.BitBoard,
    moves: anytype,
    ctx: *const LegalContext,
    comptime mode: Mode,
) void {
    const us = b.move;
    const them = oppositeColor(us);
    const ours = b.getColorBitboard(us);
    const enemy = b.getColorBitboard(them);
    const empty = ~(ours | enemy);

    const enemy_atk = enemyAttackMapXKing(b, them, ctx.king_sq);

    const target = modeTarget(mode, ~ours, enemy, empty);
    var atk = attacks.getKingAttacks(ctx.king_sq) & target & ~enemy_atk;
    while (atk != 0) {
        const to: u6 = @intCast(@ctz(atk));
        atk &= atk - 1;
        appendMove(moves, ctx.king_sq, to, null);
    }

    // Castling: only in "all" or "quiets" modes, and never while in check.
    if (mode == .captures) return;
    if (ctx.checkers != 0) return;

    const occ = ours | enemy;
    if (us == .white) {
        if (b.castle_rights.white_kingside) {
            const path: u64 = (@as(u64, 1) << 5) | (@as(u64, 1) << 6); // f1, g1 empty
            const safe: u64 = (@as(u64, 1) << 4) | (@as(u64, 1) << 5) | (@as(u64, 1) << 6); // e1,f1,g1 not attacked
            if ((occ & path) == 0 and (enemy_atk & safe) == 0) {
                appendMove(moves, 4, 6, null);
            }
        }
        if (b.castle_rights.white_queenside) {
            const path: u64 = (@as(u64, 1) << 1) | (@as(u64, 1) << 2) | (@as(u64, 1) << 3); // b1,c1,d1 empty
            const safe: u64 = (@as(u64, 1) << 4) | (@as(u64, 1) << 3) | (@as(u64, 1) << 2); // e1,d1,c1 not attacked
            if ((occ & path) == 0 and (enemy_atk & safe) == 0) {
                appendMove(moves, 4, 2, null);
            }
        }
    } else {
        if (b.castle_rights.black_kingside) {
            const path: u64 = (@as(u64, 1) << 61) | (@as(u64, 1) << 62);
            const safe: u64 = (@as(u64, 1) << 60) | (@as(u64, 1) << 61) | (@as(u64, 1) << 62);
            if ((occ & path) == 0 and (enemy_atk & safe) == 0) {
                appendMove(moves, 60, 62, null);
            }
        }
        if (b.castle_rights.black_queenside) {
            const path: u64 = (@as(u64, 1) << 57) | (@as(u64, 1) << 58) | (@as(u64, 1) << 59);
            const safe: u64 = (@as(u64, 1) << 60) | (@as(u64, 1) << 59) | (@as(u64, 1) << 58);
            if ((occ & path) == 0 and (enemy_atk & safe) == 0) {
                appendMove(moves, 60, 58, null);
            }
        }
    }
}

const PROMO_TYPES = [_]piece.Type{ .queen, .knight, .rook, .bishop };

inline fn fromForWhitePush(to: u6) u6 {
    return @intCast(@as(u32, to) - 8);
}

inline fn fromForWhiteDoublePush(to: u6) u6 {
    return @intCast(@as(u32, to) - 16);
}

inline fn fromForWhiteCapLeft(to: u6) u6 {
    return @intCast(@as(u32, to) - 7);
}

inline fn fromForWhiteCapRight(to: u6) u6 {
    return @intCast(@as(u32, to) - 9);
}

inline fn fromForBlackPush(to: u6) u6 {
    return @intCast(@as(u32, to) + 8);
}

inline fn fromForBlackDoublePush(to: u6) u6 {
    return @intCast(@as(u32, to) + 16);
}

inline fn fromForBlackCapLeft(to: u6) u6 {
    return @intCast(@as(u32, to) + 7);
}

inline fn fromForBlackCapRight(to: u6) u6 {
    return @intCast(@as(u32, to) + 9);
}

inline fn pinFilter(ctx: *const LegalContext, from: u6, to: u6) bool {
    if (!ctx.isPinned(from)) return true;
    return ((@as(u64, 1) << to) & attacks.line(ctx.king_sq, from)) != 0;
}

fn emitPawnSet(
    moves: anytype,
    dest_bb: u64,
    fromFn: anytype,
    ctx: *const LegalContext,
    comptime promotions: bool,
) void {
    var bb = dest_bb & ctx.check_mask;
    while (bb != 0) {
        const to: u6 = @intCast(@ctz(bb));
        bb &= bb - 1;
        const from: u6 = fromFn(to);
        if (!pinFilter(ctx, from, to)) continue;
        if (promotions) {
            inline for (PROMO_TYPES) |pt| {
                appendMove(moves, from, to, pt);
            }
        } else {
            appendMove(moves, from, to, null);
        }
    }
}

fn generatePawnMovesFast(
    b: board.BitBoard,
    moves: anytype,
    ctx: *const LegalContext,
    comptime mode: Mode,
) void {
    const us = b.move;
    const ours = b.getColorBitboard(us);
    const enemy = b.getColorBitboard(if (us == .white) piece.Color.black else piece.Color.white);
    const occ = ours | enemy;
    const empty = ~occ;
    const pawns = ours & b.getKindBitboard(.pawn);
    if (pawns == 0) return;

    if (us == .white) {
        const promo_rank: u64 = RANK_8;
        const push_one_all = (pawns << 8) & empty;
        const push_one_no_promo = push_one_all & ~promo_rank;
        const push_one_promo = push_one_all & promo_rank;

        // Quiet pushes (excluded from captures-only mode).
        if (mode != .captures) {
            emitPawnSet(moves, push_one_no_promo, fromForWhitePush, ctx, false);
            const push_two = ((push_one_all & RANK_3) << 8) & empty;
            emitPawnSet(moves, push_two, fromForWhiteDoublePush, ctx, false);
        }
        // Push-promotions (tactical → captures or all, not quiets).
        if (mode != .quiets) {
            emitPawnSet(moves, push_one_promo, fromForWhitePush, ctx, true);
        }

        // Captures (and capture-promotions) — only in captures or all modes.
        if (mode != .quiets) {
            const left_caps_all = ((pawns & NOT_A) << 7) & enemy;
            emitPawnSet(moves, left_caps_all & ~promo_rank, fromForWhiteCapLeft, ctx, false);
            emitPawnSet(moves, left_caps_all & promo_rank, fromForWhiteCapLeft, ctx, true);

            const right_caps_all = ((pawns & NOT_H) << 9) & enemy;
            emitPawnSet(moves, right_caps_all & ~promo_rank, fromForWhiteCapRight, ctx, false);
            emitPawnSet(moves, right_caps_all & promo_rank, fromForWhiteCapRight, ctx, true);

            // En passant.
            if (b.en_passant_square) |ep_sq_u8| {
                emitWhiteEnPassant(b, moves, ctx, @intCast(ep_sq_u8));
            }
        }
    } else {
        const promo_rank: u64 = RANK_1;
        const push_one_all = (pawns >> 8) & empty;
        const push_one_no_promo = push_one_all & ~promo_rank;
        const push_one_promo = push_one_all & promo_rank;

        if (mode != .captures) {
            emitPawnSet(moves, push_one_no_promo, fromForBlackPush, ctx, false);
            const push_two = ((push_one_all & RANK_6) >> 8) & empty;
            emitPawnSet(moves, push_two, fromForBlackDoublePush, ctx, false);
        }
        if (mode != .quiets) {
            emitPawnSet(moves, push_one_promo, fromForBlackPush, ctx, true);
        }

        if (mode != .quiets) {
            const left_caps_all = ((pawns & NOT_H) >> 7) & enemy;
            emitPawnSet(moves, left_caps_all & ~promo_rank, fromForBlackCapLeft, ctx, false);
            emitPawnSet(moves, left_caps_all & promo_rank, fromForBlackCapLeft, ctx, true);

            const right_caps_all = ((pawns & NOT_A) >> 9) & enemy;
            emitPawnSet(moves, right_caps_all & ~promo_rank, fromForBlackCapRight, ctx, false);
            emitPawnSet(moves, right_caps_all & promo_rank, fromForBlackCapRight, ctx, true);

            if (b.en_passant_square) |ep_sq_u8| {
                emitBlackEnPassant(b, moves, ctx, @intCast(ep_sq_u8));
            }
        }
    }
}

fn epIsLegal(
    b: board.BitBoard,
    ctx: *const LegalContext,
    from: u6,
    to: u6,
    captured_sq: u6,
) bool {
    // Pin (non-rank): pinned-pawn diagonal still allowed if `to` lies on the pin line.
    if (!pinFilter(ctx, from, to)) return false;

    // Check resolution: legal iff destination is in check_mask, OR the captured
    // pawn is itself the (single) checker.
    if (ctx.check_mask != ~@as(u64, 0)) {
        const to_bit = @as(u64, 1) << to;
        const cap_bit = @as(u64, 1) << captured_sq;
        if ((to_bit & ctx.check_mask) == 0 and (cap_bit & ctx.checkers) == 0) {
            return false;
        }
    }

    // Horizontal (rank) x-ray pin: simulate occupancy after the EP and check
    // if a rook/queen now attacks the king on the rank.
    const them = oppositeColor(b.move);
    const enemy = b.getColorBitboard(them);
    const occ = b.occupied();
    const occ_after =
        (occ & ~(@as(u64, 1) << from) & ~(@as(u64, 1) << captured_sq)) | (@as(u64, 1) << to);

    const enemy_rq = enemy & (b.getKindBitboard(.rook) | b.getKindBitboard(.queen));
    if (enemy_rq != 0) {
        if ((attacks.getRookAttacks(ctx.king_sq, occ_after) & enemy_rq) != 0) return false;
    }

    // Diagonal x-ray pin: same shape, with bishops/queens. Rare but possible
    // when our king sits on a diagonal to an enemy bishop and the captured
    // pawn was the only blocker.
    const enemy_bq = enemy & (b.getKindBitboard(.bishop) | b.getKindBitboard(.queen));
    if (enemy_bq != 0) {
        if ((attacks.getBishopAttacks(ctx.king_sq, occ_after) & enemy_bq) != 0) return false;
    }

    return true;
}

fn emitWhiteEnPassant(b: board.BitBoard, moves: anytype, ctx: *const LegalContext, ep_sq: u6) void {
    const us = piece.Color.white;
    const pawns = b.getColorBitboard(us) & b.getKindBitboard(.pawn);
    // Squares from which a white pawn could capture into ep_sq.
    const attackers = attacks.getPawnAttacks(ep_sq, .black) & pawns;
    if (attackers == 0) return;
    const captured_sq: u6 = @intCast(@as(u32, ep_sq) - 8);

    var src = attackers;
    while (src != 0) {
        const from: u6 = @intCast(@ctz(src));
        src &= src - 1;
        if (epIsLegal(b, ctx, from, ep_sq, captured_sq)) {
            appendMove(moves, from, ep_sq, null);
        }
    }
}

fn emitBlackEnPassant(b: board.BitBoard, moves: anytype, ctx: *const LegalContext, ep_sq: u6) void {
    const us = piece.Color.black;
    const pawns = b.getColorBitboard(us) & b.getKindBitboard(.pawn);
    const attackers = attacks.getPawnAttacks(ep_sq, .white) & pawns;
    if (attackers == 0) return;
    const captured_sq: u6 = @intCast(@as(u32, ep_sq) + 8);

    var src = attackers;
    while (src != 0) {
        const from: u6 = @intCast(@ctz(src));
        src &= src - 1;
        if (epIsLegal(b, ctx, from, ep_sq, captured_sq)) {
            appendMove(moves, from, ep_sq, null);
        }
    }
}

/// Public entry point: emit all legal moves matching `mode`.
pub fn generateLegalMovesFast(
    b: board.BitBoard,
    moves: anytype,
    ctx: *const LegalContext,
    comptime mode: Mode,
) void {
    const us = b.move;
    const ours = b.getColorBitboard(us);
    const enemy = b.getColorBitboard(oppositeColor(us));
    const empty = ~(ours | enemy);

    // Double check → only the king can move; skip everything else.
    if (ctx.checkCount() >= 2) {
        generateKingMovesFast(b, moves, ctx, mode);
        return;
    }

    const target = modeTarget(mode, ~ours, enemy, empty);

    generatePawnMovesFast(b, moves, ctx, mode);
    generateKnightMovesFast(b, moves, ctx, target);
    generateBishopMovesFast(b, moves, ctx, target);
    generateRookMovesFast(b, moves, ctx, target);
    generateQueenMovesFast(b, moves, ctx, target);
    generateKingMovesFast(b, moves, ctx, mode);
}
