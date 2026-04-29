const piece = @import("../piece.zig");
const board = @import("../bitboard.zig");
const attacks = @import("attacks.zig");

/// Per-node legality precompute. Compute once per search node, then each move's
/// legality reduces to a few bit ops (no make/unmake required for non-king,
/// non-EP, non-castle moves).
///
/// Semantics:
///   - `king_sq`: square of the side-to-move's king.
///   - `checkers`: bitboard of enemy pieces currently giving check.
///   - `check_mask`: set of legal target squares for non-king moves under check.
///       * not in check: ~0 (every square allowed)
///       * single check: the checker's square OR squares strictly between the
///                       checker and the king (i.e. you may capture or block).
///                       For a contact / leaper checker this collapses to the
///                       checker bit alone (`between` is 0).
///       * double check: 0 (only the king may move).
///   - `pinned`: bitboard of side-to-move's pieces pinned to the king.
///       Pinned pieces may only move along the line through king and pinner;
///       use `attacks.line(king_sq, from_sq)` as the move-target mask.
pub const LegalContext = struct {
    king_sq: u6,
    checkers: u64,
    check_mask: u64,
    pinned: u64,

    pub inline fn checkCount(self: LegalContext) u32 {
        return @popCount(self.checkers);
    }

    pub inline fn isPinned(self: LegalContext, sq: u6) bool {
        return ((self.pinned >> sq) & 1) != 0;
    }
};

pub fn computeLegalContext(b: board.BitBoard) LegalContext {
    const us = b.move;
    const them: piece.Color = if (us == .white) .black else .white;

    const ours = b.getColorBitboard(us);
    const enemy = b.getColorBitboard(them);
    const king_bb = ours & b.getKindBitboard(.king);

    if (king_bb == 0) {
        // No king on board (e.g. partial test position). Treat as "no
        // restrictions" so callers fall through to whatever movegen they use.
        return .{
            .king_sq = 0,
            .checkers = 0,
            .check_mask = ~@as(u64, 0),
            .pinned = 0,
        };
    }

    const king_sq: u6 = @intCast(@ctz(king_bb));
    const occ = ours | enemy;

    const enemy_pawns = enemy & b.getKindBitboard(.pawn);
    const enemy_knights = enemy & b.getKindBitboard(.knight);
    const enemy_bishops = enemy & b.getKindBitboard(.bishop);
    const enemy_rooks = enemy & b.getKindBitboard(.rook);
    const enemy_queens = enemy & b.getKindBitboard(.queen);
    const enemy_bq = enemy_bishops | enemy_queens;
    const enemy_rq = enemy_rooks | enemy_queens;

    var checkers: u64 = 0;
    // Pawns: a pawn of color `us` at king_sq attacks the squares from which an
    // enemy pawn could attack king_sq (pawn attacks are direction-symmetric
    // when you flip color).
    checkers |= attacks.getPawnAttacks(king_sq, us) & enemy_pawns;
    checkers |= attacks.getKnightAttacks(king_sq) & enemy_knights;
    checkers |= attacks.getBishopAttacks(king_sq, occ) & enemy_bq;
    checkers |= attacks.getRookAttacks(king_sq, occ) & enemy_rq;

    const check_mask: u64 = blk: {
        const n = @popCount(checkers);
        if (n == 0) break :blk ~@as(u64, 0);
        if (n >= 2) break :blk 0;
        const c_sq: u6 = @intCast(@ctz(checkers));
        break :blk checkers | attacks.between(king_sq, c_sq);
    };

    // Pinned-piece scan. For each enemy slider whose ray on an empty board
    // crosses the king, the piece is a pinner iff exactly one piece (and it
    // must be ours) sits strictly between the slider and the king.
    var pinned: u64 = 0;

    var bq_candidates = attacks.getBishopAttacks(king_sq, 0) & enemy_bq;
    while (bq_candidates != 0) {
        const sl_sq: u6 = @intCast(@ctz(bq_candidates));
        bq_candidates &= bq_candidates - 1;
        const between_occ = attacks.between(king_sq, sl_sq) & occ;
        if (@popCount(between_occ) == 1 and (between_occ & ours) != 0) {
            pinned |= between_occ;
        }
    }

    var rq_candidates = attacks.getRookAttacks(king_sq, 0) & enemy_rq;
    while (rq_candidates != 0) {
        const sl_sq: u6 = @intCast(@ctz(rq_candidates));
        rq_candidates &= rq_candidates - 1;
        const between_occ = attacks.between(king_sq, sl_sq) & occ;
        if (@popCount(between_occ) == 1 and (between_occ & ours) != 0) {
            pinned |= between_occ;
        }
    }

    return .{
        .king_sq = king_sq,
        .checkers = checkers,
        .check_mask = check_mask,
        .pinned = pinned,
    };
}
