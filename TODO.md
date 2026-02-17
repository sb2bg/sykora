High impact:

1. Check extension limit (bug fix) — Your MEMORY.md warns about this but  
   src/search.zig still has no ply cap on check extensions. This causes seldepth  
   explosion and phantom tactics. Add ply < 2 \* depth + 8 guard.
2. LMR formula upgrade — You're using a step-function capped at R=3. Modern engines
   use R = base + ln(depth) \* ln(moveIndex) / C with history-based modulation. This
   is likely 20-40 Elo.
3. Singular Extensions — When the TT move is clearly best (re-search at reduced
   depth with TT score - margin as beta), extend it. ~20-30 Elo.
4. Internal Iterative Reduction (IIR) — When no TT move exists, reduce depth by 1.
   Trivial to implement, ~10-15 Elo.

Medium impact: 5. Multi-bucket TT — Single entry per index loses a lot to collisions. 2-4 entries
per bucket is standard. 6. History-based LMR modulation — Moves with bad history should reduce more; good
history reduces less. Your LMR doesn't consult history at all. 7. TT prefetch — @prefetch the child TT entry after making a move to hide cache
latency. 8. Continuation history / capture history — You only track from-to history.
Continuation history (indexed by previous move's piece-to) improves move ordering
significantly.

Evaluation

High impact: 9. Texel tuning — All ~100+ eval weights are hand-tuned. Automated tuning against
labeled positions would be the single biggest HCE improvement (potentially 100-200
Elo). 10. NNUE incremental updates — Your NNUE does a full accumulator recompute every
call. This is extremely slow for hidden sizes of 256+. Without incremental
make/unmake updates, HCE is probably faster at blitz time controls. 11. Attack-count king safety — Current king safety is basic pawn shield + open
files. A non-linear danger table indexed by attacker count (weighted per piece
type) is the standard approach.

Medium impact: 12. Passed pawn refinements — Missing: rook behind passer (Tarrasch rule), king
proximity in endgame, blockade penalty, unstoppable passer detection. 13. Pawn hash table — Pawn structure is recomputed from scratch every eval call. A
small dedicated cache saves significant time since pawns rarely change.

Performance / Movegen

14. Add a mailbox array (piece_on[64]) — getPieceAt loops through 6 bitboards every
    time. This is called constantly. A parallel [64]Piece array makes it O(1).
15. Iterate piece bitboards in eval, not all 64 squares — evaluate() loops over all
    64 squares calling getPieceAt twice. Iterating set bits of each piece bitboard is
    much faster, especially in endgames.
16. Selection sort in legacy move ordering — orderMoves/orderCaptures do full
    O(n^2) sorts. The staged move picker already does incremental pick-best, but the
    legacy functions still exist and may be called.

Quick Wins (least effort, still useful)

- IIR: ~5 lines of code
- Check extension cap: 1 line
- TT prefetch: 1 line
- Mailbox array: maintain alongside bitboards in make/unmake
