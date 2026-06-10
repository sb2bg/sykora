# Search & Movegen Improvement Findings

## SYKNNUE6 regression investigation (SPRT dev=6e9aa0a3 vs base=9d97fa45, -178.4 ± 40.9)

What was verified locally (June 2026):

- **The committed `src/net.sknnue` is byte-identical** to re-running
  `repack_v3_to_v6.py` on `src/net.sknnue.v3.bak` — the conversion is correct.
- **Eval parity**: `nnuecheck` on the v3 net through the base binary vs the v6 net
  through the dev binary over `utils/nnue/parity.fens`: max |diff| = 1 cp,
  mean 0.59 cp (expected divTrunc → round-to-nearest difference per the spec).
- **The incremental accumulator path is untouched** by the migration
  (`updateAccumulators`/`initAccumulators` identical; only loader + output head changed).
- **The embedded net loads** in the dev binary (UseNNUE on/off produces clearly
  different searches), and **NPS is equal** (~660K dev vs ~648K base, same position).
- So the v6 engine is functionally equivalent to v3 in eval and speed.

What is NOT fine: **time forfeits**. At fast TC the engine flags frequently
(local 5+0.05 match: 18 forfeits in 67 games across both engines; 10+0.1: dev
forfeited 2/20). Two compounding bugs, both *pre-dating* v6 (symmetric in
dev/base — they add noise/losses for both sides):

1. `calculateTimeLimit` floors the budget at **100ms regardless of remaining
   clock** — once the clock is under ~150ms the engine plans to spend more than
   it has. Measured: unfixed binary answers in ~108ms even with 50ms on the clock.
2. **No mid-iteration time check** — time is only checked between ID iterations,
   so one blown iteration overruns the budget arbitrarily.

Both fixed on `exp/search-bugfixes`. With the fix, go-to-bestmove scales with the
clock (20ms at 50ms clock vs 108ms unfixed).

If the rig's -178 was not dominated by forfeits (check the rig PGNs for
"loses on time" / disconnections), the remaining suspects are environmental:
build mode on the worker, an `EvalFile` option pinned to an old v3 net
(dev rejects it with InvalidArgument — check how the rig handles setoption
errors), or worker CPU contention. The engine code itself diffs clean.

---

Audit of `src/search.zig`, `src/search/*`, `src/board/*`, `src/bitboard.zig` (June 2026).
The search is already mature (PVS, aspiration windows, log-table LMR, LMP, NMP with
verification, singular extensions + multicut, SEE pruning, razoring, RFP, continuation
history), so the largest remaining wins are in infrastructure (board representation,
TT, check detection), a few search gaps, and three outright bugs.

## Bugs (fixed on `exp/search-bugfixes`)

### 1. TT pollution after a stopped search
`alphaBeta` checks `stop_search` only at function entry. When the stop flag goes up
mid-tree, every child call returns 0 immediately, the parent's move loop keeps running
through all remaining moves with bogus 0 scores, and `storeAlphaBetaResult` stores that
garbage into the TT **at full depth**. Since `probe` doesn't check age, those entries can
produce wrong cutoffs in the *next* search. Same applies to `quiescence` stores.
Fix: refuse TT stores once the stop flag is set, and bail out of the move loop early.

### 2. NNUE accumulator stack overflow at extreme ply
The accumulator stack is 128 entries and `pushAccumulator` writes `stack[acc_ply + 1]`
unguarded. There is no hard ply cap anywhere — check extensions allow `ply < 2*depth + 8`,
so `go depth 64` on a forcing position plus a deep qsearch tail can exceed 128 plies and
write out of bounds.
Fix: hard ply cap (`MAX_SEARCH_PLY`, tied to the accumulator stack size) in both
`alphaBeta` and `quiescence` that returns static eval.

### 3. No mid-iteration time check + 100ms budget floor
Time was only checked between iterative-deepening iterations. The "next iteration won't
finish" heuristic helps, but an aspiration-fail storm or sudden tree blowup can overrun
`movetime` arbitrarily — nothing inside `alphaBeta` ever set the stop flag on time-up.
On top of that, `calculateTimeLimit` floored the budget at 100ms regardless of the
remaining clock, guaranteeing forfeits once the clock drops below ~150ms (see the
regression investigation above).
Fix: every 2048 nodes, compare elapsed time against the limit and set `stop_search`;
hard-cap the budget at `remaining - 30ms`.

## Search improvements (Elo)

- **Singular-extension loop has no early exit** (`search.zig`, `trySingularExtension`).
  It runs a reduced-depth search over *every* move at depth ≥ 8. Once one beater is
  found the extension is already dead, and multicut only needs two. Break as soon as
  `beaters >= 1` when multicut is impossible (`tt_bound != .lower_bound or
  tt_score < beta_adj`), and at 2 otherwise. Large node saving at exactly the expensive
  deep nodes.
- **Internal Iterative Reduction (IIR).** When there's no TT move at depth ≥ 4, reduce
  `search_depth` by 1. Two lines after the TT probe; typically ~5–15 Elo, very low risk.
- **Fail-soft quiescence.** `quiescence` returns `beta` on cutoff and clamps to `alpha`
  otherwise, which weakens the bounds stored in the TT. Return the real best score
  (fail-soft, as `alphaBeta` already does).
- **Capture history.** Captures are ordered purely by SEE + MVV-LVA
  (`move_picker.zig:scoreCaptures`). A `[piece][to][captured_type]` history table updated
  on capture cutoffs is the standard next step, and also gives LMR/pruning a signal for
  captures (currently never reduced at all).
- **LMP is shallow and hard-coded** (`lmpQuietMoveLimit`, depth ≤ 3 only). The usual
  formula `(3 + depth*depth) / (improving ? 1 : 2)` applied up to depth ~8 prunes
  meaningfully more.
- **Use the TT score to refine `static_eval`** for RFP/NMP/futility decisions when the
  bound direction allows it — `tt_score`/`tt_bound` are already in hand from
  `probeTransposition`.

## Speed improvements (NPS)

- **Mailbox (`[64]` piece-on-square array).** The single biggest speed lever.
  `BitBoard.getPieceAt` loops over 6 kind bitboards and is called constantly: in the
  search move loop (capture/EP detection, moving piece, continuation keys), MovePicker
  scoring, SEE setup, and make/unmake. Worse, `clearSquare` blindly clears all 8
  bitboards and `unmakeMoveUnchecked` uses it, while make uses direct masks. A mailbox
  maintained in make/unmake turns all of these into single array reads.
- **Cheap `moveGivesCheck`.** `SearchEngine.moveGivesCheck` does a full make/unmake —
  including zobrist, EP-file, and castle-rights work — just to ask whether a move gives
  check, and it sits in three pruning hot paths (main SEE prune, futility, qsearch SEE
  prune). A direct test ("does the piece attack the enemy king from `to`, or does moving
  off `from` discover a slider?") needs only attack lookups already in `attacks.zig`.
- **TT: drop the mutexes, shrink the entries.** Every probe and store takes a striped
  mutex (`tt.zig`) — pure overhead even uncontended, and the shared-TT Lazy SMP helpers
  contend on it. Standard fix: lockless validation (store `key ^ data`, or atomic
  16-byte entries). Separately, `TTEntry` is ~24 bytes padded, so a 4-entry bucket is
  96 bytes (1.5 cache lines); compressing to 16 bytes (u32 key part, i16 score, u16 move,
  u8 depth, u8 age+bound) makes a bucket exactly one 64-byte cache line. `@prefetch` on
  the child hash right after make is nearly free on top.
- **Compute the legal context once per node; vectorize the pawn attack map.**
  `MovePicker` calls `generateLegalCaptures` and later `generateLegalQuietMoves`, and
  each recomputes `computeLegalContext` *and* `enemyAttackMapXKing`. Hoist the context
  (and enemy attack map) so they're computed once and shared by both stages. Inside
  `enemyAttackMapXKing` (`legal_movegen.zig`), replace the per-pawn loop with two setwise
  shifts: `((pawns & NOT_A) << 7) | ((pawns & NOT_H) << 9)` for white, mirrored for black.
- **SEE is computed twice per capture in qsearch** — once in `orderCaptures` and again in
  the SEE-pruning check — and the recursive `staticExchangeRec` rebuilds the full
  attacker set at every level. Cache the ordering score for reuse, and/or switch to the
  standard iterative swap-list SEE which discovers x-ray attackers incrementally.
- **Eval cache is small** (16384 entries) and only active under NNUE. Bumping to 2^17+
  is a trivial change.

## Suggested order

1. The three bugs (done on `exp/search-bugfixes`).
2. Speed batch: mailbox + cheap `givesCheck` + TT rework — these compound, likely
   20–40% NPS together.
3. Cheap Elo batch: singular early-exit, IIR, fail-soft qsearch.
4. Then capture history / LMP extension / TT-eval refinement, one at a time.

All Elo-related changes should go through `utils/benchmark.py` or self-play one change
at a time — pruning interactions are notoriously non-additive.
