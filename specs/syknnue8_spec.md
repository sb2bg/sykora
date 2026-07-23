# SYKNNUE8 Threat-Input Architecture and Experiment Spec

Status: T1024/T768 training and sectioned-format support implemented on main;
the runtime currently uses the required scalar reference threat path while
incremental/lazy SIMD optimisation remains follow-up work

Research date: 2026-07-14

## 1. Decision

`SYKNNUE8` should add full threat inputs to the proven v7 graph. The first v8
experiment must change only the input feature set:

```text
10 mirrored king buckets × 768 PSQ features, i16 weights
+ 60,720 shared full-threat features, i8 weights
  -> separate PSQ and threat accumulators, summed per SIMD lane
  -> H=1024 dual perspective
  -> pairwise product pooling within each perspective
  -> concat(us, them), 1024 u8 values total
  -> selected material head: 1024 -> 16
  -> dual activation: CReLU(16) || CSReLU(16)
  -> 32 -> 32 -> 1
```

This candidate is named **T1024** in this document. It preserves v7's:

- 10-bucket mirrored PSQ layout;
- H=1024 accumulator width;
- pairwise pooling;
- eight material heads;
- 16 -> 32 -> 1 nonlinear tail;
- quantisation and evaluation scales.

The performance sibling **T768** makes only one further change: H=1024 ->
H=768. T1024 establishes whether threat inputs improve evaluation quality;
T768 establishes whether giving up some width converts that improvement into
more clock-time Elo.

Do not add pawn-pair inputs, more king buckets, a wider accumulator, a new
tail, a skip connection, or search changes to the first v8 experiment. Those
are later ablations. A data-only fine-tune of the existing graph is `v7.1`,
not v8.

## 2. Why This Is the Next Architecture

The deployed v7 network is not obviously capacity-starved in its accumulator.
It has H=1024, which is wider than current Pawnocchio's H=768 and equal to the
current Stockfish feature-transformer width. The main difference is the input
vocabulary:

| Network | Real sparse inputs | H | Deployed parameters | File size |
| --- | ---: | ---: | ---: | ---: |
| Sykora v7 | 7,680 PSQ | 1,024 | 8,005,256 | 15,872,384 bytes |
| Pawnocchio | 12,288 PSQ + 59,808 threats + 4,560 pawn pairs | 768 | 58,979,720 | 68,444,224 bytes |
| Stockfish dev, 2026-07 | 22,528 PSQ + 60,720 threats | 1,024 | 86,193,032 | 67,391,455 bytes compressed |

These rows pin Pawnocchio commit `bd860905` with `pp_new2.nnue` and Stockfish
commit `9a8dd81d` with default `nn-0ee0657fb25e.nnue`. Their containers are not
directly comparable: Stockfish's on-disk net is compressed, while parameter
count, resident memory, and executed operations are separate quantities.

The parameter gap is therefore mostly a feature-vocabulary gap, not a hidden
width gap. This is consistent with:

- [Pawnocchio's current architecture](https://github.com/JonathanHallstrom/pawnocchio/blob/bd860905d38bcf4293d66845ad79ce21764abc40/src/nnue/arch.zig),
  which uses threats and pawn-pair features with H=768;
- [Pawnocchio's bundled PSQ/threat/pawn-pair result](https://github.com/JonathanHallstrom/pawnocchio/commit/8069d014),
  which reported a large gain but cannot isolate its components;
- [Stockfish's full-threat merge](https://github.com/official-stockfish/Stockfish/pull/6411),
  which passed STC, LTC, and very-long-TC testing, and the
  [2026-07 source snapshot](https://github.com/official-stockfish/Stockfish/blob/9a8dd81dd7f98cbf02f16c59b4377d174d6eb4b5/src/nnue/features/full_threats.h),
  plus its
  [feature enumerator](https://github.com/official-stockfish/Stockfish/blob/9a8dd81dd7f98cbf02f16c59b4377d174d6eb4b5/src/nnue/features/full_threats.cpp),
  which fix the current vocabulary at 60,720 inputs;
- [Stockfish's architecture history](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md),
  where the current main accumulator is H=1024 rather than a continually
  widening PSQ-only transformer.

The strongest independently supported next feature is full threats. Pawn
pairs are plausible, but Stockfish supplies a cleaner threat-only result and
there is not yet a clean upstream pawn-pair ablation strong enough to justify
bundling both features.

## 3. Goals and Non-Goals

### 3.1 Goals

V8 must:

1. test threat inputs without confounding the first result;
2. preserve exact integer inference and cross-language parity;
3. make threat updates lazy and incrementally maintainable;
4. store threat weights as i8 while accumulating into a proven-safe type;
5. support both T1024 and T768 in the same format and runtime;
6. distinguish evaluation quality from runtime cost with fixed-node and
   clock-time tests;
7. support warm-start training from the retained v7 float checkpoint;
8. retain v7 loading for regression matches.

### 3.2 Non-goals for the first candidate

The first candidate does not attempt to:

- reproduce Pawnocchio's entire network or data pipeline;
- copy GPL implementation code into the MIT-licensed Sykora repository;
- add pawn-pair features;
- change the 10-bucket layout;
- make H larger than 1024;
- change output buckets or the dense tail;
- tune search parameters simultaneously;
- prove that a larger file is slower or faster by itself.

The feature semantics may be implemented independently from the published
architecture descriptions and verified with Sykora-owned fixtures. Upstream
source is evidence for the design, not code to transplant.

## 4. Normative Network Graph

### 4.1 Dimensions

| Parameter | T1024 | T768 |
| --- | ---: | ---: |
| PSQ features per input bucket | 768 | 768 |
| Input buckets | 10 | 10 |
| Full-threat features | 60,720 | 60,720 |
| FT width `H` | 1,024 | 768 |
| Output buckets `O` | 8 | 8 |
| First dense width `D1` | 16 | 16 |
| Second dense width `D2` | 32 | 32 |
| PSQ quantisation `Q0` | 255 | 255 |
| Threat weight storage | i8 | i8 |
| Pool quantisation `QP` | 128 | 128 |
| Dense quantisation `Q` | 64 | 64 |
| Evaluation scale | 400 | 400 |

`H` must be even for pairwise pooling and divisible by every registered SIMD
tile width. The format may allow other even widths, but only T1024 and T768
are v8 release candidates until separately tested.

### 4.2 PSQ contribution

The PSQ path is the exact v7 feature transformer. For perspective `p`:

```text
psq_acc_p[h] = ft_bias[h]
             + sum(psq_weight[bucket(p), psq_feature(p, piece), h])
```

Training retains the existing shared 768 -> H factoriser plus bucket
residuals. Only merged `[10, 768, H]` i16 PSQ weights are exported.

### 4.3 Full-threat contribution

A full-threat feature represents one member of a compressed attack/defence
vocabulary, not every possible directed attack. The Sykora-owned
`full_threats_v1` mapping must implement these semantics:

1. rotate the board by 180 degrees for Black's perspective;
2. horizontally mirror it when the perspective king is on the opposite half
   of the board, using one frozen orientation rule;
3. express piece colours relative to the perspective as `ours` or `theirs`;
4. consider occupied targets reached by an attacker's occupancy-aware attack
   set, stopping a sliding ray at its first occupied square;
5. for pawns, consider diagonal attacks plus a one-square forward relation
   only when a pawn occupies that square; do not encode empty pushes or double
   pushes;
6. include both attacks on enemy pieces and defences of friendly pieces;
7. exclude kings as attackers and targets;
8. apply the retained attacker/target type table below;
9. deduplicate symmetric same-type relations by oriented square order for
   enemy pairs and for friendly non-pawn pairs. Keep directional friendly-pawn
   relations because a blocked pawn relation is not symmetric.

The retained type table is normative. Rows are attacker types, columns are
occupied target types:

| Attacker / Target | Pawn | Knight | Bishop | Rook | Queen | King |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pawn | keep | keep | drop | keep | drop | drop |
| Knight | keep | keep | keep | keep | keep | drop |
| Bishop | keep | keep | keep | keep | drop | drop |
| Rook | keep | keep | keep | keep | drop | drop |
| Queen | keep | keep | keep | keep | keep | drop |
| King | drop | drop | drop | drop | drop | drop |

This removes relations whose reverse supplies the same information, such as a
bishop attacking a queen when the queen-attacking-bishop relation is retained.
It also explains why 60,720 is smaller than an uncompressed Cartesian product.
V1 has no standalone Pawnocchio-style pawn-pair table; the local blocked-pawn
relation above is part of the full-threat vocabulary.

The dimension is independently checkable. Per attacker colour, the board has
132 pawn, 336 knight, 560 bishop, 896 rook, and 1,456 queen directed geometries.
The table retains 6, 10, 8, 8, and 10 target colour/type slots respectively:

```text
2 * (132*6 + 336*10 + 560*8 + 896*8 + 1,456*10) = 60,720
```

The later symmetric-relation filter creates unreachable slots inside this
dimension; it does not change the v1 tensor shape.

Pins and move legality do not suppress a feature. The input describes board
attacks, not legal moves. Castling rights, en-passant rights, side to move,
move counters, checks as a separate flag, and empty attacked squares are not
features.

The logical key is:

```text
(attacker_relative_colour,
 attacker_piece_type,
 oriented_attacker_square,
 target_relative_colour,
 target_piece_type,
 oriented_target_square)
```

`full_threats_v1` has a deterministic 60,720-slot ID space. Symmetric
deduplication intentionally leaves some slots unreachable; v1 does not compact
those holes because an arithmetic indexer is simpler and keeps the first
experiment close to the upstream evidence. Unreachable rows must export as
zero and must never be activated. Before trainer work starts, the implementation
must add the generated packing table and its content hash to this spec. The
mapping generator must:

- be deterministic and independent of iteration/hash-map order;
- reject structurally impossible pawn squares and piece relations;
- assign candidate IDs in documented arithmetic or lexicographic order;
- emit inverse/decode data and unreachable-slot markers for tests;
- prove a dimension of 60,720, prove every emitted ID is in `0..60_720`, and
  report the exact reachable-ID count;
- produce golden IDs for every colour, piece-type, orientation, and board
  edge class.

The frozen generated table is `utils/nnue/full_threats_v1.bin`. It contains
60,720 eight-byte decode records in ID order and has SHA-256:

```text
964591edbe856c9f90694dcbfabe42d58b011a469e3275a8aaa9e4249b21988a
```

There are 51,130 structurally reachable IDs. The remaining slots are the
intentional holes described above. Generation metadata and edge/type golden
records live in `utils/nnue/full_threats_v1_manifest.json`; regenerate both
files with `python utils/nnue/full_threats_v1.py --write`, or verify them with
`python utils/nnue/full_threats_v1.py --check`.

Training must not begin while the index mapping or orientation rule is still
mutable. A trainer/runtime mismatch would create a valid-looking but useless
network.

Threat weights are shared across PSQ king buckets:

```text
threat_acc_p[h] = sum(threat_weight[active_threat_id, h])
```

They are exported as `[60_720, H]` i8 values and sign-extended during SIMD
accumulation. Full threats have no virtual factoriser in the initial trainer;
the existing factoriser applies only to the bucketed PSQ path.

### 4.4 Combined accumulator and tail

The logical accumulator is:

```text
acc_p[h] = psq_acc_p[h] + threat_acc_p[h]
```

The runtime should not materialise a third full accumulator merely to compute
this sum. A resolved SIMD read adds the PSQ and threat lanes as they enter
pooling.

After that read, inference is byte-for-byte the v7 integer contract:

```text
pooled_p[i] = clamp_u8(
    (clamp(acc_p[i], 0, 255) * clamp(acc_p[i + H/2], 0, 255)) >> 9
)

pooled = pooled_us || pooled_them
z1 = affine_i8(pooled, selected_head.l1)        # H -> 16
a1 = CReLU(z1) || CSReLU(z1)                    # 16 -> 32
z2 = affine_i8(a1, selected_head.l2)             # 32 -> 32
a2 = SCReLU(z2)
raw = affine_i8(a2, selected_head.output)         # 32 -> 1
score = rescale(raw, 400)
```

The material-head selector remains:

```text
bucket = min((popcount(occupied) - 2) / 4, 7)
```

No search correction, handcrafted term, or root-only heuristic is part of
the v8 evaluation contract.

## 5. Parameter and File-Size Budget

For `B` PSQ buckets and width `H`, the deployed parameter count is:

```text
PSQ weights       = B * 768 * H
threat weights    = 60,720 * H
FT bias           = H
L1 weights        = 8 * H * 16
L1 bias           = 8 * 16
L2 weights        = 8 * 32 * 32
L2 bias           = 8 * 32
output weights    = 8 * 32
output bias       = 8
```

With i16 PSQ weights and i8 threat/dense weights:

| Candidate | Parameters | Raw tensor payload | Approx. MiB |
| --- | ---: | ---: | ---: |
| v7, B10/H1024, no threats | 8,005,256 | 15,871,776 bytes | 15.14 |
| T1024, B10/H1024 | 70,182,536 | 78,049,056 bytes | 74.43 |
| T768, B10/H768 | 52,639,112 | 58,539,296 bytes | 55.83 |
| Later B16/H768 threat-only | 56,178,056 | 65,617,184 bytes | 62.58 |

Container headers add less than a few KiB. T1024 therefore requires raising
the current 64 MiB loader guard to 128 MiB. T768 and the later B16/H768 shape
fit under 64 MiB, but the guard should be a format-wide safety limit rather
than a hidden architecture selector.

Parameter count and file size are not inference-operation counts. Only active
features are accumulated, and only one material head is evaluated.

## 6. Runtime State and Updates

### 6.1 Separate accumulator state

Each search ply needs logical access to:

- the existing PSQ accumulator for both perspectives;
- a threat accumulator for both perspectives;
- the orientation used by each threat accumulator;
- pending threat removals/additions or a refresh marker.

PSQ refresh caching remains unchanged. Threat refresh and update state must be
separate so a threat change cannot invalidate a reusable PSQ accumulator.

### 6.2 Reference feature enumerator

Correctness starts with a slow reference function:

```text
enumerateThreats(board, perspective) -> sorted unique [threat_id]
```

It recomputes occupancy-aware attacks from the complete board. It is required
for trainer samples, golden fixtures, refreshes, and differential tests even
after an optimised updater exists.

The enumerator must define behaviour for:

- friendly and enemy targets;
- all non-king attacker/target type combinations;
- pawn direction under both perspectives;
- horizontal mirroring;
- sliding blockers and x-rays;
- promotion and promotion capture;
- en passant, including the removed pawn changing slider rays;
- both rook and king movement during castling.

### 6.3 Incremental dirty-threat calculation

The optimised updater compares threat relations affected by a move. Dirty
relations include:

- relations whose attacker or target moved, appeared, or disappeared;
- pawn attacks from the old and new squares;
- slider relations changed because an occupied square was added or removed;
- relations along every rook/bishop/queen ray crossing a changed square;
- all consequences of both squares changed by en passant;
- all consequences of all four castling squares;
- the old piece type disappearing and the promoted type appearing.

The result is a deduplicated removal/addition list for each perspective. If a
fixed update buffer would overflow, mark the accumulator for refresh instead
of truncating it.

### 6.4 Lazy materialisation

Threat updates should be recorded when a child position is created but
applied only when that ply requests an evaluation. Materialisation walks from
the nearest resolved ancestor, applying pending deltas in order. A full
refresh is required when:

- no resolved ancestor exists;
- the perspective's horizontal orientation changes;
- an update buffer overflows;
- a debug/reference comparison detects a mismatch.

A null move does not change board attacks. It changes which already-maintained
perspective is read first; it must not rebuild either accumulator.

### 6.5 Bounds and arithmetic

The exporter and loader must prove accumulator safety using independently
bounded PSQ and threat contributions. Do not infer safety from v7's PSQ-only
bound. The reference accumulator uses i32. An i16 production accumulator is
allowed only if the exported maximum reachable sum, including bias, PSQ, and
the maximum active threat contribution, fits for every lane.

If an i16 proof cannot be made without unrealistic assumptions, retain i16
PSQ/threat storage but widen the resolved addition or the accumulator itself
to i32 and measure the cost. Silent saturation is forbidden.

## 7. Performance Model: What “Smaller H” Actually Buys

T768 is 25% narrower than both v7 and T1024. That reduces:

- bytes touched for each changed PSQ feature by 25%;
- bytes touched for each changed threat feature by 25% versus T1024;
- pairwise pooling lanes by 25%;
- first-dense dot-product work by 25%;
- accumulator stack and refresh-cache footprint by 25%.

It does **not** imply that T768 performs less total accumulator work than
PSQ-only v7. V7 updates a small number of piece-square features. Threat inputs
introduce additional changing relations; the
[Stockfish NNUE documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#i8-quantization-for-threat-feature-weights)
reports that a typical midgame can have roughly three to four times as many
changing threat features as changing piece features.

A useful first-order model is:

```text
v7 update work
  ~= 1024 * changed_psq_features * i16_cost

T768 update work
  ~= 768 * changed_psq_features * i16_cost
   + 768 * changed_threat_features * i8_expand_cost
```

The second expression is likely larger on many evaluated nodes. T768 is
expected to be faster than an otherwise identical T1024 threat net, but it is
not expected or required to have higher raw NPS than v7.

There are three distinct possible wins:

1. **Raw speed:** more nodes/second. This is possible but not the base
   expectation versus v7.
2. **Tree efficiency:** better evaluations and move ordering search fewer
   nodes for the same depth, offsetting lower NPS.
3. **Playing strength:** the engine chooses better moves at the same clock
   even when raw NPS is lower.

Only measurement can establish which occurred. Benchmarks must record:

- raw NPS and evaluations/second;
- average active and changed PSQ/threat feature counts;
- percentage of created plies that materialise their pending threats;
- threat refresh frequency and causes;
- accumulator, pooling, and dense-tail time separately;
- average completed depth and nodes to completed depth;
- fixed-node and clock-time Elo.

The runtime work order is correctness first, followed by i8 SIMD expansion,
lazy deltas, refresh caching, prefetching, and architecture-specific kernels.
Do not reject threat inputs based on a scalar reference path, and do not
promote them based on fixed-node Elo alone.

## 8. SYKNNUE8 Container Requirements

V8 should extend the sectioned v7 design rather than overload a v7 feature
set with undocumented extra bytes.

Required top-level changes:

- magic `SYKNNUE8` and format version 8;
- architecture ID `pairwise_mlp_threats`;
- feature-set ID `mirrored_psq_full_threats_v1`;
- explicit PSQ feature count, threat feature count, and threat scheme ID;
- explicit PSQ and threat element types and quantisation scales;
- separate sections for PSQ weights and threat weights;
- stored hash of the threat-index packing table;
- stored accumulator-bound evidence;
- a 128 MiB maximum network-file guard;
- all existing v7 dense shape, activation, selector, and section checks.

Required tensor sections:

| Section | Type | Shape |
| --- | --- | --- |
| FT bias | i16 | `[H]` |
| PSQ FT weights | i16 | `[B, 768, H]` |
| Threat FT weights | i8 | `[60_720, H]` |
| L1 bias | i32 | `[8, 16]` |
| L1 weights | i8 | `[8, H, 16]` |
| L2 bias | i32 | `[8, 32]` |
| L2 weights | i8 | `[8, 32, 32]` |
| Output bias | i32 | `[8]` |
| Output weights | i8 | `[8, 32]` |

The loader must reject:

- a missing or duplicate required section;
- overlapping, out-of-order, truncated, or trailing tensor data;
- a threat count or mapping hash different from the registered scheme;
- an unsupported tensor type or quantisation scale;
- dimensions whose multiplication overflows;
- an unproven accumulator bound;
- a file larger than the guard;
- a content-hash mismatch.

V7 parsing must remain unchanged and testable.

## 9. Training Program

### 9.1 Data-only control: v7.1

Before attributing a gain to v8, run a same-architecture data control:

```text
v7 float checkpoint
  -> broad external-data component
  -> current-main Sykora self-play at controlled nodes
  -> lower-learning-rate mixed fine-tune
  -> v7.1
```

Generate Sykora data from the current main search, including the search-speed
and time-management improvements. Prefer fixed nodes for labels/data
generation so hardware speed does not alter target quality. Keep WDL results
and source/game IDs. Do not train from scratch on only Sykora data.

V7.1 answers whether engine-matched data improves the existing graph. It also
provides the strongest warm start for v8.

### 9.2 Exact warm start

T1024 must start from the retained v7.1 full-precision checkpoint when
available:

- copy PSQ factoriser, PSQ residuals, bias, and every dense tensor;
- initialise all threat weights to zero;
- verify on a fixed validation batch that the float T1024 output equals v7.1
  before the first optimiser step;
- verify that the quantised zero-threat export is bit-exact with v7.1.

If only a quantised v7 net remains, dequantisation is a fallback and must be
recorded as a different experiment. It is not equivalent to resuming the
float checkpoint or optimiser state.

T768 cannot be an exact v7 warm start because its width differs. It should be
trained only after T1024 establishes that threats are useful. Any neuron
selection/distillation method used to initialise T768 is its own recorded
ablation.

### 9.3 Staged training

Recommended T1024 stages:

| Stage | Samples | Purpose |
| --- | ---: | --- |
| Z | one fixed batch | prove zero-threat equivalence |
| P | 100-200 superbatches | pipeline, loss, clipping, and corruption screen |
| B | up to 800 superbatches | mature on broad mixed external data |
| S | 200-400 superbatches | lower-LR fine-tune with Sykora-matched data |

At the current trainer settings, one superbatch is 100,007,936 sample visits;
800 superbatches are about 80.0 billion visits. These are sampled visits, not
necessarily unique positions.

An early pilot can reject a broken pipeline but cannot prove that an
architecture is weak. Sykora's earlier checkpoints changed rank late in
training. Mature the best validated threat candidate before a release SPRT.

### 9.4 Data composition and validation

Each record must retain source provenance. Use whole games or immutable
shards for validation, never randomly split positions from the same games
across train and validation.

Report the mixture by:

- source/teacher and self-play engine revision;
- standard chess versus FRC if FRC is included;
- material/output bucket;
- side to move;
- game phase and piece count;
- score/WDL band;
- tactical versus quiet status;
- active and changing threat-feature count.

Keep the current WDL blend as the control. A different blend, lambda schedule,
or target scaling is a training ablation and cannot be bundled with the first
architecture comparison.

Validate every saved checkpoint on:

- the immutable broad validation set;
- a Sykora-self-play validation set;
- per-output-bucket loss and calibration;
- float versus quantised error;
- clipping counts for every tensor family;
- maximum accumulator sums and active threat counts.

## 10. Correctness Gates

Before self-play, all of the following must pass:

1. mapping-generator determinism and the frozen table hash;
2. logical threat encode/decode round trips;
3. reference active-feature sets for hand-authored positions;
4. trainer feature IDs equal runtime reference feature IDs;
5. incremental threat sets equal full recomputation after every legal child
   in the existing parity corpus;
6. random legal make/unmake games, checking every ply and undo;
7. explicit castling, en passant, every promotion, and promotion capture;
8. mirror-boundary king moves for both perspectives;
9. update-buffer overflow falling back to refresh;
10. null moves leaving feature state unchanged;
11. scalar i32, optimised accumulator, Python/reference float, and exported
    integer inference agreement under their declared contracts;
12. v7 regression loading and evaluation.

Threat-set differential tests must compare the feature IDs themselves, not
only the final score. Score equality can hide cancelling update bugs.

## 11. Strength and Performance Gates

Use the same search revision and settings on both sides of every EvalFile
match. Do not mix SPSA-tuned search changes into a net comparison.

Required sequence:

1. **Zero-threat equivalence:** v8 container/runtime with zero threat weights
   is bit-exact with v7.1.
2. **Microbench:** feature enumeration, delta generation, materialisation,
   refresh, pooling, and tail timings.
3. **Fixed positions:** NPS, evaluations/second, node count, and depth on the
   repository benchmark set.
4. **Fixed-node paired match:** T1024 versus v7.1 to measure evaluation/tree
   quality without charging wall-clock update cost.
5. **Clock STC SPRT:** T1024 versus v7.1 using the current release test time
   control and paired openings.
6. **T768 only if needed:** train/test the narrower sibling when T1024 gains
   at fixed nodes but loses too much under the clock, or when profiling shows
   accumulator bandwidth dominates.
7. **Clock LTC confirmation:** required for either threat architecture because
   feature cost and evaluation value can change with time control.
8. **Release regression:** winning net plus unchanged search versus the
   released engine.

Interpretation:

| Result | Decision |
| --- | --- |
| Fixed-node loss | Do not optimise runtime yet; debug data/features/training |
| Fixed-node win, clock loss | Profile and try T768/lazy-update optimisation |
| T1024 clock win | Prefer T1024 unless T768 wins a direct confirmation |
| T768 clock win but lower NPS than v7 | Valid promotion; tree/eval quality paid for its cost |
| STC win, LTC loss | Do not promote without understanding the reversal |
| Loss improvement only | Continue checkpoint testing; loss is not Elo |

Promotion requires a pentanomial-aware SPRT and an LTC confirmation, not a
small fixed-game score.

## 12. Follow-Up Ablation Ladder

Only after a threat candidate wins:

| Order | Change | Reason |
| ---: | --- | --- |
| 1 | T1024 -> T768 | Trade width for update/tail speed |
| 2 | 10 -> 12 or 16 PSQ buckets at winning H | Add spatial capacity separately |
| 3 | add pawn-pair inputs | Measure value beyond full threats |
| 4 | L1-to-output skip | Test material/extreme-score representation |
| 5 | threat compression/index revision | Reduce memory only after a strength baseline |
| 6 | broader dense tail or H > 1024 | Lowest-priority capacity expansion |

Each row needs its own format ID or declared architecture flag, validation
run, fixed-node test, clock SPRT, and profile. Do not rename a data-only net or
an implementation-only optimisation as a new architecture generation.

## 13. Implementation Work Order

1. Freeze `full_threats_v1` semantics, generator, table, and hash.
2. Add the slow reference enumerator and fixtures.
3. Extend the trainer feature loader and cross-check its IDs.
4. Define and test the SYKNNUE8 container, exporter, and loader.
5. Implement separate reference threat accumulators with full refreshes.
6. Prove zero-threat v7.1 equivalence.
7. Implement incremental dirty-threat lists and make/unmake differential tests.
8. Add lazy materialisation and refresh fallback.
9. Add portable i8-to-i16 SIMD accumulation, then AVX2/AVX-512/ARM kernels.
10. Add per-stage profiling counters and benchmark reporting.
11. Run v7.1 data control.
12. Warm-start and train T1024.
13. Run fixed-node and clock gates.
14. Train T768 only when T1024 evidence or profiling justifies it.
15. Run final search-margin SPSA only after the release net is selected.

## 14. Acceptance Criteria

V8 is eligible to become the default only when:

- the threat index scheme and table hash are frozen and reproducible;
- trainer and runtime emit identical active IDs;
- incremental, refresh, make/unmake, and reference threat sets agree;
- zero-threat v8 is bit-exact with its v7.1 warm start;
- exported quantisation has no unexplained clipping or overflow;
- accumulator bounds are proven at load time;
- the candidate beats v7.1 at fixed nodes;
- it passes a clock-time STC SPRT;
- it is confirmed at LTC;
- its raw NPS, tree efficiency, depth, and update costs are recorded rather
  than inferred from file size;
- v7 remains loadable for regression tests;
- the final embedded file has complete architecture, data, trainer, mapping,
  checkpoint, exporter, and test provenance.

The intended strength hypothesis is that explicit attack/defence relations
improve evaluation and tree efficiency enough to repay their update cost. It
is not a promise that a 50-70 million parameter net will execute faster than
the 8 million parameter PSQ-only v7 network.
