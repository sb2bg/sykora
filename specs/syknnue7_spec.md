# SYKNNUE7 Research and Design Spec

Status: trainer, sectioned exporter, validated Zig loader, integer inference,
and cross-language parity implemented; full CUDA training run pending
Research date: 2026-07-09

## 1. Decision

`SYKNNUE7` should be a sectioned, self-describing container that can carry two
explicit architecture profiles:

1. `legacy_linear`: the exact v3 architecture, for pipeline reproduction and
   regression testing.
2. `pairwise_mlp`: the strength target, using a factorised king-bucket feature
   transformer, pairwise product pooling, eight material heads, and a small
   nonlinear tail.

The first strength candidate should be PSQ-only:

```text
10 mirrored king buckets × 768 PSQ features
  -> factorised sparse FT, H=1024, dual perspective
  -> pairwise product pooling within each perspective
  -> concat(us, them), 1024 u8 values total
  -> selected material head: 1024 -> 16
  -> dual activation: CReLU(16) || CSReLU(16)
  -> 32 -> 32 -> 1
```

Threat and pawn-pair inputs should be a later profile/feature-set extension,
not part of the first v7 training run. They are promising, but they change
feature generation, accumulator updates, memory footprint, and NPS all at
once. The first objective is to establish a trustworthy staircase from v3.

The binary format itself does not produce Elo. The likely strength gains come
from the architecture, data, training duration, and testing discipline. The
new container exists to make those experiments explicit and prevent another
spec/trainer/runtime divergence.

## 2. What the Repository Actually Contains

### 2.1 Current net

`src/net.sknnue` is a `SYKNNUE6` container containing the proven v3 model:

```text
H=512
activation=SCReLU
input buckets=10, v3_10 mirrored layout
output buckets=1, single-head scheme
Q0=255, Q=64, scale=400
```

Its size and payload match `src/net.sknnue.v3.bak` after the documented v3 to
v6 repack. It is not a separately trained v6 network. This is good as a known
baseline, but naming it “v6” can hide that no v6 architecture has yet been
validated end to end.

### 2.2 v3 matured much later than the existing spec says

The promoted v3 checkpoint is the 600-superbatch checkpoint under
`nnue/models/bullet/v3_512/checkpoints/run_20260323T063759Z-600`, not a
320-superbatch model. The local match history is noisy but important:

| Match | Games | Candidate result |
| --- | ---: | ---: |
| v2 vs v3 checkpoint 400 | 80 | v3 about -75 Elo, p=0.057 |
| v2 vs v3 checkpoint 540 | 80 | v3 about +53 Elo, p=0.180 |
| v3 checkpoint 540 vs 600 | 80 | checkpoint 600 about +80 Elo, p=0.044 |

These samples are too small to treat the Elo values as precise. They do show
that checkpoint maturity can reverse the conclusion of an architecture test.
Future runs must not reject a wider model merely because an early checkpoint
loses.

There is also an arithmetic error in the v6 spec. With batch size 16,384 and
6,104 batches per superbatch, one superbatch is about 100 million sampled
positions. Therefore 640 superbatches are about 64.0 billion position samples,
not about 670 million. This is sampled throughput, not necessarily unique
positions, because datasets can cycle.

### 2.3 The v4/v5 postmortem is not proven

Factorisation is a strong and sensible requirement, but the repository does
not contain a clean ablation proving that its removal was the single cause of
the v4/v5 failures. Those generations simultaneously changed several of:

- input bucket count and layout;
- FT width;
- head depth and activation;
- product pooling;
- output buckets;
- WDL blend;
- training data and duration;
- factorisation.

The correct conclusion is: **the failed runs are confounded**. Restore the
factoriser because it improves sample sharing between buckets, but do not
declare other techniques retired. Stronger engines successfully use 12–16
input buckets, product pooling, and nonlinear heads.

## 3. Primary-Source Research

Dimensions below were read from current engine source on the research date.
Repository ratings are deliberately not used to infer the Elo value of one
architectural component; only controlled upstream tests are useful for that.

### 3.1 Pawnocchio

[Pawnocchio's current architecture](https://github.com/JonathanHallstrom/pawnocchio/blob/bd860905d38bcf4293d66845ad79ce21764abc40/src/nnue/arch.zig)
uses 16 mirrored king buckets, a 768-wide accumulator, 8 material heads,
pairwise pooling, a 16 -> 32 tail, threat features, and pawn-pair features.
Its output implementation is fully integer and uses sparse u8 x i8 dot
products. This is particularly relevant to a portable Zig implementation.

Useful upstream experiments include:

- [the multilayer head](https://github.com/JonathanHallstrom/pawnocchio/commit/51f830345),
  reported at roughly +10 Elo in its test;
- [dual activation](https://github.com/JonathanHallstrom/pawnocchio/commit/c91fce7),
  reported positive at both STC and LTC;
- [the large PSQ/threat/pawn-pair change](https://github.com/JonathanHallstrom/pawnocchio/commit/8069d014),
  reported around +27 to +30 Elo, but too bundled to attribute to one feature;
- [fine-tuning on new data](https://github.com/JonathanHallstrom/pawnocchio/commit/52a23e3),
  whose larger LTC result is a reminder that data and time control interact.

Pawnocchio also states that its nets use self-play data from zero knowledge
plus games from the Vine MCTS engine. That supports testing engine-matched
fine-tuning rather than relying on only one external distribution.

### 3.2 PlentyChess

[PlentyChess's current NNUE](https://github.com/Yoshie2000/PlentyChess/blob/04e07a98ee6ac104c30e7374450c94b96d94ef4d/src/nnue.h)
uses 12 mirrored king buckets, H=1024, 8 material heads, pairwise pooling,
a 16 -> 32 tail, dual activation, threat inputs, pawn-pair inputs, and a skip
from the first dense activation to the output.

[Threat-input PR #400](https://github.com/Yoshie2000/PlentyChess/pull/400) is
the most useful cautionary result. The new net was very strong at fixed nodes,
slightly negative at STC, and only modestly positive at LTC. The lesson is not
that threats are weak; it is that feature-update cost can consume the gained
evaluation quality. Sykora must test both nodes and clock time.

PlentyChess currently describes its network as trained on more than 15 billion
self-generated standard and Fischer-random positions, with some
self-distillation. That is strong evidence for adding a Sykora-generated data
phase once the basic pipeline is reliable.

### 3.3 Alexandria

[Alexandria's current NNUE](https://github.com/PGG106/Alexandria/blob/2ebf11435fb0d458795f0a8dca58f651595995f9/src/nnue.h)
uses 16 mirrored buckets, H=1536, factorised PSQ inputs, pairwise pooling,
8 material heads, dual activation, and a 16 -> 32 tail. Its history is useful
because several ideas were tested separately:

- [8 output buckets](https://github.com/PGG106/Alexandria/commit/96346ab)
  passed its reported STC and LTC tests;
- [horizontal mirroring](https://github.com/PGG106/Alexandria/commit/cba722a)
  passed both reported tests;
- [pairwise pooling plus the dense tail](https://github.com/PGG106/Alexandria/commit/39a847e)
  lost at STC but won at LTC;
- [16 input buckets](https://github.com/PGG106/Alexandria/commit/073910a)
  similarly lost at STC and won strongly at LTC.

Again, NPS and time control decide whether additional knowledge is useful.
Alexandria trains on Leela Chess Zero data, showing that there is no single
mandatory data source; the distribution must be validated against the target
engine.

### 3.4 Stockfish

The official [Stockfish NNUE architecture history](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
documents the same broad trajectory: HalfKAv2-style king-bucket features,
product pooling, shallow dense layers, output buckets, and later full threat
features. The documentation also explains why the large sparse first layer
and a small dense tail are a good CPU tradeoff.

[Stockfish PR #6411](https://github.com/official-stockfish/Stockfish/pull/6411)
is primary evidence that efficiently updated threat inputs can pass STC, LTC,
and very-long-TC testing. It is a destination for v7, not proof that Sykora's
first v7 run should include them.

### 3.5 Research conclusion

The repeated pattern across stronger engines is:

```text
factorised and mirrored king-bucket PSQ features
  + a wide but not enormous accumulator
  + pairwise pooling to expose feature interactions
  + material-dependent heads
  + a very small nonlinear tail
  + aggressive SIMD/sparsity optimisation
```

Threat inputs, pawn-pair inputs, more king buckets, and self-generated data are
credible follow-ups. None should be bundled into the first proof-of-pipeline
run.

## 4. Architecture Profiles

### 4.1 Profile 0: `legacy_linear`

This profile exists for reproducibility, converters, and A/B isolation. Its
inference contract is the existing v3/v6 contract:

```text
factorised 10-bucket PSQ FT, H=512
  -> SCReLU(us) || SCReLU(them)
  -> one linear output
```

The exported FT weights contain the merged factoriser. The runtime never
loads a separate factoriser.

### 4.2 Profile 1: `pairwise_mlp`

Normative first strength target:

| Parameter | Value |
| --- | ---: |
| PSQ features per bucket | 768 |
| Input buckets | 10, v3_10 mirrored layout |
| FT width `H` | 1024 |
| Output buckets `O` | 8 |
| First dense width `D1` | 16 |
| Second dense width `D2` | 32 |
| FT quantisation `Q0` | 255 |
| Pooled activation scale `QP` | 128 |
| Dense activation/weight scale `Q` | 64 |
| Evaluation scale | 400 |

The 10-bucket layout is retained for the initial target because it is proven
inside Sykora. A 12- or 16-bucket layout remains a valid later ablation after
the factorised v7 core wins.

#### Sparse feature transformer

For perspective `p`:

```text
acc_p[h] = ft_bias[h] + sum(ft_weight[feature(p, piece), h])
```

The runtime accumulator is `i16` if a static/export-time bound proves all
reachable sums fit; otherwise it must use `i32`. The reference path always
uses `i32`. An optimised `i16` path must be checked against it after every ply
in differential tests.

Training is factorised:

```text
effective_weight[bucket, base_feature, h]
  = shared_factoriser[base_feature, h]
  + bucket_residual[bucket, base_feature, h]
```

Only the merged effective weights are exported.

#### Pairwise product pooling

Split each H-wide accumulator into two H/2 halves. For `i in 0..H/2`:

```text
x = clamp(acc[i],       0, 255)
y = clamp(acc[i + H/2], 0, 255)
pooled[i] = clamp_u8((x * y) >> 9)  # 0..127, QP=128 domain
```

Calculate this independently for `us` and `them`, then concatenate:

```text
pooled_input = pooled_us || pooled_them  # H=1024 bytes total
```

Product pooling adds an inexpensive learned interaction between pairs of FT
channels while returning the dense input to H elements instead of 2H.

#### Selected material head

The material bucket is selected before dense inference:

```text
non_king_count = popcount(occupied) - 2
divisor = ceil(32 / O)
bucket = min(non_king_count / divisor, O - 1)
```

For the required `O=8`, `divisor=4`. Storing the selector ID in the file is
important; a generic implicit formula has already diverged between engines.

Each output bucket owns a complete small head:

```text
z1 = affine_i8(pooled_input, l1_weight[b], l1_bias[b])  # 1024 -> 16
a1 = CReLU(z1) || CSReLU(z1)                            # 16 -> 32
z2 = affine_i8(a1, l2_weight[b], l2_bias[b])             # 32 -> 32
a2 = SCReLU(z2)                                          # 32
raw = affine_i8(a2, out_weight[b], out_bias[b])           # 32 -> 1
score = rescale(raw, 400)
```

An L1-to-output skip, as used by current PlentyChess, is an optional later
architecture flag and must be its own ablation. It is not enabled in the
initial target.

### 4.3 Exact integer dense contract

All signed divisions use round-to-nearest with ties away from zero.

First affine:

```text
s1[j] = l1_bias[b,j] + sum(pooled_input[i] * l1_weight[b,i,j])
r1[j] = round(s1[j] * Q / (QP * Q)) = round(s1[j] / QP)

linear[j]  = clamp(r1[j], 0, Q)
squared[j] = clamp(round(r1[j] * r1[j] / Q), 0, Q)
a1 = linear || squared
```

The squared branch intentionally squares the signed preactivation before
clipping. Negative preactivations can therefore contribute to the squared
channel, matching the dual-activation pattern in the researched engines.

Second affine and activation:

```text
s2[j] = l2_bias[b,j] + sum(a1[i] * l2_weight[b,i,j])
r2[j] = round(s2[j] / Q)
c2[j] = clamp(r2[j], 0, Q)
a2[j] = round(c2[j] * c2[j] / Q)
```

Output:

```text
s3 = out_bias[b] + sum(a2[i] * out_weight[b,i])
score = round(s3 * SCALE / (Q * Q))
```

Tensor scales are therefore:

| Tensor | Type | Scale |
| --- | --- | ---: |
| `ft_bias`, `ft_weight` | `i16` | 255 |
| pooled activations | `u8` | 128 |
| `l1_weight`, `l2_weight`, `out_weight` | `i8` | 64 |
| `l1_bias` | `i32` | 128 × 64 |
| `l2_bias`, `out_bias` | `i32` | 64 × 64 |

The exporter must reject values that require clipping unless an explicit
`--allow-clipping` diagnostic option is supplied. Silent saturation can make
a good float checkpoint weak after export.

All affine sums use at least `i32`; output rescaling uses `i64`. The loader
must validate every dimension and byte count before allocation.

## 5. SYKNNUE7 Container

### 5.1 Goals

The container must:

- distinguish architecture, feature set, activation, pooling, and material
  selector explicitly;
- declare tensor names, dimensions, types, offsets, and sizes;
- preserve optional training provenance without affecting inference;
- be exactly hashable and corruption-detecting;
- allow a reference loader to reject a trainer/export mismatch before play;
- remain little-endian and simple to parse in Zig, Rust, and Python.

It is not a general neural-network graph format. Only registered Sykora
architecture profiles are executable.

### 5.2 Fixed header

All multibyte integers are little-endian. The fixed header is 160 bytes:

```text
u8[8]   magic                 = "SYKNNUE7"
u16     version               = 7
u16     header_bytes          = 160
u16     section_count
u16     section_entry_bytes   = 48
u32     flags
u16     architecture_id       # 0=legacy_linear, 1=pairwise_mlp
u16     feature_set_id        # 1=mirrored_psq, future 2=psq_threats_pairs
u16     input_bucket_count
u16     output_bucket_count
u16     ft_hidden_size
u16     dense1_size
u16     dense2_size
u8      ft_activation_id
u8      pooling_id
u8      dense1_activation_id
u8      dense2_activation_id
u8      output_selector_id
u8[3]   reserved0             # zero
u16     ft_quant              # Q0
u16     pool_quant            # QP
u16     dense_quant           # Q
u16     score_scale
u8[64]  input_bucket_layout
u8[32]  content_sha256        # whole file with this field zeroed
u8[14]  reserved1             # zero
```

IDs are closed enums for version 7. Unknown architecture or required feature
IDs are rejected rather than guessed.

The registered `pairwise_mlp` profile uses these IDs:

```text
architecture_id=1, feature_set_id=1
ft_activation_id=0             # CReLU before pairwise pooling
pooling_id=1                   # pairwise product
dense1_activation_id=1         # CReLU || CSReLU
dense2_activation_id=1         # SCReLU
output_selector_id=1           # material popcount, ceil(32/O)
```

### 5.3 Section table

The table immediately follows the header. Each 48-byte entry is:

```text
u16     section_id
u8      element_type          # 1=i8, 2=u8, 3=i16, 4=i32, 5=utf8
u8      rank                  # 0..4
u32     flags                 # bit 0=required; all unknown bits zero
u32[4]  dimensions            # unused dimensions are 1
u64     file_offset
u64     byte_length
u32     crc32
u32     reserved              # zero
```

Sections begin on 64-byte boundaries, may not overlap, must lie inside the
file, and must have exactly `product(dimensions) * sizeof(type)` bytes except
for UTF-8 metadata. Gaps and alignment padding must be zero so the content
hash is deterministic.

Required section IDs for `pairwise_mlp`:

| ID | Name | Type | Dimensions |
| ---: | --- | --- | --- |
| 1 | `ft_bias` | i16 | `[H]` |
| 2 | `ft_weight` | i16 | `[768*B, H]` |
| 10 | `l1_bias` | i32 | `[O, D1]` |
| 11 | `l1_weight` | i8 | `[O, H, D1]` |
| 12 | `l2_bias` | i32 | `[O, D2]` |
| 13 | `l2_weight` | i8 | `[O, 2*D1, D2]` |
| 14 | `out_bias` | i32 | `[O]` |
| 15 | `out_weight` | i8 | `[O, D2]` |

Section 100 is reserved for future embedded UTF-8 JSON provenance. The initial
registered loader intentionally accepts exactly the eight inference sections;
complete provenance is stored beside the net in `run_meta.json`, including:

- Sykora commit and dirty diff hash;
- Bullet commit and dirty diff hash;
- trainer source hash;
- architecture and hyperparameters;
- RNG seed;
- every data shard's path-independent name, byte size, and SHA-256;
- split manifest hash;
- start/end checkpoint and sample counts;
- exporter version and clipping counts;
- GPU/backend and precision mode.

The runtime can ignore optional metadata after verifying its bounds and hash.

### 5.4 Target size

For B=10 and H=1024:

```text
FT weights = 768 * 10 * 1024 * 2 = 15,728,640 bytes (15.0 MiB)
FT bias    = 2,048 bytes
dense tail = about 138 KiB including all 8 heads
```

The complete PSQ-only target is about 15.2 MiB and fits the current 64 MiB
network guard comfortably.

## 6. Threat/Pair Extension, Not Initial Target

The later `psq_threats_pairs` feature set should use separate efficiently
updated accumulator contributions:

```text
PSQ weights:        i16, king-bucketed
threat weights:     i8, shared across king buckets
pawn-pair weights:  i8, shared across king buckets
combined acc:       i16 or i32 after proven bounds
```

Using the current Pawnocchio/Plenty-style counts, threat plus pawn-pair inputs
add `(59,808 + 4,560) * H` bytes at i8. At H=1024 this is 65,912,832 bytes
before the 15 MiB PSQ transformer, for about 77.9 MiB total. That exceeds
Sykora's current 64 MiB guard.

Practical extension choices are:

- H=768, B=10: about 58.4 MiB before the dense tail, fitting narrowly;
- H=640, B=12: about 50.5 MiB;
- raise the cap to 128 MiB and accept the cache/memory cost;
- investigate structured compression only after an uncompressed strength
  result exists.

Threat features require new incremental-differential tests and NPS gates.
Do not promote a threat net on fixed-node Elo alone.

## 7. Training Program

### 7.1 Non-negotiable controls

Every experiment must have:

- the factorised FT unless factorisation itself is the named ablation;
- one immutable train/validation split made by game/source, not individual
  positions;
- a fixed validation set that is never fed to the optimiser;
- per-output-bucket validation loss and sample counts;
- float-checkpoint vs quantised-net error histograms;
- exact provenance from section 5.3;
- the same search binary and settings on both sides of an EvalFile match;
- paired openings and a pentanomial-aware SPRT for promotion.

The current Bullet setup uses `test_set: None`; that must change before an
expensive v7 run. Training loss alone cannot select the best checkpoint or
detect overfitting.

### 7.2 Controlled ladder

Do not launch directly at the final architecture. Train/gate in this order:

| Stage | Change from previous | Purpose |
| --- | --- | --- |
| R | H512, O1, legacy linear, 600+ SB | Reproduce v3 pipeline and strength |
| A | O1 -> O8 only | Isolate material heads |
| B | H512 -> H768, still linear | Isolate width |
| C | H768 -> H1024, still linear | Find capacity/NPS tradeoff |
| D | product pool, linear head | Isolate pairwise pooling |
| E | replace linear head with 16 -> 32 -> 1 | Establish v7 core target |
| F | fine-tune on Sykora self-play mixture | Test source/engine alignment |
| G | 10 -> 12 or 16 input buckets | Test spatial capacity after factorisation |
| H | threats/pawn pairs at a size that fits | Final high-risk extension |

Stage R is a pipeline test, not expected innovation. If it cannot reach
neutral against the promoted v3 net, stop and debug export, data, duration,
quantisation, or runtime parity.

Stages D and E should be compared at both fixed nodes and clock time. If D
wins by nodes but loses by time, optimise the inference path before rejecting
the architecture.

### 7.3 Duration and checkpoint selection

Use the proven 600-superbatch v3 result as the initial duration anchor, not
the obsolete 320-superbatch statement. Wider nets may need longer. Do not
assume a fixed final checkpoint is best:

1. evaluate the held-out set at every saved checkpoint;
2. select checkpoints by validation loss/calibration, including per-bucket
   health;
3. quantise those checkpoints and run a short corruption/smoke match;
4. send the best validated candidates, not merely the newest three, to SPRT.

The existing checkpoint gate chooses the most recent checkpoints. That can
miss a better earlier model and should be changed before v7 training.

### 7.4 Data plan

Keep the existing high-quality external data for pretraining, but add a later
Sykora-matched phase:

```text
Phase 1: broad high-quality external data
Phase 2: mixture with Sykora self-play at controlled nodes
Phase 3: lower-LR fine-tune on the mixture or Sykora-heavy subset
```

Generate games with randomized, adjudication-safe openings and retain WDL
outcomes. Teacher scores may be included, but the WDL blend must be ablated;
`0.75` is a v3 baseline, not a universal optimum for every data source.

Report data balance by material bucket, side to move, phase, score band,
castling rights, and tactical/quiet status. Do not silently call the filter
“quiet” unless captures, promotions, and checks are actually excluded.

### 7.5 Match gates

The existing 80-game, 120 ms, `p <= 0.25` gate is a smoke test only. It is not
a promotion test.

Required promotion sequence:

1. format/hash/parity tests;
2. 80–200 game smoke test for catastrophic regressions;
3. fixed-node paired test to measure evaluation quality;
4. fast-TC pentanomial SPRT with a diverse opening book;
5. LTC confirmation for candidates whose architecture costs NPS;
6. only then replace the embedded default net.

Use a standard match runner such as fastchess/OpenBench statistics rather
than the current score-as-Bernoulli approximation. Record W/D/L pentanomial
pair counts and confidence bounds.

## 8. Runtime Plan

### 8.1 Reference first

The implementation retains a scalar `i32/i64` fallback and uses vector lanes
for pairwise pooling and the registered 16 -> 32 selected dense head. The
first scalar build reduced local search NPS by about 69%; vectorising the head
reduced the measured penalty to roughly 25–28% on the six-position benchmark.
The remaining optimisation options are:

- `i16` FT accumulators only after exported-bound proof;
- native-width SIMD rather than a fixed 8-lane vector;
- packed u8 product outputs;
- i8 first-dense weights;
- `dpbusd`/equivalent where available, portable fallback otherwise;
- sparse nonzero-block discovery for the first dense layer;
- optional Finny-table-style refresh caching after correctness tests.

The output tail is tiny. The performance-critical work is accumulator updates,
pool/pack, and the H -> 16 u8 x i8 multiply.

### 8.2 Differential tests

The parity command now compares Python integer inference with Sykora's full
recomputation on 77 FENs. For every legal child of those positions it also
compares the incrementally updated accumulator and evaluation with a fresh
full recomputation. The suite explicitly includes castling, en passant,
promotion, and promotion capture.

The invariant is:

```text
incremental optimised accumulator/eval
  == full scalar accumulator/eval
  == Python/reference evaluator
```

Coverage must include:

- every input and output bucket;
- both horizontal mirror states and both sides to move;
- king moves within a bucket and across bucket/mirror boundaries;
- quiet moves and captures;
- en passant;
- every promotion and promotion capture;
- both sides of castling;
- null moves if the search updates NNUE state on them;
- random legal games thousands of plies deep in aggregate;
- undo back to the root, checking at each unmake.

This test suite is a prerequisite for threat features, where changing attacks
can touch many features after one board move.

## 9. Audit Findings and Improvements

### P0: fix before training v7

1. **Historical raw checkpoints omitted the factoriser.** Bullet applies
   `SavedFormat` transforms to `quantised.bin`, but its `raw.bin` writer reads
   only the listed tensor IDs. The old runner listed `l0w` with an `l0f`
   transform but did not list `l0f` itself, leaving raw checkpoints with bucket
   residuals that could not be merged. The updated runner saves `l0f`
   explicitly, and the converter can recover old checkpoints from
   `optimiser_state/weights.bin`.
2. **The old launcher skipped the documented ablation ladder.** The updated
   `launch_training.ps1` now exposes named `repro`, `heads`, `wide768`,
   `wide1024`, `pairwise`, and `v7` stages, with `repro` as the default.
3. **Bullet has no implemented held-out validation pass.** `bullet_runner`
   necessarily configures `test_set: None`, but the updated pipeline reserves
   whole SF shards, builds a fixed validation sample, and ranks checkpoints in
   an independent float evaluator.
4. **Promotion gate is underpowered.** Eighty games with `p <= 0.25`, a small
   repeated opening set, and non-pentanomial statistics can easily promote or
   reject noise.
5. **Resolved: incremental parity was missing.** The 77-position suite covers
   all eight v7 output buckets, both mirrors, both sides to move, and explicit
   special moves. Every legal child is checked incremental-versus-full before
   the root evaluation is compared with Python.

### P1: fix for reproducibility and trustworthy results

6. **Bootstrap alone does not enforce the Bullet pin for an existing
   checkout.** The updated launcher verifies HEAD before training, snapshots a
   dirty tracked diff, and optionally requires a clean checkout.
7. **Old run metadata was incomplete.** Updated runs record Sykora/Bullet
   SHA/status/diff hashes, dataset fingerprints and roles, sample counts,
   architecture, commands, and the upstream RNG limitation.
8. **The old checkpoint gate selected newest, not best validated.** It now
   consumes the held-out MSE ranking and converts/tests only those candidates.
9. **Historical “quiet” descriptions are misleading.** The current predicate
   checks ply, check status, and score magnitude, but does not itself reject
   captures, promotions, or checking moves. If the source dataset is already
   quiet, name that invariant explicitly and verify it.
10. **The v6 postmortem overstates causality.** “No factoriser” is plausible and
   worth correcting, but no clean run proves it was the single dominant cause.
   This can prematurely discard techniques that are successful upstream.

### P2: tooling and format cleanup

11. **Training-volume math in the v6 spec is wrong by roughly two orders of
    magnitude.** Use actual sample counts from the trainer rather than hand
    estimates.
12. **Compiler version is documented but not enforced.** `zig build test`
    passes under the currently selected Zig 0.16.0, while building the engine
    fails because `GeneralPurposeAllocator` moved. The repository declares
    Zig 0.15.2, and a 0.15.2 build succeeds. Set `minimum_zig_version`, add a
    preflight version check, or use the pinned compiler in scripts.
13. **The v6 output-bucket formula is needlessly restricted.** Runtime accepts
    only bucket counts dividing 32, while Bullet's general material selector
    uses ceiling division. O=1 and O=8 agree today. V7 stores the selector ID
    and exact formula so future counts cannot silently diverge.
14. **Format validation is duplicated.** Python tooling validates fewer fields
    than the Zig loader. Build one conformance fixture set containing valid,
    truncated, overflowed, overlapping, wrong-shape, and bad-hash files and run
    every loader against it.
15. **The old trainer argument handling masked validation.** The duplicate
    non-positive hidden-size fallback/check has been replaced with one strict
    validation pass over every numeric parameter.

## 10. Acceptance Criteria

V7 is ready to become the default only when all of these are true:

- Stage R is bit-exact and statistically neutral with the promoted v3 net;
- every v7 tensor is shape/type/hash validated by Zig, Rust export, and Python;
- incremental and full recomputation agree after all special moves and random
  make/unmake sequences;
- the candidate is selected using a fixed held-out set, not training loss or
  checkpoint recency;
- fixed-node and clock-time results are both recorded;
- fast-TC SPRT passes and LTC confirms architectures with a material NPS cost;
- the final file includes complete provenance and zero unexpected quantisation
  clipping;
- the old embedded v3/v6 net remains available for regression testing.

## 11. Recommended Immediate Work Order

1. **Done:** add held-out data, a persistent validation sample, and
   reproducibility manifests to the trainer.
2. Add incremental-vs-full NNUE differential tests to the current v6 runtime.
3. Make the match gate pentanomial/SPRT-capable with a larger paired opening
   suite.
4. Run Stage R at the proven duration and establish a trustworthy baseline.
5. Run O=8 and width ablations using the existing linear runtime.
6. **Done:** implement the v7 sectioned exporter/loader, scalar fallback,
   vectorised registered head, and 77-position cross-language/incremental
   parity.
7. **Done for the first run:** benchmark the full H=1024 shape and reduce the
   initial scalar NPS penalty from about 69% to roughly 25–28%.
8. Run the full H=1024 pairwise-MLP CUDA training job and try the exported net.
9. Only after the core wins, test self-play fine-tuning, more input buckets,
   and threat/pawn-pair features.
