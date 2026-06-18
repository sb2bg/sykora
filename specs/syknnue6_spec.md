# SYKNNUE6 Design Spec

## 1. Overview

`SYKNNUE6` is Sykora's NNUE network format and training target. It is the
direct successor to `SYKNNUE3` — the last successfully trained net — and makes
two changes over v3: **wider FT** (512 → 768) and **material-count output
buckets** (1 → 8). Everything else (feature set, king-bucket layout,
activation, factorised training, quantisation) stays on the proven v3 recipe.

```text
king_buckets_mirrored (10 buckets, v3_10 layout)
→ factorised sparse FT, H hidden units, dual perspective
→ concat(screlu(A_us), screlu(A_them))
→ material-count bucketed linear output head (O buckets)
```

The v6 strength target is:

```text
shared factorised FT: 7680 → 768  (10 king buckets × 768 base features)
→ concat(us, them): 1536
→ 8 material-count output heads
```

The pipeline is validated in stages (§6.6): first a parity net at `H=512,
O=1` (architecturally identical to v3), then output buckets at `H=512, O=8`,
then the full `H=768, O=8` strength target. Each stage is SPRT-gated against
the previous.

### Non-goals

The first `SYKNNUE6` implementation does **not** include:

- dense hidden layers (no `2H → L1 → L2 → 1` head)
- product pooling / pairwise multiplication
- threat-side input features
- PSQT side-channel outputs
- more than 10 king buckets
- FT widths beyond 768

These were all tried in v4/v5 and are retired. See §2.

## 2. Postmortem: Why v4 and v5 Failed

v3 is the last net that trained successfully. v4 and v5 both failed to
produce a net stronger than v3. The root causes are documented here so the
same traps are not repeated.

### v3 (successful)

The v3 trainer used **factorised** king-bucketed inputs:

```rust
let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);
let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, hl_size);
l0.weights = l0.weights + expanded_factoriser;
```

The factoriser (`l0f`) is a shared `768 → H` weight matrix that is trained
across **all** positions regardless of king bucket. Every position
contributes to it, so it converges fast and generalises well. The per-bucket
weights (`l0w`) only need to learn the **residual** after the factoriser — a
much easier problem. At export time, the factoriser is merged into the
bucket weights:

```text
merged_l0w[bucket][feat][hid] = l0w[bucket][feat][hid] + l0f[feat % 768][hid]
```

The exported net contains only the merged weights; the runtime has no
knowledge of factorisation.

v3 parameters: 10 buckets (v3_10 layout), 512 FT, SCReLU, single linear
head `2H → 1`, factorised, i16/255 FT quantisation, i16/64 output
quantisation, WDL 0.75, 320 superbatches, cosine LR decay, weight clipping
at ±0.99 on `l0w` and `l0f`.

### v4 (failed)

v4 changed **five things at once**:

1. **16 king buckets** (sykora16 layout) — each bucket sees ~1/16 of data.
2. **1536 FT width** — 3× the parameters, needs far more data to converge.
3. **Dense head** (16 → 32 → 1 with 8 layer stacks) — spec said SCReLU +
   simple dense; trainer actually used CReLU + `pairwise_mul()` +
   `abs_pow(2.0)` expansion, a completely different architecture.
4. **Spec ↔ trainer divergence** — the spec described one architecture, the
   trainer implemented another. Neither was validated.
5. **No factorisation** in the later v4 trainer iterations — the factoriser
   was dropped, so 16 buckets were trained independently with no shared
   base.

With five simultaneous untested changes, it was impossible to isolate which
one broke training. None of the v4 nets cleared self-play gating.

### v5 (failed)

v5 was simpler than v4 but repeated v4's critical mistake:

1. **16 king buckets** (sykora16) — same undertraining problem.
2. **No factorisation** — the v5 trainer (`run_syk5`) has no `l0f`, no
   `.transform()` merge. Each of the 16 buckets' 768 weights is trained on
   ~1/16 of the dataset with no shared base. This is the single dominant
   cause of failure.
3. Output buckets (8 material) — this part was correct and is retained in
   v6. The output-side innovation was fine; the input-side regression killed
   the net.

### The temporary v6 (never trained)

The temporary v6 format was a repackaged v3 with configurable output
buckets. The format itself was sound, but the trainer (`run_syk6`) also
omitted factorisation — the same flaw as v5. The embedded `src/net.sknnue`
is a byte-for-byte repack of the v3 net (O=1, `single` scheme); no v6 net was
ever trained from scratch.

### Lesson

> **Change one thing at a time from the last working baseline.**
> The only change in v6 over v3 is material-count output buckets.
> Factorisation is non-negotiable. The bucket layout stays at v3_10.

## 3. Architecture

### 3.1 Feature Set: `king_buckets_mirrored`

Same as v3. Each position is encoded from two perspectives (side-to-move and
opponent). For each perspective, the king square determines:

1. **Horizontal mirroring** — if the king is on files e–h, the board is
   mirrored left-right so the king always appears on files a–d.
2. **King bucket** — the (mirrored) king square maps to one of N input
   buckets via `bucket_layout[64]`. Each bucket has its own 768-feature slice
   of FT weights.

Base feature index within a bucket (before mirroring):

```text
feature = side_idx * 6 * 64 + piece_idx * 64 + square
```

where `side_idx ∈ {0, 1}` (perspective-relative colour), `piece_idx ∈ 0..5`,
`square ∈ 0..63`.

After mirroring (`sq ^= 7` if king file > 3) and bucket offset:

```text
index = LEGACY_INPUT_SIZE * bucket_layout[perspective_king_sq]
        + side_idx * 6 * 64 + piece_idx * 64 + mirrored_sq
```

`LEGACY_INPUT_SIZE = 768` (2 colours × 6 piece types × 64 squares).

### 3.2 King Bucket Layout: `v3_10`

The only supported layout for v6. 10 king buckets, stored as a 32-entry
half-board layout and expanded to 64 entries via horizontal mirroring:

```text
32-entry half:
    0, 1, 2, 3,       (rank 1: a1-d1, one bucket per file pair)
    4, 4, 5, 5,       (rank 2)
    6, 6, 6, 6,       (rank 3)
    7, 7, 7, 7,       (rank 4)
    8, 8, 8, 8,       (rank 5)
    8, 8, 8, 8,       (rank 6)
    9, 9, 9, 9,       (rank 7)
    9, 9, 9, 9,       (rank 8)

64-entry expansion:
    mirror = [0, 1, 2, 3, 3, 2, 1, 0]
    layout_64[idx] = layout_32[(idx / 8) * 4 + mirror[idx % 8]]
```

This gives more granularity near the back rank (where castling happens) and
less in the endgame middle. 10 buckets is enough to capture king-position
patterns without undertraining any single bucket.

### 3.3 Sparse Feature Transformer

- One **shared** FT parameter set for both perspectives (dual-perspective
  accumulator, not two separate FT matrices).
- Width: `H = 768` (v6 target) or `H = 512` (Stage 1/2 parity baseline).
- Input weights: `i16`, scale `Q0 = 255`.
- Input biases: `i16`, scale `Q0 = 255`.
- Activation: **SCReLU** (squared clipped ReLU).

```text
a_i = clamp(acc_i, 0, Q0)          # clip to [0, 255]
v_i = a_i * a_i                      # square, range [0, 65025]
```

### 3.4 Output Head

Bucketed linear head:

```text
x = concat(v_us, v_them)             # length 2H
y = out_weights[bucket] · x + out_bias[bucket]
```

- Output weights: `i16`, scale `Q = 64`. Shape `[O, 2H]`, bucket-major.
- Output biases: `i32`, scale `Q0 * Q = 16320`. Shape `[O]`.
- `O = 1` (Stage 1, `single` scheme) or `O = 8` (Stage 2+, `material_popcount`
  scheme).

### 3.5 Output Bucket Selection

For the `material_popcount` scheme with `O` buckets (O must divide 32):

```text
piece_count    = popcount(occupied)           # 2..32
non_king_count = piece_count - 2              # 0..30
divisor        = 32 / O                        # integer division
bucket         = min(non_king_count / divisor, O - 1)
```

With `O = 8`: `divisor = 4`, bucket covers 4 non-king pieces each (bucket 0
= 0–3 pieces, ..., bucket 7 = 28–30 pieces). This matches Bullet's
`MaterialCount<N>`.

For the `single` scheme (`O = 1`): `bucket = 0` always.

## 4. File Format

All integers are little-endian.

### 4.1 Header

```text
u8[8]   magic              = "SYKNNUE6"
u16     version            = 6
u8      feature_set        = 1   (king_buckets_mirrored)
u16     ft_hidden_size     = H   (baseline 512)
u8      activation_type    = 1   (0=ReLU, 1=SCReLU; only SCReLU supported)
u8      input_bucket_count = 10  (max(bucket_layout) + 1)
u8      output_bucket_count = O  (1 or 8)
u8      output_bucket_scheme = 0 (single) or 1 (material_popcount)
u16     q0                 = 255
u16     q                  = 64
u16     scale              = 400
u8[64]  bucket_layout            (v3_10 expanded to 64 entries)
```

Header size: 8 + 2 + 1 + 2 + 1 + 1 + 1 + 1 + 2 + 2 + 2 + 64 = 87 bytes.

### 4.2 Payload

Let `H = ft_hidden_size`, `I = input_size = 768 * input_bucket_count`,
`O = output_bucket_count`.

```text
i32[O]              output_biases        (scale Q0 * Q)
i16[H]              ft_biases            (scale Q0)
i16[I * H]          ft_weights           (scale Q0, merged factoriser)
i16[O * 2 * H]      output_weights       (scale Q, bucket-major)
```

`output_weights` are bucket-major. For bucket `b`, the slice is:

```text
output_weights[b * 2H .. (b + 1) * 2H]
```

The first `H` weights apply to the STM (us) accumulator; the second `H`
apply to the NTM (them) accumulator.

### 4.3 Payload Size

```text
ft_biases_bytes      = H * 2
ft_weights_bytes     = I * H * 2
output_biases_bytes  = O * 4
output_weights_bytes = O * 2 * H * 2
total_payload        = ft_biases_bytes + ft_weights_bytes
                        + output_biases_bytes + output_weights_bytes
```

For the v6 target (`H=768, I=7680, O=8`):

```text
ft_biases:      1,536 B
ft_weights:   11,796,480 B    (11.3 MiB)
output_biases:    32 B
output_weights: 24,576 B      (24 KiB)
total_payload: 11,822,624 B   (≈ 11.3 MiB)
total file:    11,822,711 B   (≈ 11.3 MiB)
```

For the Stage 1/2 baseline (`H=512, I=7680, O=8`):

```text
ft_biases:      1,024 B
ft_weights:    7,864,320 B    (7.5 MiB)
output_biases:    32 B
output_weights: 16,384 B      (16 KiB)
total_payload: 7,881,760 B    (≈ 7.5 MiB)
total file:    7,881,847 B    (≈ 7.5 MiB)
```

### 4.4 Validation Rules

The loader must reject nets where:

- `magic != "SYKNNUE6"`
- `version != 6`
- `feature_set != 1` (only `king_buckets_mirrored` is supported)
- `activation_type` is not `0` or `1`
- `ft_hidden_size == 0` or `ft_hidden_size > 2048`
- `input_bucket_count == 0`
- any `bucket_layout` entry `>= input_bucket_count`
- `output_bucket_count == 0`
- `output_bucket_scheme == 0` (single) but `output_bucket_count != 1`
- `output_bucket_scheme == 1` (material) but `32 % output_bucket_count != 0`
- `q0 == 0` or `q == 0` or `scale == 0`
- any payload-size multiplication overflows `u64`
- `header_size + payload_size != file_size` (exact match required)
- `file_size > 64 MiB`

All payload-size computations must be performed in `u64` with overflow
checking before any allocation.

## 5. Integer Inference Contract

This section defines the exact integer arithmetic the runtime must perform.
The Python reference (`utils/nnue/bullet/check_net_parity.py`) and the Zig
runtime (`src/nnue.zig`) must produce bit-identical results.

### 5.1 Accumulator

For each perspective `p ∈ {white, black}`:

```text
acc_p[h] = ft_biases[h] + Σ ft_weights[feature(p, piece) * H + h]
```

over all pieces on the board, where `feature(p, piece)` is the feature index
from §3.1. Accumulator values are `i32` and may be negative or exceed `Q0`.

### 5.2 SCReLU Activation

```text
a = clamp(acc, 0, Q0)       # clipped to [0, 255]
v = a * a                    # squared, range [0, 65025], i32
```

### 5.3 Output Dot Product

```text
bucket = output_bucket(board)          # per §3.5
w_us   = output_weights[bucket * 2H .. bucket * 2H + H]
w_them = output_weights[bucket * 2H + H .. bucket * 2H + 2H]

raw = Σ(v_us[h] * w_us[h]) + Σ(v_them[h] * w_them[h])
```

Accumulate in `i64` to avoid overflow. Each term `v * w` has magnitude up to
`65025 * 32767 ≈ 2.1 × 10^9`, and there are `2H` terms (1024 at `H=512`,
1536 at `H=768`), so the sum fits in `i64` with large margin.

### 5.4 Scoring

For SCReLU activation (`activation_type == 1`):

```text
raw   = round_to_nearest(raw / Q0)           # divide by 255
raw  += output_biases[bucket]                 # i32, pre-scaled by Q0*Q
score = round_to_nearest(raw * SCALE / (Q0 * Q))   # = round(raw * 400 / 16320)
```

For ReLU activation (`activation_type == 0`):

```text
raw  += output_biases[bucket]
score = round_to_nearest(raw * SCALE / (Q0 * Q))
```

`round_to_nearest` is round-half-away-from-zero for signed integers:

```text
round(x, d) = sign(x) * floor((|x| + d/2) / d)
```

The score is from the **side-to-move** perspective, matching the convention
of the classical evaluation. Positive = good for STM.

### 5.5 Quantisation Scales Summary

| Tensor           | Stored type | Scale          | Float → Int              |
| ---------------- | ----------- | -------------- | ------------------------ |
| `ft_biases`      | `i16`       | `Q0 = 255`     | `round(float * 255)`     |
| `ft_weights`     | `i16`       | `Q0 = 255`     | `round(float * 255)`     |
| `output_biases`  | `i32`       | `Q0 * Q = 16320` | `round(float * 16320)` |
| `output_weights` | `i16`       | `Q = 64`       | `round(float * 64)`      |

Clipping ranges: `i16` to `[-32768, 32767]`, `i32` to
`[-2147483648, 2147483647]`.

## 6. Training

### 6.1 Factorisation (Mandatory)

The FT **must** be trained with a factoriser. This is the single most
important training-side requirement and the root cause of v4/v5 failure.

The factoriser is a `768 → H` weight matrix (`l0f`) shared across all king
buckets. During training, the effective FT weight for bucket `b`, feature
`f`, hidden `h` is:

```text
effective_l0w[b * 768 + f][h] = l0w[b * 768 + f][h] + l0f[f][h]
```

At export time, the factoriser is merged into the bucket weights (§7.2) and
the exported net contains only the merged `ft_weights`. The runtime never
sees `l0f`.

### 6.2 Bullet Trainer Configuration

The trainer uses [Bullet](https://github.com/jw1912/bullet)'s
`ValueTrainerBuilder` with:

```rust
let mut trainer = ValueTrainerBuilder::default()
    .dual_perspective()
    .optimiser(AdamW)
    .inputs(ChessBucketsMirrored::new(V3_BUCKET_LAYOUT_32))
    .output_buckets(MaterialCount::<O>)          // O = 1 or 8
    .use_threads(threads)
    .save_format(&[
        // Factoriser merge: tile l0f across all buckets and add to l0w
        SavedFormat::id("l0w")
            .transform(move |store, weights| {
                let factoriser = store.get("l0f").values.repeat(num_input_buckets);
                weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
            })
            .round()
            .quantise::<i16>(255),
        SavedFormat::id("l0b").round().quantise::<i16>(255),
        SavedFormat::id("outw").round().quantise::<i16>(64),
        SavedFormat::id("outb").round().quantise::<i32>(255 * 64),
    ])
    .loss_fn(|output, target| output.sigmoid().squared_error(target))
    .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
        // Factoriser: shared 768 → H weights, trained across all buckets
        let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
        let expanded_factoriser = l0f.repeat(num_input_buckets);

        // Per-bucket FT weights + factoriser
        let mut l0 = builder.new_affine("l0", 768 * num_input_buckets, hl_size);
        l0.init_with_effective_input_size(32);
        l0.weights = l0.weights + expanded_factoriser;

        // Bucketed output head
        let out = builder.new_affine("out", 2 * hl_size, O);

        let stm_hidden = l0.forward(stm_inputs).screlu();
        let ntm_hidden = l0.forward(ntm_inputs).screlu();
        let hidden = stm_hidden.concat(ntm_hidden);
        out.forward(hidden).select(output_buckets)
    });

// Factoriser weights need tighter clipping (they're shared, so magnitudes
// propagate across all buckets)
let stricter_clipping = AdamWParams {
    max_weight: 0.99,
    min_weight: -0.99,
    ..Default::default()
};
trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);
```

`V3_BUCKET_LAYOUT_32` is the 32-entry half-board layout from §3.2.
`ChessBucketsMirrored::new` expands it to 64 entries internally.

### 6.3 Bucket Layout Constant

```rust
#[rustfmt::skip]
const V3_BUCKET_LAYOUT_32: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
];
```

### 6.4 Hyperparameters

v6 uses v3's proven recipe, scaled for the wider FT:

| Parameter              | Stage 1 (parity) | Stage 2 (buckets) | Stage 3 (wide FT) |
| ---------------------- | ----------------- | ------------------ | ------------------ |
| FT hidden size (`H`)   | 512               | 512                | 768                |
| Input bucket layout    | `v3_10`           | `v3_10`            | `v3_10`            |
| Output buckets (`O`)   | 1                 | 8                  | 8                  |
| Activation             | SCReLU            | SCReLU             | SCReLU             |
| Batch size             | 16,384            | 16,384             | 16,384             |
| Batches per superbatch | 6,104             | 6,104              | 6,104              |
| Start superbatch       | 1                 | 1                  | 1                  |
| End superbatch         | 320               | 640                | 640                |
| Initial LR             | 0.001             | 0.001              | 0.001              |
| Final LR               | `initial * 0.3^5` | `initial * 0.3^5`  | `initial * 0.3^5`  |
| LR scheduler           | Cosine decay      | Cosine decay       | Cosine decay       |
| WDL proportion         | 0.75              | 0.75               | 0.75               |
| Weight clipping        | ±0.99             | ±0.99              | ±0.99              |
| Eval scale             | 400.0             | 400.0              | 400.0              |
| Loss                   | sigmoid²          | sigmoid²           | sigmoid²           |
| Save rate              | every superbatch  | every superbatch   | every superbatch   |

Stage 3 runs longer (640 superbatches) because the wider FT has 50% more
parameters and needs more data to converge. With ample binpack data this is
not a constraint.

### 6.5 Data Filter

For Stockfish binpack data:

```rust
fn binpack_filter(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10000
}
```

### 6.6 Staging

Each stage is SPRT-gated against the previous stage's best net. Do not skip
stages — if a stage fails, the failure isolates which variable broke.

**Stage 1 — parity net (`H=512`, `O=1`, `single` scheme, 320 superbatches)**

Train a single-head net with the exact v3 architecture + factorisation.
This validates that the v6 pipeline (trainer + export + runtime + factoriser
merge) reproduces v3-level strength. Gate against the existing v3 net
(repacked to v6) via SPRT at neutral (`elo0=-30, elo1=30`). If Stage 1 does
not clear SPRT at neutral, the pipeline has a bug — do not proceed.

**Stage 2 — material output buckets (`H=512`, `O=8`, `material_popcount`, 640 superbatches)**

Same FT width as v3 + 8 material-count output buckets. This isolates the
output-bucket change. Gate against the Stage 1 net via SPRT. Expected gain
is modest (5–15 Elo) since the FT is unchanged.

**Stage 3 — wider FT (`H=768`, `O=8`, `material_popcount`, 640 superbatches)**

The v6 strength target. 50% more FT capacity than v3, with output buckets.
Gate against the Stage 2 net via SPRT. Expected gain over Stage 2 is 15–30
Elo, for a total of 20–45 Elo over v3. This is where the data volume pays
off — the wider FT needs more data to converge, and 640 superbatches ×
6,104 batches × 16,384 = ~670M positions should be sufficient with ample
binpack data.

If Stage 3 fails to clear SPRT, fall back to diagnostic isolation:
- Stage 3a: `H=768, O=1` (isolate width without output buckets)
- Stage 3b: compare 3a vs 3 to isolate the width × bucket interaction

**Stretch targets (only after Stage 3 clears SPRT)**

- `H = 1024` (2× v3 capacity, ~15 MiB net, needs 800+ superbatches)
- More superbatches (640 → 800+)
- These are size/duration changes, not architecture changes.

## 7. Export Pipeline

### 7.1 Raw Checkpoint Layout

Bullet saves `raw.bin` as a flat `float32` stream of all graph weight
tensors in creation order. With factorisation, the order is:

```text
l0f:  [768, H]                    768 * H floats
l0w:  [768 * num_buckets, H]      768 * num_buckets * H floats
l0b:  [H]                         H floats
outw: [O, 2 * H]                  O * 2 * H floats
outb: [O]                         O floats
```

Total: `768*H + 768*B*H + H + O*2*H + O` floats (B = input bucket count).

The `checkpoint_raw_to_npz.py` script must read `l0f` and **merge** it into
`l0w` before saving the NPZ:

```python
l0f, offset = take_f32(raw, offset, 768 * ft_hidden)         # factoriser
l0w, offset = take_f32(raw, offset, input_size * ft_hidden)   # bucket weights

# Merge: tile factoriser across all buckets and add
factoriser_tiled = np.tile(l0f.reshape(768, ft_hidden), (bucket_count, 1))
ft_weights = l0w.reshape(input_size, ft_hidden) + factoriser_tiled
```

The NPZ then contains the merged `ft_weights` (no `l0f`), plus `ft_bias`,
`out_weights`, `out_bias`, and metadata.

### 7.2 NPZ → SYKNNUE6

`export_npz_to_syk6.py` quantises the float NPZ and writes the `.sknnue`
file:

```python
ft_bias_i16   = quantize(ft_bias,   q0=255,     dtype=np.int16)
ft_weights_i16 = quantize(ft_weights, q0=255,   dtype=np.int16)
out_bias_i32   = quantize(out_bias,   q0*q=16320, dtype=np.int32)
out_weights_i16 = quantize(out_weights, q=64,    dtype=np.int16)
```

Quantisation is round-half-away-from-zero followed by clipping to the
integer type's range:

```python
def quantize(values, scale, min_val, max_val, dtype):
    rounded = np.sign(values) * np.floor(np.abs(values) * scale + 0.5)
    return np.clip(rounded, min_val, max_val).astype(dtype)
```

The output bucket scheme is inferred from `output_bucket_count`: `1` →
`single`, `8` → `material_popcount`.

### 7.3 Export Validation

After export, run the bit-exact parity gate:

```bash
python utils/nnue/bullet/check_net_parity.py \
    --net output.sknnue \
    --fens utils/nnue/parity.fens \
    --engine ./zig-out/bin/sykora
```

This computes a numpy reference eval for every FEN (reproducing the §5
inference contract exactly) and compares against the engine's `nnuecheck`
output. Every position must match exactly.

The `parity.fens` suite must cover:
- both mirror states (king on files a–d and e–h)
- all output buckets (for `O = 8`)
- both side-to-move colours
- at least one endgame and one middlegame position

## 8. Runtime Requirements

### 8.1 Accumulator Storage

The search maintains an incremental accumulator stack. Each stack entry is
an `AccumulatorPair` (one `i32` array per perspective, length `H`).

```text
AccumulatorPair = 2 * H * sizeof(i32) bytes
                = 2 * 768 * 4 = 6144 bytes  (v6 target, H=768)
                = 2 * 512 * 4 = 4096 bytes  (Stage 1/2, H=512)
```

The stack is heap-allocated and bounded by `MAX_SEARCH_PLY` (currently 128).
Total per thread: `128 * 6144 = 768 KiB` (H=768) or `512 KiB` (H=512).

`MAX_HIDDEN_SIZE = 2048` is the compile-time upper bound on `H`. The
accumulator arrays are statically sized to `MAX_HIDDEN_SIZE` to avoid
dynamic allocation per node; only the first `H` entries are used.

### 8.2 Incremental Updates

The FT is updated incrementally during search (make/unmake). When the king
moves to a square in a different bucket or different mirror state, the
affected perspective's accumulator is fully recomputed. All other updates
are feature-delta updates (add/subtract weight slices).

The incremental update path is identical to v3 — only the output head
changed. The SIMD update loops (`applyFeatureSlice`,
`applyFeatureSlicesAddSub`, etc.) operate on `i16` weight vectors and `i32`
accumulator vectors, 8 lanes at a time.

### 8.3 Output Head Evaluation

The output head is evaluated only at leaf nodes (not incrementally). Cost:

```text
2H multiply-accumulates + 1 bucket select + 1 bias add + 2 divisions
= 1536 MACs (H=768) or 1024 MACs (H=512)
```

This is negligible compared to the FT update cost.

### 8.4 Memory Budget

| Component               | H=512 (Stage 1/2) | H=768 (Stage 3, v6 target) |
| ----------------------- | ------------------ | --------------------------- |
| FT weights              | 7.5 MiB            | 11.3 MiB                    |
| FT biases               | 1 KiB              | 1.5 KiB                     |
| Output weights (O=8)    | 16 KiB             | 24 KiB                      |
| Output biases           | 32 B               | 32 B                        |
| Total net file          | ≈ 7.5 MiB          | ≈ 11.3 MiB                  |
| Accumulator stack/thread| 512 KiB            | 768 KiB                     |

The loader guard is `MAX_NETWORK_BYTES = 64 MiB`. Both sizes fit easily.

## 9. Validation Workflow

### 9.1 Bit-Exact Parity Gate

Every exported net must pass `check_net_parity.py` (§7.3) before being
considered for embedding. This catches any quantisation or inference
contract mismatch between the Python reference and the Zig runtime.

### 9.2 Self-Play Smoke Test

```bash
python utils/nnue/bullet/gate_checkpoints.py \
    --checkpoints-dir nnue/models/bullet/<run_id>/checkpoints \
    --engine ./zig-out/bin/sykora \
    --blend 100 --nnue-scale 100 \
    --selfplay-games 80 --selfplay-movetime-ms 120 --selfplay-top-k 3 \
    --threads 1 --hash-mb 64 \
    --min-elo 0 --max-p-value 0.25 \
    --promote-to nnue/syk_nnue_best.sknnue
```

The 80-game / 120 ms self-play gate has a wide Elo error bar; treat it as a
smoke test only.

### 9.3 SPRT Promotion

The canonical promotion signal is SPRT at fast TC via the archived
`history.py` flow:

```bash
python utils/history/history.py sprt baseline candidate \
    --elo0 -30 --elo1 30 \
    --games-per-batch 12 --max-games 360 \
    --movetime-ms 80 --max-plies 220 \
    --threads 1 --hash-mb 64 --shuffle-openings
```

For Stage 1, the baseline is the current v3 net (repacked to v6). For
Stage 2, the baseline is the Stage 1 net. For Stage 3, the baseline is the
Stage 2 net.

### 9.4 NPS Regression Check

After embedding, verify NPS has not regressed:

```bash
python utils/bench/nps.py --engine ./zig-out/bin/sykora --depth 10
```

The v6 output head (bucket selection + `O`-way weight lookup) has negligible
NPS impact. The wider FT (`H=768`) makes each accumulator update ~50% more
expensive than v3 (`H=512`), so expect a measurable NPS drop at Stage 3
(roughly 15–25%). This is the expected cost of the extra capacity — the Elo
gain from the wider FT must outweigh the NPS loss at the target time
control. If the NPS drop is larger than ~25%, investigate the FT update
hot path (SIMD utilisation, cache behaviour on the larger weight table).
