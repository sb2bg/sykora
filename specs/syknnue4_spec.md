# SYKNNUE4 Design Spec

## Goal

`SYKNNUE4` is the first Sykora NNUE format that moves beyond a pure
`sparse FT -> linear output` network. The target is a stronger architecture
family already used by strong public engines:

- wider king-conditioned sparse feature transformer
- small dense head
- piece-count-selected layer stacks

The baseline `SYKNNUE4` net for implementation is:

```text
king_buckets_mirrored(16 buckets, factorized in training only)
-> accumulator width 1536 per perspective
-> concat(us, them)
-> 16
-> 32
-> 1
with 8 piece-count layer stacks
```

Short form:

```text
shared FT: 12288 -> 1536, dual perspective
-> concat(us_ft, them_ft)            # 3072 inputs
-> 16 -> 32 -> 1, with 8 piece-count layer stacks
```

This keeps the expensive incremental part sparse and moves additional capacity
into a cheap dense head.

## Why This Shape

- It is materially larger than the current shipped `SYKNNUE3` net.
- It stays in the architecture family used by engines such as Integral and
  Lizard.
- The first-layer update cost grows with accumulator width, so the dense head
  should stay small.
- Piece-count buckets are cheap to select at inference time and are a proven
  discriminator for endgame vs middlegame behavior.

## Non-Goals For First `SYKNNUE4`

The first `SYKNNUE4` implementation should not include:

- threat-side input features
- PSQT side-channel outputs
- dual-net switching
- mixed float/int inference

Those may be added later, but they should not block the first format and
runtime implementation.

## Architecture

### Inputs

- Feature set: `king_buckets_mirrored`
- Per-bucket base feature size: `768`
- Default bucket count: `16`
- Bucket layout: stored explicitly in the file, same as `SYKNNUE3`
- Training-only factorization is allowed, but exported nets must contain merged
  real weights only

### Sparse Feature Transformer

- One shared FT parameter set for both perspectives
- One accumulator per perspective
- Width: `1536`
- Input weights: `i16`
- Input biases: `i16`
- Activation: `SCReLU`

The `x2` in informal architecture discussions refers to two accumulators
(`us`/`them`), not two copies of FT parameters.

For a perspective accumulator entry `a`:

```text
ft(a) = trunc_toward_zero(clamp(a, 0, QA)^2 / QA)
```

with `QA = 255`.

### Dense Head

After building `us` and `them` transformed activations:

```text
x0 = concat(us_ft, them_ft)            # length 3072
x1 = clipped_relu(l1(x0))              # length 16
x2 = clipped_relu(l2(x1))              # length 32
y  = out(x2)                           # scalar
```

There are `8` independent layer stacks. The stack is selected by piece count:

```text
stack_index = min(saturating_sub(piece_count, 1) / 4, 7)
piece_count = popcount(all occupied squares)
```

This matches the widely used 8-bucket piece-count split.

### Quantization

The first `SYKNNUE4` should remain integer-only end to end.

Recommended scales:

- `QA = 255` for feature-transformer activations
- `Q1 = 64` for `l1` weights
- `Q2 = 64` for `l2` weights
- `QO = 64` for output weights
- `SCALE = 400` for centipawn conversion, matching current eval scale

Recommended stored types:

- FT biases: `i16`
- FT weights: `i16`
- `l1` biases: `i32`
- `l1` weights: `i16`
- `l2` biases: `i32`
- `l2` weights: `i16`
- output biases: `i32`
- output weights: `i16`

Runtime arithmetic types:

- FT accumulators: `i32`
- FT activation outputs: `u16` in `[0, QA]`
- dense hidden activations: `u16` in `[0, QA]`
- dense dot-product accumulators `z1`, `z2`, `z3`: `i64`

All divisions in inference use integer truncation toward zero.
In Zig, use `@divTrunc(...)` for these operations.
All multiply-accumulate intermediates should be widened before multiplication.

Dense math:

```text
ft_i = trunc_toward_zero(clamp(acc_i, 0, QA)^2 / QA)

z1 = b1 + sum(x0_i * w1_i)             # scale QA * Q1
a1 = clamp(trunc_toward_zero(z1 / Q1), 0, QA)

z2 = b2 + sum(a1_i * w2_i)             # scale QA * Q2
a2 = clamp(trunc_toward_zero(z2 / Q2), 0, QA)

z3 = b3 + sum(a2_i * w3_i)             # scale QA * QO
cp = trunc_toward_zero(z3 * SCALE / (QA * QO))
```

This keeps inference simple and close to the current `SYKNNUE3` scaling model.

## File Format

### Summary

New magic:

```text
"SYKNNUE4"
```

The format stores a merged sparse FT plus a bucketed dense head.

### Header Layout

All values are little-endian.

```text
8 bytes   magic: "SYKNNUE4"
u16       version = 4
u8        feature_set              # 0=legacy_psqt, 1=king_buckets_mirrored
u8        ft_activation_type       # 0=ReLU, 1=SCReLU
u16       ft_hidden_size           # baseline 1536
u8        dense_activation_type    # 0=clipped_relu, 1=SCReLU; baseline clipped_relu
u16       dense_layer_1_size       # baseline 16
u16       dense_layer_2_size       # baseline 32
u8        layer_stack_count        # baseline 8
u8        bucket_count             # baseline 16
u16       qa                       # baseline 255
u16       q1                       # baseline 64
u16       q2                       # baseline 64
u16       qo                       # baseline 64
u8[64]    bucket layout by king square
```

### Payload Layout

Let:

- `H = ft_hidden_size`
- `L1 = dense_layer_1_size`
- `L2 = dense_layer_2_size`
- `S = layer_stack_count`
- `I = input_size = 768 * bucket_count` for mirrored king buckets

Payload:

```text
i16[H]                    ft_biases
i16[I * H]                ft_weights
i32[S * L1]               l1_biases
i16[S * L1 * (2 * H)]     l1_weights
i32[S * L2]               l2_biases
i16[S * L2 * L1]          l2_weights
i32[S]                    out_biases
i16[S * L2]               out_weights
```

Weight order must be row-major with the output neuron as the major dimension:

- `l1_weights[stack][out][in]`
- `l2_weights[stack][out][in]`
- `out_weights[stack][in]`

### Validation Rules

The loader must reject nets where:

- `version != 4`
- any dimension is zero
- `feature_set` is unsupported
- `bucket_count == 0` for mirrored king buckets
- any bucket layout entry is `>= bucket_count`
- any payload-size multiplication overflows
- payload size does not exactly match the header dimensions
- file size exceeds `MAX_NETWORK_BYTES`

All payload-size computations must be performed in `u64`.
Reject on overflow during any intermediate size computation before allocating.

## Size Budget

The current 8 MiB limit is too small for any serious `SYKNNUE4`.

Approximate sizes:

- shared FT `16 buckets, 1536 FT, 16->32->1 x8`: about `36.8 MiB`
- shared FT `16 buckets, 2048 FT, 16->32->1 x8`: about `49.0 MiB`

Recommended new loader guard:

- `MAX_NETWORK_BYTES = 64 * 1024 * 1024`

Do not remove the guard entirely. A large but finite cap is still useful since
the loader reads the full file before parsing.

## Zig Runtime Changes

### Data Model

Replace the current single-layer `Network` representation with a tagged union or
single `Network` struct that can represent both `SYKNNUE3` and `SYKNNUE4`.

Recommended shape:

```text
Network {
  format_version
  feature_set
  bucket_count
  bucket_layout
  ft_hidden_size
  ft_activation_type
  dense_activation_type
  dense_l1_size
  dense_l2_size
  layer_stack_count
  qa, q1, q2, qo
  ft_biases
  ft_weights
  l1_biases
  l1_weights
  l2_biases
  l2_weights
  out_biases
  out_weights
}
```

Keep `SYKNNUE2` and `SYKNNUE3` loading intact.

### Remove Fixed 512 Assumptions

The following assumptions must be removed:

- `MAX_HIDDEN_SIZE = 512`
- fixed `[MAX_HIDDEN_SIZE]` accumulator arrays
- any stack-local assumptions that hidden size is compile-time bounded by 512

Replace accumulator storage with dynamically allocated slices sized from the net
header.

Recommended search-local structure:

```text
AccumulatorPair {
  white: []i32
  black: []i32
}
```

Allocate one contiguous block for the full accumulator stack:

```text
acc_stack_len * 2 * ft_hidden_size * sizeof(i32)
```

For `128` plies and `H=1536`, that is about `1.5 MiB` per search thread.

### Inference Path

The `SYKNNUE4` evaluation path should be:

1. Incrementally update or refresh the two FT accumulators exactly as today.
2. Apply FT activation (`SCReLU`) to produce `us_ft` and `them_ft`.
3. Select `stack_index` from piece count.
4. Run the selected dense stack `2H -> L1 -> L2 -> 1`.
5. Convert to centipawns with `SCALE`.

Dense head evaluation should use a scratch buffer allocated once per searcher:

- `x0`: optional direct streaming, no persistent allocation required
- `a1`: `[]u16` of length `L1`
- `a2`: `[]u16` of length `L2`

Dense scratch buffers are per-searcher, not per-node.

### SIMD

No new SIMD work is required for the first implementation beyond the current
feature-update path.

The FT update loops should remain SIMD-friendly.
The dense head is small enough to start with scalar loops.
If needed later, optimize only `l1`, since it dominates dense-head cost.

### Loader Safety

The loader should:

- keep a finite max file size guard
- compute the exact payload size from parsed dimensions in `u64`
- reject on overflow during dimension multiplication or byte-count conversion
- require an exact payload-size match

Do not infer payload shapes from trailing bytes in `SYKNNUE4`.
The header already carries all required dimensions.

## Trainer Changes

### Target Network

Update `utils/nnue/bullet_runner/src/main.rs` to train:

```text
inputs(ChessBucketsMirrored::new(bucket_layout))
-> dual_perspective
-> factorized FT at training time
-> concat(stm_ft, ntm_ft)
-> stack-selected 16->32->1 head
```

The FT should keep the current factorized-training idea:

- real bucketed FT weights
- one shared factorizer over the `768` base features
- merge the factorizer into exported FT weights

### Stack Selection

Training data must compute the same `stack_index` as inference:

```text
stack_index = min(saturating_sub(piece_count, 1) / 4, 7)
```

At training time, the simplest implementation is:

- produce all stack outputs in one batched tensor
- gather the output for the chosen stack per sample

This is the same broad strategy used by Stockfish-style layer-stack training.

Training should also log per-stack sample counts and per-stack loss.
Low-piece-count stacks are likely to be underrepresented in raw self-play
corpora, so reweighting or oversampling may be needed if those stacks lag.

### Dense Head Sizes

Baseline:

- `H = 1536`
- `L1 = 16`
- `L2 = 32`
- `S = 8`

Stretch target after baseline validation:

- `H = 2048`
- same dense head

Do not start with `2048` if the runtime path is not yet stable.

### Training Export

The current `quantised.bin -> SYKNNUE3` converter is not enough for `SYKNNUE4`.

Required additions:

- a new export path that writes the `SYKNNUE4` header
- export of merged FT weights
- export of `l1`, `l2`, and output stack parameters
- export of all quantization scales stored in the header

The easiest first route is:

1. export a float-domain checkpoint with explicit tensors
2. quantize in Python
3. write `.sknnue4`

This is simpler than trying to force the existing `quantised.bin` layout to
carry a more complex architecture.

## Python Exporter Changes

Extend `utils/nnue/common.py` with:

- `MAGIC_V4`
- `FORMAT_VERSION_V4`
- `write_syk_nnue_v4(...)`
- exact payload-size computation helper

Add a new exporter script instead of overloading the current v3 exporter:

```text
utils/nnue/bullet/export_npz_to_syk4.py
```

Expected tensors:

- `ft_weights`: `[input_size, H]`
- `ft_bias`: `[H]`
- `l1_weights`: `[S, L1, 2H]`
- `l1_bias`: `[S, L1]`
- `l2_weights`: `[S, L2, L1]`
- `l2_bias`: `[S, L2]`
- `out_weights`: `[S, L2]`
- `out_bias`: `[S]`

The exporter should:

- validate shapes strictly
- quantize according to header scales
- merge any FT factorizer weights before serialization
- write exact dimensions and scales into the header

The exporter should also support a bit-exact verification mode against the Zig
runtime on at least:

- one opening or midgame position
- one reduced-material endgame position

The quantized Python eval and Zig eval must match exactly.

## Migration Plan

### Phase 1

- add `SYKNNUE4` file format support
- raise network size guard to `64 MiB`
- replace fixed accumulators with dynamic slices
- keep `SYKNNUE3` evaluation unchanged

### Phase 2

- implement `SYKNNUE4` inference
- add scalar dense-head path
- add unit tests for loader and eval determinism
- add required bit-exact Python-exporter vs Zig-runtime eval tests

### Phase 3

- train and export baseline `1536 / 16 / 32 / 8-stack` nets
- benchmark NPS and memory
- run selfplay gating

### Phase 4

- consider `2048` FT width
- consider threat inputs or PSQT side-channel only after the baseline is stable

## Plan

Implement exactly one baseline `SYKNNUE4` first:

```text
shared FT 12288 -> 1536, dual perspective
-> concat(1536, 1536)
-> 16 -> 32 -> 1, 8 piece-count stacks
```

Will not mix in threats, PSQT forwarding, or dual-network switching until this
architecture is trained, exported, and measured.
