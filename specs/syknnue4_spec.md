# SYKNNUE4 Design Spec

## Goal

`SYKNNUE4` is the first Sykora NNUE format that moves beyond a pure
`sparse FT -> linear output` network while keeping the implementation simple
enough to verify end to end.

The baseline `SYKNNUE4` net is:

```text
king_buckets_mirrored(16 buckets)
-> shared sparse feature transformer, width 2048, two color-fixed accumulators
-> product pooling per accumulator half-pair
-> concat(us_pooled, them_pooled)            # 2048 inputs
-> 16
-> [linear, squared] expansion to 32
-> 32
-> 1
with 8 output buckets by non-king piece count
```

Short form:

```text
shared FT: 12288 -> 2048, color-fixed dual perspective
-> product_pool(1024, 1024) per perspective
-> concat(1024, 1024)
-> 16 -> expand(32) -> 32 -> 1, with 8 output buckets
```

This keeps the expensive incremental part sparse and pushes the nonlinearity
into a very small dense head.

## Non-Goals

The first `SYKNNUE4` implementation should not include:

- threat-side input features
- PSQT side-channel outputs
- dual-net switching
- mixed float/int inference
- approximate pooling or approximate rescale rules in the reference path

Those may be added later, but they should not block the first correct
implementation.

## Architecture

### Inputs

- Feature set: `king_buckets_mirrored`
- Per-bucket base feature size: `768`
- Default input bucket count: `16`
- Bucket layout: stored explicitly in the file
- Horizontal mirroring: enabled
- Training-only factorization is allowed, but exported nets must contain merged
  real FT weights only

Per perspective:

```text
INPUT_SIZE = 768
INPUT_BUCKET_COUNT = 16
OUTPUT_BUCKET_COUNT = 8
HORIZONTAL_MIRRORING = true
```

Feature indexing is defined for color-fixed perspectives `white` and `black`,
not for side-to-move / side-not-to-move.

For a perspective `p`:

```text
feature =
    king_bucket(p.king_sq) * 768
  + relative_color(piece, p) * (6 * 64)
  + piece_type * 64
  + mirrored_square(p.king_sq, sq)
```

Where:

- `relative_color(piece, p)` is whether the piece is friendly or enemy from
  the perspective `p`
- `mirrored_square(...)` applies horizontal mirroring so the king is always
  treated as living on one side of the board

### Sparse Feature Transformer

The FT is shared between perspectives.

For each perspective independently:

```text
FT: SparseAffine(768, 2048) per king bucket
```

Maintain two color-fixed accumulators:

- `A_white[2048]`
- `A_black[2048]`

For the reference implementation, store these accumulators as `i32`.

This is a bring-up choice, not a final optimization target. It reduces risk
while validating:

- incremental update correctness
- trainer/export/runtime agreement
- bias quantization
- pooling exactness

At evaluation time:

```text
if side_to_move == white:
    A_us   = A_white
    A_them = A_black
else:
    A_us   = A_black
    A_them = A_white
```

### Product Pooling

This is the main structural nonlinearity before the head.

Split each accumulator into two halves:

```text
A_us   = [U0, U1], each length 1024
A_them = [T0, T1], each length 1024
```

Clamp each element first:

```text
u0 = clamp(U0[i], 0, Q0)
u1 = clamp(U1[i], 0, Q0)
t0 = clamp(T0[i], 0, Q0)
t1 = clamp(T1[i], 0, Q0)
```

Then compute pooled activations with exact divide-by-`255` normalization:

```text
P_us[i]   = div_round_nearest_nonneg(u0 * u1, Q0)
P_them[i] = div_round_nearest_nonneg(t0 * t1, Q0)
```

With:

```text
Q0 = 255
```

Concatenate:

```text
P = [P_us, P_them]
```

So:

- `P` has length `2048`
- `P[i]` is stored as `u8`
- `P[i]` lies in `[0, 255]`

Use the exact divide-by-255 form in the reference implementation.
Only after correctness is validated may the runtime replace it with a proven
equivalent optimization.

### Output Bucket Selection

Use `8` output buckets by non-king piece count:

```text
non_king_piece_count = popcount(all occupied squares) - 2
output_bucket = min(7, non_king_piece_count / 4)
```

For legal positions, `non_king_piece_count` is in `[0, 30]`.

### Dense Head

After pooling, the `2048 -> 16` stage is architecturally dense.

The head for the selected output bucket is:

```text
L1: Affine(2048, 16)
-> clipped activation to Q domain
-> mixed [linear, squared] expansion to 32 dims
-> L2: Affine(32, 32)
-> clipped activation to Q domain
-> L3: Affine(32, 1)
```

## Quantization Contract

Use the following constants:

```text
Q0 = 255
Q1 = 128
Q  = 64
SCALE = 400
```

Interpretation:

- `Q0`: pooled activation clamp / scale
- `Q1`: first dense layer weight scale
- `Q`: head activation and later dense-layer weight scale
- `SCALE`: final centipawn conversion

All float-to-int quantization in this spec uses:

```text
quantize_round(x, scale) =
    if x >= 0:
        floor(x * scale + 0.5)
    else:
        -floor((-x) * scale + 0.5)
```

This is “round to nearest, ties away from zero”.

### FT Storage

Export the FT as:

- FT biases: `i16`
- FT weights: `i16`

Quantization:

```text
ft_bias_int   = quantize_round(ft_bias_float, Q0)
ft_weight_int = quantize_round(ft_weight_float, Q0)
```

The reference accumulator domain is therefore the raw integer sum of these
stored FT values. Pooling clamps accumulator entries into `[0, Q0]` before any
multiply.

### Layer 1: `Affine(2048, 16)`

Storage:

- weights: `i8`
- biases: `i32`

Input:

- `P[i]` has scale `Q0`

Quantization:

```text
W1_int = quantize_round(W1_float, Q1)
b1_int = quantize_round(b1_float, Q0 * Q1)
```

Preactivation:

```text
z1_int[j] = b1_int[j] + sum_i(P[i] * W1_int[j][i])
```

So the preactivation scale is:

```text
Q0 * Q1 = 255 * 128 = 32640
```

Rescale into the `Q = 64` activation domain using exact signed rounding:

```text
R1_DEN = (Q0 * Q1) / Q = 510
u1[j] = div_round_nearest_signed(z1_int[j], R1_DEN)
t1[j] = clamp(u1[j], 0, Q)
```

So:

- `t1[j]` lies in `[0, 64]`
- `t1` is in the `Q` domain

### Mixed Linear/Squared Expansion

For each of the 16 values in `t1`:

```text
lin[j] = t1[j] * Q
sq[j]  = t1[j] * t1[j]
```

Since `Q = 64`, both `lin[j]` and `sq[j]` are in the `Q^2` domain:

```text
Q^2 = 4096
```

Concatenate:

```text
H1 = [lin[0..15], sq[0..15]]
```

So:

- `H1` has length `32`
- `H1` scale is `Q^2`
- `H1` may be stored as `u16` or `i32` scratch in the reference runtime

### Layer 2: `Affine(32, 32)`

Storage:

- weights: `i8`
- biases: `i32`

Quantization:

```text
W2_int = quantize_round(W2_float, Q)
b2_int = quantize_round(b2_float, Q^3)
```

Preactivation:

```text
z2_int[k] = b2_int[k] + sum_j(H1[j] * W2_int[k][j])
```

Input scale is `Q^2` and weight scale is `Q`, so the preactivation scale is:

```text
Q^3 = 262144
```

Rescale back to the `Q = 64` activation domain:

```text
R2_DEN = Q^2 = 4096
u2[k] = div_round_nearest_signed(z2_int[k], R2_DEN)
a2[k] = clamp(u2[k], 0, Q)
```

So after layer 2:

- `a2` has length `32`
- `a2` scale is `Q`

### Final Layer: `Affine(32, 1)`

Storage:

- weights: `i8`
- bias: `i32`

Quantization:

```text
W3_int = quantize_round(W3_float, Q)
b3_int = quantize_round(b3_float, Q^2)
```

Output:

```text
z3_int = b3_int + sum_k(a2[k] * W3_int[k])
```

Since input scale is `Q` and weight scale is `Q`, the output scale is:

```text
Q^2
```

Convert to centipawns:

```text
eval_cp = div_round_nearest_signed(z3_int * SCALE, Q^2)
```

with:

```text
SCALE = 400
```

## Exact Rounding Rules

The reference implementation must not leave rounding behavior implicit.

Use the following helpers with positive denominator `d > 0`.

For nonnegative integers:

```text
div_round_nearest_nonneg(x, d) = (x + d / 2) / d
```

For signed integers:

```text
div_round_nearest_signed(x, d) =
    if x >= 0:
        (x + d / 2) / d
    else:
        -(((-x) + d / 2) / d)
```

This is “round to nearest, ties away from zero”.

All languages in the project must implement these rules exactly for:

- pooling
- `z1 -> u1`
- `z2 -> u2`
- final centipawn conversion

No approximate right-shift substitutions belong in the reference path.

## Full Architecture Summary

Features:

- HalfKA-style king-bucketed mirrored inputs
- 16 king buckets
- horizontal mirroring
- 768 inputs per bucket

Transformer:

- shared sparse FT
- `12288 -> 2048`
- two color-fixed accumulators
- reference accumulator type `i32`

Pooling:

- split `2048 -> 1024 + 1024`
- clamp to `[0, 255]`
- pooled entry = `div_round_nearest_nonneg(a * b, 255)`
- concatenate `us` and `them` pooled vectors to `2048`

Head:

- 8 output buckets by non-king piece count
- `L1: 2048 -> 16`
- activation clipped to `Q = 64`
- mixed linear/squared expansion `16 -> 32`
- `L2: 32 -> 32`
- activation clipped to `Q = 64`
- `L3: 32 -> 1`

Quantization:

- `Q0 = 255`
- `Q1 = 128`
- `Q = 64`
- `SCALE = 400`

## File Format

### Summary

Magic:

```text
"SYKNNUE4"
```

The format stores merged FT weights plus bucketed dense-head parameters.

### Header Layout

All values are little-endian.

```text
8 bytes   magic: "SYKNNUE4"
u16       version = 4
u8        feature_set                  # 1 = king_buckets_mirrored
u16       ft_hidden_size               # baseline 2048
u16       dense_layer_1_size           # baseline 16
u16       dense_layer_2_size           # baseline 32
u8        output_bucket_count          # baseline 8
u8        input_bucket_count           # baseline 16
u16       q0                           # baseline 255
u16       q1                           # baseline 128
u16       q                            # baseline 64
u16       scale                        # baseline 400
u8[64]    bucket layout by king square
```

### Payload Layout

Let:

- `H = ft_hidden_size`
- `L1 = dense_layer_1_size`
- `L2 = dense_layer_2_size`
- `S = output_bucket_count`
- `I = input_size = 768 * input_bucket_count`

Payload:

```text
i16[H]                ft_biases
i16[I * H]            ft_weights
i32[S * L1]           l1_biases
i8[S * L1 * H]        l1_weights
i32[S * L2]           l2_biases
i8[S * L2 * 32]       l2_weights
i32[S]                out_biases
i8[S * L2]            out_weights
```

Weight order is row-major with output neuron as the major dimension:

- `l1_weights[bucket][out][in]`
- `l2_weights[bucket][out][in]`
- `out_weights[bucket][in]`

### Validation Rules

The loader must reject nets where:

- `version != 4`
- any dimension is zero
- `feature_set` is unsupported
- `input_bucket_count == 0`
- `output_bucket_count == 0`
- `ft_hidden_size` is odd
- any bucket-layout entry is `>= input_bucket_count`
- any payload-size multiplication overflows
- payload size does not exactly match the header dimensions
- file size exceeds `MAX_NETWORK_BYTES`

All payload-size computations must be performed in `u64`.
Reject on overflow during any intermediate size computation before allocating.

## Size Budget

This format is materially larger than earlier Sykora nets.

Approximate size for the baseline:

- shared FT `16 buckets, 2048 FT, 16 -> expand32 -> 32 -> 1 x8`: about `50 MiB`

Recommended loader guard:

```text
MAX_NETWORK_BYTES = 64 * 1024 * 1024
```

Do not remove the guard entirely.

## Inference Path

The `SYKNNUE4` evaluation path is:

1. Incrementally update or refresh `A_white` and `A_black`.
2. Select `A_us` and `A_them` from side to move.
3. Product-pool each accumulator into `P_us` and `P_them`.
4. Concatenate into `P`.
5. Select `output_bucket` from non-king piece count.
6. Run `L1`.
7. Clip into the `Q` domain.
8. Expand `16 -> 32` via `[linear, squared]`.
9. Run `L2`.
10. Clip into the `Q` domain.
11. Run `L3`.
12. Convert to centipawns.

Dense scratch buffers are per-searcher, not per-node.

Recommended scratch:

- `pooled_us`: `[]u8` length `1024`
- `pooled_them`: `[]u8` length `1024`
- `t1`: `[]i32` length `16`
- `h1`: `[]i32` length `32`
- `a2`: `[]i32` length `32`

## Trainer / Exporter Requirements

The training/export pipeline must follow the same integer contract as runtime.

Required properties:

- training may use FT factorization, but exported FT weights must be merged
- exported tensor shapes must match the header exactly
- quantization scales must match the header exactly
- reference exporter and Zig runtime must agree on:
  - pooling
  - bucket selection
  - all rescale rules
  - final centipawn conversion

The exporter must support a bit-exact verification mode against the Zig runtime
on at least:

- one opening or middlegame position
- one reduced-material endgame position

The quantized Python eval and Zig eval must match exactly.

## Reference Implementation Priorities

Implementation order:

1. make the math match exactly
2. verify trainer/export/runtime agreement
3. verify incremental update correctness
4. only then optimize:
   - replace exact `/255` with a faster equivalent if proven identical
   - reduce accumulator storage width if safe
   - add sparse execution tricks where useful

This architecture should be treated as:

```text
a product-pooled king-bucket NNUE with an explicit integer inference contract
```

That makes it a better starting point for a first correct implementation than
an aggressively optimized but underspecified design.
