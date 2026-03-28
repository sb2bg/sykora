# SYKNNUE4 Design Spec

## Goal

`SYKNNUE4` is the baseline Sykora NNUE format for a large, stable, shared-head
network.

The design goal is:

- keep the sparse incremental part large
- remove phase-fragmenting independent output heads
- add a direct PSQT/material side path so the net can represent large material
  imbalances and winning endgames cleanly
- define the integer inference contract explicitly enough to make
  trainer/export/runtime agreement testable

The baseline `SYKNNUE4` net is:

```text
king_buckets_mirrored(16 buckets)
-> shared sparse hidden branch, width 2048, two color-fixed accumulators
-> shared sparse PSQT branch, width 1, two color-fixed accumulators

hidden path:
    concat(screlu(A_us_hidden), screlu(A_them_hidden))   # 4096 inputs
    -> 32
    -> 32
    -> 1

psqt path:
    psqt_us - psqt_them

final:
    eval_cp = dense_cp + psqt_cp
```

Short form:

```text
shared hidden FT: 12288 -> 2048, color-fixed dual perspective
shared PSQT FT:   12288 -> 1,    color-fixed dual perspective
-> concat(us, them) -> 32 -> 32 -> 1
-> add PSQT side output
```

This follows the older Stockfish-style family more closely than the previous
bucketed micro-head design.

## Non-Goals

The first `SYKNNUE4` implementation should not include:

- multiple output heads / phase stacks
- product pooling
- threat-side input features
- dual-net switching
- mixed float/int inference
- approximate rescale rules in the reference path

Those may be revisited later, but they should not block the first correct and
stable implementation.

## Architecture

### Inputs

- Feature set: `king_buckets_mirrored`
- Per-bucket base feature size: `768`
- Default input bucket count: `16`
- Output bucket count: `1`
- Bucket layout: stored explicitly in the file
- Horizontal mirroring: enabled
- Training-only factorization is allowed, but exported nets must contain merged
  real sparse weights only

Per perspective:

```text
INPUT_SIZE = 768
INPUT_BUCKET_COUNT = 16
OUTPUT_BUCKET_COUNT = 1
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

### Sparse Branches

The sparse part has two shared branches:

```text
hidden FT: SparseAffine(768, 2048) per king bucket
psqt FT:   SparseAffine(768, 1)    per king bucket
```

Maintain four color-fixed accumulators:

- `A_white_hidden[2048]`
- `A_black_hidden[2048]`
- `A_white_psqt`
- `A_black_psqt`

For the reference implementation, store these accumulators as `i32`.

At evaluation time:

```text
if side_to_move == white:
    A_us_hidden   = A_white_hidden
    A_them_hidden = A_black_hidden
    A_us_psqt     = A_white_psqt
    A_them_psqt   = A_black_psqt
else:
    A_us_hidden   = A_black_hidden
    A_them_hidden = A_white_hidden
    A_us_psqt     = A_black_psqt
    A_them_psqt   = A_white_psqt
```

### Hidden Activation

For each hidden accumulator entry:

```text
u = clamp(A_us_hidden[i],   0, Q0)
t = clamp(A_them_hidden[i], 0, Q0)
```

Then apply SCReLU:

```text
U[i] = u * u
T[i] = t * t
```

Concatenate:

```text
X = [U, T]
```

So:

- `X` has length `4096`
- each entry is in the `Q0^2` domain

This is the only nonlinearity between the sparse transformer and the dense
head in the baseline design.

### PSQT Side Path

The PSQT branch is evaluated directly from the sparse scalar outputs:

```text
psqt_delta_int = A_us_psqt - A_them_psqt
```

This path is intentionally simple. It exists to give the net a direct way to
represent material imbalance and large coarse advantages that tiny dense heads
learn poorly from scratch.

### Dense Head

The dense head is shared. There are no phase-specific output stacks.

```text
L1: Affine(4096, 32)
-> clipped activation to Q domain
-> L2: Affine(32, 32)
-> clipped activation to Q domain
-> L3: Affine(32, 1)
```

Final score:

```text
eval_cp = dense_cp + psqt_cp
```

## Quantization Contract

Use the following constants:

```text
Q0 = 255
Q1 = 64
Q  = 64
QPSQT = 128
SCALE = 400
```

Interpretation:

- `Q0`: sparse hidden branch clamp / scale
- `Q1`: first dense-layer weight scale
- `Q`: later dense-layer activation and weight scale
- `QPSQT`: PSQT branch scale
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

### Hidden FT Storage

Export the hidden sparse branch as:

- hidden biases: `i16`
- hidden weights: `i16`

Quantization:

```text
hidden_bias_int   = quantize_round(hidden_bias_float, Q0)
hidden_weight_int = quantize_round(hidden_weight_float, Q0)
```

The reference hidden-accumulator domain is therefore the raw integer sum of
these stored values. SCReLU clamps hidden accumulator entries into `[0, Q0]`
before squaring.

### PSQT FT Storage

Export the PSQT sparse branch as:

- PSQT bias: `i32`
- PSQT weights: `i16`

Quantization:

```text
psqt_bias_int   = quantize_round(psqt_bias_float, QPSQT)
psqt_weight_int = quantize_round(psqt_weight_float, QPSQT)
```

The PSQT branch is a direct sparse scalar accumulator in the `QPSQT` domain.

### Layer 1: `Affine(4096, 32)`

Storage:

- weights: `i8`
- biases: `i32`

Input:

- `X[i]` is in the `Q0^2` domain

Quantization:

```text
W1_int = quantize_round(W1_float, Q1)
b1_int = quantize_round(b1_float, Q0^2 * Q1)
```

Preactivation:

```text
z1_int[j] = b1_int[j] + sum_i(X[i] * W1_int[j][i])
```

So the preactivation scale is:

```text
Q0^2 * Q1 = 255 * 255 * 64 = 4161600
```

Rescale into the `Q = 64` activation domain using exact signed rounding:

```text
R1_DEN = (Q0^2 * Q1) / Q = 65025
u1[j] = div_round_nearest_signed(z1_int[j], R1_DEN)
a1[j] = clamp(u1[j], 0, Q)
```

So:

- `a1[j]` lies in `[0, 64]`
- `a1` is in the `Q` domain

### Layer 2: `Affine(32, 32)`

Storage:

- weights: `i8`
- biases: `i32`

Quantization:

```text
W2_int = quantize_round(W2_float, Q)
b2_int = quantize_round(b2_float, Q^2)
```

Preactivation:

```text
z2_int[k] = b2_int[k] + sum_j(a1[j] * W2_int[k][j])
```

Input scale is `Q` and weight scale is `Q`, so the preactivation scale is:

```text
Q^2
```

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
dense_cp = div_round_nearest_signed(z3_int * SCALE, Q^2)
```

### PSQT Conversion

Convert the direct PSQT side output to centipawns:

```text
psqt_cp = div_round_nearest_signed(psqt_delta_int * SCALE, QPSQT)
```

### Final Evaluation

```text
eval_cp = dense_cp + psqt_cp
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

- dense rescale after `L1`
- dense rescale after `L2`
- final dense centipawn conversion
- PSQT centipawn conversion

No approximate right-shift substitutions belong in the reference path.

## Full Architecture Summary

Features:

- HalfKA-style king-bucketed mirrored inputs
- 16 king buckets
- horizontal mirroring
- 768 inputs per bucket

Sparse branches:

- shared hidden FT
- `12288 -> 2048`
- shared PSQT FT
- `12288 -> 1`
- four color-fixed accumulators
- reference accumulator type `i32`

Dense path:

- clamp hidden accumulators to `[0, 255]`
- apply SCReLU
- concatenate `us` and `them` hidden vectors to `4096`
- `L1: 4096 -> 32`
- activation clipped to `Q = 64`
- `L2: 32 -> 32`
- activation clipped to `Q = 64`
- `L3: 32 -> 1`

Side path:

- direct scalar PSQT delta from color-fixed sparse accumulators

Quantization:

- `Q0 = 255`
- `Q1 = 64`
- `Q = 64`
- `QPSQT = 128`
- `SCALE = 400`

## File Format

### Summary

Magic:

```text
"SYKNNUE4"
```

The format stores:

- merged hidden sparse branch
- merged PSQT sparse branch
- one shared dense head

### Header Layout

All values are little-endian.

```text
8 bytes   magic: "SYKNNUE4"
u16       version = 4
u8        feature_set                  # 1 = king_buckets_mirrored
u16       ft_hidden_size               # baseline 2048
u16       dense_layer_1_size           # baseline 32
u16       dense_layer_2_size           # baseline 32
u8        output_bucket_count          # baseline 1
u8        input_bucket_count           # baseline 16
u16       q0                           # baseline 255
u16       q1                           # baseline 64
u16       q                            # baseline 64
u16       qpsqt                        # baseline 128
u16       scale                        # baseline 400
u8[64]    bucket layout by king square
```

### Payload Layout

Let:

- `H = ft_hidden_size`
- `L1 = dense_layer_1_size`
- `L2 = dense_layer_2_size`
- `I = input_size = 768 * input_bucket_count`

Payload:

```text
i16[H]                hidden_ft_biases
i16[I * H]            hidden_ft_weights
i32[1]                psqt_bias
i16[I]                psqt_weights
i32[L1]               l1_biases
i8[L1 * 2H]           l1_weights
i32[L2]               l2_biases
i8[L2 * L1]           l2_weights
i32[1]                out_bias
i8[L2]                out_weights
```

Weight order is row-major with output neuron as the major dimension:

- `l1_weights[out][in]`
- `l2_weights[out][in]`
- `out_weights[in]`

## Validation Rules

The loader must reject nets where:

- `version != 4`
- any dimension is zero
- `feature_set` is unsupported
- `input_bucket_count == 0`
- `output_bucket_count != 1`
- `ft_hidden_size == 0`
- any bucket-layout entry is `>= input_bucket_count`
- any payload-size multiplication overflows
- payload size does not exactly match the header dimensions
- file size exceeds `MAX_NETWORK_BYTES`

All payload-size computations must be performed in `u64`.
Reject on overflow during any intermediate size computation before allocating.

## Size Budget

Approximate size for the baseline:

- hidden sparse branch `16 buckets, 2048 hidden`
- PSQT sparse branch `16 buckets, 1 scalar`
- shared dense `4096 -> 32 -> 32 -> 1`
- about `52 MiB`

Recommended loader guard:

```text
MAX_NETWORK_BYTES = 64 * 1024 * 1024
```

Do not remove the guard entirely.

## Inference Path

The `SYKNNUE4` evaluation path is:

1. Incrementally update or refresh `A_white_hidden`, `A_black_hidden`,
   `A_white_psqt`, and `A_black_psqt`.
2. Select `us` and `them` accumulators from side to move.
3. Clamp hidden accumulators to `[0, Q0]`.
4. Apply SCReLU on the hidden branch.
5. Concatenate `us` and `them` hidden activations.
6. Run `L1`.
7. Clip into the `Q` domain.
8. Run `L2`.
9. Clip into the `Q` domain.
10. Run `L3`.
11. Convert the dense output to centipawns.
12. Convert the PSQT delta to centipawns.
13. Add them.

Dense scratch buffers are per-searcher, not per-node.

Recommended scratch:

- `hidden_cat`: `[]i32` length `4096`
- `a1`: `[]i32` length `32`
- `a2`: `[]i32` length `32`

## Trainer / Exporter Requirements

The training/export pipeline must follow the same integer contract as runtime.

Required properties:

- training may use sparse factorization, but exported sparse weights must be
  merged
- exported tensor shapes must match the header exactly
- quantization scales must match the header exactly
- reference exporter and Zig runtime must agree on:
  - hidden activation clamp and SCReLU
  - PSQT delta sign convention
  - all rescale rules
  - final centipawn conversion

The exporter must support a bit-exact verification mode against the Zig runtime
on at least:

- one quiet opening or middlegame position
- one simple winning endgame position

The quantized Python eval and Zig eval must match exactly.

## Reference Implementation Priorities

Implementation order:

1. make the math match exactly
2. verify trainer/export/runtime agreement
3. verify incremental update correctness
4. only then optimize:
   - reduce accumulator storage width if safe
   - add sparse execution tricks where useful
   - revisit output buckets only after the shared-head baseline is stable

This architecture should be treated as:

```text
a shared-head king-bucket NNUE with a direct PSQT side output
```

That makes it a better starting point for a first strong and stable
implementation than a fragmented bucketed-head design.
