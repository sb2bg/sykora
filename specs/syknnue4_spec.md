# SYKNNUE4 Design Spec

## Goal

`SYKNNUE4` is the simple, stable baseline Sykora NNUE format.

The design goal is:

- keep the sparse incremental part large
- keep the head shared
- stay close to the already-working v3 math
- make the file format self-describing for mirrored king-bucket inputs

The baseline `SYKNNUE4` net is:

```text
king_buckets_mirrored(16 buckets)
-> shared sparse FT, width 768, two color-fixed accumulators
-> concat(screlu(A_us), screlu(A_them))   # 1536 inputs
-> shared linear output
```

Short form:

```text
shared FT: 12288 -> 768, color-fixed dual perspective
-> concat(us, them) -> 1
```

This is intentionally a monotonic upgrade from the v3 family:

- same shared-head philosophy
- same SCReLU inference contract
- wider FT
- explicit mirrored king-bucket layout stored in the file

## Non-Goals

The first `SYKNNUE4` implementation should not include:

- multiple output heads
- multi-layer dense heads
- PSQT side channels
- product pooling
- mixed float/int inference
- approximate rescale rules in the reference path

## Architecture

### Inputs

- Feature set: `king_buckets_mirrored`
- Per-bucket base feature size: `768`
- Default input bucket count: `16`
- Bucket layout: stored explicitly in the file
- Horizontal mirroring: enabled
- Training-only factorization is allowed, but exported nets must contain merged
  sparse weights only

Per perspective:

```text
INPUT_SIZE = 768
INPUT_BUCKET_COUNT = 16
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

### Sparse Transformer

The sparse transformer is:

```text
SparseAffine(768, 768) per king bucket
```

Maintain two color-fixed accumulators:

- `A_white[768]`
- `A_black[768]`

For the reference implementation, store these accumulators as `i32`.

At evaluation time:

```text
if side_to_move == white:
    A_us   = A_white
    A_them = A_black
else:
    A_us   = A_black
    A_them = A_white
```

### Hidden Activation

For each hidden accumulator entry:

```text
u = clamp(A_us[i],   0, Q0)
t = clamp(A_them[i], 0, Q0)
```

Apply the activation selected by `activation_type`:

- `0 = ReLU`
- `1 = SCReLU`

Baseline `SYKNNUE4` uses `SCReLU`.

For `SCReLU`:

```text
U[i] = u * u
T[i] = t * t
```

Concatenate:

```text
X = [U, T]
```

So:

- `X` has length `2 * H`
- with the baseline `H = 768`, `X` has length `1536`
- each entry is in the `Q0^2` domain for `SCReLU`

### Output Head

The output head is shared. There are no phase-specific output stacks.

```text
Out: Affine(2 * H, 1)
```

## Quantization Contract

Use the following constants:

```text
Q0 = 255
Q  = 64
SCALE = 400
```

Interpretation:

- `Q0`: sparse hidden clamp / scale
- `Q`: output-weight scale
- `SCALE`: final centipawn conversion

All float-to-int quantization in this spec uses:

```text
quantize_round(x, scale) =
    if x >= 0:
        floor(x * scale + 0.5)
    else:
        -floor((-x) * scale + 0.5)
```

This is round-to-nearest with ties away from zero.

### Hidden FT Storage

Export the sparse branch as:

- hidden biases: `i16`
- hidden weights: `i16`

Quantization:

```text
hidden_bias_int   = quantize_round(hidden_bias_float, Q0)
hidden_weight_int = quantize_round(hidden_weight_float, Q0)
```

### Output Head Storage

Export the shared output head as:

- output weights: `i16`
- output bias: `i32`

Quantization:

```text
out_weight_int = quantize_round(out_weight_float, Q)
out_bias_int   = quantize_round(out_bias_float, Q0 * Q)
```

## Integer Inference Contract

### Hidden Accumulators

The reference accumulator update path sums stored sparse integers directly:

```text
A_white[i] = hidden_bias_int[i] + sum(active white-perspective feature weights)
A_black[i] = hidden_bias_int[i] + sum(active black-perspective feature weights)
```

### Output Evaluation

For `SCReLU`:

```text
sum_int =
    Σ_i (clamp(A_us[i],   0, Q0)^2 * out_weight_int[i])
  + Σ_i (clamp(A_them[i], 0, Q0)^2 * out_weight_int[H + i])
```

Rescale by one factor of `Q0` before adding bias:

```text
sum_rescaled = div_round_nearest_signed(sum_int, Q0)
z_int = sum_rescaled + out_bias_int
```

Convert to centipawns:

```text
eval_cp = div_round_nearest_signed(z_int * SCALE, Q0 * Q)
```

For `ReLU`, omit the squaring and the intermediate `/Q0` rescale:

```text
sum_int =
    Σ_i (clamp(A_us[i],   0, Q0) * out_weight_int[i])
  + Σ_i (clamp(A_them[i], 0, Q0) * out_weight_int[H + i])

z_int = sum_int + out_bias_int
eval_cp = div_round_nearest_signed(z_int * SCALE, Q0 * Q)
```

### Signed Division

The reference path uses signed round-to-nearest:

```text
div_round_nearest_signed(x, d) =
    if x >= 0:
        (x + d / 2) / d
    else:
        -(((-x) + d / 2) / d)
```

This is the reference contract to match across trainer, exporter, and runtime.

## File Format

All integers are little-endian.

### Header

```text
u8[8]     magic                 = "SYKNNUE4"
u16       format_version        = 4
u8        feature_set           = 1   # king_buckets_mirrored
u16       ft_hidden_size        # baseline 768
u8        activation_type       # baseline 1 = SCReLU
u8        input_bucket_count    # baseline 16
u16       q0                    # baseline 255
u16       q                     # baseline 64
u16       scale                 # baseline 400
u8[64]    bucket_layout_64
```

### Payload

Let:

- `I = 768 * input_bucket_count`
- `H = ft_hidden_size`

Payload order:

```text
i32                    output_bias
i16[H]                 ft_biases
i16[I * H]             ft_weights
i16[2 * H]             output_weights
```

Weight order:

- `ft_weights[input_feature][hidden]`
- `output_weights[0..H]` are `us`
- `output_weights[H..2H]` are `them`

## Loader Validation

A loader should reject nets where:

- `magic != "SYKNNUE4"`
- `format_version != 4`
- `feature_set != 1`
- `ft_hidden_size == 0`
- `input_bucket_count == 0`
- any `bucket_layout_64` entry is `>= input_bucket_count`
- `q0 == 0`
- `q == 0`
- `scale == 0`
- payload size does not match the header

## Baseline Defaults

Baseline values:

```text
feature_set       = king_buckets_mirrored
input_bucket_count = 16
ft_hidden_size    = 768
activation_type   = SCReLU
q0                = 255
q                 = 64
scale             = 400
```

## Reference Implementation Priorities

If implementing or training this architecture, the recommended order is:

1. make the sparse update path correct
2. make exporter and runtime agree bit-for-bit on fixed FENs
3. validate the shared-head model against v3-like sanity positions
4. only then consider widening the FT or adding extra head complexity
