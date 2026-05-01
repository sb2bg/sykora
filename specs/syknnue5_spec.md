# SYKNNUE5 Design Spec

`SYKNNUE5` is Sykora's current king-bucketed SCReLU training target with
material-count output buckets.

## Architecture

```text
king_buckets_mirrored(16 buckets)
-> shared sparse FT, H hidden units, color-fixed dual perspective
-> concat(screlu(A_us), screlu(A_them))
-> material-count bucketed linear output head
```

The first intended training target is:

```text
shared FT: 12288 -> 512
-> concat(us, them): 1024
-> 8 material-count output heads
```

`H = 768` is the larger follow-up target.

## Output Buckets

The output bucket selector matches Bullet's `MaterialCount<8>`:

```text
piece_count = popcount(occupied)
non_king_count = piece_count - 2
divisor = ceil(32 / output_bucket_count)
output_bucket = min(non_king_count / divisor, output_bucket_count - 1)
```

With the default `output_bucket_count = 8`, the divisor is `4`.

## File Format

All integers are little-endian.

```text
u8[8]  magic = "SYKNNUE5"
u16    version = 5
u8     feature_set = 1                  # king_buckets_mirrored
u16    ft_hidden_size = H
u8     activation_type                  # 0 = ReLU, 1 = SCReLU
u8     input_bucket_count
u8     output_bucket_count
u16    q0
u16    q
u16    scale
u8[64] bucket_layout
i16[H] ft_biases
i16[input_bucket_count * 768 * H] ft_weights
i32[output_bucket_count] output_biases
i16[output_bucket_count * 2 * H] output_weights
```

`output_weights` are bucket-major. For bucket `b`, the slice is:

```text
output_weights[b * 2H .. (b + 1) * 2H]
```

The first `H` weights apply to `A_us`; the second `H` apply to `A_them`.

## Quantization

The baseline constants are:

```text
Q0 = 255
Q  = 64
SCALE = 400
```

SCReLU output is divided by `Q0` before adding the selected output bias, then the
final score is converted to centipawns with:

```text
score = round(sum * SCALE / (Q0 * Q))
```
