# SYKNNUE9 P³-ANOVA Runtime and Container Contract

SYKNNUE9 extends the registered SYKNNUE8 T1024 graph with a rank-32
pawn–pawn–piece adapter. The v8 PSQ transformer, full-threat accumulator,
pairwise product pool, material buckets, and `16 -> 32 -> 1` tail are unchanged.

## P³ feature state

For each perspective, pawn identities are indexed by relative colour and the
same oriented/mirrored square used by the factorised PSQ input (`2 * 64 = 128`
rows). Non-pawn identities use relative colour, type from knight through king,
and oriented square (`2 * 5 * 64 = 640` rows). Both kings are context entities.

The production accumulator stores, for every rank channel:

- eight file-local pawn sums `S_f`;
- the maintained distinct same-file pair sum `Q_same`;
- the maintained adjacent-file pair sum `Q_adj`;
- the non-pawn context sum `C`.

Pawn insertion on file `f` applies

```text
Q_same += x * S_f
Q_adj  += x * (S_(f-1) + S_(f+1))
S_f    += x
```

Removal subtracts `x` from `S_f` first, then subtracts the products with the
post-removal same-file sum and unchanged neighbour sums. Evaluation forms
`[Q_same * C; Q_adj * C]` for the side-to-move perspective followed by the
other perspective. The material-bucketed `128 -> 16` projection is added to
the existing l1 sums immediately before the v8 dual activation.

King moves refresh a perspective only when its horizontal mirror state changes.
All other moves use atomic removals and insertions, including capture-square
removal for en passant, pawn removal plus context insertion for promotion, and
the two context moves required by castling.

## Integer contract

Pawn and context embeddings are signed i8 with quantisation scale 64. The raw
triadic moments therefore have scale `64^3`. They are rounded to nearest,
ties away from zero, with a right shift by 12 and clipped to signed i8. The P³
projection uses scale 128, so its packed dot product lands directly in the
existing l1-sum scale `128 * 64 = 8192`.

The signed activation `t` is encoded as unsigned `t + 128`. For every output,
the loader precomputes `-128 * sum(weights)` and adds that correction before
using the same packed `u8 * i8` dot-product kernel as the base l1 layer. The
exporter and loader both verify that adjacent weight-pair magnitudes cannot
saturate the AVX2 `pmaddubsw` intermediate. The loader also proves from the
embedding maxima and 32-entity population bound that raw moment products fit
signed i32, permitting vectorised i32 moment formation.

Before quantisation, the exporter uses the per-rank CP scale symmetry to balance
the maximum pawn, context, and projection magnitudes. This does not change the
float function. The exporter rejects clipping and writes a conservative l1
contribution bound; the loader recomputes and verifies that bound.

## Container

The fixed header and section entry sizes remain 224 and 48 bytes. SYKNNUE9 uses
magic `SYKNNUE9`, version 9, architecture id 3, and twelve sections. The nine
v8 sections retain their ids and shapes. New signed-i8 sections are:

| Id | Tensor | Shape |
|---:|---|---|
| 20 | pawn embeddings | `[128, 32]` |
| 21 | non-pawn context embeddings | `[640, 32]` |
| 22 | material-bucketed P³ projection | `[8, 128, 16]` |

The 20 bytes after the SHA-256 field contain, in order, rank `u16`, pawn count
`u16`, context count `u16`, embedding quantisation `u16`, activation shift `u8`,
three zero bytes, the declared P³ l1 absolute bound `u32`, and four zero bytes.
Payload alignment, per-section CRC32 checks, and the whole-file SHA-256 rule are
identical to SYKNNUE8.

## Required gates

- brute-force moments must equal maintained-pair moments;
- incremental state must equal full recomputation after every tested legal move;
- trainer/export/engine evaluation must be bit-exact on the parity FEN suite;
- all eight material heads, both perspectives, and both mirror states must be
  covered;
- fixed-node strength and clock strength must be reported separately because
  the adapter has a measurable NPS cost.
