# SYKNNUE8 b16: from-scratch 16-bucket profile

Status: implemented in trainer, exporter, and engine; first training run pending

Date: 2026-11-05

## Summary

A from-scratch run of the v8 graph with 16 king buckets, trained on all
BT4-rescored data except Sykora self-play, with WDL 0.1. It needs no new
container: SYKNNUE8 already stores the bucket layout, so the format version
stays at 8 and the next version remains reserved for the two-speed design
(`syknnue10_spec.md`).

Only changes with outside evidence are made in this run. Everything more
speculative is kept for later function-preserving fine-tunes, so a scratch run
is not spent on several untested ideas at once.

## Changes from v8 T1024

| Item | v8 T1024 | b16 profile |
| --- | --- | --- |
| King buckets | 10 (`v3_10`) | 16 (`b16`) |
| Initialisation | warm start from v7 | random |
| WDL proportion | 0.75 default | 0.1 |
| Position filter | ply >= 16, not in check, \|score\| <= 10000 | the same, plus the recorded move must not be a capture or promotion |
| Data order | files in path order | sources interleaved by bytes |

Unchanged: `full_threats_v1`, H = 1024 with pairwise pooling, eight
material-count heads, the 1024 -> 16 -> 32 -> 32 -> 1 tail, the virtual
factoriser, and the quantisation contract.

## King buckets

Mirrored half-board (files a-d), rank 1 at the top:

```text
 0  1  2  3
 4  5  6  7
 8  8  9  9
10 10 11 11
12 12 13 13
12 12 13 13
14 14 15 15
14 14 15 15
```

- Rank 1 keeps one bucket per file, as in `v3_10`.
- Rank 2, where castled kings stand, gains four buckets instead of two.
- Ranks 3-8 split into wing and centre halves instead of one bucket per band.

Every `b16` bucket lies inside one `v3_10` bucket. A `v3_10` net therefore
expands into `b16` without changing its output
(`utils/nnue/bullet/expand_net_to_b16.py`), which gives an exact engine test
and an optional function-preserving warm start.

Evidence: the v7 spec kept 10 buckets as "proven" and listed a 12- or
16-bucket layout as a later ablation, which was never run. ZigQueen measured
+15 to +23 Elo for a change that included moving to 16 buckets. The
factoriser shares statistics across buckets, so the extra buckets do not
starve for data.

Cost: the PSQ table grows from 15.0 to 24.0 MiB (file 78 -> 87.5 MiB), and
refreshes happen slightly more often. Incremental update cost is unchanged.

## Training recipe

- Data: every BT4-rescored source except Sykora self-play, one subdirectory
  per source under `train/`. Split any source held in a few large files with
  `split_binpack.py` so interleaving can spread it.
- `-Stage scratch -BucketLayout b16 -Wdl 0.1 -SkipTactical -InterleaveSources`.
- Superbatches: 1000 (100B samples, matching v8's 200 + 800) as the floor;
  more if the data and GPU time allow without many repeated passes.
- Learning rate: cosine from 0.001 to 2.43e-6 over the run, as for v8.
- Checkpoint choice: best held-out validation loss, then a short fixed-node
  match between the last few candidates.

## Validation before training

All of these pass on the implementation commit:

1. The engine loads `b16` nets. A `v3_10` net expanded to `b16` gives the
   same evaluation on 1,500 STS positions and the same bench at depths 10 and
   13 as the original, so incremental updates and refreshes are exact under
   the new layout.
2. CPU smoke runs with both layouts go through training, raw checkpoint,
   factoriser merge, `.npz`, and SYKNNUE8 export. The engine matches the
   Python reference on all 77 parity positions.
3. The quiet-position filter has a unit test, including the castling
   encoding (king onto its own rook), en passant, and promotion.

## Promotion

1. STC SPRT against the embedded v8 on the same engine build.
2. LTC confirmation.
3. Search SPSA on the new net, since search margins were fitted to v8.

## Later experiments (warm-started from this net)

Each can start with no effect on the net's output:

- pawn-pair inputs or the Pawn-King tower (`syknnue10_spec.md` §6.1);
- material conditioning (§6.2) and the error head (§7);
- power-2.5 loss and quantisation-aware training.
