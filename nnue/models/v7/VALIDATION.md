# v7_20260710T055911Z checkpoint 800 — validation record

First full SYKNNUE7 training run (H=1024 pairwise-MLP, 8 material heads).

## Run config
- Architecture: pairwise-mlp, FT H=1024 factorised, v3_10 input buckets, O=8
- Head: pairwise pool -> bucketed 1024 -> 16 -> dual 32 -> 32 -> 1
- Data: SF test80 binpacks 2023-06..2024-05 (2024-06 held out)
- Filter: ply>=16, not in check, |score|<=10000
- WDL 0.25 constant, LR cosine 0.001 -> 2.43e-6, 800 superbatches (~80B positions)
- Hardware: RTX 4070 Ti Super, ~35.5 s/superbatch (~2.8M pos/sec), ~8h wall

## Validation (2026-07-10, checkpoint 800)
- Final running loss 0.01266 (matches WDL-0.25 cohort floor prediction ~0.012)
- Export: raw -> npz -> syk7, zero quantisation clipping on all tensors
- Parity: 77/77 bit-exact engine vs Python reference (all 8 buckets, both
  mirrors, both stms); incremental == full on every legal child
- Weight health: FT/L1/L2 clean; L3 output layer has 8.2% of weights pinned
  at the +/-1.98 clip rail — quant scale is binding there (v7.1 candidate fix)
- Piece values monotonic at stm: N +1842 < B +1965 < R +2063 < Q +2188;
  B-for-N -24; KPK fortress draw reads 2cp; symmetric positions ~0
- Speed (M-series Mac, 1 thread): 453K NPS vs 794K embedded v3 (-43%), but
  depth 13 from startpos needs 1.03M nodes vs 2.90M — faster in wall clock
- SPRT vs embedded v3 (3s+30ms, elo0=0, elo1=30): in progress at commit time

## Files
- `v7_20260710T055911Z-800.sknnue` — exported SYKNNUE7 net (load via EvalFile)
- `run_meta.synthesized.json` — architecture metadata reconstructed from the
  training preamble (original run_meta.json lives on the training box)
