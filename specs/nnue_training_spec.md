# Sykora NNUE Training Spec (v0)

## Scope

This spec defines a practical first NNUE training pipeline for Sykora:

- Train a Sykora-native NNUE file (`SYKNNUE1`) from archived game data.
- Keep every dataset/model artifact reproducible and versioned by metadata.
- Gate promotions using STS + self-play instead of STS-only decisions.

This is intentionally an initial implementation, not a final production trainer.

## Repository Layout

Training utilities live in `utils/nnue/`:

- `utils/nnue/extract_positions.py`: build raw position dataset from PGNs.
- `utils/nnue/label_with_stockfish.py`: add teacher eval labels.
- `utils/nnue/train_syknnue.py`: train and export `SYKNNUE1`.
- `utils/nnue/common.py`: shared feature indexing + binary export helpers.

Local NNUE artifacts live under `nnue/` (gitignored except keep files):

- `nnue/data/positions.jsonl`: extracted raw training rows.
- `nnue/data/labeled.jsonl`: teacher-labeled rows.
- `nnue/models/`: optional checkpoints/metadata.
- `nnue/*.sknnue`: exported Sykora-ready nets.

## Data Pipeline

### 1) Extract Positions

```bash
~/.pyenv/shims/python utils/nnue/extract_positions.py \
  --pgn-glob "history/matches/*/pgn/*.pgn" \
  --output nnue/data/positions.jsonl \
  --min-ply 8 \
  --max-ply 180 \
  --sample-rate 0.30 \
  --seed 1
```

Output rows include:

- `fen`
- `target_result_stm` (game-result target from side to move perspective)
- provenance (`source_pgn`, `event`, `ply`)

### 2) Label With Teacher

```bash
~/.pyenv/shims/python utils/nnue/label_with_stockfish.py \
  --input nnue/data/positions.jsonl \
  --output nnue/data/labeled.jsonl \
  --stockfish /opt/homebrew/bin/stockfish \
  --depth 12 \
  --result-mix 0.0 \
  --cp-clip 2000 \
  --eval-scale 400
```

Targets are blended:

- teacher target: `sigmoid(cp / eval_scale)`
- result target: `target_result_stm`
- blended: `result_mix * result + (1 - result_mix) * teacher`

If `--eval-file` is incompatible with the selected Stockfish binary, labeling will fail fast with an explicit error.

### 3) Train + Export Sykora Net

```bash
~/.pyenv/shims/python utils/nnue/train_syknnue.py \
  --input nnue/data/labeled.jsonl \
  --output-net nnue/syk_v0.sknnue \
  --hidden-size 512 \
  --epochs 4 \
  --batch-size 128 \
  --lr 0.005 \
  --weight-decay 1e-6 \
  --eval-scale 400 \
  --backend numpy \
  --augment-mirror \
  --seed 1
```

Export format:

- magic: `SYKNNUE1`
- version: `1`
- quantized params for accumulator/output layers

Backend notes:

- `numpy` (default): portable baseline backend.
- `torch`: faster option on machines with stable PyTorch/OpenMP runtime.

## Engine Integration Contract

Sykora should be configured via UCI:

```text
setoption name EvalFile value /absolute/path/to/net.sknnue
setoption name UseNNUE value true
setoption name NnueBlend value 10
isready
```

Note: Sykora does not execute Stockfish `.nnue` files directly. Use Stockfish as a teacher during data labeling, then train/export a `SYKNNUE1` net for Sykora.

For direct comparison against Stockfish with a specific net, run self-play with Stockfish as the opponent and pass `EvalFile` through engine options.

## Acceptance Gates

A new NNUE net is a promotion candidate only if all pass:

1. STS non-regression on selected themes at fixed time/depth.
2. Archived self-play win-rate advantage over current baseline.
3. Statistically acceptable p-value threshold (configurable in tune loop).

Suggested first gate:

- STS themes: `STS1,STS2,STS4,STS8,STS9,STS15`
- self-play: at least 80 games at fixed movetime

## Experiment Metadata (Required)

Each trained net should have a sidecar metadata JSON, e.g.:

`nnue/models/syk_v0.meta.json`

Required keys:

- data source glob
- extraction params (`min_ply`, `max_ply`, `sample_rate`, `seed`)
- teacher params (`stockfish`, `depth`, `result_mix`, `eval_scale`)
- training params (`hidden_size`, `epochs`, `batch_size`, `lr`, `weight_decay`, `seed`)
- output net path + sha256
- benchmark links (`history` match IDs, STS summary)

## Near-Term Next Steps

1. Add a script to generate the metadata sidecar automatically.
2. Add a history command to archive NNUE runs and connect them to match IDs.
3. Add quick ablation runner (material-only vs NNUE vs hybrid).
4. Expand trainer to SCReLU and/or deeper head once baseline pipeline is stable.
