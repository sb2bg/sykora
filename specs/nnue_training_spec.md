# Sykora NNUE Training Spec (v1, Bullet-First)

## Scope

This spec defines the target NNUE workflow for Sykora:

- Use high-quality large datasets (Leela/lc0 + self-play data) instead of tiny ad-hoc samples.
- Train with Bullet on stronger hardware.
- Export into Sykora's native `SYKNNUE1` format for engine use.
- Archive every run with reproducible metadata and promotion gates.

## Repository Layout

Core utilities:

- `utils/nnue/extract_positions.py`: PGN -> JSONL extractor (fallback pipeline).
- `utils/nnue/label_with_stockfish.py`: Stockfish teacher labels for fallback pipeline.
- `utils/nnue/train_syknnue.py`: in-repo baseline trainer.
- `utils/nnue/common.py`: `SYKNNUE1` constants + writer.

Bullet pipeline helpers:

- `utils/nnue/bullet/prepare_lc0_dataset.py`: validate/manifest lc0 shard sets.
- `utils/nnue/bullet/inspect_lc0_v6.py`: sample-inspect lc0 v6 records.
- `utils/nnue/bullet/export_npz_to_sknnue.py`: convert float checkpoint (`.npz`) -> `SYKNNUE1`.

Artifacts live under `nnue/`:

- `nnue/data/bullet/*`: manifests and dataset summaries.
- `nnue/models/*`: training metadata/checkpoints (if kept locally).
- `nnue/*.sknnue`: Sykora-ready nets.

## Data Sources

### A) Leela/lc0 training chunks (preferred scale)

The provided `training.*.gz` files are lc0 binary chunks with fixed 8356-byte records (v6 format).

Prepare and validate:

```bash
~/.pyenv/shims/python utils/nnue/bullet/prepare_lc0_dataset.py \
  --source-dir /Users/sullivanbognar/Downloads/training-run3--20210605-0521 \
  --output-dir nnue/data/bullet/leela_run3 \
  --manifest-name shards.txt
```

Optional deep inspection on a sample:

```bash
~/.pyenv/shims/python utils/nnue/bullet/inspect_lc0_v6.py \
  --source-dir /Users/sullivanbognar/Downloads/training-run3--20210605-0521 \
  --max-files 32 \
  --records-per-file 8 \
  --output-json nnue/data/bullet/leela_run3/inspect.json
```

### B) Fishtest/engine PGNs (fallback + augmentation)

```bash
~/.pyenv/shims/python utils/nnue/extract_positions.py \
  --pgn-glob "datasets/fishtest/**/*.pgn.gz" \
  --output nnue/data/positions.jsonl

~/.pyenv/shims/python utils/nnue/label_with_stockfish.py \
  --input nnue/data/positions.jsonl \
  --output nnue/data/labeled.jsonl \
  --stockfish /opt/homebrew/bin/stockfish \
  --depth 12 \
  --result-mix 0.0 \
  --cp-clip 2000
```

## Bullet Training Pipeline

### Step 1: Ensure dataset format compatibility

Bullet commonly trains from `sfbinpack`, `bulletformat`, `montyformat`, or `viriformat` datasets.
If your Bullet build cannot ingest lc0 v6 chunks directly, convert lc0 shards first using your chosen converter.

Store conversion outputs and metadata under:

- `nnue/data/bullet/<dataset_id>/`

### Step 2: Train with Bullet (external trainer repo/tooling)

Run training on your stronger machine using the prepared dataset manifest. Keep run artifacts under:

- `nnue/models/bullet/<run_id>/`

Required run metadata (JSON sidecar):

- trainer commit/version
- dataset ID and manifest path
- architecture definition (inputs/layers/activations)
- optimizer + LR schedule
- batch size, epochs/steps, hardware
- loss curves and final validation metrics
- exported checkpoint path and sha256

### Step 3: Export to Sykora format

Once Bullet (or your export script) emits float parameters in NPZ form:

- `input_weights` shape `[768, hidden]`
- `input_bias` shape `[hidden]`
- `output_weights` shape `[2*hidden]` or `[2, hidden]`
- `output_bias` scalar

Convert to engine net:

```bash
~/.pyenv/shims/python utils/nnue/bullet/export_npz_to_sknnue.py \
  --input nnue/models/bullet/<run_id>/checkpoint.npz \
  --output-net nnue/syk_nnue_<run_id>.sknnue
```

## Engine Integration Contract

UCI setup:

```text
setoption name EvalFile value /absolute/path/to/net.sknnue
setoption name UseNNUE value true
setoption name NnueBlend value 2
isready
```

Notes:

- Sykora does not load Stockfish `.nnue` directly.
- `NnueBlend=100` means pure NNUE; keep this for final strength checks.
- For development, test both hybrid (`1..10`) and pure (`100`) to separate eval quality from search-speed effects.

## Acceptance Gates

Promote a net only if all pass:

1. STS non-regression on fixed subset and full suite sanity pass.
2. Self-play win-rate vs current baseline with confidence interval.
3. Pure NNUE (`NnueBlend=100`) does not collapse vs classical baseline.

Suggested minimum confirmation:

- STS full suite at fixed budget.
- 200+ game self-play at one short TC and one longer TC.

## Historical Logging (Required)

Every candidate must be archived through `utils/history/history.py`:

- snapshot engine
- run match(es)
- recompute ratings
- archive STS output

This keeps a connected graph of versions, Elo estimates, and eval progress over time.
