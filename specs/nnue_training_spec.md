# Sykora NNUE Training Spec (Bullet Production Pipeline)

## Goal

Replace the legacy small-data NumPy training path with a Bullet-first pipeline that:

- scales to large datasets,
- runs long productive sessions on RTX 4070 Ti SUPER class GPUs,
- promotes only checkpoints that pass STS/self-play gates,
- keeps run metadata reproducible.

## What Was Removed

The legacy fallback trainer/data scripts were removed:

- `utils/nnue/train_syknnue.py`
- `utils/nnue/extract_positions.py`
- `utils/nnue/label_with_stockfish.py`

Do not use that path anymore.

## Canonical Pipeline

### 1) Build teacher-labeled Bullet text from PGNs

Use this for your own/self-play PGNs (augmentation path):

```bash
python utils/nnue/bullet/make_teacher_text_dataset.py \
  --pgn-glob "history/matches/**/*.pgn" \
  --pgn-glob "datasets/fishtest/**/*.pgn.gz" \
  --output nnue/data/bullet/train/teacher_text.txt \
  --stockfish /path/to/stockfish \
  --depth 12 \
  --threads 1 \
  --hash-mb 256 \
  --sample-rate 0.2 \
  --min-ply 12 \
  --max-ply 220 \
  --skip-check \
  --skip-captures \
  --cp-clip 2500
```

### 2) Pack into BulletFormat (`.data`)

```bash
python utils/nnue/bullet/pack_dataset.py \
  --text-input nnue/data/bullet/train/teacher_text.txt \
  --output nnue/data/bullet/train/train_main.data \
  --shuffle-mem-mb 4096 \
  --convert-threads 8
```

You can mix external `.data` shards with your own via repeated `--data-input`.

### 3) Train with Bullet (4070 tuned defaults)

```bash
python utils/nnue/bullet/train_cuda_longrun.py \
  --dataset nnue/data/bullet/train/train_main.data \
  --bullet-repo nnue/bullet_repo \
  --output-root nnue/models/bullet \
  --hidden 256 \
  --end-superbatch 320 \
  --threads 8
```

This writes run metadata + checkpoints under `nnue/models/bullet/<run_id>/`.

Apple Silicon / CPU sanity run (pipeline validation, not final strength):

```bash
python utils/nnue/bullet/train_apple_silicon_test.py \
  --dataset nnue/data/bullet/train/train_main.data \
  --bullet-repo nnue/bullet_repo \
  --output-root nnue/models/bullet_cpu_test \
  --hidden 64 \
  --end-superbatch 1 \
  --threads 6
```

### 4) Gate checkpoints and promote

```bash
python utils/nnue/bullet/gate_checkpoints.py \
  --checkpoints-dir nnue/models/bullet/<run_id>/checkpoints \
  --engine ./zig-out/bin/sykora \
  --blend 2 \
  --nnue-scale 100 \
  --sts-epd epd \
  --sts-movetime-ms 40 \
  --sts-max-positions 400 \
  --selfplay-games 80 \
  --selfplay-movetime-ms 120 \
  --selfplay-top-k 3 \
  --threads 1 \
  --hash-mb 64 \
  --min-elo 0 \
  --max-p-value 0.25 \
  --promote-to nnue/syk_nnue_best.sknnue
```

## Data Strategy (Important)

- External large datasets should be the backbone.
- Your own engine positions are useful as augmentation, not sole source.
- If you use own positions, always teacher-label with stronger analysis before training.
- Prefer quality and diversity over raw volume from one weak generator.

## 4070 Ti SUPER Notes

- Bullet CUDA backend is the intended path.
- Keep batch/schedule long and checkpoint frequently (`save_rate=1`) for gate-driven progress.
- Avoid architecture inflation before engine-side incremental NNUE updates exist.

## Promotion Rule

Only keep nets that pass both:

1. STS non-regression (or improvement), and
2. self-play confidence gate (Elo/p-value thresholds).

Anything else is experimentation, not promotion.
