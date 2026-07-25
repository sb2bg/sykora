# Sykora NNUE training

`launch_training.ps1` accepts a local dataset root instead of naming datasets
in tracked source. The root must use one of these layouts:

```text
dataset-root/
  train/
    shard-001.binpack
    shard-002.binpack
  validation/
    held-out-001.binpack
```

```text
dataset-root/
  train/
    shard-001.data
    shard-002.data
  validation/
    held-out.data
```

Directories are scanned recursively and sorted by full path. A root must use
only Stockfish `.binpack` files or only BulletFormat `.data` files; the
launcher rejects mixed formats. BulletFormat validation currently requires
exactly one `.data` file. Validation positions must come from games that do
not appear in the training set.

The data root, generated validation cache, checkpoints, and `run_meta.json`
are local artifacts. They are ignored under `nnue/` when stored in the
repository, but metadata should still be kept private if it contains local
paths that should not be published.

## Fine-tuning

Fine-tuning requires a full-precision checkpoint directory containing
`optimiser_state/weights.bin`, `momentum.bin`, and `velocity.bin`. An exported
`.sknnue` file cannot be resumed.

```powershell
$resume = Resolve-Path `
  ".\nnue\models\bullet\<run>\checkpoints\<run>-800"

.\launch_training.ps1 `
  -Stage finetune `
  -DataDir "D:\nnue-data\fine-tune" `
  -Resume $resume `
  -Superbatches 200 `
  -Wdl 0.25 `
  -DryRun
```

Inspect the dry run, then repeat without `-DryRun`. By default, `finetune`
trains for 200 superbatches and restarts cosine decay at the inferred next
superbatch, from `0.0001` to `0.00001`. Override these with `-Superbatches`,
`-LrStart`, and `-LrFinal`.

Keep the WDL proportion used by the source run for the first data-only
experiment. Read it from the source run metadata:

```powershell
$meta = Get-Content `
  ".\nnue\models\bullet\<run>\run_meta.json" |
  ConvertFrom-Json
$meta.training.wdl
```

The launcher resumes both weights and optimiser state. It numbers new
checkpoints after the resumed checkpoint, validates every saved checkpoint,
and exports the checkpoint with the best held-out validation loss.

## Preparing BulletFormat data

`utils/nnue/bullet/pack_dataset.py` shuffles and interleaves text or existing
BulletFormat inputs into `.data`. This is not a converter to Stockfish
`.binpack`; no conversion is necessary because the launcher supports both
formats.

```powershell
python .\utils\nnue\bullet\pack_dataset.py `
  --data-input "D:\source-data\train-*.data" `
  --output "D:\nnue-data\fine-tune\train\mixed.data"
```

Build validation data separately from completely held-out games and place the
single resulting file under `validation/`.
