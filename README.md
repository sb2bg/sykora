# Sykora

[![Lichess bullet rating](https://lichess-shield.vercel.app/api?username=sykorabot&format=bullet)](https://lichess.org/@/sykorabot/perf/bullet)
[![Lichess blitz rating](https://lichess-shield.vercel.app/api?username=sykorabot&format=blitz)](https://lichess.org/@/sykorabot/perf/blitz)
[![Lichess rapid rating](https://lichess-shield.vercel.app/api?username=sykorabot&format=rapid)](https://lichess.org/@/sykorabot/perf/rapid)
[![Release SPRT](https://github.com/sb2bg/sykora/actions/workflows/sprt.yml/badge.svg)](https://github.com/sb2bg/sykora/actions/workflows/sprt.yml)

<img src="https://github.com/sb2bg/sykora/blob/main/assets/logo.png" width="200" alt="Sykora Logo">

Sykora is a UCI chess engine written from scratch in Zig. It features magic bitboard move generation, a full alpha-beta search with LMR/null-move/futility pruning, Lazy SMP parallel search, a hand-tuned classical evaluation, and NNUE evaluation trained via the [Bullet](https://github.com/jw1912/bullet) trainer. An NNUE net is embedded in the binary and enabled by default. Sykora plays live on Lichess as [SykoraBot](https://lichess.org/@/sykorabot).

## Features

### Engine Core

- Bitboard-based board representation with fast occupancy/piece set operations.
- Precomputed attack tables for king/knight/pawn moves.
- Magic bitboards for rook and bishop attacks (queen attacks via composition).
- Full legal move generation including castling, en passant, promotions, and check legality filtering.
- Incremental make/unmake move pipeline with Zobrist hashing.
- Polyglot-compatible en-passant hash handling.

### Search

- Negamax alpha-beta search with iterative deepening.
- Aspiration windows around prior iteration score.
- Principal Variation Search (PVS).
- Transposition table with aging and depth-preferred replacement.
- Move ordering pipeline:
  - TT move
  - SEE-scored captures
  - Killer moves
  - History and counter-move scoring for quiets
  - Deferred bad captures
- Pruning and reduction framework:
  - Mate distance pruning
  - Check extension
  - Null-move pruning with verification at higher depths
  - Reverse futility pruning
  - Futility pruning
  - Razoring
  - Late Move Reductions (LMR)
  - Late Move Pruning (LMP)
- Quiescence search with:
  - Check evasions when in check
  - Delta pruning
  - SEE-based pruning of clearly losing captures
- Repetition detection with contempt shaping and cycle penalties.
- 50-move-rule handling.
- Basic time management for clocked play.
- Evaluation cache for expensive eval paths.

### Evaluation

- **NNUE evaluation** (default, embedded in binary):
  - `SYKNNUE3` net generation with backward-compatible `SYKNNUE2` loading
  - Legacy `768 -> Nx2 -> 1` and mirrored king-bucketed sparse-input nets
  - SCReLU activation with incremental accumulators during search
  - Trained on high-depth self-play data via the Bullet trainer
  - King-bucket training path via `nnue/bullet_repo/examples/sykora_bucketed.rs`
  - Blendable with classical eval via `NnueBlend` (default: 100 = pure NNUE)
- **Classical handcrafted evaluation** (fallback):
  - Material and piece-square tables
  - Pawn structure terms (isolated/doubled/backward/passed)
  - Mobility terms
  - King safety and castling terms
  - Endgame mop-up/king activity terms

### Parallel Search

- Lazy SMP-style parallel search with helper threads.
- Shared transposition table across threads.
- Best-move voting across main/helper results.

### UCI and Developer Commands

- Standard UCI command support (`uci`, `isready`, `position`, `go`, `stop`, `setoption`, `ucinewgame`, `quit`).
- `display` helper command for board/FEN/hash inspection.
- `perft` helper command with:
  - Fast node count mode
  - `stats` mode (captures, checks, promotions, mates, etc.)
  - `divide` mode

### Tooling

- Perft and movegen shell tests (`utils/test`).
- NPS benchmarking (`utils/bench/nps.py`).
- STS runner (`utils/sts/sts.py`).
- Engine-vs-engine runners (`utils/match`, used internally by `utils/history`).
- Long-term archived experiment history, SPRT, and ratings workflow (`utils/history`).
- NNUE data prep/training/export pipelines (`utils/nnue`, `utils/data`).

## UCI Options

| Option           | Type   | Default | Description                                                |
| ---------------- | ------ | ------- | ---------------------------------------------------------- |
| `Debug Log File` | string | `""`    | Path for debug logging                                     |
| `UseNNUE`        | bool   | `true`  | Enable NNUE evaluation (embedded net loads automatically)  |
| `EvalFile`       | string | `""`    | Path to external `.sknnue` file (overrides embedded net)   |
| `NnueBlend`      | int    | `100`   | NNUE/classical blend (0 = classical only, 100 = pure NNUE) |
| `NnueScale`      | int    | `100`   | NNUE output scaling factor (10..400)                       |
| `Threads`        | int    | `1`     | Search threads (1..64, Lazy SMP)                           |
| `Hash`           | int    | `64`    | Transposition table size in MB (1..4096)                   |

The activation function (ReLU or SCReLU) is auto-detected from the network file header -- no manual configuration needed.

## Prerequisites

- [Zig](https://ziglang.org/) compiler (latest stable version recommended)
- Python 3.10+ (for benchmark, STS, match, history, NNUE, and bot utilities)
- A UCI-compatible chess GUI (like Arena, Cutechess, or similar)

For Python tooling, install dependencies as needed:

```bash
# Core analysis/benchmark utilities
python -m pip install chess
```

For NNUE training on a fresh machine, bootstrap the Bullet dependency once:

```bash
python3 utils/nnue/bullet/bootstrap.py --build-utils
```

The tracked training runner lives under `utils/nnue/bullet_runner/`; datasets, checkpoints, and Bullet build output remain untracked.

## Building

To build the project, run:

```bash
zig build -Doptimize=ReleaseFast
```

This will create an executable named `sykora` in the `zig-out/bin` directory. The embedded NNUE net (`src/net.sknnue`) is compiled into the binary -- no external files needed to play at full strength.

## Running

To run the engine:

```bash
zig build -Doptimize=ReleaseFast run
```

Or directly run the executable:

```bash
./zig-out/bin/sykora
```

The engine starts with NNUE enabled and the embedded net loaded. No configuration is required for normal use.

## Play On Lichess

Sykora is available on Lichess at [SykoraBot](https://lichess.org/@/sykorabot).

## Testing

To run the test suite:

```bash
zig build -Doptimize=ReleaseFast test
```

To run perft validation:

```bash
utils/test/test_perft_suite.sh
```

To benchmark search speed (NPS):

```bash
python utils/bench/nps.py --engine ./zig-out/bin/sykora --depth 10
python utils/bench/nps.py --engine ./zig-out/bin/sykora --movetime-ms 500 --runs 2
```

To run Strategic Test Suite (STS) EPD files (requires `python-chess`):

```bash
python utils/sts/sts.py --epd /path/to/sts --pattern "STS*.epd" --engine ./zig-out/bin/sykora --movetime-ms 300
```

STS is still available as a diagnostic tool, but it is not the recommended promotion signal for NNUE changes.

The canonical engine-vs-engine workflow is the archived `history.py` flow:

```bash
# Initialize ledger folders
python utils/history/history.py init

# Snapshot two builds you want to compare
python utils/history/history.py snapshot --engine ./old_sykora --label "baseline" --engine-id baseline
python utils/history/history.py snapshot --engine ./zig-out/bin/sykora --label "candidate" --engine-id candidate

# Archived fixed-game selfplay
python utils/history/history.py list-engines
python utils/history/history.py selfplay baseline candidate --games 120 --movetime-ms 200

# Archived SPRT
python utils/history/history.py sprt baseline candidate \
  --elo0 -30 --elo1 30 \
  --games-per-batch 12 --max-games 360 \
  --movetime-ms 80 --max-plies 220 \
  --threads 1 --hash-mb 64 --shuffle-openings

# Ratings and graph data are built from archived selfplay only
python utils/history/history.py ratings --plot
python utils/history/history.py network --top-n 12 --min-games 10 --min-edge-games 2

# Optional: diagnostic STS run for a snapshot
python utils/history/history.py sts candidate --movetime-ms 100
```

Every archived selfplay/SPRT run writes settings, summary JSON, stdout/stderr logs, and reproducibility metadata under `history/`.

The `Release SPRT` workflow (`.github/workflows/sprt.yml`) uses the same archived `history.py sprt` path.

Low-level runners under `utils/match/` remain available, but they are implementation details rather than the recommended user/agent entrypoints.

See `history/README.md` for folder schema and the archived workflow.

## NNUE

Sykora supports both legacy `768 -> Nx2 -> 1` nets and mirrored king-bucketed `SYKNNUE3` nets with dual-perspective accumulator updates and SCReLU activation, trained via the [Bullet](https://github.com/jw1912/bullet) trainer. The current embedded net in `src/net.sknnue` is a `SYKNNUE3` mirrored king-bucketed net with 10 buckets and hidden size 512, equivalent to `7680 -> 512x2 -> 1`.

### How It Works

- The embedded net (`src/net.sknnue`) is compiled into the binary and loaded automatically at startup.
- NNUE is enabled by default (`UseNNUE = true`, `NnueBlend = 100`).
- The activation function (ReLU or SCReLU) is stored in the network file header and auto-detected on load.
- To use a different net, set `EvalFile` to the path of an external `.sknnue` file.
- `NnueBlend` remains available for fallback/debugging, but the primary workflow is pure NNUE.

### Network Formats (`SYKNNUE2` / `SYKNNUE3`)

New tooling writes `SYKNNUE3` only. `SYKNNUE2` remains loadable for backward compatibility with older external nets.

Legacy `SYKNNUE2`:

```
8 bytes   magic: "SYKNNUE2"
u16       version: 2
u16       hidden_size
u8        activation_type (0=ReLU, 1=SCReLU)
i32       output_bias
i16[hidden_size]                accumulator biases
i16[768 * hidden_size]          input -> accumulator weights
i16[2 * hidden_size]            output weights (stm half, nstm half)
```

Bucketed `SYKNNUE3`:

```
8 bytes   magic: "SYKNNUE3"
u16       version: 3
u8        feature_set (0=legacy_psqt, 1=king_buckets_mirrored)
u16       hidden_size
u8        activation_type (0=ReLU, 1=SCReLU)
u16       bucket_count
u8[64]    bucket layout by king square
i32       output_bias
i16[hidden_size]                         accumulator biases
i16[input_size * hidden_size]            input -> accumulator weights
i16[2 * hidden_size]                     output weights (stm half, nstm half)
```

For `king_buckets_mirrored`, `input_size = 768 * bucket_count`. The current embedded net uses `bucket_count = 10`, so its effective input size is `7680`. For `legacy_psqt`, `SYKNNUE3` uses `bucket_count = 1` with a zero bucket layout, so old 768-input nets can be represented without using `SYKNNUE2`. All values are little-endian.

### Training Pipeline

Training uses the [Bullet](https://github.com/jw1912/bullet) NNUE trainer with CUDA. Two data formats are supported. `SYKNNUE3` remains the default training format; `SYKNNUE4` training/export support is now available behind `--network-format syk4`.

**Using binpack data:**

```bash
python utils/nnue/bullet/train_cuda_longrun.py \
  --dataset data/training.binpack \
  --data-format binpack \
  --bullet-repo nnue/bullet_repo \
  --output-root nnue/models/bullet \
  --network-format syk3 \
  --hidden 256 --end-superbatch 320 --threads 8
```

**Using BulletFormat .data files:**

```bash
python utils/nnue/bullet/train_cuda_longrun.py \
  --dataset nnue/data/bullet/train/train_main.data \
  --bullet-repo nnue/bullet_repo \
  --output-root nnue/models/bullet \
  --network-format syk3 \
  --hidden 256 --end-superbatch 320 --threads 8
```

**Multiple datasets** can be passed space-separated:

```bash
python utils/nnue/bullet/train_cuda_longrun.py \
  --dataset data/set1.binpack data/set2.binpack \
  --data-format binpack \
  ...
```

**Resuming from a checkpoint:**

```bash
python utils/nnue/bullet/train_cuda_longrun.py \
  --dataset data/test80.binpack \
  --data-format binpack \
  --resume nnue/models/bullet/<run_id>/checkpoints/<checkpoint>/raw.bin \
  --start-superbatch 161 --end-superbatch 320 \
  ...
```

**Training a SYKNNUE4 baseline:**

```bash
python utils/nnue/bullet/train_cuda_longrun.py \
  --dataset data/training.binpack \
  --data-format binpack \
  --network-format syk4 \
  --bucket-layout sykora16 \
  --hidden 1536 \
  --dense-l1 16 --dense-l2 32 \
  --end-superbatch 320 --threads 8
```

### Self-Play Data Generation

Sykora can generate its own training data via the `gensfen` command:

```bash
# Generate positions using the engine's own evaluation
./zig-out/bin/sykora gensfen output.data depth 7 count 1000000 threads 4
```

### Exporting a Trained Net

Convert a Bullet checkpoint to `.sknnue`:

```bash
# From quantised.bin (default after Bullet training)
python utils/nnue/bullet/bullet_quantised_to_sknnue.py \
  nnue/models/bullet/<run_id>/checkpoints/<checkpoint>/quantised.bin \
  output.sknnue --activation screlu

# From NPZ export
python utils/nnue/bullet/export_npz_to_sknnue.py \
  checkpoint.npz output.sknnue --activation screlu
```

Export a `SYKNNUE4` checkpoint:

```bash
python utils/nnue/bullet/checkpoint_raw_to_npz.py \
  --input nnue/models/bullet/<run_id>/checkpoints/<checkpoint> \
  --output checkpoint_syk4.npz

python utils/nnue/bullet/export_npz_to_syk4.py \
  --input checkpoint_syk4.npz \
  --output-net output.sknnue
```

`SYKNNUE4` engine-side loading/eval is still separate work. The new tooling here covers training metadata, checkpoint export, and file generation only.

### Embedding a New Net

To update the embedded net in the binary:

```bash
cp output.sknnue src/net.sknnue
zig build -Doptimize=ReleaseFast
```

### Gating Checkpoints

```bash
python utils/nnue/bullet/gate_checkpoints.py \
  --checkpoints-dir nnue/models/bullet/<run_id>/checkpoints \
  --engine ./zig-out/bin/sykora \
  --blend 100 --nnue-scale 100 \
  --selfplay-games 80 --selfplay-movetime-ms 120 --selfplay-top-k 3 \
  --threads 1 --hash-mb 64 \
  --min-elo 0 --max-p-value 0.25 \
  --promote-to nnue/syk_nnue_best.sknnue
```

This gate now evaluates recent checkpoints by selfplay only. STS is intentionally not part of the checkpoint promotion path.

SYKNNUE4 design spec: `specs/syknnue4_spec.md`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
