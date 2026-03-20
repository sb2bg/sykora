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
  - `768 -> Nx2 -> 1` architecture with SCReLU activation
  - Trained on high-depth self-play data via the Bullet trainer
  - Incremental accumulator updates during search
  - Custom `SYKNNUE2` network format with auto-detected activation type
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
- Engine-vs-engine self-play tooling (`utils/match`).
- Long-term experiment history and ratings workflow (`utils/history`).
- Automated tuning loop (`utils/tuning/tune_loop.py`).
- NNUE data prep/training/export pipelines (`utils/nnue`, `utils/data`).
- Lichess play/challenge bot tooling (`utils/bot`).

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
- Python 3.10+ (for benchmark, STS, match, history, tuning, NNUE, and bot utilities)
- A UCI-compatible chess GUI (like Arena, Cutechess, or similar)

For Python tooling, install dependencies as needed:

```bash
# Core analysis/benchmark utilities
python -m pip install chess

# Lichess bot utilities
python -m pip install berserk python-dotenv
```

## Building

To build the project, run:

```bash
zig build
```

This will create an executable named `sykora` in the `zig-out/bin` directory. The embedded NNUE net (`src/net.sknnue`) is compiled into the binary -- no external files needed to play at full strength.

## Running

To run the engine:

```bash
zig build run
```

Or directly run the executable:

```bash
./zig-out/bin/sykora
```

The engine starts with NNUE enabled and the embedded net loaded. No configuration is required for normal use.

## Play On Lichess

Sykora is available on Lichess at [SykoraBot](https://lichess.org/@/sykorabot).

To run your own bot instance against Lichess:

```bash
export LICHESS_API_TOKEN="<your_lichess_bot_token>"
export ENGINE_PATH="./zig-out/bin/sykora"
python utils/bot/lichess_bot.py
```

To issue a challenge from your token/account:

```bash
# challenge a specific user
python utils/bot/challenge_bot.py some_username --minutes 3 --increment 2

# or pick a random online bot
python utils/bot/challenge_bot.py --random-online-bot --minutes 3 --increment 2
```

## Testing

To run the test suite:

```bash
zig build test
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

To run engine-vs-engine self-play (also requires `python-chess`):

```bash
# Baseline vs candidate, 80 games, 200ms/move, balanced openings
python utils/match/selfplay.py ./old_sykora ./zig-out/bin/sykora --name1 old --name2 new --games 80 --movetime-ms 200
```

Useful variants:

```bash
# More stable signal (recommended for commits you may keep)
python utils/match/selfplay.py ./old_sykora ./zig-out/bin/sykora --games 200 --movetime-ms 200

# Fixed-depth comparison
python utils/match/selfplay.py ./old_sykora ./zig-out/bin/sykora --games 120 --depth 8

# Save all PGNs for manual inspection
python utils/match/selfplay.py ./old_sykora ./zig-out/bin/sykora --games 80 --output-dir ./selfplay_pgn
```

The script prints an estimated Elo difference (`candidate - baseline`), a 95% confidence interval, and a p-value versus equal strength.

To run a batched SPRT locally (faster decision-oriented A/B test):

```bash
python utils/match/sprt.py ./old_sykora ./zig-out/bin/sykora \
  --name1 old --name2 new \
  --elo0 -30 --elo1 30 \
  --alpha 0.10 --beta 0.10 \
  --games-per-batch 12 --max-games 360 \
  --movetime-ms 80 --max-plies 220 \
  --threads 1 --hash-mb 64 \
  --shuffle-openings \
  --summary-json sprt_summary.json
```

Exit codes from `sprt.py`:
- `0`: candidate accepted as stronger (or practical stronger threshold reached)
- `1`: candidate accepted as weaker
- `2`: inconclusive at max games
- `3`: runner/configuration failure

The `Release SPRT` workflow (`.github/workflows/sprt.yml`) runs automatically on `v*` tag pushes and compares the new tag against the previous `v*` tag.

Status codes (useful for CI):

- `utils/sts/sts.py`
  - `0`: success
  - `1`: runtime/input error (missing EPD, engine startup failure, parse failure, etc.)
  - `2`: invalid CLI options (for example malformed `--engine-opt`)
- `utils/match/selfplay.py`
  - `0`: candidate score > baseline score
  - `1`: baseline score > candidate score
  - `2`: exact tie
  - `>2`: unexpected failure (engine/protocol/runtime error)

To compare current working tree against your configured baseline in one command:

```bash
# history/current_baseline.txt can contain either:
# 1) a snapshot id (history/engines/<id>/engine), or
# 2) a direct path to a baseline binary
utils/match/selfplay_vs_ref.sh --games 120 --movetime-ms 200

# or override baseline at runtime
utils/match/selfplay_vs_ref.sh --baseline history/engines/<snapshot_id>/engine --games 200 --depth 8
```

This wrapper builds only the current working tree in `ReleaseFast`, runs self-play vs the baseline, and prints the same Elo/confidence summary.

For long-term version tracking, use the history ledger:

```bash
# Initialize ledger folders
python utils/history/history.py init

# Snapshot current engine build
python utils/history/history.py snapshot --label "experiment-a" --notes "describe your change"

# List snapshots, run archived match, recompute global ratings
python utils/history/history.py list-engines
python utils/history/history.py match <engine_id_A> <engine_id_B> --games 120 --movetime-ms 200
python utils/history/history.py ratings --plot
python utils/history/history.py sts <engine_id> --movetime-ms 100

# Auto pit strongest vs weakest and render a network graph
python utils/history/history.py match-extremes --min-games 20 --games 80 --movetime-ms 120
python utils/history/history.py network --top-n 12 --min-games 10 --min-edge-games 2
```

See `history/README.md` for folder schema and full workflow.

One-command tuning loop (STS gate + archived self-play + auto-promotion):

```bash
# First run: create baseline from a known engine binary
python utils/tuning/tune_loop.py --bootstrap-baseline-engine old_versions/old_sykora --candidate-label "first-pass"

# Normal run: compare current code to history/current_baseline.txt
python utils/tuning/tune_loop.py --candidate-label "eval-tweak" --candidate-notes "describe change"
```

Quick vs serious settings:

```bash
# Quick loop (default): 6 STS themes, 20 self-play games at 80ms/move
python utils/tuning/tune_loop.py --candidate-label "quick-iter"

# Serious confirmation before keeping a major change
python utils/tuning/tune_loop.py --candidate-label "confirm" --sp-games 120 --sp-movetime-ms 150 --max-p-value 0.2
```

## NNUE

Sykora uses a `768 -> Nx2 -> 1` NNUE architecture with dual-perspective accumulator updates and SCReLU activation, trained via the [Bullet](https://github.com/jw1912/bullet) trainer.

### How It Works

- The embedded net (`src/net.sknnue`) is compiled into the binary and loaded automatically at startup.
- NNUE is enabled by default (`UseNNUE = true`, `NnueBlend = 100`).
- The activation function (ReLU or SCReLU) is stored in the network file header and auto-detected on load.
- To use a different net, set `EvalFile` to the path of an external `.sknnue` file.
- To blend NNUE with classical eval, lower `NnueBlend` (e.g., `50` for 50/50, `0` for classical only).

### Network Format (SYKNNUE2)

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

All values are little-endian.

### Training Pipeline

Training uses the [Bullet](https://github.com/jw1912/bullet) NNUE trainer with CUDA. Two data formats are supported:

**Using binpack data:**

```bash
python utils/nnue/bullet/train_cuda_longrun.py \
  --dataset data/training.binpack \
  --data-format binpack \
  --bullet-repo nnue/bullet_repo \
  --output-root nnue/models/bullet \
  --hidden 256 --end-superbatch 320 --threads 8
```

**Using BulletFormat .data files:**

```bash
python utils/nnue/bullet/train_cuda_longrun.py \
  --dataset nnue/data/bullet/train/train_main.data \
  --bullet-repo nnue/bullet_repo \
  --output-root nnue/models/bullet \
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

### Embedding a New Net

To update the embedded net in the binary:

```bash
cp output.sknnue src/net.sknnue
zig build
```

### Gating Checkpoints

```bash
python utils/nnue/bullet/gate_checkpoints.py \
  --checkpoints-dir nnue/models/bullet/<run_id>/checkpoints \
  --engine ./zig-out/bin/sykora \
  --blend 100 --nnue-scale 100 \
  --sts-epd epd --sts-movetime-ms 40 --sts-max-positions 400 \
  --selfplay-games 80 --selfplay-movetime-ms 120 --selfplay-top-k 3 \
  --threads 1 --hash-mb 64 \
  --min-elo 0 --max-p-value 0.25 \
  --promote-to nnue/syk_nnue_best.sknnue
```

Full process spec: `specs/nnue_training_spec.md`.

## Documentation

- `engine-interface.md` - Detailed documentation of the engine interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
