# Sykora

[![Lichess rapid rating](https://lichess-shield.vercel.app/api?username=sykorabot&format=rapid)](https://lichess.org/@/sykorabot/perf/rapid)
[![Lichess blitz rating](https://lichess-shield.vercel.app/api?username=sykorabot&format=blitz)](https://lichess.org/@/sykorabot/perf/blitz)
[![Lichess bullet rating](https://lichess-shield.vercel.app/api?username=sykorabot&format=bullet)](https://lichess.org/@/sykorabot/perf/bullet)

<img src="https://github.com/sb2bg/sykora/blob/main/assets/logo.png" width="200" alt="Sykora Logo">

Sykora is a chess engine written in Zig that implements the Universal Chess Interface (UCI) protocol. It provides a robust and efficient implementation of chess game logic and UCI communication.

You can play against the live bot on Lichess: [SykoraBot](https://lichess.org/@/sykorabot).

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

- Classical handcrafted evaluation with:
  - Material and piece-square tables
  - Pawn structure terms (isolated/doubled/backward/passed)
  - Mobility terms
  - King safety and castling terms
  - Endgame mop-up/king activity terms
- Optional NNUE evaluation (`SYKNNUE1` format), including:
  - Classical/NNUE blend (`NnueBlend`)
  - NNUE output scaling (`NnueScale`)
  - Optional SCReLU path (`NnueSCReLU`)

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

Sykora currently exposes the following options:

- `Debug Log File` (string)
- `UseNNUE` (`true`/`false`)
- `EvalFile` (path to `.sknnue`)
- `NnueBlend` (`0..100`)
- `NnueScale` (`10..400`)
- `NnueSCReLU` (`true`/`false`)
- `Threads` (`1..64`)
- `Hash` (`1..4096` MB)

## Prerequisites

- [Zig](https://ziglang.org/) compiler (latest stable version recommended)
- Python 3.10+ (for benchmark, STS, match, history, tuning, NNUE, and bot utilities)
- A UCI-compatible chess GUI (like Arena, Cutechess, or similar)

For Python tooling, install dependencies as needed:

```bash
# Core analysis/benchmark utilities
python3 -m pip install chess

# Lichess bot utilities
python3 -m pip install berserk python-dotenv
```

## Building

To build the project, run:

```bash
zig build
```

This will create an executable named `sykora` in the `zig-out/bin` directory.

## Running

To run the engine:

```bash
zig build run
```

Or directly run the executable:

```bash
./zig-out/bin/sykora
```

## Play On Lichess

Sykora is available on Lichess at [SykoraBot](https://lichess.org/@/sykorabot).

To run your own bot instance against Lichess:

```bash
export LICHESS_API_TOKEN="<your_lichess_bot_token>"
export ENGINE_PATH="./zig-out/bin/sykora"
python3 utils/bot/lichess_bot.py
```

To issue a challenge from your token/account:

```bash
# challenge a specific user
python3 utils/bot/challenge_bot.py some_username --minutes 3 --increment 2

# or pick a random online bot
python3 utils/bot/challenge_bot.py --random-online-bot --minutes 3 --increment 2
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
python3 utils/bench/nps.py --engine ./zig-out/bin/sykora --depth 10
python3 utils/bench/nps.py --engine ./zig-out/bin/sykora --movetime-ms 500 --runs 2
```

To run Strategic Test Suite (STS) EPD files (requires `python-chess`):

```bash
python3 utils/sts/sts.py --epd /path/to/sts --pattern "STS*.epd" --engine ./zig-out/bin/sykora --movetime-ms 300

# Example with NNUE options
python3 utils/sts/sts.py --epd /path/to/sts --engine ./zig-out/bin/sykora --movetime-ms 300 \
  --engine-opt UseNNUE=true --engine-opt EvalFile=nnue/syk_nnue_v11_h128_d10.sknnue --engine-opt NnueBlend=2
```

To run engine-vs-engine self-play (also requires `python-chess`):

```bash
# Baseline vs candidate, 80 games, 200ms/move, balanced openings
python3 utils/match/selfplay.py ./old_sykora ./zig-out/bin/sykora --name1 old --name2 new --games 80 --movetime-ms 200
```

Useful variants:

```bash
# More stable signal (recommended for commits you may keep)
python3 utils/match/selfplay.py ./old_sykora ./zig-out/bin/sykora --games 200 --movetime-ms 200

# Fixed-depth comparison
python3 utils/match/selfplay.py ./old_sykora ./zig-out/bin/sykora --games 120 --depth 8

# Save all PGNs for manual inspection
python3 utils/match/selfplay.py ./old_sykora ./zig-out/bin/sykora --games 80 --output-dir ./selfplay_pgn
```

The script prints an estimated Elo difference (`candidate - baseline`), a 95% confidence interval, and a p-value versus equal strength.

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
python3 utils/history/history.py init

# Snapshot current engine build
python3 utils/history/history.py snapshot --label "experiment-a" --notes "describe your change"

# List snapshots, run archived match, recompute global ratings
python3 utils/history/history.py list-engines
python3 utils/history/history.py match <engine_id_A> <engine_id_B> --games 120 --movetime-ms 200
python3 utils/history/history.py ratings --plot
python3 utils/history/history.py sts <engine_id> --movetime-ms 100

# Auto pit strongest vs weakest and render a network graph
python3 utils/history/history.py match-extremes --min-games 20 --games 80 --movetime-ms 120
python3 utils/history/history.py network --top-n 12 --min-games 10 --min-edge-games 2
```

See `history/README.md` for folder schema and full workflow.

One-command tuning loop (STS gate + archived self-play + auto-promotion):

```bash
# First run: create baseline from a known engine binary
python3 utils/tuning/tune_loop.py --bootstrap-baseline-engine old_versions/old_sykora --candidate-label "first-pass"

# Normal run: compare current code to history/current_baseline.txt
python3 utils/tuning/tune_loop.py --candidate-label "eval-tweak" --candidate-notes "describe change"
```

Quick vs serious settings:

```bash
# Quick loop (default): 6 STS themes, 20 self-play games at 80ms/move
python3 utils/tuning/tune_loop.py --candidate-label "quick-iter"

# Serious confirmation before keeping a major change
python3 utils/tuning/tune_loop.py --candidate-label "confirm" --sp-games 120 --sp-movetime-ms 150 --max-p-value 0.2
```

Legacy wrapper paths under `utils/` are kept for compatibility, but new docs use the canonical subfolders.

## NNUE (Experimental)

Sykora now exposes two UCI options:

- `UseNNUE` (`true`/`false`)
- `EvalFile` (path to network file)

Example UCI sequence:

```text
setoption name EvalFile value /absolute/path/to/net.sknnue
setoption name UseNNUE value true
setoption name NnueBlend value 2
isready
```

Current implementation uses a simple Sykora-specific NNUE format (`SYKNNUE1`), not Stockfish `.nnue`.
To use pretrained weights immediately, use a net trained/exported for this format.

Bullet-first dataset prep (for large Leela/lc0 chunk sets):

```bash
python3 utils/nnue/bullet/prepare_lc0_dataset.py \
  --source-dir /Users/sullivanbognar/Downloads/training-run3--20210605-0521 \
  --output-dir nnue/data/bullet/leela_run3 \
  --manifest-name shards.txt

python3 utils/nnue/bullet/inspect_lc0_v6.py \
  --source-dir /Users/sullivanbognar/Downloads/training-run3--20210605-0521 \
  --max-files 32 --records-per-file 8 \
  --output-json nnue/data/bullet/leela_run3/inspect.json
```

If your Bullet build exports float checkpoints (`.npz`), convert to Sykora net:

```bash
python3 utils/nnue/bullet/export_npz_to_sknnue.py \
  --input nnue/models/bullet/<run_id>/checkpoint.npz \
  --output-net nnue/syk_nnue_<run_id>.sknnue
```

Fallback in-repo trainer pipeline:

```bash
python3 utils/nnue/extract_positions.py --output nnue/data/positions.jsonl
python3 utils/nnue/label_with_stockfish.py --input nnue/data/positions.jsonl --output nnue/data/labeled.jsonl --depth 8 --result-mix 0.0 --cp-clip 2000
python3 utils/nnue/train_syknnue.py --input nnue/data/labeled.jsonl --output-net nnue/syk_v0.sknnue --augment-mirror
```

Fishtest PGN ingestion example:

```bash
# Download .pgn.gz runs (network required)
python3 utils/data/download_fishtest_pgns.py --path datasets/fishtest --time-delta 720 --ltc-only true

# Extract positions directly from .pgn.gz (recursive glob supported)
python3 utils/nnue/extract_positions.py --pgn-glob "datasets/fishtest/**/*.pgn.gz" --output nnue/data/positions.jsonl
```

Optional cp-regression training mode:

```bash
python3 utils/nnue/train_syknnue.py --input nnue/data/labeled.jsonl --output-net nnue/syk_v0_cp.sknnue --target-mode cp --cp-target-key teacher_cp_stm --cp-norm 400 --augment-mirror
```

To compare Sykora against Stockfish running a specific net:

```bash
python3 utils/match/selfplay.py ./zig-out/bin/sykora /opt/homebrew/bin/stockfish --name1 sykora --name2 stockfish-nnue --engine2-opt EvalFile=nnue/nn-49c1193b131c.nnue --games 40 --movetime-ms 150
```

Some Stockfish builds reject external `EvalFile`; in that case use the default embedded net or a matching Stockfish build.

`train_syknnue.py` defaults to a portable NumPy backend; use `--backend torch` on machines with a stable PyTorch/OpenMP setup.
`NnueBlend` lets you blend NNUE with classical eval (0 = classical only, 100 = pure NNUE). Current default is `2`.

Full process spec: `specs/nnue_training_spec.md`.

## Restructuring Roadmap

The project has outgrown a few single large files. A staged refactor is planned to improve maintainability while keeping behavior stable and benchmark-driven.

### Phase 1 (Search split)

- Split `src/search.zig` into focused modules under `src/search/`:
  - `core.zig` (iterative deepening, alpha-beta entry points)
  - `move_picker.zig`
  - `tt.zig`
  - `history.zig` (killer/history/counter tables)
  - `quiescence.zig`
  - `see.zig`
  - `repetition.zig`

### Phase 2 (Board/movegen split)

- Split `src/bitboard.zig` into modules under `src/board/`:
  - `state.zig` (bitboard state + helpers)
  - `attacks.zig` (tables/magic attacks)
  - `movegen.zig`
  - `make_unmake.zig`
  - `perft.zig`
  - `fen_io.zig`

### Phase 3 (UCI/interface split)

- Split `src/interface.zig` into:
  - UCI loop and command dispatch
  - option handlers
  - search thread orchestration / SMP glue

### Safety Rules During Refactor

- Preserve current UCI behavior and option names.
- Require perft suite pass after each phase.
- Require STS/self-play sanity checks before landing large refactors.

## Documentation

- `engine-interface.md` - Detailed documentation of the engine interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
