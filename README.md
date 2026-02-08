# Sykora

<img src="https://github.com/sb2bg/sykora/blob/main/assets/logo.png" width="200" alt="Sykora Logo">

Sykora is a chess engine written in Zig that implements the Universal Chess Interface (UCI) protocol. It provides a robust and efficient implementation of chess game logic and UCI communication.

## Features

- Currently in progress, full feature list coming soon

## Prerequisites

- [Zig](https://ziglang.org/) compiler (latest stable version recommended)
- A UCI-compatible chess GUI (like Arena, Cutechess, or similar)

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

## Testing

To run the test suite:

```bash
zig build test
```

To run perft validation:

```bash
utils/test/test_perft_suite.sh
```

To run Strategic Test Suite (STS) EPD files (requires `python-chess`):

```bash
utils/sts/sts.py --epd /path/to/sts --pattern "STS*.epd" --engine ./zig-out/bin/sykora --movetime-ms 300
```

To run engine-vs-engine self-play (also requires `python-chess`):

```bash
# Baseline vs candidate, 80 games, 200ms/move, balanced openings
utils/match/selfplay.py ./old_sykora ./zig-out/bin/sykora --name1 old --name2 new --games 80 --movetime-ms 200
```

Useful variants:

```bash
# More stable signal (recommended for commits you may keep)
utils/match/selfplay.py ./old_sykora ./zig-out/bin/sykora --games 200 --movetime-ms 200

# Fixed-depth comparison
utils/match/selfplay.py ./old_sykora ./zig-out/bin/sykora --games 120 --depth 8

# Save all PGNs for manual inspection
utils/match/selfplay.py ./old_sykora ./zig-out/bin/sykora --games 80 --output-dir ./selfplay_pgn
```

The script prints an estimated Elo difference (`candidate - baseline`), a 95% confidence interval, and a p-value versus equal strength.

To compare current working tree against a git ref in one command:

```bash
utils/match/selfplay_vs_ref.sh HEAD~1 --games 120 --movetime-ms 200
utils/match/selfplay_vs_ref.sh v0.2.0 --games 200 --depth 8
```

This wrapper builds both versions in `ReleaseFast`, runs self-play, and prints the same Elo/confidence summary.

For long-term version tracking, use the history ledger:

```bash
# Initialize ledger folders
~/.pyenv/shims/python utils/history/history.py init

# Snapshot current engine build
~/.pyenv/shims/python utils/history/history.py snapshot --label "experiment-a" --notes "describe your change"

# List snapshots, run archived match, recompute global ratings
~/.pyenv/shims/python utils/history/history.py list-engines
~/.pyenv/shims/python utils/history/history.py match <engine_id_A> <engine_id_B> --games 120 --movetime-ms 200
~/.pyenv/shims/python utils/history/history.py ratings --plot
~/.pyenv/shims/python utils/history/history.py sts <engine_id> --movetime-ms 100

# Auto pit strongest vs weakest and render a network graph
~/.pyenv/shims/python utils/history/history.py match-extremes --min-games 20 --games 80 --movetime-ms 120
~/.pyenv/shims/python utils/history/history.py network --top-n 12 --min-games 10 --min-edge-games 2
```

See `history/README.md` for folder schema and full workflow.

One-command tuning loop (STS gate + archived self-play + auto-promotion):

```bash
# First run: create baseline from a known engine binary
~/.pyenv/shims/python utils/tuning/tune_loop.py --bootstrap-baseline-engine old_versions/old_sykora --candidate-label "first-pass"

# Normal run: compare current code to history/current_baseline.txt
~/.pyenv/shims/python utils/tuning/tune_loop.py --candidate-label "eval-tweak" --candidate-notes "describe change"
```

Quick vs serious settings:

```bash
# Quick loop (default): 6 STS themes, 20 self-play games at 80ms/move
~/.pyenv/shims/python utils/tuning/tune_loop.py --candidate-label "quick-iter"

# Serious confirmation before keeping a major change
~/.pyenv/shims/python utils/tuning/tune_loop.py --candidate-label "confirm" --sp-games 120 --sp-movetime-ms 150 --max-p-value 0.2
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
setoption name NnueBlend value 10
isready
```

Current implementation uses a simple Sykora-specific NNUE format (`SYKNNUE1`), not Stockfish `.nnue`.
To use pretrained weights immediately, use a net trained/exported for this format.

Starter training pipeline:

```bash
~/.pyenv/shims/python utils/nnue/extract_positions.py --output nnue/data/positions.jsonl
~/.pyenv/shims/python utils/nnue/label_with_stockfish.py --input nnue/data/positions.jsonl --output nnue/data/labeled.jsonl --depth 8 --result-mix 0.0 --cp-clip 2000
~/.pyenv/shims/python utils/nnue/train_syknnue.py --input nnue/data/labeled.jsonl --output-net nnue/syk_v0.sknnue --augment-mirror
```

To compare Sykora against Stockfish running a specific net:

```bash
~/.pyenv/shims/python utils/match/selfplay.py ./zig-out/bin/sykora /opt/homebrew/bin/stockfish --name1 sykora --name2 stockfish-nnue --engine2-opt EvalFile=nnue/nn-49c1193b131c.nnue --games 40 --movetime-ms 150
```

Some Stockfish builds reject external `EvalFile`; in that case use the default embedded net or a matching Stockfish build.

`train_syknnue.py` defaults to a portable NumPy backend; use `--backend torch` on machines with a stable PyTorch/OpenMP setup.
`NnueBlend` lets you blend NNUE with classical eval (0 = classical only, 100 = pure NNUE).

Full process spec: `specs/nnue_training_spec.md`.

## Documentation

- `engine-interface.md` - Detailed documentation of the engine interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
