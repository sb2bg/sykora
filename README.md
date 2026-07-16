<div align="center">

<img src="https://github.com/sb2bg/sykora/blob/main/assets/logo.png" width="180" alt="Sykora Logo">

# Sykora

**A UCI chess engine written from scratch in Zig.**

Magic bitboards, alpha-beta with modern pruning and reductions, Lazy SMP, embedded NNUE trained via [Bullet](https://github.com/jw1912/bullet).

[![Sykora CCRL badge](https://ccrl-badges.vercel.app/badge?engine=Sykora&list=blitz&showList=false&showRank=true)](<https://computerchess.org.uk/ccrl/404/cgi/engine_details.cgi?match_length=30&print=Details+(text)&eng=Sykora%200.2.2%2064-bit#Sykora_0_2_2_64-bit>)
[![Lichess bullet rating](https://lichess-shield.vercel.app/api?username=sykorabot&format=bullet)](https://lichess.org/@/sykorabot/perf/bullet)
[![Lichess blitz rating](https://lichess-shield.vercel.app/api?username=sykorabot&format=blitz)](https://lichess.org/@/sykorabot/perf/blitz)
[![Lichess rapid rating](https://lichess-shield.vercel.app/api?username=sykorabot&format=rapid)](https://lichess.org/@/sykorabot/perf/rapid)
[![Release SPRT](https://github.com/sb2bg/sykora/actions/workflows/sprt.yml/badge.svg)](https://github.com/sb2bg/sykora/actions/workflows/sprt.yml)

[Play live on Lichess →](https://lichess.org/@/sykorabot)

</div>

---

## Quick Start

Requires [Zig](https://ziglang.org/) `0.15.2`.

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/sykora
```

Engine is compatible with any UCI GUI (Arena, Cutechess, etc.)

## Strength

Sykora is tested by [CCRL](https://computerchess.org.uk/ccrl/404/). Current entries:

| Version               | CCRL Rating | Rank  | Elo vs `0.1.0` |
| --------------------- | ----------- | ----- | -------------- |
| `Sykora 0.2.2 64-bit` | `3240`      | `163` | `+872`         |
| `Sykora 0.2.1 64-bit` | N/A         | N/A   | N/A            |
| `Sykora 0.1.0 64-bit` | `2368`      | `423` | baseline       |

## Features

<details>
<summary><b>Engine Core</b>: bitboards, magic move generation, Zobrist hashing</summary>

- Bitboard-based board representation with fast occupancy/piece set operations.
- Precomputed attack tables for king/knight/pawn moves.
- Magic bitboards for rook and bishop attacks (queen attacks via composition).
- Full legal move generation including castling, en passant, promotions, and check legality filtering.
- Incremental make/unmake move pipeline with Zobrist hashing.
- Polyglot-compatible en-passant hash handling.

</details>

<details>
<summary><b>Search</b>: PVS, aspiration windows, full pruning/reduction stack</summary>

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
- Repetition detection.
- 50-move-rule handling.
- Clock-aware time management with soft iteration budgets and absolute hard deadlines.
- Evaluation cache for expensive eval paths.

</details>

<details>
<summary><b>Evaluation</b>: NNUE (default) with classical fallback</summary>

- **NNUE evaluation** (default, embedded in binary):
  - `SYKNNUE7` pairwise MLP: factorised mirrored king-bucketed sparse inputs (10 buckets), H=1024 feature transformer, and eight material heads
  - Pairwise product pooling with a CReLU/CSReLU dense tail and incremental accumulators during search
  - Factorised training: shared `768 → H` factoriser merged into per-bucket weights at export
  - Trained on Stockfish binpack data via the Bullet trainer
  - Blendable with classical eval via `NnueBlend` (default: `100` = pure NNUE)
- **Classical handcrafted evaluation** (fallback):
  - Material and piece-square tables
  - Pawn structure terms (isolated/doubled/backward/passed)
  - Mobility terms
  - King safety and castling terms
  - Endgame mop-up/king activity terms

</details>

<details>
<summary><b>Parallel Search</b>: Lazy SMP with shared TT</summary>

- Lazy SMP-style parallel search with helper threads.
- Shared transposition table across threads.
- Best-move voting across main/helper results.

</details>

<details>
<summary><b>UCI and Developer Commands</b></summary>

- Standard UCI command support (`uci`, `isready`, `position`, `go`, `stop`, `setoption`, `ucinewgame`, `quit`).
- `display` helper command for board/FEN/hash inspection.
- `perft` helper command with:
  - Fast node count mode
  - `stats` mode (captures, checks, promotions, mates, etc.)
  - `divide` mode

</details>

<details>
<summary><b>Tooling</b>: perft, NPS, STS, SPRT, OpenBench, ratings, NNUE pipelines</summary>

- Perft and movegen shell tests (`utils/test`).
- NPS benchmarking (`utils/bench/nps.py`).
- STS runner (`utils/sts/sts.py`).
- Engine-vs-engine runners (`utils/match`, used internally by `utils/history`).
- OpenBench build, benchmark, SPSA input, and instance scaffolding (`utils/openbench`).
- Long-term archived experiment history, SPRT, and ratings workflow (`utils/history`).
- NNUE data prep/training/export pipelines (`utils/nnue`, `utils/data`).

</details>

## UCI Options

| Option           | Type   | Default   | Description                                                |
| ---------------- | ------ | --------- | ---------------------------------------------------------- |
| `Debug Log File` | string | `<empty>` | Path for debug logging                                     |
| `UseNNUE`        | bool   | `true`    | Enable NNUE evaluation (embedded net loads automatically)  |
| `EvalFile`       | string | `<empty>` | Path to external `.sknnue` file (overrides embedded net)   |
| `NnueBlend`      | int    | `100`     | NNUE/classical blend (0 = classical only, 100 = pure NNUE) |
| `NnueScale`      | int    | `100`     | NNUE output scaling factor (10..400)                       |
| `Threads`        | int    | `1`       | Search threads (1..64, Lazy SMP)                           |
| `Hash`           | int    | `128`     | Transposition table size in MB (1..4096)                   |
| `Move Overhead`  | int    | `30`      | Clock reserve in milliseconds (0..5000)                    |
| `LMRScale`       | int    | `109`     | Fixed-point LMR scale for OpenBench tuning (100 = 1.00)    |
| `LMRHistoryScale`| int    | `101`     | Fixed-point LMR history influence (100 = 1.00)             |
| `LMPMoveScale`   | int    | `94`      | Fixed-point LMP move-count scale (100 = 1.00)              |
| `HistoryMaxBonus`| int    | `380`     | Maximum quiet/continuation history update                  |

The activation function (ReLU or SCReLU) is auto-detected from the network file header.

## Prerequisites

- [Zig](https://ziglang.org/) compiler (`0.15.2` currently used for local builds)
- Python 3.10+ (for benchmark, STS, match, history, NNUE, and bot utilities)
- A UCI-compatible chess GUI (like Arena, Cutechess, or similar)

For Python tooling, install dependencies as needed:

```bash
# Core analysis, benchmark, and NNUE validation/export utilities
python -m pip install chess numpy
```

For NNUE training on a fresh machine, bootstrap the Bullet dependency once:

```bash
python3 utils/nnue/bullet/bootstrap.py --build-utils
```

The tracked training runner lives under `utils/nnue/bullet_runner/`; datasets, checkpoints, and Bullet build output remain untracked.

## Building & Running

Build a release binary:

```bash
zig build -Doptimize=ReleaseFast
```

The executable lands at `zig-out/bin/sykora`. The embedded NNUE net (`src/net.sknnue`) is compiled in.

Run directly, or via the build system:

```bash
./zig-out/bin/sykora
zig build run -Doptimize=ReleaseFast
```

For OpenBench, the repository also implements its `EXE=` Makefile and
deterministic CLI benchmark contracts:

```bash
make EXE=sykora-openbench
./sykora-openbench bench
python utils/openbench/validate.py ./sykora-openbench
```

See [`utils/openbench/README.md`](utils/openbench/README.md) for instance and
SPSA setup.

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

Sykora embeds the `v7_20260710T055911Z-800` `SYKNNUE7` pairwise-MLP candidate and can also load compatible nets from an external `EvalFile`. The previous mature v3 weights remain in `src/net.sknnue.v3.bak` as a regression baseline.

The key training requirement is **factorisation**: a shared `768 → H` matrix is trained across all king buckets and merged into each bucket's residual weights at export time. V7 keeps that proven sample-sharing mechanism while changing the activation and dense architecture.

### Runtime

- The embedded net (`src/net.sknnue`) is compiled into the binary and loaded automatically at startup.
- NNUE is enabled by default (`UseNNUE = true`, `NnueBlend = 100`, `NnueScale = 100`).
- The activation function and quantization constants are stored in the header and read on load.
- To use a different net, set `EvalFile` to the path of an external `.sknnue` file.
- `NnueScale` scales the NNUE score before it is fed into the search.

For the v7 architecture, file format, research, and integer inference contract, see `specs/syknnue7_spec.md`. The v6 specification is retained for legacy-net and regression-test compatibility.

### Training Pipeline

Training uses the [Bullet](https://github.com/jw1912/bullet) trainer. The registered v7 profile is a factorised H=1024 feature transformer, pairwise product pooling, eight material heads, and a selected `1024 -> 16 -> 32 -> 32 -> 1` nonlinear tail. The launcher holds out one SF shard for a final loss check, then automatically converts, exports, and bit-exactness-checks the trained net.

```powershell
# Compile and exercise the full CUDA -> checkpoint -> SYKNNUE7 -> engine path.
.\launch_training.ps1 -Smoke

# Launch the full v7 run.
.\launch_training.ps1
```

The ready-to-load `.sknnue` path is printed when training finishes. Resume an interrupted run with `-Resume <checkpoint-directory>`; the launcher derives the next superbatch from the checkpoint name.

### Self-Play Data Generation

Sykora can generate its own training data via the `gensfen` command:

```bash
# Generate Bullet-format training data
./zig-out/bin/sykora gensfen --output output.data --games 100000 --depth 7
```

### Exporting a Trained Net

The launcher exports automatically. To repeat it manually, convert the final Bullet checkpoint to NPZ and then to SYKNNUE7:

```bash
python utils/nnue/bullet/checkpoint_raw_to_npz.py \
  --input nnue/models/bullet/<run_id>/checkpoints/<checkpoint> \
  --output checkpoint.npz

python utils/nnue/bullet/export_npz_to_syk7.py \
  --input checkpoint.npz \
  --output-net output.sknnue
```

### Embedding a New Net

To update the embedded net in the binary:

```bash
cp output.sknnue src/net.sknnue
zig build -Doptimize=ReleaseFast
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
