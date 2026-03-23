# History Ledger

This folder is the long-term experiment database for Sykora.

It stores four things:
- immutable engine snapshots (`engines/`)
- archived fixed-game selfplay runs (`matches/`)
- archived SPRT runs (`sprt/`)
- computed rating/graph artifacts (`ratings/`)
- archived STS evaluations (`sts/`)

## Layout

- `engines/<engine_id>/engine`: frozen engine binary for that snapshot
- `engines/<engine_id>/metadata.json`: snapshot metadata (git state, hash, notes, UCI id)
- `matches/<run_id>/summary.json`: machine-readable archived selfplay summary
- `matches/<run_id>/metadata.json`: reproducibility metadata (command, ids, timestamps)
- `matches/<run_id>/stdout.log`: captured selfplay stdout
- `matches/<run_id>/stderr.log`: captured selfplay stderr
- `matches/<run_id>/pgn/`: one PGN per game
- `sprt/<run_id>/summary.json`: machine-readable archived SPRT summary
- `sprt/<run_id>/metadata.json`: reproducibility metadata (command, ids, timestamps)
- `sprt/<run_id>/stdout.log`: captured SPRT stdout
- `sprt/<run_id>/stderr.log`: captured SPRT stderr
- `ratings/latest.json`: current leaderboard from archived selfplay only
- `ratings/latest.csv`: table version of leaderboard
- `ratings/edges.csv`: pairwise selfplay edge data (for network graphs)
- `ratings/timeline.csv`: cumulative rating timeline after each selfplay run
- `ratings/timeline.png`: optional chart (if matplotlib installed)
- `sts/<engine_id>/<timestamp>.json`: immutable STS run payload for that snapshot
- `sts/<engine_id>/latest.json`: latest STS run for that snapshot
- `sts/latest.csv`: latest STS totals for all snapshots (ranked by score%)
- `index/engines.jsonl`: append-only engine snapshot log
- `index/matches.jsonl`: append-only selfplay log
- `index/sprt_runs.jsonl`: append-only SPRT log
- `index/sts_runs.jsonl`: append-only STS run log

## Workflow

Initialize once:

```bash
~/.pyenv/shims/python utils/history/history.py init
```

Snapshot the current build:

```bash
zig build -Doptimize=ReleaseFast
~/.pyenv/shims/python utils/history/history.py snapshot --label "tune-castling" --notes "castling rights tweak"
```

List snapshot IDs:

```bash
~/.pyenv/shims/python utils/history/history.py list-engines
```

Run archived selfplay:

```bash
~/.pyenv/shims/python utils/history/history.py selfplay <engine_id_A> <engine_id_B> --games 120 --movetime-ms 200
```

Run archived SPRT:

```bash
~/.pyenv/shims/python utils/history/history.py sprt <engine_id_A> <engine_id_B> --elo0 -30 --elo1 30 --games-per-batch 12 --max-games 360 --movetime-ms 80
```

Auto-play strongest vs weakest by current selfplay ratings:

```bash
~/.pyenv/shims/python utils/history/history.py selfplay-extremes --min-games 20 --games 80 --movetime-ms 120
```

Recompute all ratings and export graph data:

```bash
~/.pyenv/shims/python utils/history/history.py ratings --plot
```

Render a version network map from selfplay data:

```bash
~/.pyenv/shims/python utils/history/history.py network --top-n 12 --min-games 10 --min-edge-games 2
```

Optional diagnostic STS for one snapshot:

```bash
~/.pyenv/shims/python utils/history/history.py sts <engine_id> --movetime-ms 100
```

NNUE note:

- STS is retained as a diagnostic utility, but it is not the canonical promotion signal for NNUE work.
- Promotion-quality comparison should use archived selfplay or archived SPRT.

## Notes

- Ratings are relative Elo values fit from archived selfplay outcomes only.
- Better confidence comes from more games and diversified openings.
- Keep selfplay settings consistent when comparing deltas.
- `history/.gitignore` excludes large artifacts (`engine` binaries, PGN folders) from accidental commits while keeping all metadata/versioning files trackable.
