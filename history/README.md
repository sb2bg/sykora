# History Ledger

This folder is the long-term experiment database for Sykora.

It stores three things:
- immutable engine snapshots (`engines/`)
- archived head-to-head matches (`matches/`)
- computed rating/graph artifacts (`ratings/`)
- archived STS evaluations (`sts/`)

## Layout

- `engines/<engine_id>/engine`: frozen engine binary for that snapshot
- `engines/<engine_id>/metadata.json`: snapshot metadata (git state, hash, notes, UCI id)
- `matches/<match_id>/summary.json`: machine-readable match summary from `utils/selfplay.py`
- `matches/<match_id>/metadata.json`: reproducibility metadata (command, ids, timestamps)
- `matches/<match_id>/pgn/`: one PGN per game
- `ratings/latest.json`: current leaderboard from all archived matches
- `ratings/latest.csv`: table version of leaderboard
- `ratings/edges.csv`: pairwise edge data (for network graphs)
- `ratings/timeline.csv`: cumulative rating timeline after each match
- `ratings/timeline.png`: optional chart (if matplotlib installed)
- `sts/<engine_id>/<timestamp>.json`: immutable STS run payload for that snapshot
- `sts/<engine_id>/latest.json`: latest STS run for that snapshot
- `sts/latest.csv`: latest STS totals for all snapshots (ranked by score%)
- `index/engines.jsonl`: append-only engine snapshot log
- `index/matches.jsonl`: append-only match log
- `index/sts_runs.jsonl`: append-only STS run log

## Workflow

Initialize once:

```bash
~/.pyenv/shims/python utils/history.py init
```

Snapshot the current build:

```bash
zig build -Doptimize=ReleaseFast
~/.pyenv/shims/python utils/history.py snapshot --label "tune-castling" --notes "castling rights tweak"
```

List snapshot IDs:

```bash
~/.pyenv/shims/python utils/history.py list-engines
```

Run a tracked match:

```bash
~/.pyenv/shims/python utils/history.py match <engine_id_A> <engine_id_B> --games 120 --movetime-ms 200
```

Auto-play strongest vs weakest (by current ratings):

```bash
~/.pyenv/shims/python utils/history.py match-extremes --min-games 20 --games 80 --movetime-ms 120
```

Recompute all ratings and export graph data:

```bash
~/.pyenv/shims/python utils/history.py ratings --plot
```

Render a version network map (nodes=engines, edges=matchups):

```bash
~/.pyenv/shims/python utils/history.py network --top-n 12 --min-games 10 --min-edge-games 2
```

Run and persist STS for one snapshot:

```bash
~/.pyenv/shims/python utils/history.py sts <engine_id> --movetime-ms 100
```

Backfill STS for all snapshots:

```bash
~/.pyenv/shims/python utils/history.py sts --all --movetime-ms 100 --continue-on-error
```

One-command loop (build + STS gate + self-play gate + promotion):

```bash
# Bootstrap baseline once
~/.pyenv/shims/python utils/tune_loop.py --bootstrap-baseline-engine old_versions/old_sykora --candidate-label "bootstrap"

# Regular iteration
~/.pyenv/shims/python utils/tune_loop.py --candidate-label "tweak-name" --candidate-notes "what changed"
```

## Notes

- Ratings are relative Elo values fit from all stored pairwise match outcomes.
- Better confidence comes from more games and diversified openings.
- Keep match settings consistent (time control, openings, options) when comparing deltas.
- `history/.gitignore` excludes large artifacts (`engine` binaries, PGN folders) from accidental commits while keeping all metadata/versioning files trackable.
