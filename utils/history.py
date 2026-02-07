#!/usr/bin/env python3
"""Experiment ledger and rating pipeline for Sykora engine iterations.

This script creates a structured, durable history of:
- engine snapshots (binary + metadata),
- head-to-head matches (PGN + machine-readable summary),
- network ratings across all archived matches.

Typical workflow:
  1) Snapshot binaries you want to compare.
  2) Run matches between snapshot IDs.
  3) Recompute ratings and graph exports from accumulated match data.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_ROOT = REPO_ROOT / "history"


@dataclass
class MatchRecord:
    match_id: str
    started_at_utc: str
    engine1_id: str
    engine2_id: str
    engine1_wins: int
    engine2_wins: int
    draws: int

    @property
    def total_games(self) -> int:
        return self.engine1_wins + self.engine2_wins + self.draws

    @property
    def engine1_score(self) -> float:
        return self.engine1_wins + 0.5 * self.draws

    @property
    def engine2_score(self) -> float:
        return self.engine2_wins + 0.5 * self.draws


@dataclass
class PairStats:
    games: int = 0
    score_first: float = 0.0


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def slugify(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    value = value.strip("-._")
    return value or "snapshot"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def run_git(args: List[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip()


def git_metadata() -> dict:
    commit = run_git(["rev-parse", "HEAD"])
    short_commit = run_git(["rev-parse", "--short", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = run_git(["status", "--porcelain"]) or ""

    return {
        "commit": commit,
        "short_commit": short_commit,
        "branch": branch,
        "dirty": bool(status.strip()),
        "status_porcelain": status,
    }


def probe_uci_identity(engine_path: Path) -> dict:
    try:
        proc = subprocess.run(
            [str(engine_path)],
            input="uci\nquit\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return {"error": str(exc)}

    name = None
    author = None
    for line in proc.stdout.splitlines():
        if line.startswith("id name "):
            name = line[len("id name ") :].strip()
        if line.startswith("id author "):
            author = line[len("id author ") :].strip()

    return {
        "id_name": name,
        "id_author": author,
        "exit_code": proc.returncode,
    }


def ensure_layout(root: Path) -> None:
    (root / "engines").mkdir(parents=True, exist_ok=True)
    (root / "matches").mkdir(parents=True, exist_ok=True)
    (root / "ratings").mkdir(parents=True, exist_ok=True)
    (root / "index").mkdir(parents=True, exist_ok=True)


def load_engine_metadata(root: Path, engine_id: str) -> dict:
    meta_path = root / "engines" / engine_id / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Unknown engine id '{engine_id}' (missing {meta_path})")
    return json.loads(meta_path.read_text())


def default_engine_id(label: str, git: dict) -> str:
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    commit = git.get("short_commit") or "nogit"
    return f"{ts}_{commit}_{slugify(label)}"


def uniquify_engine_id(root: Path, engine_id: str) -> str:
    candidate = engine_id
    counter = 2
    while (root / "engines" / candidate).exists():
        candidate = f"{engine_id}_{counter}"
        counter += 1
    return candidate


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.history_root)
    ensure_layout(root)
    print(f"Initialized history layout at {root}")
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    root = Path(args.history_root)
    ensure_layout(root)

    engine_src = Path(args.engine)
    if not engine_src.is_file():
        raise FileNotFoundError(f"Engine binary not found: {engine_src}")

    git = git_metadata()
    label = args.label or engine_src.stem

    engine_id = args.engine_id or default_engine_id(label, git)
    engine_id = uniquify_engine_id(root, slugify(engine_id))

    engine_dir = root / "engines" / engine_id
    engine_dir.mkdir(parents=True, exist_ok=False)

    engine_dst = engine_dir / "engine"
    shutil.copy2(engine_src, engine_dst)
    engine_dst.chmod(engine_dst.stat().st_mode | 0o111)

    metadata = {
        "engine_id": engine_id,
        "created_at_utc": utc_now_iso(),
        "label": label,
        "notes": args.notes,
        "source_engine_path": str(engine_src.resolve()),
        "binary": {
            "path": "engine",
            "size_bytes": engine_dst.stat().st_size,
            "sha256": sha256_file(engine_dst),
        },
        "git": git,
        "uci": probe_uci_identity(engine_dst),
    }
    write_json(engine_dir / "metadata.json", metadata)

    append_jsonl(
        root / "index" / "engines.jsonl",
        {
            "engine_id": engine_id,
            "created_at_utc": metadata["created_at_utc"],
            "label": label,
            "sha256": metadata["binary"]["sha256"],
            "git_commit": git.get("commit"),
            "dirty": git.get("dirty"),
            "metadata_relpath": str((engine_dir / "metadata.json").relative_to(root)),
        },
    )

    print(f"Snapshot created: {engine_id}")
    print(f"  binary: {engine_dst}")
    print(f"  metadata: {engine_dir / 'metadata.json'}")
    return 0


def run_selfplay_for_match(
    root: Path,
    engine1_id: str,
    engine2_id: str,
    args: argparse.Namespace,
) -> Tuple[int, Path, Path, str]:
    engine1_meta = load_engine_metadata(root, engine1_id)
    engine2_meta = load_engine_metadata(root, engine2_id)

    engine1_path = root / "engines" / engine1_id / engine1_meta["binary"]["path"]
    engine2_path = root / "engines" / engine2_id / engine2_meta["binary"]["path"]

    if not engine1_path.is_file() or not engine2_path.is_file():
        raise FileNotFoundError("Snapshot engine binary missing")

    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    match_id = f"{ts}__{slugify(engine1_id)}__vs__{slugify(engine2_id)}"

    match_dir = root / "matches" / match_id
    pgn_dir = match_dir / "pgn"
    summary_path = match_dir / "summary.json"
    meta_path = match_dir / "metadata.json"

    match_dir.mkdir(parents=True, exist_ok=False)

    python_bin = args.python or sys.executable

    cmd: List[str] = [
        python_bin,
        str(REPO_ROOT / "utils" / "selfplay.py"),
        str(engine1_path),
        str(engine2_path),
        "--name1",
        engine1_id,
        "--name2",
        engine2_id,
        "--games",
        str(args.games),
        "--movetime-ms",
        str(args.movetime_ms),
        "--openings",
        args.openings,
        "--seed",
        str(args.seed),
        "--max-plies",
        str(args.max_plies),
        "--output-dir",
        str(pgn_dir),
        "--summary-json",
        str(summary_path),
    ]

    if args.depth is not None:
        cmd.extend(["--depth", str(args.depth)])
    if args.shuffle_openings:
        cmd.append("--shuffle-openings")
    if args.threads is not None:
        cmd.extend(["--threads", str(args.threads)])
    if args.hash_mb is not None:
        cmd.extend(["--hash-mb", str(args.hash_mb)])
    for opt in args.engine1_opt:
        cmd.extend(["--engine1-opt", opt])
    for opt in args.engine2_opt:
        cmd.extend(["--engine2-opt", opt])
    if args.quiet:
        cmd.append("--quiet")

    started = utc_now_iso()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    finished = utc_now_iso()

    if proc.returncode not in (0, 1, 2):
        raise RuntimeError(f"selfplay.py failed with exit code {proc.returncode}")

    if not summary_path.is_file():
        raise RuntimeError(f"Expected summary JSON not produced: {summary_path}")

    summary = json.loads(summary_path.read_text())
    metadata = {
        "match_id": match_id,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "engine1_id": engine1_id,
        "engine2_id": engine2_id,
        "selfplay_exit_code": proc.returncode,
        "summary_relpath": str(summary_path.relative_to(root)),
        "pgn_relpath": str(pgn_dir.relative_to(root)),
        "command": cmd,
        "result": summary.get("result", {}),
        "elo": summary.get("elo", {}),
    }
    write_json(meta_path, metadata)

    append_jsonl(
        root / "index" / "matches.jsonl",
        {
            "match_id": match_id,
            "started_at_utc": started,
            "engine1_id": engine1_id,
            "engine2_id": engine2_id,
            "selfplay_exit_code": proc.returncode,
            "summary_relpath": metadata["summary_relpath"],
            "metadata_relpath": str(meta_path.relative_to(root)),
        },
    )

    return proc.returncode, summary_path, meta_path, match_id


def load_matches(root: Path) -> List[MatchRecord]:
    matches: List[MatchRecord] = []

    for summary_path in sorted((root / "matches").glob("*/summary.json")):
        try:
            data = json.loads(summary_path.read_text())
        except Exception:
            continue

        engine1_id = str(data.get("engine1", {}).get("name", "")).strip()
        engine2_id = str(data.get("engine2", {}).get("name", "")).strip()
        result = data.get("result", {})

        if not engine1_id or not engine2_id:
            continue

        try:
            engine1_wins = int(result.get("engine1_wins", 0))
            engine2_wins = int(result.get("engine2_wins", 0))
            draws = int(result.get("draws", 0))
        except Exception:
            continue

        match_id = summary_path.parent.name
        started = data.get("generated_at_utc") or ""

        matches.append(
            MatchRecord(
                match_id=match_id,
                started_at_utc=str(started),
                engine1_id=engine1_id,
                engine2_id=engine2_id,
                engine1_wins=engine1_wins,
                engine2_wins=engine2_wins,
                draws=draws,
            )
        )

    matches.sort(key=lambda m: (m.started_at_utc, m.match_id))
    return matches


def score_rate_to_elo(score_rate: float) -> float:
    p = min(max(score_rate, 1e-6), 1.0 - 1e-6)
    return 400.0 * math.log10(p / (1.0 - p))


def aggregate_pairs(matches: Iterable[MatchRecord]) -> Dict[Tuple[str, str], PairStats]:
    stats: Dict[Tuple[str, str], PairStats] = {}

    for m in matches:
        a, b = sorted([m.engine1_id, m.engine2_id])
        key = (a, b)
        pair = stats.setdefault(key, PairStats())

        if m.engine1_id == a:
            score_a = m.engine1_score
        else:
            score_a = m.engine2_score

        pair.games += m.total_games
        pair.score_first += score_a

    return stats


def solve_ratings(pair_stats: Dict[Tuple[str, str], PairStats], iterations: int = 400) -> Dict[str, float]:
    engines: set[str] = set()
    for a, b in pair_stats:
        engines.add(a)
        engines.add(b)

    ratings: Dict[str, float] = {engine: 0.0 for engine in engines}

    if not ratings:
        return ratings

    damping = 0.35
    tolerance = 1e-3

    for _ in range(iterations):
        max_delta = 0.0

        for engine in sorted(ratings.keys()):
            numer = 0.0
            denom = 0.0

            for (a, b), pair in pair_stats.items():
                if pair.games <= 0:
                    continue
                if engine != a and engine != b:
                    continue

                if engine == a:
                    opponent = b
                    score = pair.score_first
                else:
                    opponent = a
                    score = pair.games - pair.score_first

                score_rate = score / pair.games
                d_elo = score_rate_to_elo(score_rate)
                weight = float(pair.games)

                numer += weight * (d_elo + ratings[opponent])
                denom += weight

            if denom > 0.0:
                target = numer / denom
                updated = ratings[engine] + damping * (target - ratings[engine])
                max_delta = max(max_delta, abs(updated - ratings[engine]))
                ratings[engine] = updated

        mean = sum(ratings.values()) / len(ratings)
        for engine in ratings:
            ratings[engine] -= mean

        if max_delta < tolerance:
            break

    return ratings


def build_games_by_engine(matches: Iterable[MatchRecord]) -> Dict[str, int]:
    games: Dict[str, int] = {}
    for m in matches:
        games[m.engine1_id] = games.get(m.engine1_id, 0) + m.total_games
        games[m.engine2_id] = games.get(m.engine2_id, 0) + m.total_games
    return games


def write_ratings_outputs(root: Path, matches: List[MatchRecord]) -> dict:
    pair_stats = aggregate_pairs(matches)
    ratings = solve_ratings(pair_stats)
    games_by_engine = build_games_by_engine(matches)

    leaderboard = [
        {
            "engine_id": engine,
            "rating": ratings.get(engine, 0.0),
            "games": games_by_engine.get(engine, 0),
        }
        for engine in sorted(ratings.keys(), key=lambda e: ratings[e], reverse=True)
    ]

    summary = {
        "generated_at_utc": utc_now_iso(),
        "method": "weighted_pairwise_elo_fit",
        "match_count": len(matches),
        "engine_count": len(leaderboard),
        "leaderboard": leaderboard,
    }

    write_json(root / "ratings" / "latest.json", summary)

    csv_path = root / "ratings" / "latest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "engine_id", "rating", "games"])
        for idx, row in enumerate(leaderboard, start=1):
            writer.writerow([idx, row["engine_id"], f"{row['rating']:.2f}", row["games"]])

    edges_path = root / "ratings" / "edges.csv"
    with edges_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["engine_a", "engine_b", "games", "score_a", "score_rate_a", "elo_a_minus_b"])
        for (a, b), pair in sorted(pair_stats.items()):
            rate = pair.score_first / pair.games if pair.games else 0.5
            writer.writerow([a, b, pair.games, f"{pair.score_first:.1f}", f"{rate:.6f}", f"{score_rate_to_elo(rate):.2f}"])

    timeline_path = root / "ratings" / "timeline.csv"
    with timeline_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "match_id", "started_at_utc", "engine_id", "rating"])

        cumulative: List[MatchRecord] = []
        for step, match in enumerate(matches, start=1):
            cumulative.append(match)
            partial_pair = aggregate_pairs(cumulative)
            partial_ratings = solve_ratings(partial_pair)
            for engine, rating in sorted(partial_ratings.items()):
                writer.writerow([step, match.match_id, match.started_at_utc, engine, f"{rating:.2f}"])

    return summary


def maybe_plot_timeline(root: Path, top_n: int = 8) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    latest_csv = root / "ratings" / "latest.csv"
    timeline_csv = root / "ratings" / "timeline.csv"
    if not latest_csv.is_file() or not timeline_csv.is_file():
        return None

    top_engines: List[str] = []
    with latest_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if len(top_engines) >= top_n:
                break
            top_engines.append(row["engine_id"])

    if not top_engines:
        return None

    points: Dict[str, List[Tuple[int, float]]] = {engine: [] for engine in top_engines}
    with timeline_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            engine = row["engine_id"]
            if engine not in points:
                continue
            points[engine].append((int(row["step"]), float(row["rating"])))

    fig, ax = plt.subplots(figsize=(10, 6))
    for engine in top_engines:
        series = points[engine]
        if not series:
            continue
        xs = [item[0] for item in series]
        ys = [item[1] for item in series]
        ax.plot(xs, ys, label=engine)

    ax.set_title("Sykora Engine Rating Timeline")
    ax.set_xlabel("Match Step")
    ax.set_ylabel("Relative Elo")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    out_path = root / "ratings" / "timeline.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def cmd_match(args: argparse.Namespace) -> int:
    root = Path(args.history_root)
    ensure_layout(root)

    code, summary_path, meta_path, match_id = run_selfplay_for_match(
        root=root,
        engine1_id=args.engine1_id,
        engine2_id=args.engine2_id,
        args=args,
    )

    print(f"Match stored: {match_id}")
    print(f"  summary: {summary_path}")
    print(f"  metadata: {meta_path}")

    if not args.no_rate:
        matches = load_matches(root)
        rating_summary = write_ratings_outputs(root, matches)
        if args.plot:
            plot_path = maybe_plot_timeline(root)
            if plot_path is not None:
                print(f"  timeline plot: {plot_path}")
            else:
                print("  timeline plot: skipped (matplotlib not installed)")

        print(
            f"  ratings updated: {root / 'ratings' / 'latest.json'}"
            f" ({rating_summary['engine_count']} engines, {rating_summary['match_count']} matches)"
        )

    return code


def cmd_ratings(args: argparse.Namespace) -> int:
    root = Path(args.history_root)
    ensure_layout(root)

    matches = load_matches(root)
    if not matches:
        print("No matches found. Run 'history.py match' first.")
        return 1

    summary = write_ratings_outputs(root, matches)
    plot_path = None
    if args.plot:
        plot_path = maybe_plot_timeline(root)

    print(f"Ratings updated: {root / 'ratings' / 'latest.json'}")
    print(f"Engines: {summary['engine_count']}, Matches: {summary['match_count']}")

    print("Top engines:")
    for idx, row in enumerate(summary["leaderboard"][: min(10, len(summary["leaderboard"]))], start=1):
        print(f"  {idx:2d}. {row['engine_id']}  Elo {row['rating']:+.1f}  games={row['games']}")

    if args.plot:
        if plot_path is not None:
            print(f"Timeline plot: {plot_path}")
        else:
            print("Timeline plot skipped (matplotlib not installed)")

    return 0


def cmd_list_engines(args: argparse.Namespace) -> int:
    root = Path(args.history_root)
    ensure_layout(root)

    entries = sorted((root / "engines").glob("*/metadata.json"))
    if not entries:
        print("No snapshots found.")
        return 0

    print(f"Snapshots in {root / 'engines'}:")
    for meta_path in entries:
        data = json.loads(meta_path.read_text())
        print(
            f"  {data.get('engine_id')} | {data.get('created_at_utc')}"
            f" | label={data.get('label')} | commit={data.get('git', {}).get('short_commit')}"
        )
    return 0


def cmd_list_matches(args: argparse.Namespace) -> int:
    root = Path(args.history_root)
    ensure_layout(root)

    matches = load_matches(root)
    if not matches:
        print("No matches found.")
        return 0

    print(f"Matches in {root / 'matches'}:")
    for m in matches:
        print(
            f"  {m.match_id} | {m.engine1_id} vs {m.engine2_id}"
            f" | {m.engine1_wins}-{m.draws}-{m.engine2_wins}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage historical Sykora engine snapshots, matches, and ratings.",
    )
    parser.add_argument(
        "--history-root",
        default=str(DEFAULT_HISTORY_ROOT),
        help=f"History root directory (default: {DEFAULT_HISTORY_ROOT})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize history folder layout")
    p_init.set_defaults(func=cmd_init)

    p_snapshot = sub.add_parser("snapshot", help="Snapshot an engine binary with metadata")
    p_snapshot.add_argument("--engine", default=str(REPO_ROOT / "zig-out" / "bin" / "sykora"), help="Path to engine binary")
    p_snapshot.add_argument("--label", default="working", help="Human label for this snapshot")
    p_snapshot.add_argument("--notes", default="", help="Optional notes")
    p_snapshot.add_argument("--engine-id", default=None, help="Optional explicit snapshot id")
    p_snapshot.set_defaults(func=cmd_snapshot)

    p_match = sub.add_parser("match", help="Run and archive a match between two snapshot IDs")
    p_match.add_argument("engine1_id", help="Snapshot ID for baseline engine")
    p_match.add_argument("engine2_id", help="Snapshot ID for candidate engine")

    p_match.add_argument("--games", type=int, default=80, help="Total games")
    p_match.add_argument("--movetime-ms", type=int, default=200, help="Per-move time in ms")
    p_match.add_argument("--depth", type=int, default=None, help="Optional fixed depth")
    p_match.add_argument("--openings", default="default", help="Opening source for selfplay.py")
    p_match.add_argument("--shuffle-openings", action="store_true", help="Shuffle opening order")
    p_match.add_argument("--seed", type=int, default=1, help="Opening shuffle seed")
    p_match.add_argument("--max-plies", type=int, default=300, help="Move-limit draw adjudication")
    p_match.add_argument("--threads", type=int, default=None, help="UCI Threads for both engines")
    p_match.add_argument("--hash-mb", type=int, default=None, help="UCI Hash MB for both engines")
    p_match.add_argument("--engine1-opt", action="append", default=[], help="Extra UCI option for engine1 (Key=Value)")
    p_match.add_argument("--engine2-opt", action="append", default=[], help="Extra UCI option for engine2 (Key=Value)")
    p_match.add_argument("--python", default=None, help="Python interpreter for selfplay.py (default: current interpreter)")
    p_match.add_argument("--no-rate", action="store_true", help="Skip automatic ratings recompute")
    p_match.add_argument("--plot", action="store_true", help="Attempt timeline plot after ratings update")
    p_match.add_argument("--quiet", action="store_true", help="Pass --quiet to selfplay runner")
    p_match.set_defaults(func=cmd_match)

    p_ratings = sub.add_parser("ratings", help="Recompute ratings from archived matches")
    p_ratings.add_argument("--plot", action="store_true", help="Attempt timeline plot generation")
    p_ratings.set_defaults(func=cmd_ratings)

    p_list_engines = sub.add_parser("list-engines", help="List snapshot IDs and metadata")
    p_list_engines.set_defaults(func=cmd_list_engines)

    p_list_matches = sub.add_parser("list-matches", help="List archived matches")
    p_list_matches.set_defaults(func=cmd_list_matches)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
