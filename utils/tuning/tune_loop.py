#!/usr/bin/env python3
"""One-command tuning loop: STS gate + archived self-play + auto-promotion.

Workflow per run:
1) Build current engine (`zig build -Doptimize=ReleaseFast`)
2) Resolve baseline snapshot ID
3) Snapshot candidate binary
4) Run STS gate (baseline vs candidate on selected themes)
5) If STS gate passes, run archived self-play via `utils/history/history.py match`
6) Auto-promote candidate to baseline only if gates pass

Artifacts are recorded under `history/` via `utils/history/history.py`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_ROOT = REPO_ROOT / "history"
CURRENT_BASELINE_FILE = HISTORY_ROOT / "current_baseline.txt"


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def parse_total_score(output: str) -> Tuple[int, int]:
    # Example line:
    # TOTAL           1500       8159      15000   54.39%   42.20%   42.20%
    m = re.search(r"^TOTAL\s+\d+\s+(\d+)\s+(\d+)\s+", output, flags=re.MULTILINE)
    if not m:
        raise ValueError("Could not parse TOTAL score from STS output")
    return int(m.group(1)), int(m.group(2))


def run_sts(engine_path: Path, epd_files: List[Path], movetime_ms: int, python_bin: str) -> Tuple[int, int, Dict[str, Tuple[int, int]]]:
    total_points = 0
    total_max = 0
    per_theme: Dict[str, Tuple[int, int]] = {}

    for epd in epd_files:
        cmd = [
            python_bin,
            str(REPO_ROOT / "utils" / "sts" / "sts.py"),
            "--engine",
            str(engine_path),
            "--epd",
            str(epd),
            "--movetime-ms",
            str(movetime_ms),
            "--show",
            "none",
        ]
        proc = run_cmd(cmd, capture=True)
        if proc.returncode != 0:
            raise RuntimeError(f"STS failed on {epd}:\n{proc.stdout}")

        pts, mx = parse_total_score(proc.stdout)
        total_points += pts
        total_max += mx
        per_theme[epd.stem] = (pts, mx)

    return total_points, total_max, per_theme


def history_cmd(python_bin: str, args: List[str], capture: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [python_bin, str(REPO_ROOT / "utils" / "history" / "history.py"), *args]
    return run_cmd(cmd, capture=capture)


def snapshot_engine(python_bin: str, label: str, notes: str, engine_path: Optional[Path] = None) -> str:
    args = ["snapshot", "--label", label, "--notes", notes]
    if engine_path is not None:
        args.extend(["--engine", str(engine_path)])

    proc = history_cmd(python_bin, args, capture=True)
    if proc.returncode != 0:
        raise RuntimeError(f"history snapshot failed:\n{proc.stdout}")

    m = re.search(r"Snapshot created:\s*(\S+)", proc.stdout)
    if not m:
        raise RuntimeError(f"Could not parse snapshot ID from output:\n{proc.stdout}")
    return m.group(1)


def get_engine_binary(snapshot_id: str) -> Path:
    path = HISTORY_ROOT / "engines" / snapshot_id / "engine"
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot engine binary missing: {path}")
    return path


def read_current_baseline() -> Optional[str]:
    if not CURRENT_BASELINE_FILE.is_file():
        return None
    for raw in CURRENT_BASELINE_FILE.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            return line
    return None


def write_current_baseline(snapshot_id: str) -> None:
    CURRENT_BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_BASELINE_FILE.write_text(snapshot_id + "\n")


def last_match_index() -> int:
    path = HISTORY_ROOT / "index" / "matches.jsonl"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def read_new_matches(from_index: int) -> List[dict]:
    path = HISTORY_ROOT / "index" / "matches.jsonl"
    if not path.is_file():
        return []

    out: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if idx <= from_index:
                continue
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_match_summary(match_id: str) -> dict:
    summary_path = HISTORY_ROOT / "matches" / match_id / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary for match {match_id}: {summary_path}")
    return json.loads(summary_path.read_text())


def select_epd_files(epd_dir: Path, themes: List[str]) -> List[Path]:
    files: List[Path] = []
    for theme in themes:
        stem = theme.strip()
        if not stem:
            continue
        if not stem.lower().endswith(".epd"):
            stem += ".epd"
        p = epd_dir / stem
        if not p.is_file():
            raise FileNotFoundError(f"Theme file not found: {p}")
        files.append(p)
    if not files:
        raise ValueError("No EPD theme files selected")
    return files


def append_tune_run(run_data: dict) -> None:
    path = HISTORY_ROOT / "index" / "tune_runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run_data) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one tuning evaluation loop and auto-promote only winning candidates.",
    )

    parser.add_argument("--python", default=str(Path.home() / ".pyenv" / "shims" / "python"), help="Python interpreter")

    parser.add_argument(
        "--baseline-id",
        default=None,
        help="Explicit baseline snapshot ID (default: history/current_baseline.txt)",
    )
    parser.add_argument(
        "--bootstrap-baseline-engine",
        default=None,
        help="If no baseline exists, snapshot this engine binary as baseline",
    )

    parser.add_argument("--candidate-label", default="working", help="Label for candidate snapshot")
    parser.add_argument("--candidate-notes", default="", help="Notes for candidate snapshot")

    parser.add_argument("--epd-dir", default=str(REPO_ROOT / "epd"), help="EPD directory")
    parser.add_argument(
        "--themes",
        default="STS1,STS2,STS4,STS8,STS9,STS15",
        help="Comma-separated STS theme stems/files",
    )
    parser.add_argument("--sts-movetime-ms", type=int, default=100, help="STS movetime (ms)")
    parser.add_argument(
        "--min-sts-delta",
        type=int,
        default=0,
        help="Required candidate-baseline point delta across selected STS themes",
    )

    parser.add_argument("--sp-games", type=int, default=20, help="Self-play games")
    parser.add_argument("--sp-movetime-ms", type=int, default=80, help="Self-play movetime (ms)")
    parser.add_argument("--sp-depth", type=int, default=None, help="Optional self-play depth")
    parser.add_argument("--sp-openings", default="default", help="Self-play openings source")
    parser.add_argument("--sp-seed", type=int, default=1, help="Self-play opening seed")
    parser.add_argument("--sp-max-plies", type=int, default=220, help="Self-play move-limit adjudication")
    parser.add_argument("--sp-threads", type=int, default=None, help="Self-play Threads")
    parser.add_argument("--sp-hash-mb", type=int, default=None, help="Self-play Hash MB")
    parser.add_argument("--sp-shuffle-openings", action="store_true", help="Shuffle self-play openings")
    parser.add_argument(
        "--max-p-value",
        type=float,
        default=1.0,
        help="Require self-play p-value <= this to promote (default: 1.0 = disabled)",
    )

    parser.add_argument("--plot-ratings", action="store_true", help="Attempt timeline plot after match")
    parser.add_argument("--quiet-selfplay", action="store_true", help="Pass --quiet to self-play")

    args = parser.parse_args()

    if args.sts_movetime_ms <= 0:
        parser.error("--sts-movetime-ms must be > 0")
    if args.sp_games <= 0:
        parser.error("--sp-games must be > 0")
    if args.sp_movetime_ms <= 0:
        parser.error("--sp-movetime-ms must be > 0")
    if args.sp_depth is not None and args.sp_depth <= 0:
        parser.error("--sp-depth must be > 0")
    if args.max_p_value <= 0.0 or args.max_p_value > 1.0:
        parser.error("--max-p-value must be in (0, 1]")

    return args


def main() -> int:
    args = parse_args()

    print("==> Ensuring history layout")
    proc = history_cmd(args.python, ["init"], capture=True)
    if proc.returncode != 0:
        print(proc.stdout or "history init failed", file=sys.stderr)
        return 1
    print(proc.stdout.strip())

    print("==> Building candidate engine (ReleaseFast)")
    build = run_cmd(["zig", "build", "-Doptimize=ReleaseFast"], capture=True)
    if build.returncode != 0:
        print(build.stdout or "build failed", file=sys.stderr)
        return 1

    baseline_id = args.baseline_id or read_current_baseline()
    if baseline_id is None:
        if not args.bootstrap_baseline_engine:
            print(
                "No baseline configured. Provide --baseline-id or --bootstrap-baseline-engine.",
                file=sys.stderr,
            )
            return 1
        print("==> Bootstrapping baseline snapshot")
        baseline_id = snapshot_engine(
            args.python,
            label="baseline-bootstrap",
            notes="bootstrap baseline",
            engine_path=Path(args.bootstrap_baseline_engine),
        )
        write_current_baseline(baseline_id)
        print(f"Baseline set to {baseline_id}")

    print(f"==> Baseline: {baseline_id}")
    baseline_bin = get_engine_binary(baseline_id)

    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    candidate_label = f"{args.candidate_label}-{ts}"
    candidate_id = snapshot_engine(
        args.python,
        label=candidate_label,
        notes=args.candidate_notes or "tune-loop candidate",
        engine_path=REPO_ROOT / "zig-out" / "bin" / "sykora",
    )
    candidate_bin = get_engine_binary(candidate_id)
    print(f"==> Candidate snapshot: {candidate_id}")

    theme_names = [part.strip() for part in args.themes.split(",") if part.strip()]
    epd_files = select_epd_files(Path(args.epd_dir), theme_names)

    print("==> STS gate")
    base_pts, base_max, base_theme = run_sts(baseline_bin, epd_files, args.sts_movetime_ms, args.python)
    cand_pts = 0
    cand_max = 0
    cand_theme: Dict[str, Tuple[int, int]] = {}
    delta = -1_000_000
    sts_error: Optional[str] = None

    try:
        cand_pts, cand_max, cand_theme = run_sts(candidate_bin, epd_files, args.sts_movetime_ms, args.python)
        delta = cand_pts - base_pts
        print(f"STS baseline: {base_pts}/{base_max}")
        print(f"STS candidate: {cand_pts}/{cand_max}")
        print(f"STS delta: {delta:+d}")
        for theme in theme_names:
            t = theme if theme.lower().endswith(".epd") else theme
            stem = Path(t).stem
            b = base_theme.get(stem, (0, 0))[0]
            c = cand_theme.get(stem, (0, 0))[0]
            print(f"  {stem}: {b} -> {c} ({c - b:+d})")
    except Exception as exc:
        sts_error = str(exc)
        print(f"STS baseline: {base_pts}/{base_max}")
        print("STS candidate: ERROR")
        print(f"STS failure detail: {sts_error}")

    sts_pass = sts_error is None and delta >= args.min_sts_delta

    match_id = None
    selfplay_summary = None
    selfplay_pass = False

    if sts_pass:
        print("==> Self-play gate")
        start_idx = last_match_index()
        match_args: List[str] = [
            "match",
            baseline_id,
            candidate_id,
            "--games",
            str(args.sp_games),
            "--movetime-ms",
            str(args.sp_movetime_ms),
            "--openings",
            args.sp_openings,
            "--seed",
            str(args.sp_seed),
            "--max-plies",
            str(args.sp_max_plies),
            "--no-rate",
        ]
        if args.sp_depth is not None:
            match_args.extend(["--depth", str(args.sp_depth)])
        if args.sp_threads is not None:
            match_args.extend(["--threads", str(args.sp_threads)])
        if args.sp_hash_mb is not None:
            match_args.extend(["--hash-mb", str(args.sp_hash_mb)])
        if args.sp_shuffle_openings:
            match_args.append("--shuffle-openings")
        if args.plot_ratings:
            match_args.append("--plot")
        if args.quiet_selfplay:
            match_args.append("--quiet")

        match_proc = history_cmd(args.python, match_args, capture=True)
        print(match_proc.stdout or "")
        if match_proc.returncode not in (0, 1, 2):
            print("Self-play match execution failed", file=sys.stderr)
            return 1

        new_matches = read_new_matches(start_idx)
        if not new_matches:
            print("No new archived match record found", file=sys.stderr)
            return 1

        match_id = new_matches[-1]["match_id"]
        selfplay_summary = load_match_summary(match_id)

        result = selfplay_summary.get("result", {})
        cand_score = float(result.get("engine2_score", 0.0))
        base_score = float(result.get("engine1_score", 0.0))
        p_value = float(selfplay_summary.get("elo", {}).get("p_value_two_sided", 1.0))

        selfplay_pass = (cand_score > base_score) and (p_value <= args.max_p_value)

        print(f"Self-play score: baseline {base_score:.1f} vs candidate {cand_score:.1f}")
        print(f"Self-play p-value: {p_value:.4f} (threshold {args.max_p_value})")
    else:
        print("Skipping self-play: STS gate failed")

    promoted = sts_pass and selfplay_pass
    if promoted:
        write_current_baseline(candidate_id)
        print(f"PROMOTED: {candidate_id} is new baseline")
    else:
        print("NOT PROMOTED")

    # Always refresh ratings from all archived matches.
    ratings_proc = history_cmd(args.python, ["ratings", *(["--plot"] if args.plot_ratings else [])], capture=True)
    if ratings_proc.returncode == 0:
        print(ratings_proc.stdout or "")

    run_record = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "themes": [Path(f).stem for f in epd_files],
        "sts_movetime_ms": args.sts_movetime_ms,
        "sts_baseline_points": base_pts,
        "sts_candidate_points": cand_pts,
        "sts_delta": delta,
        "sts_error": sts_error,
        "sts_pass": sts_pass,
        "selfplay_match_id": match_id,
        "selfplay_pass": selfplay_pass,
        "promoted": promoted,
        "current_baseline_after": read_current_baseline(),
    }
    append_tune_run(run_record)

    return 0 if promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
