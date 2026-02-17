#!/usr/bin/env python3
"""Build a mixed Bullet dataset from fishtest downloads + self-play PGNs.

Pipeline:
1) Download recent fishtest PGNs (optional).
2) Generate self-play PGNs (optional).
3) Stage selected PGNs into one folder.
4) Teacher-label staged PGNs to Bullet text format.
5) Pack text into BulletFormat .data.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path


def default_python_bin() -> str:
    python_path = shutil.which("python")
    if python_path:
        return python_path
    pyenv_path = shutil.which("pyenv")
    if pyenv_path:
        try:
            result = subprocess.run(
                [pyenv_path, "which", "python"],
                check=True,
                capture_output=True,
                text=True,
            )
            resolved = result.stdout.strip()
            if resolved:
                return resolved
        except subprocess.SubprocessError:
            pass
    if sys.executable:
        return sys.executable
    python3_path = shutil.which("python3")
    if python3_path:
        return python3_path
    return "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed NNUE/Bullet training data.")
    parser.add_argument(
        "--python-bin",
        default=default_python_bin(),
        help="Python interpreter used to run helper scripts",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Run id (default: utc timestamp)",
    )
    parser.add_argument(
        "--stage-root",
        default="nnue/data/staging/mixed",
        help="Root dir for staged PGNs and metadata",
    )
    parser.add_argument(
        "--teacher-output",
        default="nnue/data/bullet/train/teacher_text_{run_id}.txt",
        help="Teacher text output path (supports {run_id})",
    )
    parser.add_argument(
        "--packed-output",
        default="nnue/data/bullet/train/train_main_{run_id}.data",
        help="Packed BulletFormat output path (supports {run_id})",
    )

    # Fishtest download options.
    parser.add_argument("--skip-fishtest-download", action="store_true", help="Skip fishtest download step")
    parser.add_argument("--fishtest-path", default="datasets/fishtest", help="Local fishtest storage directory")
    parser.add_argument("--fishtest-hours", type=float, default=72.0, help="Download tests updated within this many hours")
    parser.add_argument("--fishtest-ltc-only", action="store_true", help="Download LTC-only tests")
    parser.add_argument("--fishtest-success-only", action="store_true", help="Download green tests only")
    parser.add_argument(
        "--fishtest-standard-chess-only",
        action="store_true",
        help="Filter out FRC book tests",
    )
    parser.add_argument("--fishtest-tc-lower-limit", type=float, default=20.0, help="Optional lower tc base filter")
    parser.add_argument("--fishtest-tc-upper-limit", type=float, default=180.0, help="Optional upper tc base filter")
    parser.add_argument("--fishtest-username", default="", help="Optional username filter")
    parser.add_argument(
        "--fishtest-max-runs",
        type=int,
        default=4,
        help="Maximum number of fishtest runs to download (0 = unlimited)",
    )
    parser.add_argument(
        "--fishtest-max-games",
        type=int,
        default=150000,
        help="Skip fishtest runs with more than this many games (0 = unlimited)",
    )
    parser.add_argument(
        "--max-fishtest-pgns",
        type=int,
        default=16,
        help="Max fishtest PGNs to stage (0 = no cap)",
    )
    parser.add_argument(
        "--include-existing-fishtest",
        action="store_true",
        help="Allow previously downloaded fishtest PGNs to fill stage if new set is small",
    )

    # Self-play generation options.
    parser.add_argument("--skip-selfplay", action="store_true", help="Skip self-play generation")
    parser.add_argument("--selfplay-engine1", default="./zig-out/bin/sykora", help="Self-play engine1 path")
    parser.add_argument("--selfplay-engine2", default="./zig-out/bin/sykora", help="Self-play engine2 path")
    parser.add_argument("--selfplay-name1", default="sykora-a", help="Label for self-play engine1")
    parser.add_argument("--selfplay-name2", default="sykora-b", help="Label for self-play engine2")
    parser.add_argument("--selfplay-games", type=int, default=40, help="Self-play game count")
    parser.add_argument("--selfplay-movetime-ms", type=int, default=100, help="Self-play movetime in ms")
    parser.add_argument("--selfplay-openings", default="default", help="Self-play openings source")
    parser.add_argument("--selfplay-shuffle-openings", action="store_true", help="Shuffle self-play openings")
    parser.add_argument("--selfplay-seed", type=int, default=1, help="Self-play opening seed")
    parser.add_argument("--selfplay-threads", type=int, default=1, help="Threads option for both engines")
    parser.add_argument("--selfplay-hash-mb", type=int, default=64, help="Hash option for both engines")
    parser.add_argument(
        "--selfplay-output-root",
        default="datasets/selfplay/mixed",
        help="Root dir for self-play PGN outputs",
    )
    parser.add_argument(
        "--selfplay-engine1-opt",
        action="append",
        default=[],
        help="Extra UCI option for engine1: Key=Value (repeatable)",
    )
    parser.add_argument(
        "--selfplay-engine2-opt",
        action="append",
        default=[],
        help="Extra UCI option for engine2: Key=Value (repeatable)",
    )

    # Extra external PGNs (e.g. lichess dumps unpacked locally).
    parser.add_argument(
        "--extra-pgn-glob",
        action="append",
        default=[],
        help="Extra PGN glob(s) to include in staging (repeatable)",
    )

    # Teacher settings.
    parser.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish", help="Teacher engine path")
    parser.add_argument("--teacher-depth", type=int, default=12, help="Teacher depth")
    parser.add_argument("--teacher-threads", type=int, default=1, help="Teacher Threads")
    parser.add_argument("--teacher-hash-mb", type=int, default=256, help="Teacher Hash MB")
    parser.add_argument("--sample-rate", type=float, default=0.2, help="Position sample probability")
    parser.add_argument("--min-ply", type=int, default=12, help="Min ply for sampling")
    parser.add_argument("--max-ply", type=int, default=220, help="Max ply for sampling")
    parser.add_argument("--cp-clip", type=int, default=2500, help="Teacher CP clipping")
    parser.add_argument("--max-positions", type=int, default=0, help="Hard cap positions (0 = all)")
    parser.add_argument("--skip-check", action="store_true", help="Skip in-check positions")
    parser.add_argument("--skip-captures", action="store_true", help="Skip positions after captures")
    parser.add_argument("--dedupe-fen", action="store_true", help="Deduplicate exact FENs")

    # Packing settings.
    parser.add_argument(
        "--bullet-utils",
        default="nnue/bullet_repo/target/release/bullet-utils",
        help="Path to bullet-utils binary",
    )
    parser.add_argument("--shuffle-mem-mb", type=int, default=4096, help="Shuffle memory budget")
    parser.add_argument("--convert-threads", type=int, default=8, help="Text conversion threads")

    parser.add_argument("--dry-run", action="store_true", help="Print planned commands and exit")
    return parser.parse_args()


def run_cmd(cmd: list[str], dry_run: bool = False, ok_codes: set[int] | None = None) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, check=False)
    allowed = ok_codes if ok_codes is not None else {0}
    if result.returncode not in allowed:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def list_pgns(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    out.extend(p for p in root.rglob("*.pgn") if p.is_file())
    out.extend(p for p in root.rglob("*.pgn.gz") if p.is_file())
    return sorted(set(out))


def choose_recent(paths: list[Path], max_count: int) -> list[Path]:
    if max_count <= 0:
        return sorted(paths)
    recent = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:max_count]
    return sorted(recent)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def render_with_run_id(template: str, run_id: str) -> Path:
    return Path(template.format(run_id=run_id))


def main() -> int:
    args = parse_args()

    run_id = args.run_id.strip() or datetime.datetime.now(datetime.UTC).strftime("mix_%Y%m%dT%H%M%SZ")
    stage_dir = Path(args.stage_root) / run_id
    stage_pgn_dir = stage_dir / "pgn_inputs"
    stage_pgn_dir.mkdir(parents=True, exist_ok=True)

    teacher_output = render_with_run_id(args.teacher_output, run_id)
    packed_output = render_with_run_id(args.packed_output, run_id)
    teacher_output.parent.mkdir(parents=True, exist_ok=True)
    packed_output.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "run_id": run_id,
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "stage_dir": str(stage_dir.resolve()),
        "teacher_output": str(teacher_output.resolve()),
        "packed_output": str(packed_output.resolve()),
        "sources": {},
    }

    selected_fishtest: list[Path] = []
    if not args.skip_fishtest_download:
        fishtest_root = Path(args.fishtest_path)
        fishtest_root.mkdir(parents=True, exist_ok=True)

        before = set(list_pgns(fishtest_root))
        cmd = [
            args.python_bin,
            "utils/data/download_fishtest_pgns.py",
            "--path",
            str(fishtest_root),
            "--time-delta",
            str(args.fishtest_hours),
            "--ltc-only",
            "true" if args.fishtest_ltc_only else "false",
            "--success-only",
            "true" if args.fishtest_success_only else "false",
            "--standard-chess-only",
            "true" if args.fishtest_standard_chess_only else "false",
        ]
        if args.fishtest_tc_lower_limit is not None:
            cmd.extend(["--tc-lower-limit", str(args.fishtest_tc_lower_limit)])
        if args.fishtest_tc_upper_limit is not None:
            cmd.extend(["--tc-upper-limit", str(args.fishtest_tc_upper_limit)])
        if args.fishtest_username:
            cmd.extend(["--username", args.fishtest_username])
        if args.fishtest_max_runs > 0:
            cmd.extend(["--max-runs", str(args.fishtest_max_runs)])
        if args.fishtest_max_games > 0:
            cmd.extend(["--max-games", str(args.fishtest_max_games)])
        run_cmd(cmd, dry_run=args.dry_run)

        after = set(list_pgns(fishtest_root))
        new_paths = sorted(after - before)
        if args.include_existing_fishtest:
            pool = sorted(after)
        else:
            pool = new_paths
        selected_fishtest = choose_recent(pool, args.max_fishtest_pgns)

        summary["sources"]["fishtest"] = {
            "download_root": str(fishtest_root.resolve()),
            "new_files": len(new_paths),
            "selected_files": len(selected_fishtest),
            "include_existing": bool(args.include_existing_fishtest),
            "max_runs": args.fishtest_max_runs,
            "max_games": args.fishtest_max_games,
        }
    else:
        summary["sources"]["fishtest"] = {"skipped": True}

    selected_selfplay: list[Path] = []
    if not args.skip_selfplay:
        selfplay_out = Path(args.selfplay_output_root) / run_id
        selfplay_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python_bin,
            "utils/match/selfplay.py",
            args.selfplay_engine1,
            args.selfplay_engine2,
            "--name1",
            args.selfplay_name1,
            "--name2",
            args.selfplay_name2,
            "--games",
            str(args.selfplay_games),
            "--movetime-ms",
            str(args.selfplay_movetime_ms),
            "--openings",
            args.selfplay_openings,
            "--seed",
            str(args.selfplay_seed),
            "--threads",
            str(args.selfplay_threads),
            "--hash-mb",
            str(args.selfplay_hash_mb),
            "--output-dir",
            str(selfplay_out),
            "--summary-json",
            str(selfplay_out / "summary.json"),
        ]
        if args.selfplay_shuffle_openings:
            cmd.append("--shuffle-openings")
        for opt in args.selfplay_engine1_opt:
            cmd.extend(["--engine1-opt", opt])
        for opt in args.selfplay_engine2_opt:
            cmd.extend(["--engine2-opt", opt])

        run_cmd(cmd, dry_run=args.dry_run, ok_codes={0, 1, 2})
        selected_selfplay = sorted(selfplay_out.glob("*.pgn"))
        summary["sources"]["selfplay"] = {
            "output_dir": str(selfplay_out.resolve()),
            "selected_files": len(selected_selfplay),
        }
    else:
        summary["sources"]["selfplay"] = {"skipped": True}

    selected_extra: list[Path] = []
    for pattern in args.extra_pgn_glob:
        matches = [Path(p) for p in glob.glob(pattern, recursive=True)]
        selected_extra.extend(p for p in matches if p.is_file() and (p.name.endswith(".pgn") or p.name.endswith(".pgn.gz")))
    selected_extra = sorted(set(selected_extra))
    summary["sources"]["extra"] = {"selected_files": len(selected_extra), "patterns": list(args.extra_pgn_glob)}

    selected_all = sorted(set(selected_fishtest + selected_selfplay + selected_extra))
    if not selected_all:
        print("No PGNs selected. Adjust download/selfplay options.", file=sys.stderr)
        return 1

    for i, src in enumerate(selected_all, start=1):
        dst = stage_pgn_dir / f"{i:06d}_{src.name}"
        if args.dry_run:
            print("$", "stage", src, "->", dst)
            continue
        link_or_copy(src, dst)

    summary["staged_pgns"] = {
        "count": len(selected_all),
        "stage_dir": str(stage_pgn_dir.resolve()),
    }

    teacher_cmd = [
        args.python_bin,
        "utils/nnue/bullet/make_teacher_text_dataset.py",
        "--pgn-glob",
        str(stage_pgn_dir / "**/*.pgn"),
        "--pgn-glob",
        str(stage_pgn_dir / "**/*.pgn.gz"),
        "--output",
        str(teacher_output),
        "--stockfish",
        args.stockfish,
        "--depth",
        str(args.teacher_depth),
        "--threads",
        str(args.teacher_threads),
        "--hash-mb",
        str(args.teacher_hash_mb),
        "--sample-rate",
        str(args.sample_rate),
        "--min-ply",
        str(args.min_ply),
        "--max-ply",
        str(args.max_ply),
        "--cp-clip",
        str(args.cp_clip),
    ]
    if args.max_positions > 0:
        teacher_cmd.extend(["--max-positions", str(args.max_positions)])
    if args.skip_check:
        teacher_cmd.append("--skip-check")
    if args.skip_captures:
        teacher_cmd.append("--skip-captures")
    if args.dedupe_fen:
        teacher_cmd.append("--dedupe-fen")

    run_cmd(teacher_cmd, dry_run=args.dry_run)

    pack_cmd = [
        args.python_bin,
        "utils/nnue/bullet/pack_dataset.py",
        "--bullet-utils",
        args.bullet_utils,
        "--text-input",
        str(teacher_output),
        "--output",
        str(packed_output),
        "--shuffle-mem-mb",
        str(args.shuffle_mem_mb),
        "--convert-threads",
        str(args.convert_threads),
    ]
    run_cmd(pack_cmd, dry_run=args.dry_run)

    summary["finished_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    summary_path = stage_dir / "summary.json"
    if args.dry_run:
        print("$", "write-summary", summary_path)
    else:
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Summary: {summary_path}")
        print(f"Teacher text: {teacher_output}")
        print(f"Packed data: {packed_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
