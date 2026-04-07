#!/usr/bin/env python3
"""Gate Bullet checkpoints with self-play only, then promote the best net."""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and promote Bullet checkpoints with self-play."
    )
    parser.add_argument(
        "--checkpoints-dir",
        required=True,
        help="Directory containing Bullet checkpoint folders",
    )
    parser.add_argument(
        "--raw-to-npz",
        default="utils/nnue/bullet/checkpoint_raw_to_npz.py",
        help="Path to raw.bin -> NPZ converter",
    )
    parser.add_argument(
        "--npz-to-net",
        default="utils/nnue/bullet/export_npz_to_syk4.py",
        help="Path to NPZ -> SYKNNUE4 exporter",
    )
    parser.add_argument(
        "--engine", default="./zig-out/bin/sykora", help="Engine under test"
    )

    parser.add_argument("--blend", type=int, default=100, help="NnueBlend for evaluation")
    parser.add_argument(
        "--nnue-scale", type=int, default=100, help="NnueScale during eval"
    )

    parser.add_argument(
        "--selfplay-games",
        type=int,
        default=80,
        help="Self-play games per candidate checkpoint",
    )
    parser.add_argument(
        "--selfplay-movetime-ms", type=int, default=80, help="Self-play movetime"
    )
    parser.add_argument(
        "--selfplay-top-k",
        type=int,
        default=3,
        help="How many recent checkpoints to self-play",
    )

    parser.add_argument(
        "--threads", type=int, default=1, help="Threads for self-play"
    )
    parser.add_argument(
        "--hash-mb", type=int, default=64, help="Hash for self-play"
    )

    parser.add_argument(
        "--min-elo", type=float, default=0.0, help="Promotion threshold"
    )
    parser.add_argument(
        "--max-p-value", type=float, default=0.25, help="Promotion threshold"
    )

    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional directory for reports and converted nets",
    )
    parser.add_argument(
        "--promote-to",
        default="",
        help="Optional destination .sknnue path for promoted net",
    )
    return parser.parse_args()


def run_capture(
    cmd: list[str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    match = re.search(r"-(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return (10**9, name)


def main() -> int:
    args = parse_args()

    if args.blend < 0 or args.blend > 100:
        print("--blend must be in [0, 100]", file=sys.stderr)
        return 2
    if args.nnue_scale < 10 or args.nnue_scale > 400:
        print("--nnue-scale must be in [10, 400]", file=sys.stderr)
        return 2
    if args.selfplay_games <= 0 or args.selfplay_top_k <= 0:
        print("Invalid self-play args", file=sys.stderr)
        return 2
    if args.threads <= 0 or args.hash_mb <= 0:
        print("--threads and --hash-mb must be > 0", file=sys.stderr)
        return 2

    ckpt_root = Path(args.checkpoints_dir)
    if not ckpt_root.is_dir():
        print(f"Checkpoint dir not found: {ckpt_root}", file=sys.stderr)
        return 1

    raw_to_npz = Path(args.raw_to_npz)
    if not raw_to_npz.is_file():
        print(f"raw->npz converter not found: {raw_to_npz}", file=sys.stderr)
        return 1

    npz_to_net = Path(args.npz_to_net)
    if not npz_to_net.is_file():
        print(f"npz->net exporter not found: {npz_to_net}", file=sys.stderr)
        return 1

    engine = Path(args.engine)
    if not engine.is_file():
        print(f"Engine not found: {engine}", file=sys.stderr)
        return 1

    run_id = datetime.datetime.now(datetime.UTC).strftime("gate_%Y%m%dT%H%M%SZ")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = ckpt_root / ".." / "gates" / run_id
    out_dir = out_dir.resolve()
    nets_dir = out_dir / "nets"
    sp_dir = out_dir / "selfplay"
    nets_dir.mkdir(parents=True, exist_ok=True)
    sp_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(
        [
            p
            for p in ckpt_root.iterdir()
            if p.is_dir() and (p / "raw.bin").is_file()
        ],
        key=checkpoint_sort_key,
    )
    if not ckpts:
        print(
            f"No checkpoints with raw.bin found under: {ckpt_root}",
            file=sys.stderr,
        )
        return 1

    records: list[dict] = []

    for ckpt in ckpts:
        npz_out = nets_dir / f"{ckpt.name}.npz"
        net_out = nets_dir / f"{ckpt.name}.sknnue4"
        run_capture(
            [
                sys.executable,
                str(raw_to_npz),
                "--input",
                str(ckpt),
                "--output",
                str(npz_out),
            ]
        )
        run_capture(
            [
                sys.executable,
                str(npz_to_net),
                "--input",
                str(npz_out),
                "--output-net",
                str(net_out),
            ]
        )

        rec = {
            "checkpoint": str(ckpt.resolve()),
            "checkpoint_name": ckpt.name,
            "npz": str(npz_out.resolve()),
            "net": str(net_out.resolve()),
        }
        records.append(rec)

    records.sort(key=lambda r: checkpoint_sort_key(Path(r["checkpoint"])), reverse=True)

    top_k = records[: min(args.selfplay_top_k, len(records))]

    for rec in top_k:
        summary_json = sp_dir / f"{rec['checkpoint_name']}.json"
        stdout_log = sp_dir / f"{rec['checkpoint_name']}.stdout.log"
        stderr_log = sp_dir / f"{rec['checkpoint_name']}.stderr.log"
        selfplay_cmd = [
            sys.executable,
            "utils/match/selfplay.py",
            str(engine),
            str(engine),
            "--name1",
            "base",
            "--name2",
            rec["checkpoint_name"],
            "--games",
            str(args.selfplay_games),
            "--movetime-ms",
            str(args.selfplay_movetime_ms),
            "--threads",
            str(args.threads),
            "--hash-mb",
            str(args.hash_mb),
            "--engine2-opt",
            "UseNNUE=true",
            "--engine2-opt",
            f"EvalFile={rec['net']}",
            "--engine2-opt",
            f"NnueBlend={args.blend}",
            "--engine2-opt",
            f"NnueScale={args.nnue_scale}",
            "--summary-json",
            str(summary_json),
            "--quiet",
        ]
        print("$", " ".join(selfplay_cmd))
        with stdout_log.open("w", encoding="utf-8") as out_f, stderr_log.open("w", encoding="utf-8") as err_f:
            proc = subprocess.run(selfplay_cmd, check=False, text=True, stdout=out_f, stderr=err_f)

        rec["selfplay_exit_code"] = proc.returncode
        rec["selfplay_stdout_log"] = str(stdout_log.resolve())
        rec["selfplay_stderr_log"] = str(stderr_log.resolve())

        if summary_json.is_file():
            payload = json.loads(summary_json.read_text())
            rec["selfplay"] = payload.get("elo", {})
            rec["selfplay_result"] = payload.get("result", {})
            rec["selfplay_summary"] = str(summary_json.resolve())

    promoted: dict | None = None
    passing: list[dict] = []
    for rec in top_k:
        elo = rec.get("selfplay", {})
        elo_cp = float(elo.get("elo_engine2_minus_engine1", -1e9))
        p_val = float(elo.get("p_value_two_sided", 1.0))
        if elo_cp >= args.min_elo and p_val <= args.max_p_value:
            passing.append(rec)

    if passing:
        promoted = max(
            passing,
            key=lambda rec: (
                float(rec.get("selfplay", {}).get("elo_engine2_minus_engine1", -1e9)),
                rec["checkpoint_name"],
            ),
        )

    report = {
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "checkpoints_dir": str(ckpt_root.resolve()),
        "engine": str(engine.resolve()),
        "eval": {
            "blend": args.blend,
            "nnue_scale": args.nnue_scale,
            "selfplay_games": args.selfplay_games,
            "selfplay_movetime_ms": args.selfplay_movetime_ms,
            "selfplay_top_k": args.selfplay_top_k,
            "threads": args.threads,
            "hash_mb": args.hash_mb,
            "min_elo": args.min_elo,
            "max_p_value": args.max_p_value,
        },
        "candidate_count": len(records),
        "tested_count": len(top_k),
        "results": records,
        "promoted": promoted,
    }

    report_path = out_dir / "gate_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Report: {report_path}")
    if promoted is None:
        print("No checkpoint met promotion thresholds.")
        return 0

    print(f"Promoted: {promoted['checkpoint_name']}")
    print(f"Net: {promoted['net']}")

    if args.promote_to:
        promote_to = Path(args.promote_to)
        promote_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(promoted["net"], promote_to)
        print(f"Copied promoted net to: {promote_to}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
