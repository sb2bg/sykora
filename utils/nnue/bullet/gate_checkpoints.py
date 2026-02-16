#!/usr/bin/env python3
"""Gate Bullet checkpoints with STS + optional self-play, then promote best net."""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


TOTAL_RE = re.compile(r"^TOTAL\s+\d+\s+(\d+)\s+(\d+)\s+([0-9.]+)%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and promote Bullet checkpoints.")
    parser.add_argument("--checkpoints-dir", required=True, help="Directory containing Bullet checkpoint folders")
    parser.add_argument(
        "--converter",
        default="utils/nnue/bullet/bullet_quantised_to_sknnue.py",
        help="Path to quantised->sknnue converter",
    )
    parser.add_argument("--engine", default="./zig-out/bin/sykora", help="Engine under test")

    parser.add_argument("--blend", type=int, default=2, help="NnueBlend for evaluation")
    parser.add_argument(
        "--no-screlu",
        action="store_true",
        help="Disable NnueSCReLU during eval (SCReLU is enabled by default)",
    )
    parser.add_argument("--nnue-scale", type=int, default=100, help="NnueScale during eval")

    parser.add_argument("--sts-epd", default="epd", help="STS directory or file")
    parser.add_argument("--sts-movetime-ms", type=int, default=40, help="STS movetime")
    parser.add_argument("--sts-max-positions", type=int, default=400, help="STS position cap")

    parser.add_argument("--selfplay-games", type=int, default=0, help="Self-play games for top nets (0=skip)")
    parser.add_argument("--selfplay-movetime-ms", type=int, default=80, help="Self-play movetime")
    parser.add_argument("--selfplay-top-k", type=int, default=3, help="How many top STS nets to self-play")

    parser.add_argument("--threads", type=int, default=1, help="Threads for STS/self-play")
    parser.add_argument("--hash-mb", type=int, default=64, help="Hash for STS/self-play")

    parser.add_argument("--min-elo", type=float, default=0.0, help="Promotion threshold")
    parser.add_argument("--max-p-value", type=float, default=0.25, help="Promotion threshold")

    parser.add_argument("--output-dir", default="", help="Optional directory for reports and converted nets")
    parser.add_argument("--promote-to", default="", help="Optional destination .sknnue path for promoted net")
    return parser.parse_args()


def run_capture(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def parse_total_line(output: str) -> tuple[int, int, float]:
    for line in output.splitlines():
        m = TOTAL_RE.match(line.strip())
        if m:
            points = int(m.group(1))
            max_points = int(m.group(2))
            pct = float(m.group(3))
            return points, max_points, pct
    raise ValueError("Could not parse TOTAL line from STS output")


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    match = re.search(r"-(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return (10**9, name)


def main() -> int:
    args = parse_args()
    screlu = not args.no_screlu

    if args.blend < 0 or args.blend > 100:
        print("--blend must be in [0, 100]", file=sys.stderr)
        return 2
    if args.nnue_scale < 10 or args.nnue_scale > 400:
        print("--nnue-scale must be in [10, 400]", file=sys.stderr)
        return 2
    if args.sts_movetime_ms <= 0 or args.sts_max_positions <= 0:
        print("Invalid STS bounds", file=sys.stderr)
        return 2
    if args.selfplay_games < 0 or args.selfplay_top_k <= 0:
        print("Invalid self-play args", file=sys.stderr)
        return 2
    if args.threads <= 0 or args.hash_mb <= 0:
        print("--threads and --hash-mb must be > 0", file=sys.stderr)
        return 2

    ckpt_root = Path(args.checkpoints_dir)
    if not ckpt_root.is_dir():
        print(f"Checkpoint dir not found: {ckpt_root}", file=sys.stderr)
        return 1

    converter = Path(args.converter)
    if not converter.is_file():
        print(f"Converter not found: {converter}", file=sys.stderr)
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
    sts_dir = out_dir / "sts"
    sp_dir = out_dir / "selfplay"
    nets_dir.mkdir(parents=True, exist_ok=True)
    sts_dir.mkdir(parents=True, exist_ok=True)
    sp_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(
        [p for p in ckpt_root.iterdir() if p.is_dir() and (p / "quantised.bin").is_file()],
        key=checkpoint_sort_key,
    )
    if not ckpts:
        print(f"No checkpoints with quantised.bin found under: {ckpt_root}", file=sys.stderr)
        return 1

    records: list[dict] = []

    for ckpt in ckpts:
        net_out = nets_dir / f"{ckpt.name}.sknnue"
        run_capture(
            [
                sys.executable,
                str(converter),
                "--input",
                str(ckpt / "quantised.bin"),
                "--output-net",
                str(net_out),
            ]
        )

        sts_cmd = [
            sys.executable,
            "utils/sts/sts.py",
            "--epd",
            str(args.sts_epd),
            "--engine",
            str(engine),
            "--movetime-ms",
            str(args.sts_movetime_ms),
            "--max-positions",
            str(args.sts_max_positions),
            "--show",
            "none",
            "--threads",
            str(args.threads),
            "--hash-mb",
            str(args.hash_mb),
            "--engine-opt",
            "UseNNUE=true",
            "--engine-opt",
            f"EvalFile={net_out}",
            "--engine-opt",
            f"NnueBlend={args.blend}",
            "--engine-opt",
                f"NnueScale={args.nnue_scale}",
                "--engine-opt",
                f"NnueSCReLU={'true' if screlu else 'false'}",
        ]
        sts_proc = run_capture(sts_cmd)
        sts_log = sts_dir / f"{ckpt.name}.txt"
        sts_log.write_text(sts_proc.stdout)

        points, max_points, pct = parse_total_line(sts_proc.stdout)

        rec = {
            "checkpoint": str(ckpt.resolve()),
            "checkpoint_name": ckpt.name,
            "net": str(net_out.resolve()),
            "sts": {
                "points": points,
                "max_points": max_points,
                "score_pct": pct,
                "log": str(sts_log.resolve()),
            },
        }
        records.append(rec)

    records.sort(key=lambda r: (r["sts"]["score_pct"], r["checkpoint_name"]), reverse=True)

    top_k = records[: min(args.selfplay_top_k, len(records))]

    if args.selfplay_games > 0:
        for rec in top_k:
            summary_json = sp_dir / f"{rec['checkpoint_name']}.json"
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
                "--engine2-opt",
                f"NnueSCReLU={'true' if screlu else 'false'}",
                "--summary-json",
                str(summary_json),
                "--quiet",
            ]
            # selfplay.py exits non-zero when candidate loses/ties, but still writes summary.
            print("$", " ".join(selfplay_cmd))
            subprocess.run(selfplay_cmd, check=False)

            if summary_json.is_file():
                payload = json.loads(summary_json.read_text())
                rec["selfplay"] = payload.get("elo", {})
                rec["selfplay_summary"] = str(summary_json.resolve())

    promoted: dict | None = None
    for rec in records:
        elo = rec.get("selfplay", {})
        elo_cp = float(elo.get("elo_engine2_minus_engine1", -1e9))
        p_val = float(elo.get("p_value_two_sided", 1.0))
        if args.selfplay_games == 0:
            promoted = rec
            break
        if elo_cp >= args.min_elo and p_val <= args.max_p_value:
            promoted = rec
            break

    report = {
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "checkpoints_dir": str(ckpt_root.resolve()),
        "engine": str(engine.resolve()),
        "eval": {
            "blend": args.blend,
            "screlu": screlu,
            "nnue_scale": args.nnue_scale,
            "sts_epd": args.sts_epd,
            "sts_movetime_ms": args.sts_movetime_ms,
            "sts_max_positions": args.sts_max_positions,
            "selfplay_games": args.selfplay_games,
            "selfplay_movetime_ms": args.selfplay_movetime_ms,
            "threads": args.threads,
            "hash_mb": args.hash_mb,
            "min_elo": args.min_elo,
            "max_p_value": args.max_p_value,
        },
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
