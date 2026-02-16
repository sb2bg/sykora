#!/usr/bin/env python3
"""Run Bullet training with long-run defaults for an RTX 4070 Ti SUPER.

This wraps `cargo run -r --example 1_simple` and records run metadata.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Bullet NNUE training runs.")
    parser.add_argument("--dataset", required=True, help="Input BulletFormat .data dataset")
    parser.add_argument("--bullet-repo", default="nnue/bullet_repo", help="Path to Bullet repository")
    parser.add_argument("--output-root", default="nnue/models/bullet", help="Training run root")
    parser.add_argument("--run-id", default="", help="Run identifier (default: utc timestamp)")

    # Architecture/training knobs (defaults are intentionally long-run)
    parser.add_argument("--hidden", type=int, default=256, help="Hidden size for 1_simple")
    parser.add_argument("--start-superbatch", type=int, default=1, help="Start superbatch")
    parser.add_argument("--end-superbatch", type=int, default=320, help="End superbatch")
    parser.add_argument("--lr-start", type=float, default=0.0010, help="Initial LR")
    parser.add_argument(
        "--lr-final",
        type=float,
        default=0.0,
        help="Final LR (0 = use lr_start * 0.3^5)",
    )
    parser.add_argument("--wdl", type=float, default=0.75, help="WDL blend used by Bullet")
    parser.add_argument("--save-rate", type=int, default=1, help="Save every N superbatches")
    parser.add_argument("--threads", type=int, default=8, help="Bullet training/data threads")

    parser.add_argument(
        "--resume",
        default="",
        help="Optional checkpoint path to resume from (<checkpoint_dir>/raw.bin or checkpoint dir)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command/env and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_file():
        print(f"Dataset not found: {dataset}", file=sys.stderr)
        return 1

    bullet_repo = Path(args.bullet_repo)
    if not bullet_repo.is_dir():
        print(f"Bullet repo not found: {bullet_repo}", file=sys.stderr)
        return 1

    if args.hidden <= 0:
        print("--hidden must be > 0", file=sys.stderr)
        return 2
    if args.start_superbatch <= 0 or args.end_superbatch < args.start_superbatch:
        print("Invalid superbatch bounds", file=sys.stderr)
        return 2
    if args.lr_start <= 0:
        print("--lr-start must be > 0", file=sys.stderr)
        return 2
    if args.save_rate <= 0 or args.threads <= 0:
        print("--save-rate and --threads must be > 0", file=sys.stderr)
        return 2

    run_id = args.run_id.strip() or datetime.datetime.now(datetime.UTC).strftime("run_%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root)
    run_dir = output_root / run_id
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    final_lr = args.lr_final if args.lr_final > 0 else args.lr_start * (0.3**5)

    env = os.environ.copy()
    env.update(
        {
            "SYK_DATASET": str(dataset.resolve()),
            "SYK_HIDDEN": str(args.hidden),
            "SYK_LR_START": str(args.lr_start),
            "SYK_LR_FINAL": str(final_lr),
            "SYK_START_SUPERBATCH": str(args.start_superbatch),
            "SYK_END_SUPERBATCH": str(args.end_superbatch),
            "SYK_WDL": str(args.wdl),
            "SYK_SAVE_RATE": str(args.save_rate),
            "SYK_THREADS": str(args.threads),
            "SYK_OUTPUT_DIR": str(ckpt_dir.resolve()),
            "SYK_NET_ID": run_id,
        }
    )
    if args.resume:
        env["SYK_RESUME"] = str(Path(args.resume).resolve())

    cmd = ["cargo", "run", "-r", "--example", "1_simple"]

    meta = {
        "run_id": run_id,
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "bullet_repo": str(bullet_repo.resolve()),
        "dataset": str(dataset.resolve()),
        "output_dir": str(run_dir.resolve()),
        "checkpoint_dir": str(ckpt_dir.resolve()),
        "command": cmd,
        "env": {
            "SYK_DATASET": env["SYK_DATASET"],
            "SYK_HIDDEN": env["SYK_HIDDEN"],
            "SYK_LR_START": env["SYK_LR_START"],
            "SYK_LR_FINAL": env["SYK_LR_FINAL"],
            "SYK_START_SUPERBATCH": env["SYK_START_SUPERBATCH"],
            "SYK_END_SUPERBATCH": env["SYK_END_SUPERBATCH"],
            "SYK_WDL": env["SYK_WDL"],
            "SYK_SAVE_RATE": env["SYK_SAVE_RATE"],
            "SYK_THREADS": env["SYK_THREADS"],
            "SYK_OUTPUT_DIR": env["SYK_OUTPUT_DIR"],
            "SYK_NET_ID": env["SYK_NET_ID"],
            "SYK_RESUME": env.get("SYK_RESUME", ""),
        },
    }

    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    print(f"Dataset: {dataset}")
    print(f"Metadata: {meta_path}")
    print("$", " ".join(cmd))

    if args.dry_run:
        return 0

    try:
        subprocess.run(cmd, cwd=str(bullet_repo), env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Training failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    print("Training finished.")
    print(f"Checkpoints: {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
