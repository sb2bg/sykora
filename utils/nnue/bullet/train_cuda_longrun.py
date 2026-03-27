#!/usr/bin/env python3
"""Run Bullet training with long-run defaults for an RTX 4070 Ti SUPER.

This wraps `cargo run -r --example sykora_bucketed` and records run metadata.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from bootstrap import DEFAULT_BULLET_REPO, ensure_bullet_repo  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]

SYKORA10_BUCKET_LAYOUT_32 = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
]

SYKORA16_BUCKET_LAYOUT_32 = [
    0, 0, 1, 1,
    2, 2, 3, 3,
    4, 4, 5, 5,
    6, 6, 7, 7,
    8, 8, 9, 9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    14, 14, 15, 15,
]


def expand_mirrored_bucket_layout(layout_32: list[int]) -> list[int]:
    mirror = [0, 1, 2, 3, 3, 2, 1, 0]
    return [int(layout_32[(idx // 8) * 4 + mirror[idx % 8]]) for idx in range(64)]


def bucket_layout_64(name: str) -> list[int]:
    if name == "sykora16":
        return expand_mirrored_bucket_layout(SYKORA16_BUCKET_LAYOUT_32)
    return expand_mirrored_bucket_layout(SYKORA10_BUCKET_LAYOUT_32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Bullet NNUE training runs.")
    parser.add_argument(
        "--dataset",
        required=True,
        nargs="+",
        help="Input dataset file(s) (.data or .binpack)",
    )
    parser.add_argument(
        "--bullet-repo", default=str(DEFAULT_BULLET_REPO), help="Path to Bullet repository"
    )
    parser.add_argument(
        "--output-root", default="nnue/models/bullet", help="Training run root"
    )
    parser.add_argument(
        "--run-id", default="", help="Run identifier (default: utc timestamp)"
    )

    # Architecture/training knobs (defaults are intentionally long-run)
    parser.add_argument(
        "--hidden", type=int, default=0, help="Hidden size (default: 128 for syk3, 1536 for syk4)"
    )
    parser.add_argument(
        "--network-format",
        choices=["syk3", "syk4"],
        default="syk3",
        help="Training network format",
    )
    parser.add_argument(
        "--bucket-layout",
        choices=["sykora10", "sykora16"],
        default="",
        help="Mirrored king-bucket layout (default: sykora10 for syk3, sykora16 for syk4)",
    )
    parser.add_argument(
        "--dense-l1",
        type=int,
        default=16,
        help="SYKNNUE4 dense layer 1 width",
    )
    parser.add_argument(
        "--dense-l2",
        type=int,
        default=32,
        help="SYKNNUE4 dense layer 2 width",
    )
    parser.add_argument(
        "--start-superbatch", type=int, default=1, help="Start superbatch"
    )
    parser.add_argument(
        "--end-superbatch", type=int, default=320, help="End superbatch"
    )
    parser.add_argument("--lr-start", type=float, default=0.0010, help="Initial LR")
    parser.add_argument(
        "--lr-final",
        type=float,
        default=0.0,
        help="Final LR (0 = use lr_start * 0.3^5)",
    )
    parser.add_argument(
        "--wdl", type=float, default=0.75, help="WDL blend used by Bullet"
    )
    parser.add_argument(
        "--save-rate", type=int, default=1, help="Save every N superbatches"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="Bullet training/data threads"
    )

    # Data format
    parser.add_argument(
        "--data-format",
        choices=["bullet", "binpack"],
        default="bullet",
        help="Dataset format: bullet (.data) or binpack (.binpack) (default: bullet)",
    )
    parser.add_argument(
        "--binpack-buffer-mb",
        type=int,
        default=1024,
        help="SfBinpackLoader buffer size in MB",
    )
    parser.add_argument(
        "--binpack-threads",
        type=int,
        default=4,
        help="SfBinpackLoader decompression threads",
    )

    parser.add_argument(
        "--resume",
        default="",
        help="Optional checkpoint path to resume from (<checkpoint_dir>/raw.bin or checkpoint dir)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print command/env and exit"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.bucket_layout:
        args.bucket_layout = "sykora16" if args.network_format == "syk4" else "sykora10"
    if args.hidden <= 0:
        args.hidden = 1536 if args.network_format == "syk4" else 128

    datasets = [Path(d) for d in args.dataset]
    for dataset in datasets:
        if not dataset.is_file():
            print(f"Dataset not found: {dataset}", file=sys.stderr)
            return 1

    # Auto-detect format from file extension if not explicitly set
    data_format = args.data_format
    if data_format == "bullet" and any(str(d).endswith(".binpack") for d in datasets):
        data_format = "binpack"
        print("Auto-detected binpack format from file extension")

    # SfBinpackLoader takes semicolon-separated paths
    dataset_str = ";".join(str(d.resolve()) for d in datasets)

    bullet_repo = ensure_bullet_repo(Path(args.bullet_repo))

    if args.hidden <= 0:
        print("--hidden must be > 0", file=sys.stderr)
        return 2
    if args.dense_l1 <= 0 or args.dense_l2 <= 0:
        print("--dense-l1 and --dense-l2 must be > 0", file=sys.stderr)
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

    run_id = args.run_id.strip() or datetime.datetime.now(datetime.UTC).strftime(
        "run_%Y%m%dT%H%M%SZ"
    )
    output_root = Path(args.output_root)
    run_dir = output_root / run_id
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    final_lr = args.lr_final if args.lr_final > 0 else args.lr_start * (0.3**5)

    env = os.environ.copy()
    env.update(
        {
            "SYK_DATASET": dataset_str,
            "SYK_HIDDEN": str(args.hidden),
            "SYK_NETWORK_FORMAT": args.network_format,
            "SYK_BUCKET_LAYOUT": args.bucket_layout,
            "SYK_DENSE_L1": str(args.dense_l1),
            "SYK_DENSE_L2": str(args.dense_l2),
            "SYK_LR_START": str(args.lr_start),
            "SYK_LR_FINAL": str(final_lr),
            "SYK_START_SUPERBATCH": str(args.start_superbatch),
            "SYK_END_SUPERBATCH": str(args.end_superbatch),
            "SYK_WDL": str(args.wdl),
            "SYK_SAVE_RATE": str(args.save_rate),
            "SYK_THREADS": str(args.threads),
            "SYK_OUTPUT_DIR": str(ckpt_dir.resolve()),
            "SYK_NET_ID": run_id,
            "SYK_DATA_FORMAT": data_format,
            "SYK_BINPACK_BUFFER_MB": str(args.binpack_buffer_mb),
            "SYK_BINPACK_THREADS": str(args.binpack_threads),
        }
    )
    if args.resume:
        env["SYK_RESUME"] = str(Path(args.resume).resolve())

    manifest = REPO_ROOT / "utils" / "nnue" / "bullet_runner" / "Cargo.toml"
    cmd = ["cargo", "run", "-r", "--manifest-path", str(manifest)]

    meta = {
        "run_id": run_id,
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "bullet_repo": str(bullet_repo.resolve()),
        "datasets": [str(d.resolve()) for d in datasets],
        "data_format": data_format,
        "output_dir": str(run_dir.resolve()),
        "checkpoint_dir": str(ckpt_dir.resolve()),
        "command": cmd,
        "network": {
            "format": args.network_format,
            "bucket_layout_name": args.bucket_layout,
            "bucket_layout_64": bucket_layout_64(args.bucket_layout),
            "ft_hidden": args.hidden,
            "dense_l1": args.dense_l1,
            "dense_l2": args.dense_l2,
            "stack_count": 8,
        },
        "env": {
            "SYK_DATASET": env["SYK_DATASET"],
            "SYK_HIDDEN": env["SYK_HIDDEN"],
            "SYK_NETWORK_FORMAT": env["SYK_NETWORK_FORMAT"],
            "SYK_BUCKET_LAYOUT": env["SYK_BUCKET_LAYOUT"],
            "SYK_DENSE_L1": env["SYK_DENSE_L1"],
            "SYK_DENSE_L2": env["SYK_DENSE_L2"],
            "SYK_LR_START": env["SYK_LR_START"],
            "SYK_LR_FINAL": env["SYK_LR_FINAL"],
            "SYK_START_SUPERBATCH": env["SYK_START_SUPERBATCH"],
            "SYK_END_SUPERBATCH": env["SYK_END_SUPERBATCH"],
            "SYK_WDL": env["SYK_WDL"],
            "SYK_SAVE_RATE": env["SYK_SAVE_RATE"],
            "SYK_THREADS": env["SYK_THREADS"],
            "SYK_OUTPUT_DIR": env["SYK_OUTPUT_DIR"],
            "SYK_NET_ID": env["SYK_NET_ID"],
            "SYK_DATA_FORMAT": env["SYK_DATA_FORMAT"],
            "SYK_BINPACK_BUFFER_MB": env["SYK_BINPACK_BUFFER_MB"],
            "SYK_BINPACK_THREADS": env["SYK_BINPACK_THREADS"],
            "SYK_RESUME": env.get("SYK_RESUME", ""),
        },
    }

    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    print(f"Data format: {data_format}")
    for d in datasets:
        print(f"  Dataset: {d}")
    print(f"Metadata: {meta_path}")
    print("$", " ".join(cmd))

    if args.dry_run:
        return 0

    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Training failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    print("Training finished.")
    print(f"Checkpoints: {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
