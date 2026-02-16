#!/usr/bin/env python3
"""Convert/shuffle/interleave data into BulletFormat (.data).

This script wraps bullet-utils so large datasets are packed reproducibly.
"""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack NNUE training data for Bullet.")
    parser.add_argument(
        "--bullet-utils",
        default="nnue/bullet_repo/target/release/bullet-utils",
        help="Path to bullet-utils binary",
    )
    parser.add_argument(
        "--text-input",
        action="append",
        default=[],
        help="Text input path or glob (<FEN> | <cp> | <result>). Repeatable.",
    )
    parser.add_argument(
        "--data-input",
        action="append",
        default=[],
        help="Existing BulletFormat .data input path or glob. Repeatable.",
    )
    parser.add_argument("--output", required=True, help="Output BulletFormat .data path")
    parser.add_argument(
        "--shuffle-mem-mb",
        type=int,
        default=4096,
        help="Memory budget for bullet-utils shuffle",
    )
    parser.add_argument("--convert-threads", type=int, default=4, help="Threads for text->data convert")
    parser.add_argument(
        "--work-dir",
        default="",
        help="Optional working directory for intermediate files (default: temp dir)",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep intermediate files for debugging",
    )
    return parser.parse_args()


def expand_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            out.extend(Path(m) for m in matches)
        else:
            out.append(Path(pattern))
    uniq = sorted(set(out))
    return [p for p in uniq if p.is_file()]


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> int:
    args = parse_args()

    if args.shuffle_mem_mb <= 0:
        print("--shuffle-mem-mb must be > 0", file=sys.stderr)
        return 2
    if args.convert_threads <= 0:
        print("--convert-threads must be > 0", file=sys.stderr)
        return 2

    bullet_utils = Path(args.bullet_utils)
    if not bullet_utils.is_file():
        print(f"bullet-utils not found: {bullet_utils}", file=sys.stderr)
        return 1

    text_inputs = expand_paths(args.text_input)
    data_inputs = expand_paths(args.data_input)

    if not text_inputs and not data_inputs:
        print("No inputs provided. Use --text-input and/or --data-input.", file=sys.stderr)
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.work_dir:
        work_root = Path(args.work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        work_ctx = None
        work_dir = work_root
    else:
        work_ctx = tempfile.TemporaryDirectory(prefix="pack_dataset_")
        work_dir = Path(work_ctx.name)

    print(f"Working directory: {work_dir}")

    shuffled_parts: list[Path] = []

    try:
        for i, txt in enumerate(text_inputs, start=1):
            raw_part = work_dir / f"text_{i:03d}.raw.data"
            shuf_part = work_dir / f"text_{i:03d}.shuf.data"

            run_cmd(
                [
                    str(bullet_utils),
                    "convert",
                    "--from",
                    "text",
                    "--input",
                    str(txt),
                    "--output",
                    str(raw_part),
                    "--threads",
                    str(args.convert_threads),
                ]
            )
            run_cmd(
                [
                    str(bullet_utils),
                    "shuffle",
                    "--input",
                    str(raw_part),
                    "--output",
                    str(shuf_part),
                    "--mem-used-mb",
                    str(args.shuffle_mem_mb),
                ]
            )
            shuffled_parts.append(shuf_part)

        for i, data_path in enumerate(data_inputs, start=1):
            shuf_part = work_dir / f"data_{i:03d}.shuf.data"
            run_cmd(
                [
                    str(bullet_utils),
                    "shuffle",
                    "--input",
                    str(data_path),
                    "--output",
                    str(shuf_part),
                    "--mem-used-mb",
                    str(args.shuffle_mem_mb),
                ]
            )
            shuffled_parts.append(shuf_part)

        if len(shuffled_parts) == 1:
            shutil.copyfile(shuffled_parts[0], output)
        else:
            cmd = [str(bullet_utils), "interleave"] + [str(p) for p in shuffled_parts] + ["--output", str(output)]
            run_cmd(cmd)

        run_cmd([str(bullet_utils), "validate", "--input", str(output)])
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    finally:
        if work_ctx is not None and not args.keep_work:
            work_ctx.cleanup()

    size = output.stat().st_size
    print(f"Output: {output}")
    print(f"Bytes:  {size}")
    print(f"Records (32B): {size // 32}")

    if args.keep_work:
        print(f"Kept intermediates in: {work_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
