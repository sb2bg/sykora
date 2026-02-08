#!/usr/bin/env python3
"""Prepare an lc0 v6 shard set for Bullet training workflows.

This script does not modify shard contents. It validates basic structure,
counts records using gzip ISIZE, and writes a manifest + summary metadata.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import struct
from collections import Counter
from pathlib import Path
from typing import Iterable, List


LC0_V6_RECORD_SIZE = 8356
_HEADER_STRUCT = struct.Struct("<II")  # version, input_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and stage lc0 v6 training shards for Bullet."
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing training.*.gz shards",
    )
    parser.add_argument(
        "--glob",
        default="training.*.gz",
        help="Shard glob under --source-dir (default: training.*.gz)",
    )
    parser.add_argument(
        "--output-dir",
        default="nnue/data/bullet/lc0_dataset",
        help="Output folder for manifest + summary metadata",
    )
    parser.add_argument(
        "--manifest-name",
        default="shards.txt",
        help="Manifest filename to write under --output-dir",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Optional cap from sorted shard list (0 = all)",
    )
    parser.add_argument(
        "--sample-files",
        type=int,
        default=0,
        help="Optional random sample size from shard list (0 = disabled)",
    )
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write manifest entries relative to --source-dir instead of absolute paths",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.json",
        help="Summary JSON filename under --output-dir",
    )
    return parser.parse_args()


def gzip_isize(path: Path) -> int:
    # Gzip trailer stores uncompressed size modulo 2^32.
    with path.open("rb") as handle:
        handle.seek(-4, 2)
        raw = handle.read(4)
    if len(raw) != 4:
        raise ValueError(f"Invalid gzip trailer: {path}")
    return int.from_bytes(raw, "little", signed=False)


def read_first_header(path: Path) -> tuple[int, int]:
    with gzip.open(path, "rb") as handle:
        chunk = handle.read(8)
    if len(chunk) != 8:
        raise ValueError(f"Shard too short to contain lc0 header: {path}")
    return _HEADER_STRUCT.unpack(chunk)


def select_shards(files: List[Path], limit_files: int, sample_files: int, seed: int) -> List[Path]:
    chosen = files
    if limit_files > 0:
        chosen = chosen[:limit_files]
    if sample_files > 0:
        if sample_files > len(chosen):
            raise ValueError(
                f"--sample-files={sample_files} exceeds available shards ({len(chosen)})"
            )
        rng = random.Random(seed)
        chosen = sorted(rng.sample(chosen, sample_files))
    return chosen


def write_manifest(path: Path, files: Iterable[Path], source_dir: Path, relative_paths: bool) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in files:
            if relative_paths:
                handle.write(str(item.relative_to(source_dir)) + "\n")
            else:
                handle.write(str(item.resolve()) + "\n")


def main() -> int:
    args = parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    shards = sorted(source_dir.glob(args.glob))
    if not shards:
        raise RuntimeError(f"No shards matched '{args.glob}' under {source_dir}")

    chosen = select_shards(shards, args.limit_files, args.sample_files, args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    version_counts: Counter[int] = Counter()
    input_format_counts: Counter[int] = Counter()
    invalid_size: list[str] = []
    zero_record_files: list[str] = []
    total_records = 0
    total_uncompressed_bytes = 0
    total_compressed_bytes = 0

    for shard in chosen:
        version, input_format = read_first_header(shard)
        version_counts[version] += 1
        input_format_counts[input_format] += 1

        isize = gzip_isize(shard)
        total_uncompressed_bytes += isize
        total_compressed_bytes += shard.stat().st_size

        if isize % LC0_V6_RECORD_SIZE != 0:
            invalid_size.append(str(shard))
            continue

        records = isize // LC0_V6_RECORD_SIZE
        if records == 0:
            zero_record_files.append(str(shard))
        total_records += records

    manifest_path = out_dir / args.manifest_name
    write_manifest(manifest_path, chosen, source_dir, args.relative_paths)

    summary = {
        "source_dir": str(source_dir.resolve()),
        "glob": args.glob,
        "record_size_bytes": LC0_V6_RECORD_SIZE,
        "selected_files": len(chosen),
        "all_matching_files": len(shards),
        "total_records_estimated": total_records,
        "total_uncompressed_bytes": total_uncompressed_bytes,
        "total_compressed_bytes": total_compressed_bytes,
        "version_counts": dict(sorted(version_counts.items())),
        "input_format_counts": dict(sorted(input_format_counts.items())),
        "invalid_size_files": invalid_size,
        "zero_record_files": zero_record_files,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_entry_mode": "relative" if args.relative_paths else "absolute",
    }

    summary_path = out_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")

    print(f"Source dir: {source_dir}")
    print(f"Shards selected: {len(chosen)} / {len(shards)}")
    print(f"Estimated records: {total_records}")
    print(f"Version counts: {dict(sorted(version_counts.items()))}")
    print(f"Input format counts: {dict(sorted(input_format_counts.items()))}")
    if invalid_size:
        print(f"Invalid-size shards: {len(invalid_size)}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
