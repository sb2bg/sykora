#!/usr/bin/env python3
"""Inspect sampled lc0 v6 records.

Useful for sanity-checking shard quality before conversion/training.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import struct
from collections import Counter
from pathlib import Path


LC0_V6_RECORD_SIZE = 8356
_STRUCT = struct.Struct("<II1858f104Q8B15fIHHfI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample-inspect lc0 v6 training shards.")
    parser.add_argument("--source-dir", required=True, help="Directory with training.*.gz files")
    parser.add_argument("--glob", default="training.*.gz", help="Shard glob under source dir")
    parser.add_argument("--max-files", type=int, default=24, help="Max shard files to sample")
    parser.add_argument(
        "--records-per-file", type=int, default=8, help="How many records to sample per selected file"
    )
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument(
        "--output-json", default="", help="Optional path for JSON report (default: print only)"
    )
    return parser.parse_args()


def safe_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def sample_record_indices(total_records: int, k: int, rng: random.Random) -> list[int]:
    if total_records <= 0 or k <= 0:
        return []
    if k >= total_records:
        return list(range(total_records))
    return sorted(rng.sample(range(total_records), k))


def inspect_file(path: Path, records_per_file: int, rng: random.Random, report: dict) -> None:
    data = gzip.decompress(path.read_bytes())
    if len(data) % LC0_V6_RECORD_SIZE != 0:
        report["invalid_size_files"].append(str(path))
        return

    total_records = len(data) // LC0_V6_RECORD_SIZE
    indices = sample_record_indices(total_records, records_per_file, rng)
    if not indices:
        return

    for idx in indices:
        start = idx * LC0_V6_RECORD_SIZE
        end = start + LC0_V6_RECORD_SIZE
        row = _STRUCT.unpack(data[start:end])

        version = int(row[0])
        input_format = int(row[1])
        report["version_counts"][version] += 1
        report["input_format_counts"][input_format] += 1

        byte_offset = 2 + 1858 + 104
        castling_us_ooo = int(row[byte_offset + 0])
        castling_us_oo = int(row[byte_offset + 1])
        castling_them_ooo = int(row[byte_offset + 2])
        castling_them_oo = int(row[byte_offset + 3])
        stm_or_ep = int(row[byte_offset + 4])
        rule50 = int(row[byte_offset + 5])

        float_offset = byte_offset + 8
        # Selected head targets.
        root_q = float(row[float_offset + 0])
        root_d = float(row[float_offset + 2])
        root_m = float(row[float_offset + 4])
        plies_left = float(row[float_offset + 6])
        result_q = float(row[float_offset + 7])
        result_d = float(row[float_offset + 8])
        result_m = float(row[float_offset + 9])

        report["rule50_values"].append(rule50)
        report["stm_or_ep_values"].append(stm_or_ep)
        report["castling_patterns"][
            f"{castling_us_ooo}{castling_us_oo}{castling_them_ooo}{castling_them_oo}"
        ] += 1
        report["root_q_values"].append(root_q)
        report["root_d_values"].append(root_d)
        report["root_m_values"].append(root_m)
        report["plies_left_values"].append(plies_left)
        report["result_q_values"].append(result_q)
        report["result_d_values"].append(result_d)
        report["result_m_values"].append(result_m)


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    shards = sorted(source_dir.glob(args.glob))
    if not shards:
        raise RuntimeError(f"No shards matched '{args.glob}' under {source_dir}")

    rng = random.Random(args.seed)
    selected = shards if args.max_files <= 0 else shards[: args.max_files]

    report = {
        "source_dir": str(source_dir.resolve()),
        "glob": args.glob,
        "selected_files": len(selected),
        "total_matching_files": len(shards),
        "record_size_bytes": LC0_V6_RECORD_SIZE,
        "version_counts": Counter(),
        "input_format_counts": Counter(),
        "castling_patterns": Counter(),
        "invalid_size_files": [],
        "rule50_values": [],
        "stm_or_ep_values": [],
        "root_q_values": [],
        "root_d_values": [],
        "root_m_values": [],
        "plies_left_values": [],
        "result_q_values": [],
        "result_d_values": [],
        "result_m_values": [],
    }

    for shard in selected:
        inspect_file(shard, args.records_per_file, rng, report)

    output = {
        "source_dir": report["source_dir"],
        "glob": report["glob"],
        "record_size_bytes": report["record_size_bytes"],
        "selected_files": report["selected_files"],
        "total_matching_files": report["total_matching_files"],
        "invalid_size_file_count": len(report["invalid_size_files"]),
        "version_counts": dict(sorted(report["version_counts"].items())),
        "input_format_counts": dict(sorted(report["input_format_counts"].items())),
        "castling_patterns": dict(sorted(report["castling_patterns"].items())),
        "rule50_stats": safe_stats(report["rule50_values"]),
        "stm_or_ep_stats": safe_stats(report["stm_or_ep_values"]),
        "root_q_stats": safe_stats(report["root_q_values"]),
        "root_d_stats": safe_stats(report["root_d_values"]),
        "root_m_stats": safe_stats(report["root_m_values"]),
        "plies_left_stats": safe_stats(report["plies_left_values"]),
        "result_q_stats": safe_stats(report["result_q_values"]),
        "result_d_stats": safe_stats(report["result_d_values"]),
        "result_m_stats": safe_stats(report["result_m_values"]),
    }

    text = json.dumps(output, indent=2, sort_keys=False)
    print(text)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
