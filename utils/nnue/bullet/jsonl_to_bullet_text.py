#!/usr/bin/env python3
"""Convert labeled JSONL positions into Bullet text format.

Output line format:
<FEN> | <score_cp_white_relative> | <result_white_relative>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JSONL -> Bullet text converter")
    parser.add_argument("--input", required=True, help="Input JSONL with labeled positions")
    parser.add_argument("--output", required=True, help="Output text file for bullet-utils convert --from text")
    parser.add_argument(
        "--score-key",
        default="teacher_cp_stm",
        help="Centipawn score key, interpreted side-to-move relative unless --score-is-white-relative is set",
    )
    parser.add_argument(
        "--result-key",
        default="target_result_stm",
        help="Result target key, interpreted side-to-move relative unless --result-is-white-relative is set",
    )
    parser.add_argument(
        "--stm-key",
        default="stm_white",
        help="Boolean key: true if side-to-move is white",
    )
    parser.add_argument(
        "--score-is-white-relative",
        action="store_true",
        help="Treat --score-key as already white-relative",
    )
    parser.add_argument(
        "--result-is-white-relative",
        action="store_true",
        help="Treat --result-key as already white-relative",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to convert (0 = all)")
    parser.add_argument("--cp-clip", type=int, default=2000, help="Clamp score to [-cp-clip, cp-clip]")
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in ("1", "true", "yes", "y", "on"):
            return True
        if lower in ("0", "false", "no", "n", "off"):
            return False
    raise ValueError(f"Cannot parse bool from {value!r}")


def clamp_result(value: float) -> float:
    return max(0.0, min(1.0, value))


def result_token(value: float) -> str:
    # Bullet text parser accepts exact tokens: 1.0 / 0.5 / 0.0 (or 1/0/1/2).
    if value >= 0.75:
        return "1.0"
    if value <= 0.25:
        return "0.0"
    return "0.5"


def main() -> int:
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    with in_path.open("r", encoding="utf-8") as inp, out_path.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue

            row = json.loads(line)
            fen = row.get("fen")
            if not isinstance(fen, str) or not fen:
                skipped += 1
                continue

            try:
                stm_white = parse_bool(row.get(args.stm_key, True))
            except ValueError:
                skipped += 1
                continue

            try:
                score_raw = float(row.get(args.score_key, 0.0))
                result_raw = float(row.get(args.result_key, 0.5))
            except (TypeError, ValueError):
                skipped += 1
                continue

            if not math.isfinite(score_raw) or not math.isfinite(result_raw):
                skipped += 1
                continue

            if args.score_is_white_relative:
                score_white = score_raw
            else:
                score_white = score_raw if stm_white else -score_raw
            score_white = int(max(-args.cp_clip, min(args.cp_clip, round(score_white))))

            if args.result_is_white_relative:
                result_white = result_raw
            else:
                result_white = result_raw if stm_white else (1.0 - result_raw)
            result_white = clamp_result(result_white)

            out.write(f"{fen} | {score_white} | {result_token(result_white)}\n")
            converted += 1

            if args.limit > 0 and converted >= args.limit:
                break

    print(f"Input: {in_path}")
    print(f"Output: {out_path}")
    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
