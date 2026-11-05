#!/usr/bin/env python3
"""Expand a v3_10 SYKNNUE8 net into the b16 king-bucket layout.

Every b16 bucket lies inside exactly one v3_10 bucket, so copying that
bucket's PSQ rows gives a net whose evaluation is identical to the input's.
The engine's bench must therefore match the original net's, which tests the
16-bucket load, indexing, and refresh paths end to end. The output can also
serve as a function-preserving warm start for b16 training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

UTILS_NNUE_DIR = Path(__file__).resolve().parents[1]
if str(UTILS_NNUE_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_NNUE_DIR))

from common import (  # noqa: E402
    B16_BUCKET_LAYOUT_32,
    PSQ_INPUT_SIZE,
    SECTION_FT_BIAS,
    SECTION_FT_WEIGHT,
    SECTION_L1_BIAS,
    SECTION_L1_WEIGHT,
    SECTION_L2_BIAS,
    SECTION_L2_WEIGHT,
    SECTION_OUT_BIAS,
    SECTION_OUT_WEIGHT,
    SECTION_THREAT_WEIGHT,
    V3_BUCKET_LAYOUT_32,
    expand_mirrored_bucket_layout,
    read_syk_nnue_v8,
    write_syk_nnue_v8,
)


def parent_buckets() -> list[int]:
    """For each b16 bucket, the v3_10 bucket that contains it."""
    parents: dict[int, int] = {}
    for new, old in zip(B16_BUCKET_LAYOUT_32, V3_BUCKET_LAYOUT_32):
        if parents.setdefault(new, old) != old:
            raise AssertionError(f"b16 bucket {new} spans several v3_10 buckets")
    return [parents[bucket] for bucket in range(len(parents))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="v3_10 SYKNNUE8 net")
    parser.add_argument("output", type=Path, help="b16 SYKNNUE8 net to write")
    args = parser.parse_args()

    net = read_syk_nnue_v8(args.input)
    if net["bucket_layout_64"] != expand_mirrored_bucket_layout(V3_BUCKET_LAYOUT_32):
        raise ValueError("input net does not use the v3_10 layout")
    sections = net["sections"]
    h = net["ft_hidden_size"]
    row_bytes = PSQ_INPUT_SIZE * h * 2  # one bucket's i16 PSQ rows
    psq = sections[SECTION_FT_WEIGHT]["payload"]
    expanded = b"".join(psq[old * row_bytes : (old + 1) * row_bytes] for old in parent_buckets())

    write_syk_nnue_v8(
        args.output,
        ft_hidden_size=h,
        dense1_size=net["dense1_size"],
        dense2_size=net["dense2_size"],
        input_bucket_layout_64=expand_mirrored_bucket_layout(B16_BUCKET_LAYOUT_32),
        output_bucket_count=net["output_bucket_count"],
        ft_bias_bytes=sections[SECTION_FT_BIAS]["payload"],
        psq_weight_bytes=expanded,
        threat_weight_bytes=sections[SECTION_THREAT_WEIGHT]["payload"],
        l1_bias_bytes=sections[SECTION_L1_BIAS]["payload"],
        l1_weight_bytes=sections[SECTION_L1_WEIGHT]["payload"],
        l2_bias_bytes=sections[SECTION_L2_BIAS]["payload"],
        l2_weight_bytes=sections[SECTION_L2_WEIGHT]["payload"],
        out_bias_bytes=sections[SECTION_OUT_BIAS]["payload"],
        out_weight_bytes=sections[SECTION_OUT_WEIGHT]["payload"],
        # Each bucket's rows are copied unchanged, so the bounds carry over.
        psq_abs_bound=net["psq_abs_bound"],
        threat_abs_bound=net["threat_abs_bound"],
    )
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
