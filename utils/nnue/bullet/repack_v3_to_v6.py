#!/usr/bin/env python3
"""Repack a legacy SYKNNUE3 net into a SYKNNUE6 container (Stage 0).

This is a lossless byte reshuffle: the v3 single linear head becomes a v6
output head with O=1 and the `single` bucket scheme. No requantization is
performed, so the resulting net is architecturally identical to the source.
Evaluating it through the v6 round-to-nearest contract may differ from the old
v3 truncating-division runtime by +-1 cp on some positions (expected; see the
SYKNNUE6 spec).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

UTILS_NNUE_DIR = Path(__file__).resolve().parents[1]
if str(UTILS_NNUE_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_NNUE_DIR))

from common import (  # noqa: E402
    FEATURE_SET_KING_BUCKETS_MIRRORED,
    LEGACY_INPUT_SIZE,
    OUTPUT_BUCKET_SCHEME_SINGLE,
    write_syk_nnue_v6,
)

MAGIC_V3 = b"SYKNNUE3"
FORMAT_VERSION_V3 = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repack SYKNNUE3 -> SYKNNUE6.")
    parser.add_argument("--input", required=True, help="Input SYKNNUE3 .sknnue file")
    parser.add_argument("--output", required=True, help="Output SYKNNUE6 .sknnue file")
    return parser.parse_args()


def _read(fmt: str, data: bytes, pos: int):
    size = struct.calcsize(fmt)
    if pos + size > len(data):
        raise ValueError("SYKNNUE3 truncated header")
    return struct.unpack_from(fmt, data, pos)[0], pos + size


def read_v3(data: bytes) -> dict:
    if data[0:8] != MAGIC_V3:
        raise ValueError(f"not a SYKNNUE3 net (magic={data[0:8]!r})")
    pos = 8
    version, pos = _read("<H", data, pos)
    if version != FORMAT_VERSION_V3:
        raise ValueError(f"unsupported v3 version: {version}")
    feature_set, pos = _read("<B", data, pos)
    if feature_set != FEATURE_SET_KING_BUCKETS_MIRRORED:
        raise ValueError("repack only supports king_buckets_mirrored v3 nets")
    ft_hidden_size, pos = _read("<H", data, pos)
    activation_type, pos = _read("<B", data, pos)
    bucket_count, pos = _read("<H", data, pos)  # v3 quirk: u16 here
    bucket_layout_64 = list(data[pos : pos + 64])
    pos += 64
    if len(bucket_layout_64) != 64:
        raise ValueError("SYKNNUE3 truncated bucket layout")

    output_bias, pos = _read("<i", data, pos)

    h = ft_hidden_size
    input_size = LEGACY_INPUT_SIZE * bucket_count

    def take_i16(count: int, p: int):
        end = p + count * 2
        if end > len(data):
            raise ValueError("SYKNNUE3 truncated payload")
        return list(struct.unpack_from(f"<{count}h", data, p)), end

    ft_biases, pos = take_i16(h, pos)
    ft_weights, pos = take_i16(input_size * h, pos)
    output_weights, pos = take_i16(2 * h, pos)

    if pos != len(data):
        raise ValueError(f"trailing bytes in SYKNNUE3 net: {len(data) - pos}")

    return {
        "ft_hidden_size": h,
        "activation_type": activation_type,
        "bucket_layout_64": bucket_layout_64,
        "output_bias": output_bias,
        "ft_biases": ft_biases,
        "ft_weights": ft_weights,
        "output_weights": output_weights,
    }


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input net not found: {in_path}")

    net = read_v3(in_path.read_bytes())

    write_syk_nnue_v6(
        Path(args.output),
        ft_hidden_size=net["ft_hidden_size"],
        ft_biases_i16=net["ft_biases"],
        ft_weights_i16=net["ft_weights"],
        out_biases_i32=[net["output_bias"]],
        out_weights_i16=net["output_weights"],
        activation_type=net["activation_type"],
        feature_set=FEATURE_SET_KING_BUCKETS_MIRRORED,
        bucket_layout_64=net["bucket_layout_64"],
        output_bucket_count=1,
        output_bucket_scheme=OUTPUT_BUCKET_SCHEME_SINGLE,
    )

    print(f"Input:  {in_path} (SYKNNUE3)")
    print(f"Output: {args.output} (SYKNNUE6, O=1, scheme=single)")
    print(f"FT hidden: {net['ft_hidden_size']}, input buckets: {max(net['bucket_layout_64']) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
