#!/usr/bin/env python3
"""Convert a SYKNNUE4 Bullet checkpoint raw.bin into explicit NPZ tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UTILS_NNUE_DIR = Path(__file__).resolve().parents[1]
if str(UTILS_NNUE_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_NNUE_DIR))

from common import (  # noqa: E402
    SCALE,
    V4_Q0,
    V4_Q,
    V4_Q1,
    SYKORA16_BUCKET_LAYOUT_32,
    expand_mirrored_bucket_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a SYKNNUE4 Bullet raw checkpoint into NPZ tensors."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Checkpoint directory or raw.bin path",
    )
    parser.add_argument(
        "--run-meta",
        default="",
        help="Optional path to run_meta.json (default: infer from checkpoint path)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz path",
    )
    return parser.parse_args()


def resolve_paths(input_path: Path, run_meta_arg: str) -> tuple[Path, Path]:
    if input_path.is_dir():
        raw_path = input_path / "raw.bin"
    else:
        raw_path = input_path
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw.bin not found: {raw_path}")

    if run_meta_arg:
        run_meta_path = Path(run_meta_arg)
    else:
        try:
            run_meta_path = raw_path.parents[2] / "run_meta.json"
        except IndexError as exc:
            raise FileNotFoundError(
                "Could not infer run_meta.json from checkpoint path; pass --run-meta explicitly"
            ) from exc
    if not run_meta_path.is_file():
        raise FileNotFoundError(f"run_meta.json not found: {run_meta_path}")
    return raw_path, run_meta_path


def take_f32(buf, offset: int, count: int):
    end = offset + count
    if end > buf.shape[0]:
        raise ValueError(
            f"Unexpected EOF while decoding raw checkpoint: need {count} floats at offset {offset}, "
            f"have {buf.shape[0]}"
        )
    return buf[offset:end], end


def expected_raw_sizes(
    *, bucket_count: int, ft_hidden: int, dense_l1: int, dense_l2: int, stack_count: int
) -> dict[str, int]:
    input_size = 768 * bucket_count
    dense_expand = 2 * dense_l1
    return {
        "spec_merged_ft": (
            input_size * ft_hidden
            + ft_hidden
            + ft_hidden * (stack_count * dense_l1)
            + stack_count * dense_l1
            + dense_expand * (stack_count * dense_l2)
            + stack_count * dense_l2
            + dense_l2 * stack_count
            + stack_count
        ),
    }


def detect_layout(
    *, raw_len: int, bucket_count: int, ft_hidden: int, dense_l1: int, dense_l2: int, stack_count: int
) -> str:
    sizes = expected_raw_sizes(
        bucket_count=bucket_count,
        ft_hidden=ft_hidden,
        dense_l1=dense_l1,
        dense_l2=dense_l2,
        stack_count=stack_count,
    )
    for name, expected in sizes.items():
        if raw_len == expected:
            return name
    expected_str = ", ".join(f"{k}={v}" for k, v in sizes.items())
    raise ValueError(
        f"raw.bin length mismatch: found {raw_len} floats, expected one of {expected_str}"
    )


def parse_network_config(run_meta: dict) -> dict:
    network = dict(run_meta.get("network", {}))
    env = run_meta.get("env", {})

    network_format = network.get("format") or env.get("SYK_NETWORK_FORMAT") or "syk4"
    if network_format != "syk4":
        raise ValueError(
            f"run_meta.json does not describe a SYKNNUE4 run: {network_format!r}"
        )

    if "bucket_layout_64" in network:
        bucket_layout_64 = [int(v) for v in network["bucket_layout_64"]]
    else:
        bucket_layout_name = env.get("SYK_BUCKET_LAYOUT", "sykora16")
        if bucket_layout_name != "sykora16":
            raise ValueError(
                f"Unsupported checkpoint bucket layout without explicit bucket_layout_64: "
                f"{bucket_layout_name!r}"
            )
        bucket_layout_64 = expand_mirrored_bucket_layout(SYKORA16_BUCKET_LAYOUT_32)

    return {
        "format": network_format,
        "bucket_layout_64": bucket_layout_64,
        "ft_hidden": int(network.get("ft_hidden") or env["SYK_HIDDEN"]),
        "dense_l1": int(network.get("dense_l1") or env["SYK_DENSE_L1"]),
        "dense_expand": int(network.get("dense_expand") or (2 * int(network.get("dense_l1") or env["SYK_DENSE_L1"]))),
        "dense_l2": int(network.get("dense_l2") or env["SYK_DENSE_L2"]),
        "stack_count": int(network.get("output_bucket_count", network.get("stack_count", 8))),
    }


def main() -> int:
    args = parse_args()

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required for checkpoint export") from exc

    raw_path, run_meta_path = resolve_paths(Path(args.input), args.run_meta)
    run_meta = json.loads(run_meta_path.read_text())
    network = parse_network_config(run_meta)

    bucket_layout_64 = [int(v) for v in network["bucket_layout_64"]]
    bucket_count = max(bucket_layout_64) + 1
    ft_hidden = int(network["ft_hidden"])
    dense_l1 = int(network["dense_l1"])
    dense_expand = int(network["dense_expand"])
    dense_l2 = int(network["dense_l2"])
    stack_count = int(network["stack_count"])
    input_size = 768 * bucket_count

    raw = np.fromfile(raw_path, dtype="<f4")
    layout = detect_layout(
        raw_len=raw.shape[0],
        bucket_count=bucket_count,
        ft_hidden=ft_hidden,
        dense_l1=dense_l1,
        dense_l2=dense_l2,
        stack_count=stack_count,
    )
    offset = 0

    l0w, offset = take_f32(raw, offset, input_size * ft_hidden)
    l0b, offset = take_f32(raw, offset, ft_hidden)
    l1w, offset = take_f32(raw, offset, ft_hidden * (stack_count * dense_l1))
    l1b, offset = take_f32(raw, offset, stack_count * dense_l1)
    l2w, offset = take_f32(raw, offset, dense_expand * (stack_count * dense_l2))
    l2b, offset = take_f32(raw, offset, stack_count * dense_l2)
    l3w, offset = take_f32(raw, offset, dense_l2 * stack_count)
    l3b, offset = take_f32(raw, offset, stack_count)

    if offset != raw.shape[0]:
        raise ValueError(
            f"raw.bin length mismatch: expected {offset} floats from metadata, found {raw.shape[0]}"
        )

    ft_weights = l0w.reshape(input_size, ft_hidden)
    ft_bias = l0b.reshape(ft_hidden)

    l1_weights = (
        l1w.reshape(ft_hidden, stack_count * dense_l1)
        .T.reshape(stack_count, dense_l1, ft_hidden)
    )
    l1_bias = l1b.reshape(stack_count, dense_l1)

    l2_weights = (
        l2w.reshape(dense_expand, stack_count * dense_l2)
        .T.reshape(stack_count, dense_l2, dense_expand)
    )
    l2_bias = l2b.reshape(stack_count, dense_l2)

    out_weights = l3w.reshape(dense_l2, stack_count).T.reshape(stack_count, dense_l2)
    out_bias = l3b.reshape(stack_count)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ft_weights=ft_weights.astype(np.float32),
        ft_bias=ft_bias.astype(np.float32),
        l1_weights=l1_weights.astype(np.float32),
        l1_bias=l1_bias.astype(np.float32),
        l2_weights=l2_weights.astype(np.float32),
        l2_bias=l2_bias.astype(np.float32),
        out_weights=out_weights.astype(np.float32),
        out_bias=out_bias.astype(np.float32),
        bucket_layout_64=np.asarray(bucket_layout_64, dtype=np.uint8),
        feature_set=np.asarray([1], dtype=np.uint8),
        input_bucket_count=np.asarray([bucket_count], dtype=np.uint8),
        output_bucket_count=np.asarray([stack_count], dtype=np.uint8),
        dense_expand=np.asarray([dense_expand], dtype=np.uint16),
        q0=np.asarray([V4_Q0], dtype=np.uint16),
        q1=np.asarray([V4_Q1], dtype=np.uint16),
        q=np.asarray([V4_Q], dtype=np.uint16),
        scale=np.asarray([SCALE], dtype=np.uint16),
    )

    print(f"Input: {raw_path}")
    print(f"Run metadata: {run_meta_path}")
    print("Network format: SYKNNUE4")
    print(f"Detected raw layout: {layout}")
    print(f"Bucket count: {bucket_count}")
    print(f"FT hidden: {ft_hidden}")
    print(
        f"Dense head: pooled {ft_hidden} -> {dense_l1} -> expand({dense_expand}) -> {dense_l2} -> 1 "
        f"with {stack_count} buckets"
    )
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
