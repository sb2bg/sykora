#!/usr/bin/env python3
"""Convert a float-domain NPZ checkpoint into Sykora SYKNNUE4 format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


UTILS_NNUE_DIR = Path(__file__).resolve().parents[1]
if str(UTILS_NNUE_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_NNUE_DIR))

from common import (  # noqa: E402
    ACTIVATION_SCRELU,
    FEATURE_SET_KING_BUCKETS_MIRRORED,
    SCALE,
    V4_Q,
    V4_Q0,
    SYKORA16_BUCKET_LAYOUT_32,
    expand_mirrored_bucket_layout,
    input_size_for_feature_set,
    write_syk_nnue_v4,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export NPZ checkpoint to SYKNNUE4 net."
    )
    parser.add_argument("--input", required=True, help="Input .npz checkpoint")
    parser.add_argument("--output-net", required=True, help="Output .sknnue path")
    parser.add_argument("--q0", type=int, default=None, help="FT activation scale")
    parser.add_argument("--q", type=int, default=None, help="Output weight scale")
    parser.add_argument("--scale", type=int, default=None, help="Final centipawn scale")
    return parser.parse_args()


def expect_array(ckpt, key: str):
    if key not in ckpt:
        raise KeyError(f"Missing NPZ key: {key}")
    return ckpt[key]


def round_away_from_zero(values):
    """Round to nearest with ties away from zero."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    return np.where(values >= 0.0, np.floor(values + 0.5), -np.floor(-values + 0.5))


def quantize_clipped(values, scale: int, min_value: int, max_value: int, dtype):
    import numpy as np

    rounded = round_away_from_zero(np.asarray(values, dtype=np.float64) * float(scale))
    return np.clip(rounded, min_value, max_value).astype(dtype)


def main() -> int:
    args = parse_args()

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required for NPZ export") from exc

    in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {in_path}")

    with np.load(in_path) as ckpt:
        ft_weights = np.asarray(expect_array(ckpt, "ft_weights"), dtype=np.float32)
        ft_bias = np.asarray(expect_array(ckpt, "ft_bias"), dtype=np.float32).reshape(-1)
        out_weights = np.asarray(expect_array(ckpt, "out_weights"), dtype=np.float32).reshape(-1)
        out_bias = np.asarray(expect_array(ckpt, "out_bias"), dtype=np.float32).reshape(-1)

        if "bucket_layout_64" in ckpt:
            bucket_layout_64 = np.asarray(ckpt["bucket_layout_64"], dtype=np.uint8).reshape(-1)
            bucket_layout = [int(v) for v in bucket_layout_64.tolist()]
        else:
            bucket_layout = expand_mirrored_bucket_layout(SYKORA16_BUCKET_LAYOUT_32)

        if "feature_set" in ckpt:
            feature_set = int(np.asarray(ckpt["feature_set"]).reshape(-1)[0])
        else:
            feature_set = FEATURE_SET_KING_BUCKETS_MIRRORED

        if "activation_type" in ckpt:
            activation_type = int(np.asarray(ckpt["activation_type"]).reshape(-1)[0])
        else:
            activation_type = ACTIVATION_SCRELU

        q0 = int(np.asarray(ckpt["q0"]).reshape(-1)[0]) if "q0" in ckpt else V4_Q0
        q = int(np.asarray(ckpt["q"]).reshape(-1)[0]) if "q" in ckpt else V4_Q
        scale = int(np.asarray(ckpt["scale"]).reshape(-1)[0]) if "scale" in ckpt else SCALE

    if args.q0 is not None:
        q0 = args.q0
    if args.q is not None:
        q = args.q
    if args.scale is not None:
        scale = args.scale

    if feature_set != FEATURE_SET_KING_BUCKETS_MIRRORED:
        raise ValueError("SYKNNUE4 only supports king_buckets_mirrored inputs")
    if len(bucket_layout) != 64:
        raise ValueError(f"bucket_layout_64 must have 64 entries, got {len(bucket_layout)}")

    if ft_weights.ndim != 2:
        raise ValueError(f"ft_weights must be rank-2, got shape {ft_weights.shape}")
    ft_input_size, ft_hidden_size = ft_weights.shape
    expected_input_size = input_size_for_feature_set(feature_set, bucket_layout)
    if ft_input_size != expected_input_size:
        raise ValueError(
            f"ft_weights.shape[0] mismatch: expected {expected_input_size}, got {ft_input_size}"
        )
    if ft_bias.shape[0] != ft_hidden_size:
        raise ValueError(
            f"ft_bias length mismatch: expected {ft_hidden_size}, got {ft_bias.shape[0]}"
        )
    if out_weights.shape[0] != 2 * ft_hidden_size:
        raise ValueError(
            f"out_weights shape mismatch: expected {(2 * ft_hidden_size,)}, got {out_weights.shape}"
        )
    if out_bias.shape[0] != 1:
        raise ValueError(
            f"out_bias length mismatch: expected 1, got {out_bias.shape[0]}"
        )

    ft_bias_i16 = quantize_clipped(ft_bias, q0, -32768, 32767, np.int16)
    ft_weights_i16 = quantize_clipped(
        ft_weights.reshape(-1), q0, -32768, 32767, np.int16
    )

    out_bias_i32 = quantize_clipped(
        out_bias,
        q0 * q,
        -2147483648,
        2147483647,
        np.int32,
    )
    out_weights_i16 = quantize_clipped(
        out_weights.reshape(-1), q, -32768, 32767, np.int16
    )

    out_path = Path(args.output_net)
    write_syk_nnue_v4(
        out_path,
        ft_hidden_size=ft_hidden_size,
        ft_biases_i16=ft_bias_i16.tolist(),
        ft_weights_i16=ft_weights_i16.tolist(),
        out_bias_i32=int(out_bias_i32[0]),
        out_weights_i16=out_weights_i16.tolist(),
        activation_type=activation_type,
        feature_set=feature_set,
        bucket_layout_64=bucket_layout,
        q0=q0,
        q=q,
        scale=scale,
    )

    print(f"Input: {in_path}")
    print("Output format: SYKNNUE4")
    print(f"Bucket count: {max(bucket_layout) + 1}")
    print(f"FT hidden: {ft_hidden_size}")
    print(f"Dense head: linear {2 * ft_hidden_size} -> 1")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
