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
    FEATURE_SET_KING_BUCKETS_MIRRORED,
    SCALE,
    V4_Q,
    V4_Q0,
    V4_Q1,
    V4_QPSQT,
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
    parser.add_argument("--q0", type=int, default=None, help="FT / pooled activation scale")
    parser.add_argument("--q1", type=int, default=None, help="Dense layer 1 scale")
    parser.add_argument("--q", type=int, default=None, help="Dense layer 2 / output scale")
    parser.add_argument("--qpsqt", type=int, default=None, help="PSQT side-path scale")
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
        psqt_weights = np.asarray(expect_array(ckpt, "psqt_weights"), dtype=np.float32).reshape(-1)
        l1_weights = np.asarray(expect_array(ckpt, "l1_weights"), dtype=np.float32)
        l1_bias = np.asarray(expect_array(ckpt, "l1_bias"), dtype=np.float32).reshape(-1)
        l2_weights = np.asarray(expect_array(ckpt, "l2_weights"), dtype=np.float32)
        l2_bias = np.asarray(expect_array(ckpt, "l2_bias"), dtype=np.float32).reshape(-1)
        out_weights = np.asarray(expect_array(ckpt, "out_weights"), dtype=np.float32)
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

        q0 = int(np.asarray(ckpt["q0"]).reshape(-1)[0]) if "q0" in ckpt else V4_Q0
        q1 = int(np.asarray(ckpt["q1"]).reshape(-1)[0]) if "q1" in ckpt else V4_Q1
        q = int(np.asarray(ckpt["q"]).reshape(-1)[0]) if "q" in ckpt else V4_Q
        qpsqt = int(np.asarray(ckpt["qpsqt"]).reshape(-1)[0]) if "qpsqt" in ckpt else V4_QPSQT
        scale = int(np.asarray(ckpt["scale"]).reshape(-1)[0]) if "scale" in ckpt else SCALE

    if args.q0 is not None:
        q0 = args.q0
    if args.q1 is not None:
        q1 = args.q1
    if args.q is not None:
        q = args.q
    if args.qpsqt is not None:
        qpsqt = args.qpsqt
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
    if psqt_weights.shape[0] != ft_input_size:
        raise ValueError(
            f"psqt_weights length mismatch: expected {ft_input_size}, got {psqt_weights.shape[0]}"
        )

    if l1_weights.ndim != 2:
        raise ValueError(f"l1_weights must be rank-2, got shape {l1_weights.shape}")
    dense_l1_size, l1_inputs = l1_weights.shape
    if l1_inputs != 2 * ft_hidden_size:
        raise ValueError(
            f"l1_weights input mismatch: expected {2 * ft_hidden_size}, got {l1_inputs}"
        )
    if l1_bias.shape[0] != dense_l1_size:
        raise ValueError(
            f"l1_bias length mismatch: expected {dense_l1_size}, got {l1_bias.shape[0]}"
        )

    if l2_weights.ndim != 2:
        raise ValueError(f"l2_weights must be rank-2, got shape {l2_weights.shape}")
    dense_l2_size, l2_inputs = l2_weights.shape
    if l2_inputs != dense_l1_size:
        raise ValueError(
            f"l2_weights input mismatch: expected {dense_l1_size}, got {l2_inputs}"
        )
    if l2_bias.shape[0] != dense_l2_size:
        raise ValueError(
            f"l2_bias length mismatch: expected {dense_l2_size}, got {l2_bias.shape[0]}"
        )

    if out_weights.ndim != 1 or out_weights.shape[0] != dense_l2_size:
        raise ValueError(
            f"out_weights shape mismatch: expected {(dense_l2_size,)}, got {out_weights.shape}"
        )
    if out_bias.shape[0] != 1:
        raise ValueError(
            f"out_bias length mismatch: expected 1, got {out_bias.shape[0]}"
        )

    ft_bias_i16 = quantize_clipped(ft_bias, q0, -32768, 32767, np.int16)
    ft_weights_i16 = quantize_clipped(
        ft_weights.reshape(-1), q0, -32768, 32767, np.int16
    )
    psqt_weights_i16 = quantize_clipped(
        psqt_weights,
        qpsqt,
        -32768,
        32767,
        np.int16,
    )

    l1_bias_i32 = quantize_clipped(
        l1_bias,
        q0 * q0 * q1,
        -2147483648,
        2147483647,
        np.int32,
    )
    l1_weights_i8 = quantize_clipped(
        l1_weights.reshape(-1), q1, -128, 127, np.int8
    )

    l2_bias_i32 = quantize_clipped(
        l2_bias,
        q * q,
        -2147483648,
        2147483647,
        np.int32,
    )
    l2_weights_i8 = quantize_clipped(
        l2_weights.reshape(-1), q, -128, 127, np.int8
    )

    out_bias_i32 = quantize_clipped(
        out_bias,
        q * q,
        -2147483648,
        2147483647,
        np.int32,
    )
    out_weights_i8 = quantize_clipped(
        out_weights.reshape(-1), q, -128, 127, np.int8
    )

    out_path = Path(args.output_net)
    write_syk_nnue_v4(
        out_path,
        ft_hidden_size=ft_hidden_size,
        dense_layer_1_size=dense_l1_size,
        dense_layer_2_size=dense_l2_size,
        output_bucket_count=1,
        ft_biases_i16=ft_bias_i16.tolist(),
        ft_weights_i16=ft_weights_i16.tolist(),
        psqt_weights_i16=psqt_weights_i16.tolist(),
        l1_biases_i32=l1_bias_i32.tolist(),
        l1_weights_i8=l1_weights_i8.tolist(),
        l2_biases_i32=l2_bias_i32.tolist(),
        l2_weights_i8=l2_weights_i8.tolist(),
        out_bias_i32=int(out_bias_i32[0]),
        out_weights_i8=out_weights_i8.tolist(),
        feature_set=feature_set,
        bucket_layout_64=bucket_layout,
        q0=q0,
        q1=q1,
        q=q,
        qpsqt=qpsqt,
        scale=scale,
    )

    print(f"Input: {in_path}")
    print("Output format: SYKNNUE4")
    print(f"Bucket count: {max(bucket_layout) + 1}")
    print(f"FT hidden: {ft_hidden_size}")
    print(f"Dense head: shared {2 * ft_hidden_size} -> {dense_l1_size} -> {dense_l2_size} -> 1")
    print("PSQT side path: shared 12288 -> 1")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
