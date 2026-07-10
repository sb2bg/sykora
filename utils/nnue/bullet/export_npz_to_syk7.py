#!/usr/bin/env python3
"""Export a pairwise-MLP Bullet NPZ checkpoint as SYKNNUE7."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

UTILS_NNUE_DIR = Path(__file__).resolve().parents[1]
if str(UTILS_NNUE_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_NNUE_DIR))

from common import (  # noqa: E402
    FEATURE_SET_KING_BUCKETS_MIRRORED,
    NNUE_Q,
    NNUE_Q0,
    SCALE,
    input_size_for_feature_set,
    write_syk_nnue_v7,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pairwise-MLP NPZ to SYKNNUE7.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-net", required=True)
    parser.add_argument("--allow-clipping", action="store_true")
    return parser.parse_args()


def round_away_from_zero(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    return np.where(values >= 0.0, np.floor(values + 0.5), -np.floor(-values + 0.5))


def quantize_checked(values, scale: int, dtype, *, name: str, allow_clipping: bool):
    import numpy as np

    info = np.iinfo(dtype)
    rounded = round_away_from_zero(np.asarray(values, dtype=np.float64) * float(scale))
    clipped = int(np.count_nonzero((rounded < info.min) | (rounded > info.max)))
    if clipped and not allow_clipping:
        raise ValueError(
            f"{name}: {clipped}/{rounded.size} values exceed {dtype} after x{scale}; "
            "fix training clipping or use --allow-clipping only for diagnostics"
        )
    print(f"Quantisation {name}: clipped={clipped}/{rounded.size}")
    return np.clip(rounded, info.min, info.max).astype(dtype)


def main() -> int:
    import numpy as np

    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    with np.load(input_path) as ckpt:
        architecture = str(np.asarray(ckpt["architecture"]).reshape(-1)[0])
        if architecture != "pairwise-mlp":
            raise ValueError(f"SYKNNUE7 exporter requires pairwise-mlp, got {architecture!r}")
        ft_weights = np.asarray(ckpt["ft_weights"], dtype=np.float32)
        ft_bias = np.asarray(ckpt["ft_bias"], dtype=np.float32)
        l1_weights = np.asarray(ckpt["l1_weights"], dtype=np.float32)
        l1_bias = np.asarray(ckpt["l1_bias"], dtype=np.float32)
        l2_weights = np.asarray(ckpt["l2_weights"], dtype=np.float32)
        l2_bias = np.asarray(ckpt["l2_bias"], dtype=np.float32)
        out_weights = np.asarray(ckpt["l3_weights"], dtype=np.float32)
        out_bias = np.asarray(ckpt["l3_bias"], dtype=np.float32)
        bucket_layout = [int(v) for v in np.asarray(ckpt["bucket_layout_64"]).reshape(-1)]
        feature_set = int(np.asarray(ckpt["feature_set"]).reshape(-1)[0])

    if feature_set != FEATURE_SET_KING_BUCKETS_MIRRORED:
        raise ValueError("SYKNNUE7 currently supports mirrored PSQ features only")
    if len(bucket_layout) != 64:
        raise ValueError("bucket_layout_64 must contain 64 entries")
    input_size, h = ft_weights.shape
    if input_size != input_size_for_feature_set(feature_set, bucket_layout) or h % 2:
        raise ValueError("invalid FT shape for pairwise pooling")
    if ft_bias.shape != (h,):
        raise ValueError("ft_bias shape mismatch")
    if l1_weights.ndim != 3:
        raise ValueError("l1_weights must be [O,H,D1]")
    outputs, l1_input, d1 = l1_weights.shape
    if outputs != 8 or l1_input != h or l1_bias.shape != (outputs, d1):
        raise ValueError("l1 tensor shape mismatch")
    if l2_weights.ndim != 3:
        raise ValueError("l2_weights must be [O,2*D1,D2]")
    l2_outputs, l2_input, d2 = l2_weights.shape
    if l2_outputs != outputs or l2_input != 2 * d1 or l2_bias.shape != (outputs, d2):
        raise ValueError("l2 tensor shape mismatch")
    if out_weights.shape != (outputs, d2) or out_bias.shape != (outputs,):
        raise ValueError("output tensor shape mismatch")

    q0 = NNUE_Q0
    pool_quant = 128
    q = NNUE_Q
    ft_bias_q = quantize_checked(ft_bias, q0, np.dtype("<i2"), name="ft_bias", allow_clipping=args.allow_clipping)
    ft_weights_q = quantize_checked(ft_weights, q0, np.dtype("<i2"), name="ft_weight", allow_clipping=args.allow_clipping)
    l1_bias_q = quantize_checked(l1_bias, pool_quant * q, np.dtype("<i4"), name="l1_bias", allow_clipping=args.allow_clipping)
    l1_weights_q = quantize_checked(l1_weights, q, np.dtype("i1"), name="l1_weight", allow_clipping=args.allow_clipping)
    l2_bias_q = quantize_checked(l2_bias, q * q, np.dtype("<i4"), name="l2_bias", allow_clipping=args.allow_clipping)
    l2_weights_q = quantize_checked(l2_weights, q, np.dtype("i1"), name="l2_weight", allow_clipping=args.allow_clipping)
    out_bias_q = quantize_checked(out_bias, q * q, np.dtype("<i4"), name="out_bias", allow_clipping=args.allow_clipping)
    out_weights_q = quantize_checked(out_weights, q, np.dtype("i1"), name="out_weight", allow_clipping=args.allow_clipping)

    output_path = Path(args.output_net)
    write_syk_nnue_v7(
        output_path,
        ft_hidden_size=h,
        dense1_size=d1,
        dense2_size=d2,
        input_bucket_layout_64=bucket_layout,
        output_bucket_count=outputs,
        ft_bias_bytes=ft_bias_q.tobytes(order="C"),
        ft_weight_bytes=ft_weights_q.tobytes(order="C"),
        l1_bias_bytes=l1_bias_q.tobytes(order="C"),
        l1_weight_bytes=l1_weights_q.tobytes(order="C"),
        l2_bias_bytes=l2_bias_q.tobytes(order="C"),
        l2_weight_bytes=l2_weights_q.tobytes(order="C"),
        out_bias_bytes=out_bias_q.tobytes(order="C"),
        out_weight_bytes=out_weights_q.tobytes(order="C"),
        q0=q0,
        pool_quant=pool_quant,
        q=q,
        scale=SCALE,
    )
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Architecture: pairwise-mlp H={h} D1={d1} D2={d2} O={outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
