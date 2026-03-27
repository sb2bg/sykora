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
    QA,
    QB,
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
    parser.add_argument("--qa", type=int, default=QA, help="FT quantization scale")
    parser.add_argument("--q1", type=int, default=QB, help="Dense layer 1 scale")
    parser.add_argument("--q2", type=int, default=QB, help="Dense layer 2 scale")
    parser.add_argument("--qo", type=int, default=QB, help="Output layer scale")
    return parser.parse_args()


def expect_array(ckpt, key: str):
    if key not in ckpt:
        raise KeyError(f"Missing NPZ key: {key}")
    return ckpt[key]


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
        l1_weights = np.asarray(expect_array(ckpt, "l1_weights"), dtype=np.float32)
        l1_bias = np.asarray(expect_array(ckpt, "l1_bias"), dtype=np.float32)
        l2_weights = np.asarray(expect_array(ckpt, "l2_weights"), dtype=np.float32)
        l2_bias = np.asarray(expect_array(ckpt, "l2_bias"), dtype=np.float32)
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

        if "ft_activation_type" in ckpt:
            ft_activation_type = int(np.asarray(ckpt["ft_activation_type"]).reshape(-1)[0])
        else:
            ft_activation_type = 1

        if "dense_activation_type" in ckpt:
            dense_activation_type = int(
                np.asarray(ckpt["dense_activation_type"]).reshape(-1)[0]
            )
        else:
            dense_activation_type = 0

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

    if l1_weights.ndim != 3:
        raise ValueError(f"l1_weights must be rank-3, got shape {l1_weights.shape}")
    stack_count, dense_l1_size, l1_inputs = l1_weights.shape
    if l1_inputs != 2 * ft_hidden_size:
        raise ValueError(
            f"l1_weights input mismatch: expected {2 * ft_hidden_size}, got {l1_inputs}"
        )
    if l1_bias.shape != (stack_count, dense_l1_size):
        raise ValueError(
            f"l1_bias shape mismatch: expected {(stack_count, dense_l1_size)}, got {l1_bias.shape}"
        )

    if l2_weights.ndim != 3:
        raise ValueError(f"l2_weights must be rank-3, got shape {l2_weights.shape}")
    stack_count_l2, dense_l2_size, l2_inputs = l2_weights.shape
    if stack_count_l2 != stack_count or l2_inputs != dense_l1_size:
        raise ValueError(
            f"l2_weights shape mismatch: expected ({stack_count}, L2, {dense_l1_size}), got {l2_weights.shape}"
        )
    if l2_bias.shape != (stack_count, dense_l2_size):
        raise ValueError(
            f"l2_bias shape mismatch: expected {(stack_count, dense_l2_size)}, got {l2_bias.shape}"
        )

    if out_weights.shape != (stack_count, dense_l2_size):
        raise ValueError(
            f"out_weights shape mismatch: expected {(stack_count, dense_l2_size)}, got {out_weights.shape}"
        )
    if out_bias.shape[0] != stack_count:
        raise ValueError(
            f"out_bias length mismatch: expected {stack_count}, got {out_bias.shape[0]}"
        )

    ft_bias_i16 = np.clip(np.rint(ft_bias * float(args.qa)), -32768, 32767).astype(np.int16)
    ft_weights_i16 = np.clip(
        np.rint(ft_weights.reshape(-1) * float(args.qa)), -32768, 32767
    ).astype(np.int16)

    l1_bias_i32 = np.clip(
        np.rint(l1_bias.reshape(-1) * float(args.qa * args.q1)),
        -2147483648,
        2147483647,
    ).astype(np.int32)
    l1_weights_i16 = np.clip(
        np.rint(l1_weights.reshape(-1) * float(args.q1)), -32768, 32767
    ).astype(np.int16)

    l2_bias_i32 = np.clip(
        np.rint(l2_bias.reshape(-1) * float(args.qa * args.q2)),
        -2147483648,
        2147483647,
    ).astype(np.int32)
    l2_weights_i16 = np.clip(
        np.rint(l2_weights.reshape(-1) * float(args.q2)), -32768, 32767
    ).astype(np.int16)

    out_bias_i32 = np.clip(
        np.rint(out_bias * float(args.qa * args.qo)),
        -2147483648,
        2147483647,
    ).astype(np.int32)
    out_weights_i16 = np.clip(
        np.rint(out_weights.reshape(-1) * float(args.qo)), -32768, 32767
    ).astype(np.int16)

    out_path = Path(args.output_net)
    write_syk_nnue_v4(
        out_path,
        ft_hidden_size=ft_hidden_size,
        dense_layer_1_size=dense_l1_size,
        dense_layer_2_size=dense_l2_size,
        layer_stack_count=stack_count,
        ft_biases_i16=ft_bias_i16.tolist(),
        ft_weights_i16=ft_weights_i16.tolist(),
        l1_biases_i32=l1_bias_i32.tolist(),
        l1_weights_i16=l1_weights_i16.tolist(),
        l2_biases_i32=l2_bias_i32.tolist(),
        l2_weights_i16=l2_weights_i16.tolist(),
        out_biases_i32=out_bias_i32.tolist(),
        out_weights_i16=out_weights_i16.tolist(),
        feature_set=feature_set,
        bucket_layout_64=bucket_layout,
        ft_activation_type=ft_activation_type,
        dense_activation_type=dense_activation_type,
        qa=args.qa,
        q1=args.q1,
        q2=args.q2,
        qo=args.qo,
    )

    print(f"Input: {in_path}")
    print("Output format: SYKNNUE4")
    print(f"Bucket count: {max(bucket_layout) + 1}")
    print(f"FT hidden: {ft_hidden_size}")
    print(f"Dense head: {dense_l1_size} -> {dense_l2_size} -> 1 with {stack_count} stacks")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
