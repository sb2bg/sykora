#!/usr/bin/env python3
"""Export a P3-ANOVA Bullet NPZ checkpoint as SYKNNUE9."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

UTILS_NNUE_DIR = Path(__file__).resolve().parents[1]
if str(UTILS_NNUE_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_NNUE_DIR))

from common import (  # noqa: E402
    FEATURE_SET_MIRRORED_PSQ_FULL_THREATS_V1,
    FULL_THREATS_V1_COUNT,
    FULL_THREATS_V1_PACKING_SHA256,
    NNUE_Q,
    NNUE_Q0,
    input_size_for_feature_set,
    write_syk_nnue_v9,
)

P3_RANK = 32
P3_PAWN_COUNT = 128
P3_CONTEXT_COUNT = 640
P3_QUANT = 64
P3_PROJECTION_QUANT = 128
P3_ACTIVATION_SHIFT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export P3-ANOVA NPZ to SYKNNUE9")
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


def psq_abs_bound(ft_bias_q, ft_weights_q, bucket_count: int) -> int:
    import numpy as np

    h = ft_bias_q.size
    weights = np.asarray(ft_weights_q, dtype=np.int64).reshape(bucket_count, 768, h)
    maximum = 0
    for bucket in range(bucket_count):
        magnitudes = np.abs(weights[bucket])
        top = np.partition(magnitudes, magnitudes.shape[0] - 34, axis=0)[-34:]
        bounds = np.abs(np.asarray(ft_bias_q, dtype=np.int64)) + top.sum(axis=0)
        maximum = max(maximum, int(bounds.max()))
    return maximum


def threat_abs_bound(threat_weights_q) -> int:
    import numpy as np

    return int(np.abs(np.asarray(threat_weights_q, dtype=np.int64)).sum(axis=0).max())


def p3_l1_abs_bound(projection_q) -> int:
    """Exact conservative bound after clipping every signed activation to i8."""
    import numpy as np

    projection = np.abs(np.asarray(projection_q, dtype=np.int64))
    return 128 * int(projection.sum(axis=1).max())


def validate_packed_dot_weights(projection_q) -> None:
    """Prove the AVX2 pmaddubsw pair intermediates cannot saturate."""
    import numpy as np

    grouped = np.asarray(projection_q, dtype=np.int64).reshape(8, 32, 4, 16)
    grouped = grouped.transpose(0, 1, 3, 2)
    pair_bounds = np.maximum(
        np.abs(grouped[..., 0]) + np.abs(grouped[..., 1]),
        np.abs(grouped[..., 2]) + np.abs(grouped[..., 3]),
    )
    maximum = int(pair_bounds.max())
    if 255 * maximum > np.iinfo(np.int16).max:
        raise ValueError(
            f"P3 packed-dot pair bound {maximum} can saturate AVX2 pmaddubsw"
        )
    print(f"P3 packed-dot pair bound: {maximum} (limit 128)")


def balance_p3_channels(pawn, context, projection):
    """Use CP scale symmetry to equalise quantisation range per rank channel."""
    import numpy as np

    pawn = np.asarray(pawn, dtype=np.float64).copy()
    context = np.asarray(context, dtype=np.float64).copy()
    projection = np.asarray(projection, dtype=np.float64).copy()
    grouped = projection.reshape(8, 4, P3_RANK, 16)
    pawn_max = np.abs(pawn).max(axis=0)
    context_max = np.abs(context).max(axis=0)
    projection_max = np.abs(grouped).max(axis=(0, 1, 3))
    if np.any(pawn_max == 0.0) or np.any(context_max == 0.0) or np.any(projection_max == 0.0):
        raise ValueError("cannot balance a zero P3 rank channel")
    target = np.power(pawn_max * pawn_max * context_max * projection_max, 0.25)
    pawn_scale = target / pawn_max
    context_scale = target / context_max
    projection_scale = 1.0 / (pawn_scale * pawn_scale * context_scale)
    pawn *= pawn_scale[None, :]
    context *= context_scale[None, :]
    grouped *= projection_scale[None, None, :, None]
    print(
        "P3 channel balancing: "
        f"pre_absmax=({pawn_max.max():.6f},{context_max.max():.6f},{projection_max.max():.6f}) "
        f"post_absmax={target.max():.6f}"
    )
    return pawn.astype(np.float32), context.astype(np.float32), projection.astype(np.float32)


def main() -> int:
    import numpy as np

    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    with np.load(input_path) as ckpt:
        architecture = str(np.asarray(ckpt["architecture"]).reshape(-1)[0])
        network_format = str(np.asarray(ckpt["network_format"]).reshape(-1)[0])
        if architecture != "pairwise-mlp-p3" or network_format != "syk9":
            raise ValueError(
                f"SYKNNUE9 exporter requires syk9 pairwise-mlp-p3, got "
                f"{network_format}/{architecture}"
            )
        ft_weights = np.asarray(ckpt["ft_weights"], dtype=np.float32)
        threat_weights = np.asarray(ckpt["threat_weights"], dtype=np.float32)
        ft_bias = np.asarray(ckpt["ft_bias"], dtype=np.float32)
        l1_weights = np.asarray(ckpt["l1_weights"], dtype=np.float32)
        l1_bias = np.asarray(ckpt["l1_bias"], dtype=np.float32)
        l2_weights = np.asarray(ckpt["l2_weights"], dtype=np.float32)
        l2_bias = np.asarray(ckpt["l2_bias"], dtype=np.float32)
        out_weights = np.asarray(ckpt["l3_weights"], dtype=np.float32)
        out_bias = np.asarray(ckpt["l3_bias"], dtype=np.float32)
        p3_pawn = np.asarray(ckpt["p3_pawn_weights"], dtype=np.float32)
        p3_context = np.asarray(ckpt["p3_context_weights"], dtype=np.float32)
        p3_l1 = np.asarray(ckpt["p3_l1_weights"], dtype=np.float32)
        p3_bias = np.asarray(ckpt["p3_l1_bias"], dtype=np.float32)
        bucket_layout = [int(v) for v in np.asarray(ckpt["bucket_layout_64"]).reshape(-1)]
        feature_set = int(np.asarray(ckpt["feature_set"]).reshape(-1)[0])
        packing_hash = str(np.asarray(ckpt["threat_packing_sha256"]).reshape(-1)[0])

    if feature_set != FEATURE_SET_MIRRORED_PSQ_FULL_THREATS_V1:
        raise ValueError("SYKNNUE9 requires mirrored_psq_full_threats_v1")
    if packing_hash != FULL_THREATS_V1_PACKING_SHA256:
        raise ValueError("checkpoint threat packing hash does not match the frozen scheme")
    if len(bucket_layout) != 64:
        raise ValueError("bucket_layout_64 must contain 64 entries")
    input_size, h = ft_weights.shape
    expected_input = input_size_for_feature_set(feature_set, bucket_layout)
    if input_size != expected_input or h != 1024:
        raise ValueError("invalid SYKNNUE9 FT shape")
    if ft_bias.shape != (h,) or threat_weights.shape != (FULL_THREATS_V1_COUNT, h):
        raise ValueError("feature-transformer tensor shape mismatch")
    if l1_weights.shape != (8, h, 16) or l1_bias.shape != (8, 16):
        raise ValueError("l1 tensor shape mismatch")
    if l2_weights.shape != (8, 32, 32) or l2_bias.shape != (8, 32):
        raise ValueError("l2 tensor shape mismatch")
    if out_weights.shape != (8, 32) or out_bias.shape != (8,):
        raise ValueError("output tensor shape mismatch")
    if p3_pawn.shape != (P3_PAWN_COUNT, P3_RANK):
        raise ValueError("P3 pawn tensor shape mismatch")
    if p3_context.shape != (P3_CONTEXT_COUNT, P3_RANK):
        raise ValueError("P3 context tensor shape mismatch")
    if p3_l1.shape != (8, 4 * P3_RANK, 16):
        raise ValueError("P3 projection tensor shape mismatch")
    if p3_bias.shape != (8, 16) or np.count_nonzero(p3_bias):
        raise ValueError("SYKNNUE9 requires the constrained-zero P3 projection bias")

    p3_pawn, p3_context, p3_l1 = balance_p3_channels(p3_pawn, p3_context, p3_l1)

    table_path = UTILS_NNUE_DIR / "full_threats_v1.bin"
    table = np.frombuffer(table_path.read_bytes(), dtype=np.uint8).reshape(FULL_THREATS_V1_COUNT, 8)
    reachable = table[:, 6].astype(bool)
    unreachable_nonzero = int(np.count_nonzero(threat_weights[~reachable]))
    if unreachable_nonzero:
        print(
            f"Canonicalising {unreachable_nonzero} nonzero values in unreachable threat rows to zero"
        )
        threat_weights = threat_weights.copy()
        threat_weights[~reachable] = 0.0

    q0 = NNUE_Q0
    pool_quant = 128
    q = NNUE_Q
    ft_bias_q = quantize_checked(ft_bias, q0, np.dtype("<i2"), name="ft_bias", allow_clipping=args.allow_clipping)
    ft_weights_q = quantize_checked(ft_weights, q0, np.dtype("<i2"), name="psq_weight", allow_clipping=args.allow_clipping)
    threat_weights_q = quantize_checked(threat_weights, q0, np.dtype("i1"), name="threat_weight", allow_clipping=args.allow_clipping)
    l1_bias_q = quantize_checked(l1_bias, pool_quant * q, np.dtype("<i4"), name="l1_bias", allow_clipping=args.allow_clipping)
    l1_weights_q = quantize_checked(l1_weights, q, np.dtype("i1"), name="l1_weight", allow_clipping=args.allow_clipping)
    l2_bias_q = quantize_checked(l2_bias, q * q, np.dtype("<i4"), name="l2_bias", allow_clipping=args.allow_clipping)
    l2_weights_q = quantize_checked(l2_weights, q, np.dtype("i1"), name="l2_weight", allow_clipping=args.allow_clipping)
    out_bias_q = quantize_checked(out_bias, q * q, np.dtype("<i4"), name="out_bias", allow_clipping=args.allow_clipping)
    out_weights_q = quantize_checked(out_weights, q, np.dtype("i1"), name="out_weight", allow_clipping=args.allow_clipping)
    p3_pawn_q = quantize_checked(p3_pawn, P3_QUANT, np.dtype("i1"), name="p3_pawn", allow_clipping=args.allow_clipping)
    p3_context_q = quantize_checked(p3_context, P3_QUANT, np.dtype("i1"), name="p3_context", allow_clipping=args.allow_clipping)
    p3_l1_q = quantize_checked(p3_l1, P3_PROJECTION_QUANT, np.dtype("i1"), name="p3_l1", allow_clipping=args.allow_clipping)
    validate_packed_dot_weights(p3_l1_q)

    bucket_count = max(bucket_layout) + 1
    psq_bound = psq_abs_bound(ft_bias_q, ft_weights_q, bucket_count)
    threat_bound = threat_abs_bound(threat_weights_q)
    p3_bound = p3_l1_abs_bound(p3_l1_q)
    if psq_bound > np.iinfo(np.int16).max:
        raise ValueError(f"PSQ accumulator bound {psq_bound} exceeds i16")
    base_l1_bound = h * 127 * 128
    if base_l1_bound + p3_bound > np.iinfo(np.int32).max:
        raise ValueError("base + P3 l1 bound exceeds i32")
    print(
        f"Accumulator bounds: psq={psq_bound}, threat={threat_bound}, "
        f"resolved={psq_bound + threat_bound}, p3_l1={p3_bound}"
    )

    output_path = Path(args.output_net)
    write_syk_nnue_v9(
        output_path,
        ft_hidden_size=h,
        dense1_size=16,
        dense2_size=32,
        input_bucket_layout_64=bucket_layout,
        output_bucket_count=8,
        ft_bias_bytes=ft_bias_q.tobytes(order="C"),
        psq_weight_bytes=ft_weights_q.tobytes(order="C"),
        threat_weight_bytes=threat_weights_q.tobytes(order="C"),
        l1_bias_bytes=l1_bias_q.tobytes(order="C"),
        l1_weight_bytes=l1_weights_q.tobytes(order="C"),
        l2_bias_bytes=l2_bias_q.tobytes(order="C"),
        l2_weight_bytes=l2_weights_q.tobytes(order="C"),
        out_bias_bytes=out_bias_q.tobytes(order="C"),
        out_weight_bytes=out_weights_q.tobytes(order="C"),
        p3_pawn_weight_bytes=p3_pawn_q.tobytes(order="C"),
        p3_context_weight_bytes=p3_context_q.tobytes(order="C"),
        p3_l1_weight_bytes=p3_l1_q.tobytes(order="C"),
        p3_rank=P3_RANK,
        p3_pawn_count=P3_PAWN_COUNT,
        p3_context_count=P3_CONTEXT_COUNT,
        p3_quant=P3_QUANT,
        p3_activation_shift=P3_ACTIVATION_SHIFT,
        p3_l1_abs_bound=p3_bound,
        psq_abs_bound=psq_bound,
        threat_abs_bound=threat_bound,
    )
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Architecture: T{h} P3-R{P3_RANK} H={h} D1=16 D2=32 O=8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
