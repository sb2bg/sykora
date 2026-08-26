#!/usr/bin/env python3
"""Measure exact integer SYKNNUE8/9 inference loss on Bullet-format data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
UTILS_NNUE_DIR = THIS_DIR.parent
if str(UTILS_NNUE_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_NNUE_DIR))

from common import MAGIC_V8, MAGIC_V9, read_syk_nnue_v8  # noqa: E402
from check_net_parity import (  # noqa: E402
    SECTION_FT_BIAS,
    SECTION_FT_WEIGHT,
    SECTION_L1_BIAS,
    SECTION_L1_WEIGHT,
    SECTION_L2_BIAS,
    SECTION_L2_WEIGHT,
    SECTION_OUT_BIAS,
    SECTION_OUT_WEIGHT,
    SECTION_P3_CONTEXT_WEIGHT,
    SECTION_P3_L1_WEIGHT,
    SECTION_P3_PAWN_WEIGHT,
    SECTION_THREAT_WEIGHT,
    decode_tensors,
    read_syk_nnue_v9_for_parity,
)
from validate_checkpoints import load_validation, prepare_features, sigmoid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate deployed integer Sykora nets")
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--run-meta", required=True, help="Metadata providing bucket layout and WDL")
    parser.add_argument("--net", action="append", required=True, help="NAME=PATH (repeatable)")
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def round_signed(values, shift: int):
    import numpy as np

    half = 1 << (shift - 1)
    return np.where(values >= 0, (values + half) >> shift, -((-values + half) >> shift))


def p3_moments_int(pawn_idx, context_idx, pawn_weights, context_weights):
    import numpy as np

    rank = pawn_weights.shape[1]
    padded_pawns = np.concatenate((pawn_weights, np.zeros((1, rank), dtype=np.int64)), axis=0)
    padded_context = np.concatenate(
        (context_weights, np.zeros((1, rank), dtype=np.int64)), axis=0
    )
    pawn_safe = np.where(pawn_idx >= 0, pawn_idx, pawn_weights.shape[0])
    context_safe = np.where(context_idx >= 0, context_idx, context_weights.shape[0])
    pawn_values = padded_pawns[pawn_safe]
    file_sums = np.zeros((pawn_idx.shape[0], 8, rank), dtype=np.int64)
    file_squares = np.zeros_like(file_sums)
    for file in range(8):
        mask = ((pawn_idx >= 0) & ((pawn_idx % 64) % 8 == file))[..., None]
        values = np.where(mask, pawn_values, 0)
        file_sums[:, file] = values.sum(axis=1)
        file_squares[:, file] = (values * values).sum(axis=1)
    same = (file_sums * file_sums - file_squares).sum(axis=1) // 2
    adjacent = (file_sums[:, :-1] * file_sums[:, 1:]).sum(axis=1)
    context = padded_context[context_safe].sum(axis=1)
    return np.concatenate((same * context, adjacent * context), axis=1)


def evaluate_net(net, tensors, features, data, wdl: float, batch_size: int) -> dict:
    import numpy as np

    stm, ntm, stm_threats, ntm_threats, stm_pawns, ntm_pawns, stm_context, ntm_context, buckets = features
    ft = tensors[SECTION_FT_WEIGHT]
    ft_bias = tensors[SECTION_FT_BIAS]
    threat = tensors[SECTION_THREAT_WEIGHT]
    ft_pad = np.concatenate((ft, np.zeros((1, ft.shape[1]), dtype=np.int64)), axis=0)
    threat_pad = np.concatenate(
        (threat, np.zeros((1, threat.shape[1]), dtype=np.int64)), axis=0
    )
    predictions = np.empty(data.size, dtype=np.float64)
    for start in range(0, data.size, batch_size):
        end = min(start + batch_size, data.size)
        sf = np.where(stm[start:end] >= 0, stm[start:end], ft.shape[0])
        nf = np.where(ntm[start:end] >= 0, ntm[start:end], ft.shape[0])
        st = np.where(
            stm_threats[start:end] >= 0, stm_threats[start:end], threat.shape[0]
        )
        nt = np.where(
            ntm_threats[start:end] >= 0, ntm_threats[start:end], threat.shape[0]
        )
        sa = ft_bias + ft_pad[sf].sum(axis=1) + threat_pad[st].sum(axis=1)
        na = ft_bias + ft_pad[nf].sum(axis=1) + threat_pad[nt].sum(axis=1)
        sa = np.clip(sa, 0, net["q0"])
        na = np.clip(na, 0, net["q0"])
        half = ft.shape[1] // 2
        pooled = np.concatenate(
            ((sa[:, :half] * sa[:, half:]) // 512, (na[:, :half] * na[:, half:]) // 512),
            axis=1,
        )
        bucket = buckets[start:end]
        l1 = np.einsum(
            "bi,bij->bj", pooled, tensors[SECTION_L1_WEIGHT][bucket], optimize=True
        ) + tensors[SECTION_L1_BIAS][bucket]
        if SECTION_P3_PAWN_WEIGHT in tensors:
            pawn = tensors[SECTION_P3_PAWN_WEIGHT]
            context = tensors[SECTION_P3_CONTEXT_WEIGHT]
            stm_tri = p3_moments_int(
                stm_pawns[start:end], stm_context[start:end], pawn, context
            )
            ntm_tri = p3_moments_int(
                ntm_pawns[start:end], ntm_context[start:end], pawn, context
            )
            tri = np.concatenate((stm_tri, ntm_tri), axis=1)
            tri = np.clip(round_signed(tri, net["p3_activation_shift"]), -128, 127)
            l1 += np.einsum(
                "bi,bij->bj", tri, tensors[SECTION_P3_L1_WEIGHT][bucket], optimize=True
            )
        z1 = round_signed(l1, 7)
        linear = np.clip(z1, 0, 64)
        squared = np.minimum((np.clip(z1, -64, 64) ** 2 + 32) >> 6, 64)
        dual = np.concatenate((linear, squared), axis=1)
        l2 = np.einsum(
            "bi,bij->bj", dual, tensors[SECTION_L2_WEIGHT][bucket], optimize=True
        ) + tensors[SECTION_L2_BIAS][bucket]
        z2 = np.clip(round_signed(l2, 6), 0, 64)
        hidden = (z2 * z2 + 32) >> 6
        raw = np.einsum(
            "bi,bi->b", hidden, tensors[SECTION_OUT_WEIGHT][bucket], optimize=True
        ) + tensors[SECTION_OUT_BIAS][bucket]
        cp = round_signed(raw * 400, 12)
        predictions[start:end] = sigmoid(cp.astype(np.float64) / 400.0)

    score_target = sigmoid(data["score"].astype(np.float64) / 400.0)
    result_target = data["result"].astype(np.float64) / 2.0
    targets = wdl * result_target + (1.0 - wdl) * score_target
    error = predictions - targets
    return {
        "positions": int(data.size),
        "mse": float(np.mean(error * error)),
        "mae": float(np.mean(np.abs(error))),
        "prediction_mean": float(predictions.mean()),
        "target_mean": float(targets.mean()),
    }


def load_net(path: Path):
    magic = path.read_bytes()[:8]
    if magic == MAGIC_V8:
        return read_syk_nnue_v8(path)
    if magic == MAGIC_V9:
        return read_syk_nnue_v9_for_parity(path)
    raise ValueError(f"unsupported network magic: {magic!r}")


def main() -> int:
    args = parse_args()
    run_meta = json.loads(Path(args.run_meta).read_text())
    layout = [int(value) for value in run_meta["network"]["bucket_layout_64"]]
    output_buckets = int(run_meta["network"]["output_bucket_count"])
    wdl = float(run_meta["training"]["wdl"])
    data = load_validation(Path(args.validation_data), args.max_positions)
    print(f"Preparing features for {data.size:,} positions...")
    features = prepare_features(
        data,
        layout,
        output_buckets,
        with_threats=True,
        with_p3=True,
    )
    results = {}
    for spec in args.net:
        name, separator, raw_path = spec.partition("=")
        if not separator:
            raise ValueError("--net must use NAME=PATH")
        path = Path(raw_path)
        net = load_net(path)
        tensors = decode_tensors(net)
        metrics = evaluate_net(net, tensors, features, data, wdl, args.batch_size)
        metrics["path"] = str(path.resolve())
        results[name] = metrics
        print(f"{name}: mse={metrics['mse']:.9f} mae={metrics['mae']:.7f}")
    report = {"validation_data": str(Path(args.validation_data).resolve()), "wdl": wdl, "results": results}
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
