#!/usr/bin/env python3
"""Rank Bullet checkpoints on a fixed held-out Bullet-format dataset."""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
UTILS_NNUE_DIR = THIS_DIR.parent
if str(UTILS_NNUE_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_NNUE_DIR))

from full_threats_v1 import MAX_ACTIVE_THREATS, enumerate_board  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Sykora NNUE checkpoints.")
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--run-meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Validate every Nth saved checkpoint plus the final checkpoint",
    )
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Validate only the final saved checkpoint",
    )
    return parser.parse_args()


def checkpoint_key(path: Path) -> tuple[int, str]:
    match = re.search(r"-(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def load_validation(path: Path, max_positions: int):
    import numpy as np

    dtype = np.dtype(
        [
            ("occ", "<u8"),
            ("pcs", "u1", (16,)),
            ("score", "<i2"),
            ("result", "u1"),
            ("ksq", "u1"),
            ("opp_ksq", "u1"),
            ("extra", "u1", (3,)),
        ],
        align=False,
    )
    if dtype.itemsize != 32:
        raise AssertionError(f"unexpected BulletFormat dtype size: {dtype.itemsize}")
    data = np.fromfile(path, dtype=dtype)
    if max_positions > 0:
        data = data[:max_positions]
    if data.size == 0:
        raise ValueError("validation dataset is empty")
    return data


def packed_position_board(position):
    import chess

    board = chess.Board.empty()
    ordinal = 0
    remaining = int(position["occ"])
    while remaining:
        lsb = remaining & -remaining
        square = lsb.bit_length() - 1
        packed = int(position["pcs"][ordinal // 2])
        piece_code = (packed >> (4 * (ordinal & 1))) & 0xF
        piece_type = (piece_code & 7) + 1
        colour = chess.BLACK if piece_code & 8 else chess.WHITE
        board.set_piece_at(square, chess.Piece(piece_type, colour))
        ordinal += 1
        remaining ^= lsb
    board.turn = chess.WHITE
    return board


def prepare_features(
    data,
    bucket_layout: list[int],
    output_buckets: int,
    *,
    with_threats: bool,
):
    import numpy as np
    import chess

    count = data.size
    stm = np.full((count, 32), -1, dtype=np.int32)
    ntm = np.full((count, 32), -1, dtype=np.int32)
    buckets = np.zeros(count, dtype=np.intp)
    stm_threats = (
        np.full((count, MAX_ACTIVE_THREATS), -1, dtype=np.int32)
        if with_threats
        else None
    )
    ntm_threats = (
        np.full((count, MAX_ACTIVE_THREATS), -1, dtype=np.int32)
        if with_threats
        else None
    )

    for row, position in enumerate(data):
        occ = int(position["occ"])
        own_king = int(position["ksq"])
        opp_king = int(position["opp_ksq"])
        own_flip = 7 if own_king % 8 > 3 else 0
        opp_flip = 7 if opp_king % 8 > 3 else 0
        own_offset = 768 * bucket_layout[own_king]
        opp_offset = 768 * bucket_layout[opp_king]
        ordinal = 0
        remaining = occ
        while remaining:
            lsb = remaining & -remaining
            square = lsb.bit_length() - 1
            packed = int(position["pcs"][ordinal // 2])
            piece = (packed >> (4 * (ordinal & 1))) & 0xF
            colour = 1 if piece & 8 else 0
            piece_offset = (piece & 7) * 64
            stm_base = (384 if colour else 0) + piece_offset + square
            ntm_base = (0 if colour else 384) + piece_offset + (square ^ 56)
            stm[row, ordinal] = own_offset + (stm_base ^ own_flip)
            ntm[row, ordinal] = opp_offset + (ntm_base ^ opp_flip)
            ordinal += 1
            remaining ^= lsb
        if output_buckets > 1:
            divisor = (32 + output_buckets - 1) // output_buckets
            buckets[row] = min((occ.bit_count() - 2) // divisor, output_buckets - 1)

        if with_threats:
            board = packed_position_board(position)
            stm_features = enumerate_board(board, chess.WHITE)
            ntm_features = enumerate_board(board, chess.BLACK)
            if len(stm_features) != len(ntm_features):
                raise AssertionError(
                    f"threat perspective count mismatch at row {row}: "
                    f"{len(stm_features)} != {len(ntm_features)}"
                )
            stm_threats[row, : len(stm_features)] = stm_features
            ntm_threats[row, : len(ntm_features)] = ntm_features

    if with_threats:
        max_active = max(
            int(np.count_nonzero(stm_threats >= 0, axis=1).max()),
            int(np.count_nonzero(ntm_threats >= 0, axis=1).max()),
        )
        stm_threats = stm_threats[:, :max_active]
        ntm_threats = ntm_threats[:, :max_active]

    return stm, ntm, stm_threats, ntm_threats, buckets


def sigmoid(values):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def forward_batch(
    ckpt,
    ft_weights,
    padded_ft_weights,
    ft_bias,
    stm_idx,
    ntm_idx,
    stm_threat_idx,
    ntm_threat_idx,
    padded_threat_weights,
    buckets,
):
    import numpy as np

    sentinel = ft_weights.shape[0]
    stm_safe = np.where(stm_idx >= 0, stm_idx, sentinel)
    ntm_safe = np.where(ntm_idx >= 0, ntm_idx, sentinel)
    stm_acc = ft_bias + padded_ft_weights[stm_safe].sum(axis=1, dtype=np.float32)
    ntm_acc = ft_bias + padded_ft_weights[ntm_safe].sum(axis=1, dtype=np.float32)
    if padded_threat_weights is not None:
        sentinel = padded_threat_weights.shape[0] - 1
        stm_safe = np.where(stm_threat_idx >= 0, stm_threat_idx, sentinel)
        ntm_safe = np.where(ntm_threat_idx >= 0, ntm_threat_idx, sentinel)
        stm_acc += padded_threat_weights[stm_safe].sum(axis=1, dtype=np.float32)
        ntm_acc += padded_threat_weights[ntm_safe].sum(axis=1, dtype=np.float32)

    architecture = str(np.asarray(ckpt["architecture"]).reshape(-1)[0])
    half = ft_weights.shape[1] // 2
    stm_clipped = np.clip(stm_acc, 0.0, 1.0)
    ntm_clipped = np.clip(ntm_acc, 0.0, 1.0)
    pooled = np.concatenate(
        (
            stm_clipped[:, :half] * stm_clipped[:, half:],
            ntm_clipped[:, :half] * ntm_clipped[:, half:],
        ),
        axis=1,
    )
    if architecture == "pairwise-linear":
        weights = np.asarray(ckpt["out_weights"], dtype=np.float32)[buckets]
        biases = np.asarray(ckpt["out_bias"], dtype=np.float32)[buckets]
        return np.einsum("bi,bi->b", pooled, weights, optimize=True) + biases
    if architecture != "pairwise-mlp":
        raise ValueError(f"unsupported architecture in NPZ: {architecture}")

    l1w = np.asarray(ckpt["l1_weights"], dtype=np.float32)[buckets]
    l1b = np.asarray(ckpt["l1_bias"], dtype=np.float32)[buckets]
    z1 = np.einsum("bi,bij->bj", pooled, l1w, optimize=True) + l1b
    dual = np.concatenate((np.clip(z1, 0.0, 1.0), np.clip(z1 * z1, 0.0, 1.0)), axis=1)
    l2w = np.asarray(ckpt["l2_weights"], dtype=np.float32)[buckets]
    l2b = np.asarray(ckpt["l2_bias"], dtype=np.float32)[buckets]
    z2 = np.einsum("bi,bij->bj", dual, l2w, optimize=True) + l2b
    hidden = np.clip(z2, 0.0, 1.0) ** 2
    l3w = np.asarray(ckpt["l3_weights"], dtype=np.float32)[buckets]
    l3b = np.asarray(ckpt["l3_bias"], dtype=np.float32)[buckets]
    return np.einsum("bi,bi->b", hidden, l3w, optimize=True) + l3b


def validate_npz(
    npz_path: Path,
    data,
    stm,
    ntm,
    stm_threats,
    ntm_threats,
    buckets,
    wdl: float,
    batch_size: int,
) -> dict:
    import numpy as np

    score_target = sigmoid(data["score"].astype(np.float32) / 400.0)
    result_target = data["result"].astype(np.float32) / 2.0
    targets = wdl * result_target + (1.0 - wdl) * score_target
    predictions = np.empty(data.size, dtype=np.float32)

    with np.load(npz_path) as ckpt:
        architecture = str(np.asarray(ckpt["architecture"]).reshape(-1)[0])
        ft_weights = np.asarray(ckpt["ft_weights"], dtype=np.float32)
        ft_bias = np.asarray(ckpt["ft_bias"], dtype=np.float32)
        padded_ft_weights = np.concatenate(
            (ft_weights, np.zeros((1, ft_weights.shape[1]), dtype=np.float32)), axis=0
        )
        padded_threat_weights = None
        if "threat_weights" in ckpt.files:
            threat_weights = np.asarray(ckpt["threat_weights"], dtype=np.float32)
            padded_threat_weights = np.empty(
                (threat_weights.shape[0] + 1, threat_weights.shape[1]), dtype=np.float32
            )
            padded_threat_weights[:-1] = threat_weights
            padded_threat_weights[-1] = 0.0
            del threat_weights
        for start in range(0, data.size, batch_size):
            end = min(start + batch_size, data.size)
            predictions[start:end] = sigmoid(
                forward_batch(
                    ckpt,
                    ft_weights,
                    padded_ft_weights,
                    ft_bias,
                    stm[start:end],
                    ntm[start:end],
                    stm_threats[start:end] if stm_threats is not None else None,
                    ntm_threats[start:end] if ntm_threats is not None else None,
                    padded_threat_weights,
                    buckets[start:end],
                )
            )

    error = predictions - targets
    epsilon = 1e-7
    bce = -np.mean(
        targets * np.log(np.clip(predictions, epsilon, 1.0 - epsilon))
        + (1.0 - targets) * np.log(np.clip(1.0 - predictions, epsilon, 1.0 - epsilon))
    )
    bucket_metrics = {}
    for bucket in sorted(set(int(value) for value in buckets.tolist())):
        mask = buckets == bucket
        bucket_metrics[str(bucket)] = {
            "positions": int(mask.sum()),
            "mse": float(np.mean(error[mask] ** 2)),
            "mae": float(np.mean(np.abs(error[mask]))),
        }
    result = {
        "architecture": architecture,
        "positions": int(data.size),
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "bce": float(bce),
        "prediction_mean": float(predictions.mean()),
        "target_mean": float(targets.mean()),
        "per_output_bucket": bucket_metrics,
    }
    if stm_threats is not None:
        active = np.count_nonzero(stm_threats >= 0, axis=1)
        result["active_threats_mean"] = float(active.mean())
        result["active_threats_max"] = int(active.max())
    return result


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.every <= 0 or args.max_positions < 0:
        print("--batch-size and --every must be > 0; --max-positions must be >= 0", file=sys.stderr)
        return 2

    checkpoint_root = Path(args.checkpoints_dir)
    validation_path = Path(args.validation_data)
    run_meta_path = Path(args.run_meta)
    if not checkpoint_root.is_dir() or not validation_path.is_file() or not run_meta_path.is_file():
        print("checkpoint directory, validation data, or run metadata is missing", file=sys.stderr)
        return 1

    run_meta = json.loads(run_meta_path.read_text())
    output_buckets = int(run_meta["network"]["output_bucket_count"])
    with_threats = run_meta["network"].get("format") == "syk8"
    bucket_layout = [int(value) for value in run_meta["network"]["bucket_layout_64"]]
    wdl = float(run_meta["training"]["wdl"])
    data = load_validation(validation_path, args.max_positions)
    print(f"Preparing features for {data.size:,} held-out positions...")
    stm, ntm, stm_threats, ntm_threats, buckets = prepare_features(
        data,
        bucket_layout,
        output_buckets,
        with_threats=with_threats,
    )
    validation_batch_size = min(args.batch_size, 64) if with_threats else args.batch_size
    if validation_batch_size != args.batch_size:
        print(
            f"Capping v8 validation batch size at {validation_batch_size} to bound threat-gather memory"
        )

    checkpoints = sorted(
        [path for path in checkpoint_root.iterdir() if path.is_dir() and (path / "raw.bin").is_file()],
        key=checkpoint_key,
    )
    if not checkpoints:
        print("no checkpoints with raw.bin found", file=sys.stderr)
        return 1
    if args.final_only:
        selected = [checkpoints[-1]]
    else:
        selected = checkpoints[:: args.every]
        if checkpoints[-1] not in selected:
            selected.append(checkpoints[-1])

    results = []
    converter = THIS_DIR / "checkpoint_raw_to_npz.py"
    with tempfile.TemporaryDirectory(prefix="sykora-validation-") as temp:
        temp_dir = Path(temp)
        for index, checkpoint in enumerate(selected, start=1):
            npz_path = temp_dir / f"{checkpoint.name}.npz"
            cmd = [
                sys.executable,
                str(converter),
                "--input",
                str(checkpoint),
                "--run-meta",
                str(run_meta_path),
                "--output",
                str(npz_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            metrics = validate_npz(
                npz_path,
                data,
                stm,
                ntm,
                stm_threats,
                ntm_threats,
                buckets,
                wdl,
                validation_batch_size,
            )
            npz_path.unlink()
            metrics["checkpoint"] = str(checkpoint.resolve())
            metrics["checkpoint_name"] = checkpoint.name
            metrics["superbatch"] = checkpoint_key(checkpoint)[0]
            results.append(metrics)
            print(
                f"[{index}/{len(selected)}] {checkpoint.name}: "
                f"mse={metrics['mse']:.9f} mae={metrics['mae']:.7f}"
            )

    ranked = sorted(results, key=lambda result: (result["mse"], result["superbatch"]))
    report = {
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "checkpoints_dir": str(checkpoint_root.resolve()),
        "validation_data": str(validation_path.resolve()),
        "validation_positions": int(data.size),
        "wdl": wdl,
        "selection_metric": "mse",
        "best": ranked[0],
        "ranked": ranked,
        "results_by_superbatch": sorted(results, key=lambda result: result["superbatch"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Best checkpoint: {ranked[0]['checkpoint_name']} (mse={ranked[0]['mse']:.9f})")
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
