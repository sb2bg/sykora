#!/usr/bin/env python3
"""Train a starter Sykora NNUE network and export SYKNNUE1 weights.

Backends:
- numpy (default): portable and robust, no torch dependency
- torch: faster on suitable setups (CPU/GPU)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import chess

from common import INPUT_SIZE, QA, QB, SCALE, fen_feature_indices, write_syk_nnue


@dataclass
class Sample:
    white_features: List[int]
    black_features: List[int]
    stm_white: bool
    target: float


_TORCH = None
_F = None


def require_torch():
    global _TORCH, _F
    if _TORCH is not None and _F is not None:
        return _TORCH, _F

    try:
        import torch as torch_mod
        import torch.nn.functional as f_mod
    except Exception as exc:
        raise RuntimeError(
            "PyTorch import failed. Install torch and ensure OpenMP runtime is available."
        ) from exc

    _TORCH = torch_mod
    _F = f_mod
    return _TORCH, _F


def require_numpy():
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy import failed. Install numpy for --backend numpy.") from exc
    return np


def create_model(hidden_size: int):
    torch, _ = require_torch()

    class SykNnueModel(torch.nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.input_weights = torch.nn.Parameter(torch.empty(INPUT_SIZE, hidden_size))
            self.input_bias = torch.nn.Parameter(torch.zeros(hidden_size))
            self.output_weights = torch.nn.Parameter(torch.empty(2 * hidden_size))
            self.output_bias = torch.nn.Parameter(torch.zeros(1))

            torch.nn.init.normal_(self.input_weights, mean=0.0, std=0.15)
            torch.nn.init.normal_(self.output_weights, mean=0.0, std=0.15)

        def forward(
            self,
            white_batch: Sequence,
            black_batch: Sequence,
            stm_white_batch,
        ):
            batch_size = len(white_batch)
            device = self.input_weights.device

            white_acc = self.input_bias.unsqueeze(0).repeat(batch_size, 1)
            black_acc = self.input_bias.unsqueeze(0).repeat(batch_size, 1)

            for i in range(batch_size):
                if white_batch[i].numel() > 0:
                    white_acc[i] = white_acc[i] + self.input_weights[white_batch[i]].sum(dim=0)
                if black_batch[i].numel() > 0:
                    black_acc[i] = black_acc[i] + self.input_weights[black_batch[i]].sum(dim=0)

            stm_mask = stm_white_batch.to(device=device).unsqueeze(1)
            us_acc = torch.where(stm_mask, white_acc, black_acc)
            them_acc = torch.where(stm_mask, black_acc, white_acc)

            us = torch.clamp(us_acc, 0.0, 1.0)
            them = torch.clamp(them_acc, 0.0, 1.0)

            own_out = self.output_weights[: self.hidden_size]
            opp_out = self.output_weights[self.hidden_size :]
            raw = (us * own_out).sum(dim=1) + (them * opp_out).sum(dim=1) + self.output_bias

            cp = raw * float(SCALE)
            return cp

    return SykNnueModel(hidden_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export a starter SYKNNUE1 net.")
    parser.add_argument("--input", default="nnue/data/labeled.jsonl", help="Labeled JSONL input")
    parser.add_argument("--output-net", default="nnue/syk_starter.sknnue", help="Output net path")
    parser.add_argument("--hidden-size", type=int, default=512, help="Accumulator width")
    parser.add_argument("--epochs", type=int, default=4, help="Epoch count")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--eval-scale", type=float, default=400.0, help="Sigmoid eval scale")
    parser.add_argument(
        "--target-mode",
        choices=("prob", "cp"),
        default="prob",
        help="Training target domain: prob=sigmoid target, cp=direct centipawn regression",
    )
    parser.add_argument(
        "--cp-target-key",
        default="teacher_cp_stm",
        help="JSON key to read cp target from when --target-mode cp",
    )
    parser.add_argument(
        "--cp-norm",
        type=float,
        default=400.0,
        help="Normalization divisor for cp regression target/loss",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda (torch backend only)")
    parser.add_argument(
        "--augment-mirror",
        action="store_true",
        help="Add board-mirrored samples during data load to reduce color/orientation bias",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "torch"),
        default="numpy",
        help="Training backend (default: numpy)",
    )
    args = parser.parse_args()

    if args.hidden_size <= 0:
        parser.error("--hidden-size must be > 0")
    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.lr <= 0:
        parser.error("--lr must be > 0")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be >= 0")
    if args.eval_scale <= 0:
        parser.error("--eval-scale must be > 0")
    if args.cp_norm <= 0:
        parser.error("--cp-norm must be > 0")

    return args


def choose_device(arg: str):
    torch, _ = require_torch()
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def load_samples(
    path: Path,
    limit: int,
    augment_mirror: bool,
    target_mode: str,
    cp_target_key: str,
) -> List[Sample]:
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            fen = row["fen"]
            if target_mode == "cp":
                target = float(row.get(cp_target_key, row.get("teacher_cp_stm", 0.0)))
            else:
                target = float(row.get("target_blended_stm", row.get("target_result_stm", 0.5)))
            white_feats, black_feats, stm_white = fen_feature_indices(fen)
            samples.append(
                Sample(
                    white_features=white_feats,
                    black_features=black_feats,
                    stm_white=stm_white,
                    target=target,
                )
            )
            if augment_mirror:
                board = chess.Board(fen)
                mirrored = board.mirror()
                mfen = mirrored.fen(en_passant="fen")
                m_white_feats, m_black_feats, m_stm_white = fen_feature_indices(mfen)
                samples.append(
                    Sample(
                        white_features=m_white_feats,
                        black_features=m_black_feats,
                        stm_white=m_stm_white,
                        target=target,
                    )
                )
            if limit > 0 and len(samples) >= limit:
                break
    return samples


def to_batch_tensors(samples: Sequence[Sample], device) -> Tuple[List, List, object, object]:
    torch, _ = require_torch()
    white_batch = [torch.tensor(s.white_features, dtype=torch.long, device=device) for s in samples]
    black_batch = [torch.tensor(s.black_features, dtype=torch.long, device=device) for s in samples]
    stm = torch.tensor([s.stm_white for s in samples], dtype=torch.bool, device=device)
    target = torch.tensor([s.target for s in samples], dtype=torch.float32, device=device)
    return white_batch, black_batch, stm, target


def quantize_i16_torch(t) -> List[int]:
    torch, _ = require_torch()
    clamped = torch.clamp(torch.round(t), -32768, 32767).to(torch.int16).cpu().view(-1)
    return [int(v) for v in clamped.tolist()]


def quantize_i16_numpy(arr, np) -> List[int]:
    clamped = np.clip(np.rint(arr), -32768, 32767).astype(np.int16).reshape(-1)
    return [int(v) for v in clamped.tolist()]


def sigmoid_scalar(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def bce_loss_scalar(pred: float, target: float) -> float:
    eps = 1e-8
    p = max(eps, min(1.0 - eps, pred))
    return -(target * math.log(p) + (1.0 - target) * math.log(1.0 - p))


def train_with_torch(args: argparse.Namespace, train_samples: List[Sample], valid_samples: List[Sample]):
    torch, F = require_torch()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    model = create_model(hidden_size=args.hidden_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_samples)
        model.train()
        train_loss = 0.0
        train_count = 0

        for start in range(0, len(train_samples), args.batch_size):
            batch = train_samples[start : start + args.batch_size]
            w, b, stm, target = to_batch_tensors(batch, device)
            cp = model(w, b, stm)
            if args.target_mode == "cp":
                loss = F.smooth_l1_loss(cp / float(args.cp_norm), target / float(args.cp_norm))
            else:
                pred = torch.sigmoid(cp / float(args.eval_scale))
                loss = F.binary_cross_entropy(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            train_loss += float(loss.item()) * len(batch)
            train_count += len(batch)

        model.eval()
        valid_loss = 0.0
        valid_count = 0
        with torch.no_grad():
            for start in range(0, len(valid_samples), args.batch_size):
                batch = valid_samples[start : start + args.batch_size]
                w, b, stm, target = to_batch_tensors(batch, device)
                cp = model(w, b, stm)
                if args.target_mode == "cp":
                    loss = F.smooth_l1_loss(cp / float(args.cp_norm), target / float(args.cp_norm))
                else:
                    pred = torch.sigmoid(cp / float(args.eval_scale))
                    loss = F.binary_cross_entropy(pred, target)
                valid_loss += float(loss.item()) * len(batch)
                valid_count += len(batch)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss / max(train_count, 1):.6f} | "
            f"valid_loss={valid_loss / max(valid_count, 1):.6f}"
        )

    model = model.cpu()
    return (
        model.input_bias.data.tolist(),
        model.input_weights.data.reshape(-1).tolist(),
        model.output_weights.data.tolist(),
        float(model.output_bias.data.item()),
    )


def train_with_numpy(args: argparse.Namespace, train_samples: List[Sample], valid_samples: List[Sample]):
    np = require_numpy()
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    hidden = args.hidden_size
    input_weights = rng.normal(0.0, 0.15, size=(INPUT_SIZE, hidden)).astype(np.float32)
    input_bias = np.zeros(hidden, dtype=np.float32)
    output_weights = rng.normal(0.0, 0.15, size=(2 * hidden,)).astype(np.float32)
    output_bias = np.float32(0.0)

    own_out = output_weights[:hidden]
    opp_out = output_weights[hidden:]

    raw_to_sigmoid = float(SCALE) / float(args.eval_scale)
    cp_to_norm = float(SCALE) / float(args.cp_norm)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_samples)
        train_loss = 0.0
        train_count = 0

        for start in range(0, len(train_samples), args.batch_size):
            batch = train_samples[start : start + args.batch_size]
            bsz = len(batch)

            g_input_weights = np.zeros_like(input_weights)
            g_input_bias = np.zeros_like(input_bias)
            g_output_weights = np.zeros_like(output_weights)
            g_output_bias = 0.0

            batch_loss = 0.0

            for sample in batch:
                white_acc = input_bias.copy()
                black_acc = input_bias.copy()

                if sample.white_features:
                    white_acc += input_weights[sample.white_features].sum(axis=0)
                if sample.black_features:
                    black_acc += input_weights[sample.black_features].sum(axis=0)

                if sample.stm_white:
                    us_acc = white_acc
                    them_acc = black_acc
                else:
                    us_acc = black_acc
                    them_acc = white_acc

                us = np.clip(us_acc, 0.0, 1.0)
                them = np.clip(them_acc, 0.0, 1.0)

                raw = float(np.dot(us, own_out) + np.dot(them, opp_out) + output_bias)
                if args.target_mode == "cp":
                    pred = raw * cp_to_norm
                    target = sample.target / float(args.cp_norm)
                    diff = pred - target
                    batch_loss += 0.5 * diff * diff
                    d_raw = diff * cp_to_norm
                else:
                    pred = sigmoid_scalar(raw * raw_to_sigmoid)
                    batch_loss += bce_loss_scalar(pred, sample.target)
                    d_raw = (pred - sample.target) * raw_to_sigmoid

                g_output_weights[:hidden] += us * d_raw
                g_output_weights[hidden:] += them * d_raw
                g_output_bias += d_raw

                us_mask = ((us_acc > 0.0) & (us_acc < 1.0)).astype(np.float32)
                them_mask = ((them_acc > 0.0) & (them_acc < 1.0)).astype(np.float32)

                g_us_acc = own_out * d_raw * us_mask
                g_them_acc = opp_out * d_raw * them_mask

                if sample.stm_white:
                    g_white_acc = g_us_acc
                    g_black_acc = g_them_acc
                else:
                    g_white_acc = g_them_acc
                    g_black_acc = g_us_acc

                g_input_bias += g_white_acc + g_black_acc

                if sample.white_features:
                    g_input_weights[sample.white_features] += g_white_acc
                if sample.black_features:
                    g_input_weights[sample.black_features] += g_black_acc

            scale = 1.0 / float(max(1, bsz))
            lr = args.lr
            decay = 1.0 - (lr * args.weight_decay)

            input_weights *= decay
            input_bias *= decay
            output_weights *= decay
            output_bias *= decay

            input_weights -= (lr * scale) * g_input_weights
            input_bias -= (lr * scale) * g_input_bias
            output_weights -= (lr * scale) * g_output_weights
            output_bias -= (lr * scale) * g_output_bias

            own_out = output_weights[:hidden]
            opp_out = output_weights[hidden:]

            train_loss += batch_loss
            train_count += bsz

        valid_loss = 0.0
        valid_count = 0
        for sample in valid_samples:
            white_acc = input_bias.copy()
            black_acc = input_bias.copy()

            if sample.white_features:
                white_acc += input_weights[sample.white_features].sum(axis=0)
            if sample.black_features:
                black_acc += input_weights[sample.black_features].sum(axis=0)

            if sample.stm_white:
                us_acc = white_acc
                them_acc = black_acc
            else:
                us_acc = black_acc
                them_acc = white_acc

            us = np.clip(us_acc, 0.0, 1.0)
            them = np.clip(them_acc, 0.0, 1.0)

            raw = float(np.dot(us, own_out) + np.dot(them, opp_out) + output_bias)
            if args.target_mode == "cp":
                pred = raw * cp_to_norm
                target = sample.target / float(args.cp_norm)
                diff = pred - target
                valid_loss += 0.5 * diff * diff
            else:
                pred = sigmoid_scalar(raw * raw_to_sigmoid)
                valid_loss += bce_loss_scalar(pred, sample.target)
            valid_count += 1

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss / max(train_count, 1):.6f} | "
            f"valid_loss={valid_loss / max(valid_count, 1):.6f}"
        )

    return (
        input_bias,
        input_weights.reshape(-1),
        output_weights,
        float(output_bias),
    )


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    samples = load_samples(
        input_path,
        args.limit,
        args.augment_mirror,
        args.target_mode,
        args.cp_target_key,
    )
    if not samples:
        print("No samples loaded.", file=sys.stderr)
        return 1

    random.shuffle(samples)
    split = max(1, int(len(samples) * 0.9))
    train_samples = samples[:split]
    valid_samples = samples[split:]
    if not valid_samples:
        valid_samples = train_samples[: min(len(train_samples), 256)]

    try:
        if args.backend == "torch":
            params = train_with_torch(args, train_samples, valid_samples)
        else:
            params = train_with_numpy(args, train_samples, valid_samples)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    input_bias_f, input_weights_f, output_weights_f, output_bias_f = params

    # Quantize float-domain weights into the engine's integer-domain NNUE format.
    # input layer is scaled by QA, output layer by QB, output bias by QA*QB.
    np = require_numpy()
    input_bias_i16 = quantize_i16_numpy(np.asarray(input_bias_f, dtype=np.float32) * float(QA), np)
    input_weights_i16 = quantize_i16_numpy(np.asarray(input_weights_f, dtype=np.float32) * float(QA), np)
    output_weights_i16 = quantize_i16_numpy(np.asarray(output_weights_f, dtype=np.float32) * float(QB), np)
    output_bias_i32 = int(round(float(output_bias_f) * float(QA * QB)))

    out_net = Path(args.output_net)
    write_syk_nnue(
        out_net,
        hidden_size=args.hidden_size,
        input_biases_i16=input_bias_i16,
        input_weights_i16=input_weights_i16,
        output_weights_i16=output_weights_i16,
        output_bias_i32=output_bias_i32,
    )

    print(f"Backend: {args.backend}")
    print(f"Target mode: {args.target_mode}")
    print(f"Samples: {len(samples)} (train={len(train_samples)}, valid={len(valid_samples)})")
    print(f"Wrote {out_net}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
