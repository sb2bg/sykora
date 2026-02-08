#!/usr/bin/env python3
"""Convert a float-domain NPZ checkpoint into Sykora SYKNNUE1 format.

Expected arrays in NPZ:
- input_weights: shape [768, hidden]
- input_bias: shape [hidden]
- output_weights: shape [2 * hidden] or [2, hidden]
- output_bias: scalar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Import shared NNUE constants/exporter from utils/nnue/common.py
UTILS_NNUE_DIR = Path(__file__).resolve().parents[1]
if str(UTILS_NNUE_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_NNUE_DIR))

from common import INPUT_SIZE, QA, QB, write_syk_nnue  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NPZ checkpoint to SYKNNUE1 net.")
    parser.add_argument("--input", required=True, help="Input .npz checkpoint")
    parser.add_argument("--output-net", required=True, help="Output .sknnue path")
    parser.add_argument(
        "--input-weights-key", default="input_weights", help="NPZ key for input->hidden weights"
    )
    parser.add_argument("--input-bias-key", default="input_bias", help="NPZ key for hidden biases")
    parser.add_argument(
        "--output-weights-key", default="output_weights", help="NPZ key for output weights"
    )
    parser.add_argument("--output-bias-key", default="output_bias", help="NPZ key for output bias")
    return parser.parse_args()


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
        required = [
            args.input_weights_key,
            args.input_bias_key,
            args.output_weights_key,
            args.output_bias_key,
        ]
        missing = [k for k in required if k not in ckpt]
        if missing:
            raise KeyError(f"Missing NPZ keys: {missing}")

        input_weights = np.asarray(ckpt[args.input_weights_key], dtype=np.float32)
        input_bias = np.asarray(ckpt[args.input_bias_key], dtype=np.float32).reshape(-1)
        output_weights = np.asarray(ckpt[args.output_weights_key], dtype=np.float32)
        output_bias = float(np.asarray(ckpt[args.output_bias_key], dtype=np.float32).reshape(-1)[0])

    if input_weights.ndim != 2:
        raise ValueError(f"input_weights must be rank-2, got shape {input_weights.shape}")
    if input_weights.shape[0] != INPUT_SIZE:
        raise ValueError(
            f"input_weights first dim must be INPUT_SIZE={INPUT_SIZE}, got {input_weights.shape[0]}"
        )

    hidden_size = int(input_weights.shape[1])
    if input_bias.shape[0] != hidden_size:
        raise ValueError(
            f"input_bias length mismatch: expected {hidden_size}, got {input_bias.shape[0]}"
        )

    if output_weights.ndim == 2:
        if output_weights.shape == (2, hidden_size):
            output_weights = output_weights.reshape(-1)
        elif output_weights.shape == (hidden_size, 2):
            output_weights = output_weights.T.reshape(-1)
        else:
            raise ValueError(
                "2D output_weights must be shape [2, hidden] or [hidden, 2], "
                f"got {output_weights.shape}"
            )
    else:
        output_weights = output_weights.reshape(-1)

    if output_weights.shape[0] != 2 * hidden_size:
        raise ValueError(
            f"output_weights length mismatch: expected {2 * hidden_size}, got {output_weights.shape[0]}"
        )

    input_bias_i16 = np.clip(np.rint(input_bias * float(QA)), -32768, 32767).astype(np.int16)
    input_weights_i16 = np.clip(
        np.rint(input_weights.reshape(-1) * float(QA)), -32768, 32767
    ).astype(np.int16)
    output_weights_i16 = np.clip(
        np.rint(output_weights * float(QB)), -32768, 32767
    ).astype(np.int16)
    output_bias_i32 = int(round(output_bias * float(QA * QB)))

    out_path = Path(args.output_net)
    write_syk_nnue(
        out_path,
        hidden_size=hidden_size,
        input_biases_i16=input_bias_i16.tolist(),
        input_weights_i16=input_weights_i16.tolist(),
        output_weights_i16=output_weights_i16.tolist(),
        output_bias_i32=output_bias_i32,
    )

    print(f"Input: {in_path}")
    print(f"Hidden size: {hidden_size}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
