#!/usr/bin/env python3
"""Convert Bullet simple quantised.bin -> Sykora SYKNNUE1.

Supports the common Bullet save format from `examples/simple.rs` / `1_simple.rs`:
  l0w (i16, QA) | l0b (i16, QA) | l1w (i16, QB) | l1b (i16, QA*QB) | padding
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Bullet quantised.bin to SYKNNUE1.")
    parser.add_argument("--input", required=True, help="Path to Bullet quantised.bin")
    parser.add_argument("--output-net", required=True, help="Output SYKNNUE1 .sknnue path")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=0,
        help="Hidden size (0 = infer from file length)",
    )
    parser.add_argument("--input-size", type=int, default=768, help="Input feature size (default: 768)")
    parser.add_argument(
        "--strict-padding",
        action="store_true",
        help="Require all trailing bytes to be Bullet's known padding pattern",
    )
    return parser.parse_args()


def infer_hidden_size(total_len: int, input_size: int) -> Optional[tuple[int, int]]:
    # Core payload (without trailing pad):
    #   input_weights: input_size * h i16 => 2*input_size*h
    #   input_biases: h i16             => 2*h
    #   output_weights: 2*h i16         => 4*h
    #   output_bias: 1 i16              => 2
    # => payload_len = (2*input_size + 6) * h + 2
    stride = 2 * input_size + 6
    matches: list[tuple[int, int]] = []
    for h in range(1, 4097):
        payload_len = stride * h + 2
        pad = (64 - (payload_len % 64)) % 64
        if payload_len + pad == total_len:
            matches.append((h, payload_len))

    if len(matches) == 1:
        return matches[0]
    return None


def main() -> int:
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input not found: {in_path}")

    raw = in_path.read_bytes()
    total_len = len(raw)
    input_size = args.input_size
    stride = 2 * input_size + 6

    if args.hidden_size > 0:
        hidden = args.hidden_size
        payload_len = stride * hidden + 2
        if payload_len > total_len:
            raise ValueError(
                f"Hidden size {hidden} implies payload {payload_len} bytes > file length {total_len}"
            )
    else:
        inferred = infer_hidden_size(total_len, input_size)
        if inferred is None:
            raise ValueError(
                "Could not infer hidden size from file length; pass --hidden-size explicitly."
            )
        hidden, payload_len = inferred

    payload = raw[:payload_len]
    padding = raw[payload_len:]

    if args.strict_padding and padding:
        allowed = b"bullet"
        if any(byte not in allowed for byte in padding):
            raise ValueError("Trailing padding contains unexpected bytes")

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required for conversion") from exc

    # Interpret as little-endian i16 stream.
    vals = np.frombuffer(payload, dtype="<i2")
    expected_i16 = input_size * hidden + hidden + (2 * hidden) + 1
    if vals.size != expected_i16:
        raise ValueError(
            f"Unexpected i16 count: got {vals.size}, expected {expected_i16} "
            f"(hidden={hidden}, input={input_size})"
        )

    i = 0
    input_weights = vals[i : i + input_size * hidden]
    i += input_size * hidden
    input_biases = vals[i : i + hidden]
    i += hidden
    output_weights = vals[i : i + 2 * hidden]
    i += 2 * hidden
    output_bias_i16 = int(vals[i])

    # Shared writer is under utils/nnue/common.py
    import sys

    utils_nnue_dir = Path(__file__).resolve().parents[1]
    if str(utils_nnue_dir) not in sys.path:
        sys.path.insert(0, str(utils_nnue_dir))

    from common import write_syk_nnue  # noqa: E402

    out_path = Path(args.output_net)
    write_syk_nnue(
        out_path,
        hidden_size=hidden,
        input_biases_i16=input_biases.astype(np.int16).tolist(),
        input_weights_i16=input_weights.astype(np.int16).tolist(),
        output_weights_i16=output_weights.astype(np.int16).tolist(),
        output_bias_i32=output_bias_i16,
    )

    print(f"Input: {in_path}")
    print(f"Total bytes: {total_len}")
    print(f"Hidden size: {hidden}")
    print(f"Payload bytes: {payload_len}")
    print(f"Padding bytes: {len(padding)}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
