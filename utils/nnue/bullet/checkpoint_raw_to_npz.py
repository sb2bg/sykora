#!/usr/bin/env python3
"""Convert a Sykora Bullet checkpoint raw.bin into explicit NPZ tensors.

Supports two raw.bin layouts:
  * ``syk6_factorised`` — includes the ``l0f`` factoriser tensor before
    ``l0w``. The factoriser is merged into ``l0w`` to produce the final
    ``ft_weights``. This is the layout produced by the v6 trainer.
  * ``syk6_no_factoriser`` — no factoriser (legacy/broken v6 checkpoints).
    ``l0w`` is read directly as ``ft_weights``. Kept for backward
    compatibility; new runs should always use the factorised trainer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UTILS_NNUE_DIR = Path(__file__).resolve().parents[1]
if str(UTILS_NNUE_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_NNUE_DIR))

from common import (  # noqa: E402
    SCALE,
    NNUE_Q0,
    NNUE_Q,
    V3_BUCKET_LAYOUT_32,
    expand_mirrored_bucket_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a SYKNNUE6 Bullet raw checkpoint into NPZ tensors."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Checkpoint directory or raw.bin path",
    )
    parser.add_argument(
        "--run-meta",
        default="",
        help="Optional path to run_meta.json (default: infer from checkpoint path)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz path",
    )
    return parser.parse_args()


def resolve_paths(input_path: Path, run_meta_arg: str) -> tuple[Path, Path]:
    if input_path.is_dir():
        raw_path = input_path / "raw.bin"
    else:
        raw_path = input_path
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw.bin not found: {raw_path}")

    if run_meta_arg:
        run_meta_path = Path(run_meta_arg)
    else:
        try:
            run_meta_path = raw_path.parents[2] / "run_meta.json"
        except IndexError as exc:
            raise FileNotFoundError(
                "Could not infer run_meta.json from checkpoint path; pass --run-meta explicitly"
            ) from exc
    if not run_meta_path.is_file():
        raise FileNotFoundError(f"run_meta.json not found: {run_meta_path}")
    return raw_path, run_meta_path


def take_f32(buf, offset: int, count: int):
    end = offset + count
    if end > buf.shape[0]:
        raise ValueError(
            f"Unexpected EOF while decoding raw checkpoint: need {count} floats at offset {offset}, "
            f"have {buf.shape[0]}"
        )
    return buf[offset:end], end


def expected_raw_sizes(
    *, bucket_count: int, ft_hidden: int, network_format: str, output_bucket_count: int
) -> dict[str, int]:
    input_size = 768 * bucket_count
    if network_format != "syk6":
        raise ValueError(f"unsupported network format: {network_format}")

    factoriser = 768 * ft_hidden
    ft_weights = input_size * ft_hidden
    ft_biases = ft_hidden
    out_weights = output_bucket_count * 2 * ft_hidden
    out_biases = output_bucket_count

    return {
        "syk6_factorised": factoriser + ft_weights + ft_biases + out_weights + out_biases,
        "syk6_no_factoriser": ft_weights + ft_biases + out_weights + out_biases,
    }


def detect_layout(
    *, raw_len: int, bucket_count: int, ft_hidden: int, network_format: str, output_bucket_count: int
) -> str:
    sizes = expected_raw_sizes(
        bucket_count=bucket_count,
        ft_hidden=ft_hidden,
        network_format=network_format,
        output_bucket_count=output_bucket_count,
    )
    for name, expected in sizes.items():
        if raw_len == expected:
            return name
    expected_str = ", ".join(f"{k}={v}" for k, v in sizes.items())
    raise ValueError(
        f"raw.bin length mismatch: found {raw_len} floats, expected one of {expected_str}"
    )


def parse_network_config(run_meta: dict) -> dict:
    network = dict(run_meta.get("network", {}))
    env = run_meta.get("env", {})

    network_format = network.get("format") or env.get("SYK_NETWORK_FORMAT") or "syk6"
    if network_format != "syk6":
        raise ValueError(
            f"run_meta.json does not describe a SYKNNUE6 run: {network_format!r}"
        )

    if "bucket_layout_64" in network:
        bucket_layout_64 = [int(v) for v in network["bucket_layout_64"]]
    else:
        bucket_layout_name = env.get("SYK_BUCKET_LAYOUT", "v3_10")
        if bucket_layout_name != "v3_10":
            raise ValueError(
                f"Unsupported checkpoint bucket layout without explicit bucket_layout_64: "
                f"{bucket_layout_name!r}"
            )
        bucket_layout_64 = expand_mirrored_bucket_layout(V3_BUCKET_LAYOUT_32)

    return {
        "format": network_format,
        "bucket_layout_64": bucket_layout_64,
        "ft_hidden": int(network.get("ft_hidden") or env["SYK_HIDDEN"]),
        "output_bucket_count": int(
            network.get("output_bucket_count")
            or env.get("SYK_OUTPUT_BUCKETS")
            or 8
        ),
    }


def main() -> int:
    args = parse_args()

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required for checkpoint export") from exc

    raw_path, run_meta_path = resolve_paths(Path(args.input), args.run_meta)
    run_meta = json.loads(run_meta_path.read_text())
    network = parse_network_config(run_meta)

    bucket_layout_64 = [int(v) for v in network["bucket_layout_64"]]
    bucket_count = max(bucket_layout_64) + 1
    ft_hidden = int(network["ft_hidden"])
    network_format = str(network["format"])
    output_bucket_count = int(network["output_bucket_count"])
    input_size = 768 * bucket_count

    raw = np.fromfile(raw_path, dtype="<f4")
    layout = detect_layout(
        raw_len=raw.shape[0],
        bucket_count=bucket_count,
        ft_hidden=ft_hidden,
        network_format=network_format,
        output_bucket_count=output_bucket_count,
    )
    offset = 0

    if layout == "syk6_factorised":
        # Factoriser: [H, 768] in graph order (row-major), stored as 768*H floats.
        # Read it first, then the per-bucket l0w, and merge.
        l0f, offset = take_f32(raw, offset, 768 * ft_hidden)
        l0w, offset = take_f32(raw, offset, input_size * ft_hidden)

        # Merge: tile factoriser across all buckets and add to l0w.
        # l0f is [H, 768] in Bullet's Shape::new(hl_size, 768) convention,
        # which is row-major (output, input) = (H, 768). Flatten to [768, H]
        # for merging (feature-major, matching ft_weights layout).
        l0f_reshaped = l0f.reshape(ft_hidden, 768).T  # [768, H]
        factoriser_tiled = np.tile(l0f_reshaped, (bucket_count, 1))  # [input_size, H]
        ft_weights = l0w.reshape(input_size, ft_hidden) + factoriser_tiled
    else:
        # No factoriser (legacy checkpoint): read l0w directly.
        l0w, offset = take_f32(raw, offset, input_size * ft_hidden)
        ft_weights = l0w.reshape(input_size, ft_hidden)

    l0b, offset = take_f32(raw, offset, ft_hidden)
    outw_len = output_bucket_count * 2 * ft_hidden
    outb_len = output_bucket_count
    outw, offset = take_f32(raw, offset, outw_len)
    outb, offset = take_f32(raw, offset, outb_len)

    if offset != raw.shape[0]:
        raise ValueError(
            f"raw.bin length mismatch: expected {offset} floats from metadata, found {raw.shape[0]}"
        )

    ft_bias = l0b.reshape(ft_hidden)
    out_weights = outw.reshape(output_bucket_count, 2 * ft_hidden)
    out_bias = outb.reshape(output_bucket_count)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ft_weights=ft_weights.astype(np.float32),
        ft_bias=ft_bias.astype(np.float32),
        out_weights=out_weights.astype(np.float32),
        out_bias=out_bias.astype(np.float32),
        bucket_layout_64=np.asarray(bucket_layout_64, dtype=np.uint8),
        feature_set=np.asarray([1], dtype=np.uint8),
        input_bucket_count=np.asarray([bucket_count], dtype=np.uint8),
        output_bucket_count=np.asarray([output_bucket_count], dtype=np.uint8),
        activation_type=np.asarray([1], dtype=np.uint8),
        q0=np.asarray([NNUE_Q0], dtype=np.uint16),
        q=np.asarray([NNUE_Q], dtype=np.uint16),
        scale=np.asarray([SCALE], dtype=np.uint16),
    )

    print(f"Input: {raw_path}")
    print(f"Run metadata: {run_meta_path}")
    print(f"Network format: {network_format.upper()}")
    print(f"Detected raw layout: {layout}")
    print(f"Bucket count: {bucket_count}")
    print(f"FT hidden: {ft_hidden}")
    print(f"Output buckets: {output_bucket_count}")
    print(f"Dense head: bucketed linear {2 * ft_hidden} -> 1")
    if layout == "syk6_factorised":
        print("Factoriser: merged l0f into l0w")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
