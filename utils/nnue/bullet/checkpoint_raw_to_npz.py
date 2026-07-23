#!/usr/bin/env python3
"""Decode a Sykora Bullet raw checkpoint into explicit float-domain tensors."""

from __future__ import annotations

import argparse
import json
import struct
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
    SCALE,
    V3_BUCKET_LAYOUT_32,
    expand_mirrored_bucket_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Bullet raw checkpoint to NPZ.")
    parser.add_argument("--input", required=True, help="Checkpoint directory or raw.bin")
    parser.add_argument("--run-meta", default="", help="run_meta.json override")
    parser.add_argument("--output", required=True, help="Output .npz")
    parser.add_argument(
        "--allow-missing-factoriser",
        action="store_true",
        help="Treat l0w as complete weights for legacy checkpoints that omitted l0f",
    )
    return parser.parse_args()


def resolve_paths(input_path: Path, run_meta_arg: str) -> tuple[Path, Path]:
    raw_path = input_path / "raw.bin" if input_path.is_dir() else input_path
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw.bin not found: {raw_path}")
    run_meta_path = Path(run_meta_arg) if run_meta_arg else raw_path.parents[2] / "run_meta.json"
    if not run_meta_path.is_file():
        raise FileNotFoundError(f"run_meta.json not found: {run_meta_path}")
    return raw_path, run_meta_path


def take_f32(buf, offset: int, count: int):
    end = offset + count
    if end > buf.shape[0]:
        raise ValueError(
            f"unexpected EOF: need {count} floats at offset {offset}, have {buf.shape[0]}"
        )
    return buf[offset:end], end


def parse_network_config(run_meta: dict) -> dict:
    network = dict(run_meta.get("network", {}))
    env = run_meta.get("env", {})
    network_format = network.get("format") or env.get("SYK_NETWORK_FORMAT") or "syk7"
    architecture = network.get("architecture") or env.get("SYK_ARCHITECTURE") or "pairwise-mlp"
    if architecture not in {"pairwise-linear", "pairwise-mlp"}:
        raise ValueError(f"unsupported architecture: {architecture!r}")
    if network_format not in {"syk7", "syk8"}:
        raise ValueError(f"unsupported network format: {network_format!r}")

    if "bucket_layout_64" in network:
        layout = [int(value) for value in network["bucket_layout_64"]]
    else:
        layout_name = network.get("bucket_layout_name") or env.get("SYK_BUCKET_LAYOUT", "v3_10")
        if layout_name != "v3_10":
            raise ValueError(f"unsupported implicit bucket layout: {layout_name!r}")
        layout = expand_mirrored_bucket_layout(V3_BUCKET_LAYOUT_32)
    if len(layout) != 64:
        raise ValueError(f"bucket layout must contain 64 entries, got {len(layout)}")

    return {
        "format": network_format,
        "architecture": architecture,
        "bucket_layout_64": layout,
        "ft_hidden": int(network.get("ft_hidden") or env.get("SYK_HIDDEN") or 512),
        "dense1": int(network.get("dense1") or env.get("SYK_DENSE1") or 16),
        "dense2": int(network.get("dense2") or env.get("SYK_DENSE2") or 32),
        "output_bucket_count": int(
            network.get("output_bucket_count") or env.get("SYK_OUTPUT_BUCKETS") or 1
        ),
        "factorised": bool(network.get("factorised", True)),
    }


def tail_float_count(config: dict) -> int:
    h = config["ft_hidden"]
    outputs = config["output_bucket_count"]
    architecture = config["architecture"]
    if architecture == "pairwise-linear":
        return outputs * h + outputs
    dense1 = config["dense1"]
    dense2 = config["dense2"]
    return (
        h * outputs * dense1
        + outputs * dense1
        + 2 * dense1 * outputs * dense2
        + outputs * dense2
        + dense2 * outputs
        + outputs
    )


def expected_raw_sizes(config: dict) -> dict[str, int]:
    h = config["ft_hidden"]
    buckets = max(config["bucket_layout_64"]) + 1
    common = 768 * buckets * h + h + tail_float_count(config)
    if config["format"] == "syk8":
        return {
            "virtual_factorised": (
                768 + 768 * buckets + FULL_THREATS_V1_COUNT
            )
            * h
            + h
            + tail_float_count(config)
        }
    return {
        "factorised": 768 * h + common,
        "missing_factoriser": common,
    }


def read_optimizer_weights(path: Path) -> dict[str, object]:
    import numpy as np

    data = path.read_bytes()
    offset = 0
    tensors: dict[str, object] = {}
    while offset < len(data):
        newline = data.find(b"\n", offset)
        if newline < 0:
            raise ValueError(f"malformed optimiser weights file: {path}")
        name = data[offset:newline].decode("ascii")
        offset = newline + 1
        if offset + 8 > len(data):
            raise ValueError(f"truncated optimiser tensor header: {name}")
        count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        end = offset + count * 4
        if end > len(data):
            raise ValueError(f"truncated optimiser tensor: {name}")
        tensors[name] = np.frombuffer(data, dtype="<f4", count=count, offset=offset).copy()
        offset = end
    return tensors


def raw_from_optimizer_state(raw_path: Path, config: dict):
    import numpy as np

    state_path = raw_path.parent / "optimiser_state" / "weights.bin"
    if not state_path.is_file():
        return None
    tensors = read_optimizer_weights(state_path)
    names = ["l0w", "l0b"] if config["format"] == "syk8" else ["l0f", "l0w", "l0b"]
    if config["architecture"] == "pairwise-linear":
        if "outw" in tensors and "outb" in tensors:
            names.extend(["outw", "outb"])
        else:
            # Older Sykora trainers named the single affine head l1.
            names.extend(["l1w", "l1b"])
    else:
        names.extend(["l1w", "l1b", "l2w", "l2b", "l3w", "l3b"])
    missing = [name for name in names if name not in tensors]
    if missing:
        raise ValueError(
            f"optimiser_state/weights.bin cannot recover checkpoint; missing tensors: {missing}"
        )
    return np.concatenate([tensors[name] for name in names]).astype("<f4", copy=False)


def decode_tail(raw, offset: int, config: dict) -> tuple[dict, int]:
    import numpy as np

    h = config["ft_hidden"]
    outputs = config["output_bucket_count"]
    architecture = config["architecture"]
    tensors: dict[str, object] = {}

    if architecture == "pairwise-linear":
        outw, offset = take_f32(raw, offset, outputs * h)
        outb, offset = take_f32(raw, offset, outputs)
        tensors["out_weights"] = outw.reshape(h, outputs).T.astype(np.float32)
        tensors["out_bias"] = outb.reshape(outputs).astype(np.float32)
    else:
        dense1 = config["dense1"]
        dense2 = config["dense2"]
        l1w, offset = take_f32(raw, offset, h * outputs * dense1)
        l1b, offset = take_f32(raw, offset, outputs * dense1)
        l2w, offset = take_f32(raw, offset, 2 * dense1 * outputs * dense2)
        l2b, offset = take_f32(raw, offset, outputs * dense2)
        l3w, offset = take_f32(raw, offset, dense2 * outputs)
        l3b, offset = take_f32(raw, offset, outputs)
        tensors.update(
            {
                "l1_weights": l1w.reshape(h, outputs, dense1)
                .transpose(1, 0, 2)
                .astype(np.float32),
                "l1_bias": l1b.reshape(outputs, dense1).astype(np.float32),
                "l2_weights": l2w.reshape(2 * dense1, outputs, dense2)
                .transpose(1, 0, 2)
                .astype(np.float32),
                "l2_bias": l2b.reshape(outputs, dense2).astype(np.float32),
                "l3_weights": l3w.reshape(dense2, outputs).T.astype(np.float32),
                "l3_bias": l3b.reshape(outputs).astype(np.float32),
            }
        )
    return tensors, offset


def main() -> int:
    args = parse_args()
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required") from exc

    raw_path, run_meta_path = resolve_paths(Path(args.input), args.run_meta)
    run_meta = json.loads(run_meta_path.read_text())
    config = parse_network_config(run_meta)
    h = config["ft_hidden"]
    bucket_count = max(config["bucket_layout_64"]) + 1
    input_size = 768 * bucket_count

    raw = np.fromfile(raw_path, dtype="<f4")
    sizes = expected_raw_sizes(config)
    if config["format"] == "syk8" and raw.size == sizes["virtual_factorised"]:
        raw_layout = "virtual_factorised"
    elif config["format"] != "syk8" and raw.size == sizes["factorised"]:
        raw_layout = "factorised"
    elif config["format"] != "syk8" and raw.size == sizes["missing_factoriser"]:
        raw_layout = "missing_factoriser"
    else:
        raise ValueError(
            f"raw.bin has {raw.size} floats; expected "
            + ", ".join(f"{name}={size}" for name, size in sizes.items())
        )

    if raw_layout == "missing_factoriser" and config["factorised"]:
        recovered = raw_from_optimizer_state(raw_path, config)
        if recovered is not None:
            raw = recovered
            raw_layout = "factorised_from_optimizer_state"
        elif not args.allow_missing_factoriser:
            raise ValueError(
                "checkpoint metadata says the FT is factorised, but raw.bin omits l0f and no "
                "recoverable optimiser_state/weights.bin exists. Bullet does not apply SavedFormat "
                "transforms to raw.bin, so l0w contains residuals, not deployable FT weights. "
                "Retrain with the updated runner. Use --allow-missing-factoriser only to inspect a "
                "known non-factorised legacy checkpoint."
            )

    offset = 0
    threat_weights = None
    if raw_layout == "virtual_factorised":
        virtual_count = 768 + input_size + FULL_THREATS_V1_COUNT
        l0w, offset = take_f32(raw, offset, virtual_count * h)
        virtual = l0w.reshape(virtual_count, h)
        factoriser = virtual[:768]
        residual = virtual[768 : 768 + input_size]
        threat_weights = virtual[768 + input_size :]
        ft_weights = residual + np.tile(factoriser, (bucket_count, 1))
    elif raw_layout in {"factorised", "factorised_from_optimizer_state"}:
        l0f, offset = take_f32(raw, offset, 768 * h)
        l0w, offset = take_f32(raw, offset, input_size * h)
        # Raw affine storage is column-major, so both tensors are already
        # feature-major after reshape: [feature, hidden].
        factoriser = l0f.reshape(768, h)
        residual = l0w.reshape(input_size, h)
        ft_weights = residual + np.tile(factoriser, (bucket_count, 1))
    else:
        l0w, offset = take_f32(raw, offset, input_size * h)
        ft_weights = l0w.reshape(input_size, h)

    l0b, offset = take_f32(raw, offset, h)
    tail, offset = decode_tail(raw, offset, config)
    if offset != raw.size:
        raise ValueError(f"decoder consumed {offset} floats but raw.bin contains {raw.size}")

    payload = {
        "ft_weights": ft_weights.astype(np.float32),
        "ft_bias": l0b.reshape(h).astype(np.float32),
        "bucket_layout_64": np.asarray(config["bucket_layout_64"], dtype=np.uint8),
        "feature_set": np.asarray(
            [
                FEATURE_SET_MIRRORED_PSQ_FULL_THREATS_V1
                if config["format"] == "syk8"
                else 1
            ],
            dtype=np.uint8,
        ),
        "input_bucket_count": np.asarray([bucket_count], dtype=np.uint8),
        "output_bucket_count": np.asarray([config["output_bucket_count"]], dtype=np.uint8),
        "q0": np.asarray([NNUE_Q0], dtype=np.uint16),
        "q": np.asarray([NNUE_Q], dtype=np.uint16),
        "scale": np.asarray([SCALE], dtype=np.uint16),
        "architecture": np.asarray([config["architecture"]]),
        "network_format": np.asarray([config["format"]]),
        "dense1": np.asarray([config["dense1"]], dtype=np.uint16),
        "dense2": np.asarray([config["dense2"]], dtype=np.uint16),
        "raw_layout": np.asarray([raw_layout]),
        **tail,
    }
    if threat_weights is not None:
        payload["threat_weights"] = threat_weights.astype(np.float32)
        payload["threat_feature_count"] = np.asarray(
            [FULL_THREATS_V1_COUNT], dtype=np.uint32
        )
        payload["threat_scheme_id"] = np.asarray([1], dtype=np.uint16)
        payload["threat_packing_sha256"] = np.asarray(
            [FULL_THREATS_V1_PACKING_SHA256]
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)

    print(f"Input: {raw_path}")
    print(f"Run metadata: {run_meta_path}")
    print(f"Architecture: {config['architecture']}")
    print(f"Raw layout: {raw_layout}")
    print(f"Buckets: input={bucket_count}, output={config['output_bucket_count']}")
    print(f"FT hidden: {h}")
    if raw_layout == "virtual_factorised":
        print("Factoriser: merged virtual shared PSQ rows; retained zero-bucketed threat rows")
    elif raw_layout in {"factorised", "factorised_from_optimizer_state"}:
        print("Factoriser: merged l0f into every l0w bucket residual")
    else:
        print("WARNING: l0f unavailable; l0w treated as complete weights")
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
