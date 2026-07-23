#!/usr/bin/env python3
"""Build a SYKNNUE8 Bullet weight file from a full-precision v7 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from checkpoint_raw_to_npz import parse_network_config, read_optimizer_weights  # noqa: E402

THREAT_FEATURE_COUNT = 60_720
PACKING_SHA256 = "964591edbe856c9f90694dcbfabe42d58b011a469e3275a8aaa9e4249b21988a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create exact zero-threat T1024 warm-start weights from a v7 checkpoint"
    )
    parser.add_argument("--source", required=True, help="v7 checkpoint directory or weights.bin")
    parser.add_argument("--run-meta", default="", help="source v7 run_meta.json override")
    parser.add_argument("--output", required=True, help="output Bullet weights.bin")
    parser.add_argument("--report", default="", help="warm-start verification JSON")
    return parser.parse_args()


def resolve_source(source: Path, run_meta_arg: str) -> tuple[Path, Path]:
    if source.is_dir():
        weights = source / "optimiser_state" / "weights.bin"
        run_meta = source.parent.parent / "run_meta.json"
    else:
        weights = source
        if source.name == "weights.bin" and source.parent.name == "optimiser_state":
            run_meta = source.parents[3] / "run_meta.json"
        else:
            run_meta = Path()
    if run_meta_arg:
        run_meta = Path(run_meta_arg)
    if not weights.is_file():
        raise FileNotFoundError(f"full-precision optimiser weights not found: {weights}")
    if not run_meta.is_file():
        raise FileNotFoundError(f"source run metadata not found: {run_meta}")
    return weights.resolve(), run_meta.resolve()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_tensor(tensors: dict, name: str, count: int):
    import numpy as np

    if name not in tensors:
        raise ValueError(f"source optimiser weights are missing {name!r}")
    values = np.asarray(tensors[name], dtype="<f4").reshape(-1)
    if values.size != count:
        raise ValueError(f"{name}: found {values.size} floats, expected {count}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite source value")
    return values


def write_weights(path: Path, tensors: list[tuple[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for name, values in tensors:
            payload = values.astype("<f4", copy=False)
            handle.write(name.encode("ascii") + b"\n")
            handle.write(struct.pack("<Q", payload.size))
            handle.write(memoryview(payload).cast("B"))


def main() -> int:
    import numpy as np

    args = parse_args()
    source_weights, source_meta_path = resolve_source(Path(args.source), args.run_meta)
    source_meta = json.loads(source_meta_path.read_text())
    config = parse_network_config(source_meta)
    if config["format"] != "syk7" or config["architecture"] != "pairwise-mlp":
        raise ValueError("T1024 warm start requires a pairwise-mlp SYKNNUE7 checkpoint")
    if config["ft_hidden"] != 1024 or config["output_bucket_count"] != 8:
        raise ValueError("exact T1024 warm start requires v7 H=1024 and O=8")
    if config["dense1"] != 16 or config["dense2"] != 32:
        raise ValueError("exact T1024 warm start requires the registered 16 -> 32 dense tail")

    h = config["ft_hidden"]
    bucket_count = max(config["bucket_layout_64"]) + 1
    if bucket_count != 10:
        raise ValueError(f"exact T1024 warm start requires 10 PSQ buckets, got {bucket_count}")

    source = read_optimizer_weights(source_weights)
    factoriser = require_tensor(source, "l0f", 768 * h)
    residual = require_tensor(source, "l0w", 768 * bucket_count * h)
    bias = require_tensor(source, "l0b", h)
    combined = np.zeros((768 + 768 * bucket_count + THREAT_FEATURE_COUNT) * h, dtype="<f4")
    combined[: 768 * h] = factoriser
    combined[768 * h : (768 + 768 * bucket_count) * h] = residual

    dense_counts = {
        "l1w": h * 8 * 16,
        "l1b": 8 * 16,
        "l2w": 32 * 8 * 32,
        "l2b": 8 * 32,
        "l3w": 32 * 8,
        "l3b": 8,
    }
    dense = [(name, require_tensor(source, name, count)) for name, count in dense_counts.items()]

    # This is the exact functional equivalence proof for the sparse input
    # parameterisation: every PSQ feature activates its shared row and its
    # bucket residual, while every new threat row is zero.
    old_effective = residual.reshape(768 * bucket_count, h) + np.tile(
        factoriser.reshape(768, h), (bucket_count, 1)
    )
    new_factoriser = combined[: 768 * h].reshape(768, h)
    new_residual = combined[768 * h : (768 + 768 * bucket_count) * h].reshape(
        768 * bucket_count, h
    )
    new_effective = new_residual + np.tile(new_factoriser, (bucket_count, 1))
    if not np.array_equal(old_effective, new_effective):
        raise AssertionError("warm-start PSQ equivalence failed")
    if np.count_nonzero(combined[-THREAT_FEATURE_COUNT * h :]) != 0:
        raise AssertionError("warm-start threat rows are not zero")

    output = Path(args.output).resolve()
    write_weights(output, [("l0w", combined), ("l0b", bias), *dense])
    report_path = Path(args.report).resolve() if args.report else output.with_suffix(".json")
    report = {
        "source_weights": str(source_weights),
        "source_weights_sha256": sha256_file(source_weights),
        "source_run_meta": str(source_meta_path),
        "source_run_id": source_meta.get("run_id"),
        "source_format": config["format"],
        "target_format": "syk8",
        "target_profile": "T1024",
        "packing_sha256": PACKING_SHA256,
        "psq_float_bit_exact": True,
        "dense_float_bit_exact": True,
        "zero_threat_weights": True,
        "output_weights": str(output),
        "output_weights_sha256": sha256_file(output),
        "tensor_float_counts": {
            "l0w": int(combined.size),
            "l0b": int(bias.size),
            **dense_counts,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Source: {source_weights}")
    print(f"Output: {output}")
    print(f"Verification: {report_path}")
    print("Zero-threat equivalence: exact in float parameter space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
