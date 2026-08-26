#!/usr/bin/env python3
"""Widen a mature SYKNNUE8 Bullet checkpoint without changing its output."""

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

BASE_INPUTS = 768 + 10 * 768 + 60_720
OUTPUT_BUCKETS = 8
DENSE1 = 16
DENSE2 = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a function-preserving widened SYKNNUE8 Bullet weight file"
    )
    parser.add_argument("--source", required=True, help="source checkpoint directory or weights.bin")
    parser.add_argument("--run-meta", default="", help="source run_meta.json override")
    parser.add_argument("--target-hidden", type=int, default=1408)
    parser.add_argument("--seed", type=int, default=1408)
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


def widen_paired_axis(source, target_hidden: int, *, fill: float = 0.0):
    """Map both old pairwise halves into their corresponding widened halves."""
    import numpy as np

    source_hidden = source.shape[-1]
    if source_hidden % 2 or target_hidden % 2 or target_hidden <= source_hidden:
        raise ValueError("paired widening requires a larger even target width")
    old_half = source_hidden // 2
    new_half = target_hidden // 2
    target = np.full((*source.shape[:-1], target_hidden), fill, dtype="<f4")
    target[..., :old_half] = source[..., :old_half]
    target[..., new_half : new_half + old_half] = source[..., old_half:]
    return target


def main() -> int:
    import numpy as np

    args = parse_args()
    if args.target_hidden != 1408:
        raise ValueError("the registered wide profile is H=1408")

    source_weights, source_meta_path = resolve_source(Path(args.source), args.run_meta)
    source_meta = json.loads(source_meta_path.read_text())
    config = parse_network_config(source_meta)
    if config["format"] != "syk8" or config["architecture"] != "pairwise-mlp":
        raise ValueError("wide warm start requires a pairwise-mlp SYKNNUE8 checkpoint")
    if config["ft_hidden"] != 1024 or config["output_bucket_count"] != OUTPUT_BUCKETS:
        raise ValueError("wide warm start requires a mature H=1024, O=8 checkpoint")
    if config["dense1"] != DENSE1 or config["dense2"] != DENSE2:
        raise ValueError("wide warm start requires the registered 16 -> 32 dense tail")
    if max(config["bucket_layout_64"]) + 1 != 10:
        raise ValueError("wide warm start requires the v3_10 input bucket layout")

    source_hidden = config["ft_hidden"]
    target_hidden = args.target_hidden
    old_half = source_hidden // 2
    new_half = target_hidden // 2
    extra_per_half = new_half - old_half
    source = read_optimizer_weights(source_weights)

    source_l0 = require_tensor(source, "l0w", BASE_INPUTS * source_hidden).reshape(
        BASE_INPUTS, source_hidden
    )
    target_l0 = widen_paired_axis(source_l0, target_hidden)
    source_bias = require_tensor(source, "l0b", source_hidden).reshape(1, source_hidden)
    target_bias = widen_paired_axis(source_bias, target_hidden).reshape(-1)

    # New FT channel pairs need nonzero activations so gradients can reach them.
    # Their L1 projection starts at exactly zero, preserving the mature model's
    # output while the projection learns to admit useful new capacity.
    rng = np.random.default_rng(args.seed)
    segments = (
        (0, 768, 0.10),
        (768, 768 + 10 * 768, 0.067),
        (768 + 10 * 768, BASE_INPUTS, 0.054),
    )
    for begin, end, stdev in segments:
        target_l0[begin:end, old_half:new_half] = rng.normal(
            0.0, stdev, (end - begin, extra_per_half)
        ).astype("<f4")
        target_l0[begin:end, new_half + old_half :] = rng.normal(
            0.0, stdev, (end - begin, extra_per_half)
        ).astype("<f4")
    target_bias[old_half:new_half] = rng.normal(0.0, 0.19, extra_per_half).astype("<f4")
    target_bias[new_half + old_half :] = rng.normal(0.0, 0.19, extra_per_half).astype("<f4")

    source_l1 = require_tensor(
        source, "l1w", source_hidden * OUTPUT_BUCKETS * DENSE1
    ).reshape(source_hidden, OUTPUT_BUCKETS, DENSE1)
    target_l1 = np.zeros((target_hidden, OUTPUT_BUCKETS, DENSE1), dtype="<f4")
    target_l1[:old_half] = source_l1[:old_half]
    target_l1[new_half : new_half + old_half] = source_l1[old_half:]

    unchanged_counts = {
        "l1b": OUTPUT_BUCKETS * DENSE1,
        "l2w": 2 * DENSE1 * OUTPUT_BUCKETS * DENSE2,
        "l2b": OUTPUT_BUCKETS * DENSE2,
        "l3w": DENSE2 * OUTPUT_BUCKETS,
        "l3b": OUTPUT_BUCKETS,
    }
    unchanged = [
        (name, require_tensor(source, name, count)) for name, count in unchanged_counts.items()
    ]

    # Algebraic equivalence check for the pairwise pool and widened L1 map.
    probe = rng.normal(size=(32, source_hidden)).astype("<f4")
    wide_probe = widen_paired_axis(probe, target_hidden)
    old_pool = probe[:, :old_half] * probe[:, old_half:]
    new_pool = wide_probe[:, :new_half] * wide_probe[:, new_half:]
    if not np.array_equal(old_pool, new_pool[:, :old_half]):
        raise AssertionError("widened pairwise channel mapping is not exact")
    if np.count_nonzero(new_pool[:, old_half:]) != 0:
        raise AssertionError("algebraic probe unexpectedly activated new channels")
    old_l1 = source_l1.reshape(source_hidden, -1)
    new_l1 = target_l1.reshape(target_hidden, -1)
    dual_old = np.concatenate((old_pool, old_pool), axis=1)
    dual_new = np.concatenate((new_pool, new_pool), axis=1)
    old_projection = dual_old @ old_l1
    new_projection = dual_new @ new_l1
    projection_max_abs_error = float(np.max(np.abs(old_projection - new_projection)))
    if not np.allclose(old_projection, new_projection, rtol=0.0, atol=2e-5):
        raise AssertionError(
            "widened L1 mapping changed the mature network output beyond FP reduction noise: "
            f"max_abs_error={projection_max_abs_error}"
        )

    tensors = [
        ("l0w", target_l0.reshape(-1)),
        ("l0b", target_bias),
        ("l1w", target_l1.reshape(-1)),
        *unchanged,
    ]
    output = Path(args.output).resolve()
    write_weights(output, tensors)
    report_path = Path(args.report).resolve() if args.report else output.with_suffix(".json")
    parameter_count = 68_529 * target_hidden + 8_840
    raw_tensor_bytes = 76_210 * target_hidden + 10_016
    report = {
        "source_weights": str(source_weights),
        "source_weights_sha256": sha256_file(source_weights),
        "source_run_meta": str(source_meta_path),
        "source_run_id": source_meta.get("run_id"),
        "source_hidden": source_hidden,
        "target_hidden": target_hidden,
        "target_profile": "T1408",
        "seed": args.seed,
        "mature_channels_bit_exact": True,
        "initial_output_algebraically_equal": True,
        "float_projection_max_abs_error": projection_max_abs_error,
        "new_channels_per_half": extra_per_half,
        "new_l1_projection_zero": True,
        "parameter_count": parameter_count,
        "raw_tensor_bytes": raw_tensor_bytes,
        "output_weights": str(output),
        "output_weights_sha256": sha256_file(output),
        "tensor_float_counts": {name: int(values.size) for name, values in tensors},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Source: {source_weights}")
    print(f"Output: {output}")
    print(f"Verification: {report_path}")
    print(
        f"T1408 warm start: {parameter_count:,} parameters, mature mapping exact, "
        f"{2 * extra_per_half} new FT channels"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
