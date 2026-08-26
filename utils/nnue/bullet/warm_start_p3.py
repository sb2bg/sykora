#!/usr/bin/env python3
"""Warm-start the moment-factorised SYKNNUE9 P3 graph from a mature SYKNNUE8 checkpoint."""

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

BASE_INPUTS = 768 + 7_680 + 60_720
P3_RANK = 32
PAWN_COUNT = 2 * 64
CONTEXT_COUNT = 2 * 5 * 64
TRAINING_INPUTS = BASE_INPUTS + PAWN_COUNT + CONTEXT_COUNT
DEFAULT_SEED = 0x50335A09


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create moment-factorised rank-32 P3 Bullet weights from a full-precision v8 checkpoint"
    )
    parser.add_argument("--source", required=True, help="v8 checkpoint directory or weights.bin")
    parser.add_argument("--run-meta", default="", help="source v8 run_meta.json override")
    parser.add_argument("--output", required=True, help="output Bullet weights.bin")
    parser.add_argument("--report", default="", help="warm-start verification JSON")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
    if config["format"] != "syk8" or config["architecture"] != "pairwise-mlp":
        raise ValueError("P3 warm start requires a pairwise-mlp SYKNNUE8 checkpoint")
    if config["ft_hidden"] != 1024 or config["output_bucket_count"] != 8:
        raise ValueError("P3-ANOVA warm start requires v8 H=1024 and O=8")
    if config["dense1"] != 16 or config["dense2"] != 32:
        raise ValueError("P3-ANOVA requires the registered 16 -> 32 dense tail")

    h = config["ft_hidden"]
    source = read_optimizer_weights(source_weights)
    source_l0 = require_tensor(source, "l0w", BASE_INPUTS * h)
    target_l0 = np.zeros(TRAINING_INPUTS * h, dtype="<f4")
    target_l0[: source_l0.size] = source_l0
    l0_bias = require_tensor(source, "l0b", h)

    dense_counts = {
        "l1w": h * 8 * 16,
        "l1b": 8 * 16,
        "l2w": 32 * 8 * 32,
        "l2b": 8 * 32,
        "l3w": 32 * 8,
        "l3b": 8,
    }
    dense = [(name, require_tensor(source, name, count)) for name, count in dense_counts.items()]

    rng = np.random.default_rng(args.seed)
    p3_tensors = [
        (
            "p3_pawnw",
            rng.normal(0.0, 0.05, PAWN_COUNT * P3_RANK).astype("<f4"),
        ),
        (
            "p3_contextw",
            rng.normal(0.0, 0.05, CONTEXT_COUNT * P3_RANK).astype("<f4"),
        ),
        (
            "p3_l1w",
            rng.normal(0.0, 0.01, 4 * P3_RANK * 8 * 16).astype("<f4"),
        ),
        ("p3_l1b", np.zeros(8 * 16, dtype="<f4")),
    ]

    if not np.array_equal(target_l0[: source_l0.size], source_l0):
        raise AssertionError("v8 l0 warm-start copy is not bit exact")
    if np.count_nonzero(target_l0[source_l0.size :]) != 0:
        raise AssertionError("new l0 P3 input rows are not zero")

    output = Path(args.output).resolve()
    tensors = [("l0w", target_l0), ("l0b", l0_bias), *dense, *p3_tensors]
    write_weights(output, tensors)
    report_path = Path(args.report).resolve() if args.report else output.with_suffix(".json")
    report = {
        "source_weights": str(source_weights),
        "source_weights_sha256": sha256_file(source_weights),
        "source_run_meta": str(source_meta_path),
        "source_run_id": source_meta.get("run_id"),
        "source_format": config["format"],
        "target_format": "syk9",
        "target_architecture": "pairwise-mlp-p3",
        "target_profile": "P3-ANOVA-R32",
        "seed": args.seed,
        "base_float_bit_exact": True,
        "zero_new_l0_rows": True,
        "p3_initialisation": {
            "embedding_stdev": 0.05,
            "projection_stdev": 0.01,
            "biases_zero": True,
        },
        "p3": {
            "rank": P3_RANK,
            "pawn_count": PAWN_COUNT,
            "context_count": CONTEXT_COUNT,
            "training_inputs": TRAINING_INPUTS,
            "pair_representation": "file_local_elementary_symmetric_moments",
        },
        "output_weights": str(output),
        "output_weights_sha256": sha256_file(output),
        "tensor_float_counts": {name: int(values.size) for name, values in tensors},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Source: {source_weights}")
    print(f"Output: {output}")
    print(f"Verification: {report_path}")
    print("V8 base tensors: bit-exact; P3 adapter: deterministic tiny random initialisation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
