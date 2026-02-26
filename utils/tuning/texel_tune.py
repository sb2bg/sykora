#!/usr/bin/env python3
"""
Texel tuner for Sykora's HCE evaluation.

Algorithm: coordinate descent over all EvalParams scalar/array fields.
Each iteration tries +delta and -delta for every individual parameter;
if MSE decreases the change is kept.  When a full pass produces no
improvement the step size is halved.  Tuning stops when delta < 1 or
max_epochs is reached.

Dataset format (one position per line, pipe-separated):
  <FEN> | <cp_score> | <result>
where result is 1.0 (white wins), 0.5 (draw), 0.0 (black wins).

Usage:
  python utils/tuning/texel_tune.py \\
      --dataset datasets/texel_positions.txt \\
      --engine  zig-out/bin/sykora-tune \\
      --output  tune_params.txt \\
      --positions 200000 \\
      [--epochs 50] \\
      [--delta 5]
"""

import argparse
import math
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)


# ---------------------------------------------------------------------------
# Params I/O
# ---------------------------------------------------------------------------

def load_params(path: str) -> dict:
    """Read a params file into {name: list_of_ints | int}."""
    params = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            vals = [int(x) for x in parts[1:]]
            params[key] = vals[0] if len(vals) == 1 else vals
    return params


def save_params(params: dict, path: str) -> None:
    """Write params dict back to the flat key-value format."""
    with open(path, 'w') as f:
        for key, val in params.items():
            if isinstance(val, list):
                f.write(f"{key} {' '.join(str(v) for v in val)}\n")
            else:
                f.write(f"{key} {val}\n")


def params_to_tempfile(params: dict) -> str:
    """Write params to a temporary file and return its path."""
    fd, path = tempfile.mkstemp(suffix='.txt', prefix='sykora_params_')
    os.close(fd)
    save_params(params, path)
    return path


# ---------------------------------------------------------------------------
# Engine evaluation
# ---------------------------------------------------------------------------

def _evaluate_chunk(args: tuple) -> list[int]:
    """Evaluate a chunk of FENs in a single sykora-tune subprocess."""
    fens_chunk, params_path, engine = args
    try:
        result = subprocess.run(
            [engine, '--params', params_path],
            input='\n'.join(fens_chunk),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return []
        scores = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line:
                try:
                    scores.append(int(line))
                except ValueError:
                    scores.append(0)
        return scores
    except subprocess.TimeoutExpired:
        return []


def evaluate_batch(fens: list[str], params: dict, engine: str) -> list[int]:
    """
    Run sykora-tune with the given params and return a score for every FEN.
    Splits work across NUM_WORKERS parallel processes.
    Returns [] on engine failure.
    """
    params_path = params_to_tempfile(params)
    try:
        n = len(fens)
        w = min(NUM_WORKERS, max(1, n // 1000))  # don't spawn workers for tiny batches
        if w <= 1:
            # Single process path
            chunk_scores = _evaluate_chunk((fens, params_path, engine))
            return chunk_scores

        chunk_size = (n + w - 1) // w
        chunks = [fens[i:i + chunk_size] for i in range(0, n, chunk_size)]
        args = [(chunk, params_path, engine) for chunk in chunks]

        with multiprocessing.Pool(w) as pool:
            results = pool.map(_evaluate_chunk, args)

        scores = []
        for chunk_scores in results:
            if not chunk_scores:
                print('[warn] a worker failed, falling back to single-process',
                      file=sys.stderr)
                return _evaluate_chunk((fens, params_path, engine))
            scores.extend(chunk_scores)
        return scores
    finally:
        os.unlink(params_path)


# ---------------------------------------------------------------------------
# Sigmoid and MSE
# ---------------------------------------------------------------------------

def sigmoid(cp: np.ndarray, K: float) -> np.ndarray:
    """Map centipawn scores to win probabilities via sigmoid."""
    return 1.0 / (1.0 + np.power(10.0, -K * cp / 400.0))


def compute_mse(scores: list[int], results: np.ndarray, K: float) -> float:
    arr = np.array(scores, dtype=np.float64)
    preds = sigmoid(arr, K)
    return float(np.mean((preds - results) ** 2))


def tune_K(fens: list[str], results: np.ndarray, params: dict,
           engine: str, K_range=(0.5, 2.0), steps=20) -> float:
    """Find the optimal sigmoid scaling factor K via golden-section search."""
    print("Auto-tuning K...", end='', flush=True)
    scores = evaluate_batch(fens, params, engine)
    if not scores or len(scores) != len(fens):
        print(" failed, using K=1.0")
        return 1.0

    best_K, best_mse = 1.0, float('inf')
    for K in np.linspace(K_range[0], K_range[1], steps):
        mse = compute_mse(scores, results, float(K))
        if mse < best_mse:
            best_mse, best_K = mse, float(K)
    print(f" K={best_K:.4f}  (MSE={best_mse:.6f})")
    return best_K


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str, max_positions: int) -> tuple[list[str], np.ndarray]:
    fens, results = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3:
                continue
            fen, result_str = parts[0], parts[2]
            try:
                result = float(result_str)
            except ValueError:
                continue
            fens.append(fen)
            results.append(result)
            if len(fens) >= max_positions:
                break
    print(f"Loaded {len(fens):,} positions from {path}")
    return fens, np.array(results, dtype=np.float64)


# ---------------------------------------------------------------------------
# Coordinate descent
# ---------------------------------------------------------------------------

def flatten_params(params: dict) -> list[tuple[str, int | None]]:
    """
    Return a list of (key, array_index_or_None) for every tunable scalar.
    """
    items = []
    for key, val in params.items():
        if isinstance(val, list):
            for i in range(len(val)):
                items.append((key, i))
        else:
            items.append((key, None))
    return items


def set_scalar(params: dict, key: str, idx: int | None, value: int) -> None:
    if idx is None:
        params[key] = value
    else:
        params[key][idx] = value


def get_scalar(params: dict, key: str, idx: int | None) -> int:
    if idx is None:
        return params[key]
    return params[key][idx]


def coordinate_descent(
    fens: list[str],
    results: np.ndarray,
    params: dict,
    engine: str,
    output: str,
    K: float,
    max_epochs: int,
    initial_delta: int,
) -> dict:
    """Run coordinate descent Texel tuning."""

    all_scalars = flatten_params(params)
    delta = initial_delta

    # Evaluate baseline
    print("Evaluating baseline...", end='', flush=True)
    scores = evaluate_batch(fens, params, engine)
    if not scores:
        print(" engine failed!")
        return params
    current_mse = compute_mse(scores, results, K)
    print(f" MSE={current_mse:.6f}")

    for epoch in range(1, max_epochs + 1):
        if delta < 1:
            print(f"Delta < 1, stopping after epoch {epoch - 1}.")
            break

        epoch_start = time.time()
        improved = 0
        total = len(all_scalars)

        for param_i, (key, idx) in enumerate(all_scalars):
            orig = get_scalar(params, key, idx)

            # Try +delta
            set_scalar(params, key, idx, orig + delta)
            scores = evaluate_batch(fens, params, engine)
            if scores:
                mse_plus = compute_mse(scores, results, K)
            else:
                mse_plus = float('inf')

            if mse_plus < current_mse:
                current_mse = mse_plus
                improved += 1
                continue  # keep +delta

            # Try -delta
            set_scalar(params, key, idx, orig - delta)
            scores = evaluate_batch(fens, params, engine)
            if scores:
                mse_minus = compute_mse(scores, results, K)
            else:
                mse_minus = float('inf')

            if mse_minus < current_mse:
                current_mse = mse_minus
                improved += 1
                continue  # keep -delta

            # Neither helped — restore
            set_scalar(params, key, idx, orig)

            if (param_i + 1) % 50 == 0:
                pct = 100 * (param_i + 1) / total
                print(f"  [{pct:5.1f}%] MSE={current_mse:.6f} delta={delta} "
                      f"improved={improved}", end='\r', flush=True)

        elapsed = time.time() - epoch_start
        print(f"\nEpoch {epoch:3d} | delta={delta:3d} | improved={improved:4d} | "
              f"MSE={current_mse:.6f} | {elapsed:.1f}s")

        save_params(params, output)
        print(f"  Params saved to {output}")

        if improved == 0:
            delta = delta // 2
            print(f"  No improvement → delta halved to {delta}")

    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Texel tuner for Sykora HCE')
    parser.add_argument('--dataset', required=True,
                        help='Dataset file (FEN | cp | result per line)')
    parser.add_argument('--engine', default='zig-out/bin/sykora-tune',
                        help='Path to sykora-tune binary')
    parser.add_argument('--output', default='tune_params.txt',
                        help='Output file for tuned params')
    parser.add_argument('--positions', type=int, default=200_000,
                        help='Maximum positions to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum tuning epochs')
    parser.add_argument('--delta', type=int, default=5,
                        help='Initial step size (centipawns)')
    parser.add_argument('--params', default=None,
                        help='Starting params file (defaults: engine defaults)')
    parser.add_argument('--K', type=float, default=None,
                        help='Sigmoid scaling factor (auto-tuned if omitted)')
    args = parser.parse_args()

    # Verify engine exists
    if not os.path.isfile(args.engine):
        print(f"Error: engine not found at {args.engine}", file=sys.stderr)
        print("Build with: zig build -Doptimize=ReleaseFast", file=sys.stderr)
        sys.exit(1)

    # Load or generate starting params
    if args.params:
        params = load_params(args.params)
        print(f"Loaded starting params from {args.params}")
    else:
        # Generate defaults by running engine with no params on startpos
        # and reading its saved params
        default_params_path = tempfile.mktemp(suffix='_defaults.txt')
        # Use --save-params if we add that later; for now we use the engine's
        # defaults embedded in evaluation.zig, saved by running:
        #   echo "" | ./sykora-tune  (produces nothing, we need saveParams)
        # Workaround: parse from evaluation.zig source or just hardcode defaults.
        # For now we require --params or a pre-saved defaults file.
        print("No --params given; using engine's compiled-in defaults.")
        print("To generate a defaults file, run:")
        print("  python utils/tuning/apply_params.py --dump-defaults tune_defaults.txt")
        params = {}

    if not params:
        print("Error: no params to tune. Provide --params <file>.", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    fens, results = load_dataset(args.dataset, args.positions)
    if not fens:
        print("Error: dataset is empty.", file=sys.stderr)
        sys.exit(1)

    # Tune K
    K = args.K if args.K else tune_K(fens, results, params, args.engine)

    # Run coordinate descent
    tuned = coordinate_descent(
        fens=fens,
        results=results,
        params=params,
        engine=args.engine,
        output=args.output,
        K=K,
        max_epochs=args.epochs,
        initial_delta=args.delta,
    )

    save_params(tuned, args.output)
    print(f"\nTuning complete. Final params saved to {args.output}")
    print("Apply to engine with:")
    print(f"  python utils/tuning/apply_params.py {args.output}")


if __name__ == '__main__':
    main()
