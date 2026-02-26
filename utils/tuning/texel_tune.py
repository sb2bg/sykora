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
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

DEFAULT_WORKERS = max(1, multiprocessing.cpu_count() - 1)


class EvalError(RuntimeError):
    """Raised when engine evaluation fails or returns malformed output."""


RESULT_VALUES = {0.0, 0.5, 1.0}
PIECE_VALUE_BOUNDS = {
    'pawn_value': (50, 200),
    'knight_value': (180, 650),
    'bishop_value': (180, 650),
    'rook_value': (300, 1100),
    'queen_value': (500, 1800),
}


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

def _evaluate_chunk(args: tuple) -> tuple[list[int] | None, str | None]:
    """Evaluate a chunk of FENs in a single sykora-tune subprocess."""
    fens_chunk, params_path, engine, timeout_s = args
    if not fens_chunk:
        return [], None

    try:
        result = subprocess.run(
            [engine, '--params', params_path],
            input='\n'.join(fens_chunk),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            stderr_tail = (result.stderr or '').strip().splitlines()
            tail = stderr_tail[-1] if stderr_tail else 'no stderr'
            return None, f"engine exited with code {result.returncode}: {tail}"

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) != len(fens_chunk):
            return None, (
                f"output length mismatch: expected {len(fens_chunk)} scores, "
                f"got {len(lines)}"
            )

        scores = []
        for i, line in enumerate(lines, start=1):
            try:
                scores.append(int(line))
            except ValueError:
                return None, f"non-integer score at output line {i}: {line!r}"
        return scores, None
    except subprocess.TimeoutExpired:
        return None, f"engine timeout after {timeout_s}s"


def evaluate_batch(
    fens: list[str],
    params: dict,
    engine: str,
    workers: int,
    timeout_s: int = 120,
) -> list[int]:
    """
    Run sykora-tune with the given params and return a score for every FEN.
    Splits work across worker processes for large batches.
    Raises EvalError on any failure or malformed engine output.
    """
    params_path = params_to_tempfile(params)
    try:
        n = len(fens)
        if n == 0:
            return []

        max_workers = max(1, workers)
        w = min(max_workers, max(1, n // 20_000))  # only parallelize for large batches
        if w <= 1:
            # Single process path
            chunk_scores, err = _evaluate_chunk((fens, params_path, engine, timeout_s))
            if err or chunk_scores is None:
                raise EvalError(err or 'engine failed in single-process mode')
            return chunk_scores

        chunk_size = (n + w - 1) // w
        chunks = [fens[i:i + chunk_size] for i in range(0, n, chunk_size)]
        args = [(chunk, params_path, engine, timeout_s) for chunk in chunks]

        with multiprocessing.Pool(w) as pool:
            results = pool.map(_evaluate_chunk, args)

        scores = []
        for worker_i, (chunk_scores, err) in enumerate(results):
            if err or chunk_scores is None:
                raise EvalError(f"worker {worker_i} failed: {err or 'unknown error'}")
            scores.extend(chunk_scores)

        if len(scores) != n:
            raise EvalError(
                f"batch output mismatch after merge: expected {n}, got {len(scores)}"
            )
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
    if arr.shape[0] != results.shape[0]:
        raise ValueError(
            f"score/result length mismatch: {arr.shape[0]} vs {results.shape[0]}"
        )
    preds = sigmoid(arr, K)
    return float(np.mean((preds - results) ** 2))


def tune_K(fens: list[str], results: np.ndarray, params: dict,
           engine: str, workers: int, K_range=(0.5, 2.0), steps=20) -> float:
    """Find the optimal sigmoid scaling factor K via grid search."""
    print("Auto-tuning K...", end='', flush=True)
    scores = evaluate_batch(fens, params, engine, workers=workers)

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
    skipped_bad_rows = 0
    skipped_bad_results = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3:
                skipped_bad_rows += 1
                continue
            fen, result_str = parts[0], parts[2]
            try:
                result = float(result_str)
            except ValueError:
                skipped_bad_results += 1
                continue
            if result not in RESULT_VALUES:
                skipped_bad_results += 1
                continue
            fens.append(fen)
            results.append(result)
            if len(fens) >= max_positions:
                break
    print(f"Loaded {len(fens):,} positions from {path}")
    if skipped_bad_rows:
        print(f"  skipped {skipped_bad_rows:,} malformed rows")
    if skipped_bad_results:
        print(f"  skipped {skipped_bad_results:,} rows with invalid result labels")
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


def scalar_bounds_for_key(key: str, base_value: int, range_mult: float) -> tuple[int, int]:
    """Return conservative scalar bounds for one parameter."""
    if key in PIECE_VALUE_BOUNDS:
        return PIECE_VALUE_BOUNDS[key]

    if key == 'endgame_phase_threshold':
        return 0, 256

    if key.endswith('_penalty') or key.endswith('_bonus'):
        hi = max(20, int(max(abs(base_value) * range_mult, abs(base_value) + 30)))
        return 0, hi

    if key.endswith('_threshold'):
        span = max(16, int(abs(base_value) * range_mult))
        return max(0, base_value - span), min(512, base_value + span)

    span = max(20, int(abs(base_value) * range_mult))
    return base_value - span, base_value + span


def build_scalar_bounds(params: dict, range_mult: float) -> dict[tuple[str, int | None], tuple[int, int]]:
    bounds: dict[tuple[str, int | None], tuple[int, int]] = {}
    for key, val in params.items():
        if isinstance(val, list):
            continue
        lo, hi = scalar_bounds_for_key(key, int(val), range_mult)
        bounds[(key, None)] = (min(lo, hi), max(lo, hi))
    return bounds


def clamp_scalar(
    key: str,
    idx: int | None,
    value: int,
    scalar_bounds: dict[tuple[str, int | None], tuple[int, int]] | None,
) -> int:
    if not scalar_bounds:
        return value

    lo_hi = scalar_bounds.get((key, idx))
    if lo_hi is None:
        lo_hi = scalar_bounds.get((key, None))
    if lo_hi is None:
        return value

    lo, hi = lo_hi
    return min(hi, max(lo, value))


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
    batch_size: int = 10_000,
    workers: int = DEFAULT_WORKERS,
    scalar_bounds: dict[tuple[str, int | None], tuple[int, int]] | None = None,
    min_improvement: float = 1e-6,
    seed: int | None = None,
    full_eval_interval: int = 1,
) -> dict:
    """Run coordinate descent Texel tuning."""

    all_scalars = flatten_params(params)
    delta = initial_delta
    n = len(fens)
    rng = np.random.default_rng(seed)

    # Evaluate baseline on full dataset
    print("Evaluating baseline...", end='', flush=True)
    scores = evaluate_batch(fens, params, engine, workers=workers)
    current_mse = compute_mse(scores, results, K)
    print(f" MSE={current_mse:.6f}")

    for epoch in range(1, max_epochs + 1):
        if delta < 1:
            print(f"Delta < 1, stopping after epoch {epoch - 1}.")
            break

        # Sample a fixed subset for this epoch's per-param evaluations
        if batch_size < n:
            idx_sample = rng.choice(n, size=batch_size, replace=False)
            epoch_fens = [fens[i] for i in idx_sample]
            epoch_results = results[idx_sample]
        else:
            epoch_fens = fens
            epoch_results = results

        # Baseline MSE on the sample
        scores = evaluate_batch(epoch_fens, params, engine, workers=workers)
        current_sample_mse = compute_mse(scores, epoch_results, K)

        epoch_start = time.time()
        improved = 0
        total = len(all_scalars)

        for param_i, (key, idx) in enumerate(all_scalars):
            orig = get_scalar(params, key, idx)

            # Try +delta
            plus_candidate = clamp_scalar(key, idx, orig + delta, scalar_bounds)
            mse_plus = float('inf')
            if plus_candidate != orig:
                set_scalar(params, key, idx, plus_candidate)
                scores = evaluate_batch(epoch_fens, params, engine, workers=workers)
                mse_plus = compute_mse(scores, epoch_results, K)

            if mse_plus + min_improvement < current_sample_mse:
                current_sample_mse = mse_plus
                improved += 1
                continue  # keep +delta

            # Reset before testing -delta
            set_scalar(params, key, idx, orig)

            # Try -delta
            minus_candidate = clamp_scalar(key, idx, orig - delta, scalar_bounds)
            mse_minus = float('inf')
            if minus_candidate != orig:
                set_scalar(params, key, idx, minus_candidate)
                scores = evaluate_batch(epoch_fens, params, engine, workers=workers)
                mse_minus = compute_mse(scores, epoch_results, K)

            if mse_minus + min_improvement < current_sample_mse:
                current_sample_mse = mse_minus
                improved += 1
                continue  # keep -delta

            # Neither helped — restore
            set_scalar(params, key, idx, orig)

            if (param_i + 1) % 50 == 0:
                pct = 100 * (param_i + 1) / total
                print(f"  [{pct:5.1f}%] MSE={current_sample_mse:.6f} delta={delta} "
                      f"improved={improved}", end='\r', flush=True)

        full_eval_done = (full_eval_interval <= 1) or (epoch % full_eval_interval == 0)
        if full_eval_done:
            full_scores = evaluate_batch(fens, params, engine, workers=workers)
            current_mse = compute_mse(full_scores, results, K)

        elapsed = time.time() - epoch_start
        full_mse_display = f"{current_mse:.6f}" if full_eval_done else "skipped"
        print(
            f"\nEpoch {epoch:3d} | delta={delta:3d} | improved={improved:4d} | "
            f"sample_MSE={current_sample_mse:.6f} | full_MSE={full_mse_display} | "
            f"{elapsed:.1f}s"
        )

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
                        help='Starting params file (default: dump engine defaults via --save-params)')
    parser.add_argument('--K', type=float, default=None,
                        help='Sigmoid scaling factor (auto-tuned if omitted)')
    parser.add_argument('--batch-size', type=int, default=10_000,
                        help='Positions per batch eval during tuning (default: 10000)')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Engine eval workers (default: {DEFAULT_WORKERS})')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for epoch sampling')
    parser.add_argument('--min-improvement', type=float, default=1e-6,
                        help='Minimum MSE improvement required to keep a parameter change')
    parser.add_argument('--full-eval-interval', type=int, default=1,
                        help='Compute full-dataset MSE every N epochs (default: 1)')
    parser.add_argument('--no-scalar-bounds', action='store_true',
                        help='Disable scalar bounds/constraints')
    parser.add_argument('--scalar-bound-mult', type=float, default=3.0,
                        help='Default scalar bound span multiplier around starting values')
    args = parser.parse_args()

    # Verify engine exists
    if not os.path.isfile(args.engine):
        print(f"Error: engine not found at {args.engine}", file=sys.stderr)
        print("Build with: zig build -Doptimize=ReleaseFast", file=sys.stderr)
        sys.exit(1)
    if args.workers <= 0:
        print("Error: --workers must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.batch_size <= 0:
        print("Error: --batch-size must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.full_eval_interval <= 0:
        print("Error: --full-eval-interval must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.min_improvement < 0.0:
        print("Error: --min-improvement must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.scalar_bound_mult <= 0.0:
        print("Error: --scalar-bound-mult must be > 0", file=sys.stderr)
        sys.exit(1)

    # Load or generate starting params
    if args.params:
        params = load_params(args.params)
        print(f"Loaded starting params from {args.params}")
    else:
        fd, default_params_path = tempfile.mkstemp(
            suffix='_defaults.txt', prefix='sykora_defaults_'
        )
        os.close(fd)
        proc = subprocess.run(
            [args.engine, '--save-params', default_params_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            os.unlink(default_params_path)
            stderr_tail = (proc.stderr or '').strip()
            print("Error: failed to dump default params from engine.", file=sys.stderr)
            if stderr_tail:
                print(stderr_tail, file=sys.stderr)
            print("Try: python utils/tuning/apply_params.py --dump-defaults tune_defaults.txt", file=sys.stderr)
            sys.exit(1)

        params = load_params(default_params_path)
        os.unlink(default_params_path)
        print("No --params given; using engine's compiled-in defaults.")

    if not params:
        print("Error: no params to tune.", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    fens, results = load_dataset(args.dataset, args.positions)
    if not fens:
        print("Error: dataset is empty.", file=sys.stderr)
        sys.exit(1)

    scalar_bounds = None if args.no_scalar_bounds else build_scalar_bounds(
        params, args.scalar_bound_mult
    )
    if scalar_bounds is not None:
        print(f"Scalar bounds enabled for {len(scalar_bounds)} scalar params")

    # Tune K
    try:
        K = args.K if args.K else tune_K(
            fens, results, params, args.engine, workers=args.workers
        )
    except (EvalError, ValueError) as exc:
        print(f"Error during K tuning: {exc}", file=sys.stderr)
        sys.exit(1)

    # Run coordinate descent
    try:
        tuned = coordinate_descent(
            fens=fens,
            results=results,
            params=params,
            engine=args.engine,
            output=args.output,
            K=K,
            max_epochs=args.epochs,
            initial_delta=args.delta,
            batch_size=args.batch_size,
            workers=args.workers,
            scalar_bounds=scalar_bounds,
            min_improvement=args.min_improvement,
            seed=args.seed,
            full_eval_interval=args.full_eval_interval,
        )
    except (EvalError, ValueError) as exc:
        print(f"Error during coordinate descent: {exc}", file=sys.stderr)
        sys.exit(1)

    save_params(tuned, args.output)
    print(f"\nTuning complete. Final params saved to {args.output}")
    print("Apply to engine with:")
    print(f"  python utils/tuning/apply_params.py {args.output}")


if __name__ == '__main__':
    main()
