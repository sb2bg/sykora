#!/usr/bin/env python3
"""Bit-exactness gate for SYKNNUE6 nets.

Computes a numpy reference eval for every FEN in a suite, reproducing the
integer inference contract in src/nnue.zig exactly, then runs the engine's
`nnuecheck` subcommand and asserts every position matches.

Usage:
  python check_net_parity.py --net src/net.sknnue \
      --fens utils/nnue/parity.fens [--engine ./zig-out/bin/sykora]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import chess

UTILS_NNUE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(UTILS_NNUE_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_NNUE_DIR))

from common import (  # noqa: E402
    FEATURE_SET_KING_BUCKETS_MIRRORED,
    OUTPUT_BUCKET_SCHEME_MATERIAL,
    board_feature_indices,
    read_syk_nnue_v6,
)

ACTIVATION_SCRELU = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SYKNNUE6 engine/reference parity check.")
    p.add_argument("--net", required=True, help="SYKNNUE6 net path")
    p.add_argument("--fens", required=True, help="FEN suite (one per line)")
    p.add_argument(
        "--engine",
        default=str(REPO_ROOT / "zig-out" / "bin" / "sykora"),
        help="Path to the sykora engine binary",
    )
    return p.parse_args()


def div_round_nearest_signed(x: int, d: int) -> int:
    """Round-to-nearest, ties away from zero (matches divRoundNearestSigned)."""
    half = d // 2
    if x >= 0:
        return (x + half) // d
    return -((-x + half) // d)


def output_bucket(net: dict, board: chess.Board) -> int:
    if net["output_bucket_scheme"] != OUTPUT_BUCKET_SCHEME_MATERIAL:
        return 0
    o = net["output_bucket_count"]
    n = chess.popcount(board.occupied)  # includes kings, 2..32
    divisor = 32 // o
    non_king = n - 2 if n >= 2 else 0
    return min(non_king // divisor, o - 1)


def reference_eval(net: dict, ft_w, fen: str) -> int:
    import numpy as np

    board = chess.Board(fen)
    h = net["ft_hidden_size"]
    q0 = net["q0"]
    q = net["q"]
    scale = net["scale"]
    use_screlu = net["activation_type"] == ACTIVATION_SCRELU

    white_feats, black_feats, stm_is_white = board_feature_indices(
        board,
        feature_set=FEATURE_SET_KING_BUCKETS_MIRRORED,
        bucket_layout_64=net["bucket_layout_64"],
    )

    ft_b = np.asarray(net["ft_biases"], dtype=np.int64)
    acc_white = ft_b.copy()
    if white_feats:
        acc_white = acc_white + ft_w[np.asarray(white_feats, dtype=np.int64)].sum(axis=0)
    acc_black = ft_b.copy()
    if black_feats:
        acc_black = acc_black + ft_w[np.asarray(black_feats, dtype=np.int64)].sum(axis=0)

    us = acc_white if stm_is_white else acc_black
    them = acc_black if stm_is_white else acc_white

    b = output_bucket(net, board)
    row = np.asarray(net["out_weights"], dtype=np.int64)[b * 2 * h : b * 2 * h + 2 * h]
    w_us = row[:h]
    w_them = row[h : 2 * h]

    def activated_dot(acc, w):
        v = np.clip(acc, 0, q0)
        if use_screlu:
            v = v * v
        return int((v * w).sum())

    total = activated_dot(us, w_us) + activated_dot(them, w_them)
    if use_screlu:
        total = div_round_nearest_signed(total, q0)
    total += net["out_biases"][b]
    return div_round_nearest_signed(total * scale, q0 * q)


def run_engine(engine: str, net: str, fens: str) -> dict:
    proc = subprocess.run(
        [engine, "nnuecheck", "--net", net, "--fens", fens],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"engine nnuecheck failed (exit {proc.returncode})")
    out = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        eval_str, _, fen = line.partition("\t")
        out[fen.strip()] = int(eval_str)
    return out


def report_coverage(net: dict, fens: list[str]) -> None:
    buckets = set()
    mirror_states = set()
    stms = set()
    for fen in fens:
        board = chess.Board(fen)
        buckets.add(output_bucket(net, board))
        wk = board.king(chess.WHITE)
        mirror_states.add((wk % 8) > 3)
        stms.add(board.turn)
    print(f"Coverage: buckets={sorted(buckets)} mirror={sorted(mirror_states)} stm={sorted(stms)}")
    if net["output_bucket_scheme"] == OUTPUT_BUCKET_SCHEME_MATERIAL:
        missing = set(range(net["output_bucket_count"])) - buckets
        if missing:
            print(f"WARNING: output buckets not covered: {sorted(missing)}", file=sys.stderr)
    if len(mirror_states) < 2:
        print("WARNING: both mirror states not covered", file=sys.stderr)


def main() -> int:
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("numpy is required") from exc

    args = parse_args()
    net = read_syk_nnue_v6(Path(args.net))
    h = net["ft_hidden_size"]
    input_size = 768 * net["input_bucket_count"]
    ft_w = np.asarray(net["ft_weights"], dtype=np.int64).reshape(input_size, h)

    fens = [
        ln.strip()
        for ln in Path(args.fens).read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not fens:
        raise SystemExit("no FENs found")

    report_coverage(net, fens)

    engine_evals = run_engine(args.engine, args.net, args.fens)

    mismatches = 0
    for fen in fens:
        ref = reference_eval(net, ft_w, fen)
        eng = engine_evals.get(fen)
        if eng is None:
            print(f"MISSING engine eval for: {fen}", file=sys.stderr)
            mismatches += 1
            continue
        if ref != eng:
            mismatches += 1
            print(f"MISMATCH ref={ref} eng={eng}  {fen}", file=sys.stderr)

    if mismatches:
        print(f"FAIL: {mismatches}/{len(fens)} positions mismatch", file=sys.stderr)
        return 1
    print(f"OK: {len(fens)}/{len(fens)} positions match exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
