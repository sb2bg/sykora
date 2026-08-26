#!/usr/bin/env python3
"""Bit-exactness gate for deployed Sykora NNUE nets.

Computes a numpy reference eval for every FEN in a suite, reproducing the
integer inference contract in src/nnue.zig exactly, then runs the engine's
`nnuecheck` subcommand and asserts every position matches.

Usage:
  python check_net_parity.py --net src/net.sknnue \
      --fens utils/nnue/parity.fens [--engine ./zig-out/bin/sykora]
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import chess

UTILS_NNUE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(UTILS_NNUE_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_NNUE_DIR))

from common import (  # noqa: E402
    FEATURE_SET_MIRRORED_PSQ_FULL_THREATS_V1,
    MAGIC_V8,
    MAGIC_V9,
    OUTPUT_BUCKET_SCHEME_MATERIAL,
    SECTION_FT_BIAS,
    SECTION_FT_WEIGHT,
    SECTION_L1_BIAS,
    SECTION_L1_WEIGHT,
    SECTION_L2_BIAS,
    SECTION_L2_WEIGHT,
    SECTION_OUT_BIAS,
    SECTION_OUT_WEIGHT,
    SECTION_P3_CONTEXT_WEIGHT,
    SECTION_P3_L1_WEIGHT,
    SECTION_P3_PAWN_WEIGHT,
    SECTION_THREAT_WEIGHT,
    board_feature_indices,
    read_syk_nnue_v8,
)
from full_threats_v1 import enumerate_board  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sykora NNUE engine/reference parity check.")
    p.add_argument("--net", required=True, help="SYKNNUE8 net path")
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


def reference_eval(net: dict, tensors: dict, fen: str) -> int:
    import numpy as np

    board = chess.Board(fen)
    h = net["ft_hidden_size"]
    half = h // 2
    q0 = net["q0"]
    pool_quant = net["pool_quant"]
    q = net["q"]
    white_feats, black_feats, stm_is_white = board_feature_indices(
        board,
        feature_set=FEATURE_SET_MIRRORED_PSQ_FULL_THREATS_V1,
        bucket_layout_64=net["bucket_layout_64"],
    )
    ft_w = tensors[SECTION_FT_WEIGHT]
    ft_b = tensors[SECTION_FT_BIAS]
    acc_white = ft_b.copy()
    if white_feats:
        acc_white += ft_w[np.asarray(white_feats, dtype=np.int64)].sum(axis=0)
    acc_black = ft_b.copy()
    if black_feats:
        acc_black += ft_w[np.asarray(black_feats, dtype=np.int64)].sum(axis=0)
    if SECTION_THREAT_WEIGHT in tensors:
        threat_weights = tensors[SECTION_THREAT_WEIGHT]
        white_threats = enumerate_board(board, chess.WHITE)
        black_threats = enumerate_board(board, chess.BLACK)
        if white_threats:
            acc_white += threat_weights[np.asarray(white_threats, dtype=np.int64)].sum(axis=0)
        if black_threats:
            acc_black += threat_weights[np.asarray(black_threats, dtype=np.int64)].sum(axis=0)
    us = acc_white if stm_is_white else acc_black
    them = acc_black if stm_is_white else acc_white
    us = np.clip(us, 0, q0)
    them = np.clip(them, 0, q0)
    pooled = np.concatenate(
        (
            (us[:half] * us[half:]) // 512,
            (them[:half] * them[half:]) // 512,
        )
    )

    bucket = output_bucket(net, board)
    l1 = pooled @ tensors[SECTION_L1_WEIGHT][bucket] + tensors[SECTION_L1_BIAS][bucket]
    if SECTION_P3_PAWN_WEIGHT in tensors:
        white_p3 = p3_features(net, tensors, board, True)
        black_p3 = p3_features(net, tensors, board, False)
        p3 = np.concatenate(
            (white_p3, black_p3) if stm_is_white else (black_p3, white_p3)
        )
        p3 = np.asarray(
            [
                max(-128, min(127, div_round_nearest_signed(int(value), 1 << net["p3_activation_shift"])))
                for value in p3
            ],
            dtype=np.int64,
        )
        l1 += p3 @ tensors[SECTION_P3_L1_WEIGHT][bucket]
    l1 = np.asarray(
        [div_round_nearest_signed(int(value), pool_quant) for value in l1],
        dtype=np.int64,
    )
    dual = np.concatenate(
        (
            np.clip(l1, 0, q),
            np.minimum((l1 * l1 + q // 2) // q, q),
        )
    )
    l2 = dual @ tensors[SECTION_L2_WEIGHT][bucket] + tensors[SECTION_L2_BIAS][bucket]
    l2 = np.asarray([div_round_nearest_signed(int(value), q) for value in l2], dtype=np.int64)
    l2 = np.clip(l2, 0, q)
    l2 = (l2 * l2 + q // 2) // q
    raw = int(l2 @ tensors[SECTION_OUT_WEIGHT][bucket]) + int(
        tensors[SECTION_OUT_BIAS][bucket]
    )
    return div_round_nearest_signed(raw * net["scale"], q * q)


def p3_identity(board: chess.Board, perspective_is_white: bool, square: int, item) -> tuple[int, int]:
    sq = square if perspective_is_white else square ^ 56
    king_sq = board.king(chess.WHITE if perspective_is_white else chess.BLACK)
    if king_sq is None:
        raise ValueError("board must contain both kings")
    king_sq = king_sq if perspective_is_white else king_sq ^ 56
    relative_white = item.color if perspective_is_white else not item.color
    side = 0 if relative_white == chess.WHITE else 1
    if king_sq % 8 > 3:
        sq ^= 7
    kind = item.piece_type - 1
    identity = side * 64 + sq if kind == 0 else (side * 5 + kind - 1) * 64 + sq
    return identity, sq % 8


def p3_features(net: dict, tensors: dict, board: chess.Board, perspective_is_white: bool):
    import numpy as np

    rank = net["p3_rank"]
    pawn = tensors[SECTION_P3_PAWN_WEIGHT]
    context_weights = tensors[SECTION_P3_CONTEXT_WEIGHT]
    sums = np.zeros((8, rank), dtype=np.int64)
    squares = np.zeros((8, rank), dtype=np.int64)
    context = np.zeros(rank, dtype=np.int64)
    for square, item in board.piece_map().items():
        identity, file = p3_identity(board, perspective_is_white, square, item)
        if item.piece_type == chess.PAWN:
            row = pawn[identity]
            sums[file] += row
            squares[file] += row * row
        else:
            context += context_weights[identity]
    same = ((sums * sums - squares).sum(axis=0)) // 2
    adjacent = (sums[:-1] * sums[1:]).sum(axis=0)
    return np.concatenate((same * context, adjacent * context))


def read_syk_nnue_v9_for_parity(path: Path) -> dict:
    """Read v9 tensors; the engine performs the authoritative strict validation."""
    data = Path(path).read_bytes()
    if len(data) < 224 or data[:8] != MAGIC_V9:
        raise ValueError("not a SYKNNUE9 net")
    pos = 8

    def take(fmt: str):
        nonlocal pos
        values = struct.unpack_from(fmt, data, pos)
        pos += struct.calcsize(fmt)
        return values[0] if len(values) == 1 else values

    version = take("<H")
    header_bytes = take("<H")
    section_count = take("<H")
    section_entry_bytes = take("<H")
    flags = take("<I")
    architecture = take("<H")
    feature_set = take("<H")
    input_bucket_count = take("<H")
    output_bucket_count = take("<H")
    h = take("<H")
    d1 = take("<H")
    d2 = take("<H")
    activation_ids = tuple(data[pos : pos + 5])
    pos += 8
    q0 = take("<H")
    pool_quant = take("<H")
    q = take("<H")
    scale = take("<H")
    psq_feature_count = take("<I")
    threat_feature_count = take("<I")
    threat_scheme_id = take("<H")
    pos += 2
    threat_quant = take("<H")
    pos += 4
    bucket_layout = list(data[pos : pos + 64])
    pos += 64
    packing_hash = data[pos : pos + 32]
    pos += 32
    psq_abs_bound = take("<I")
    threat_abs_bound = take("<I")
    expected_hash = data[pos : pos + 32]
    pos += 32
    p3_rank = take("<H")
    p3_pawn_count = take("<H")
    p3_context_count = take("<H")
    p3_quant = take("<H")
    p3_activation_shift = take("<B")
    pos += 3
    p3_l1_abs_bound = take("<I")
    pos += 4
    if (
        (version, header_bytes, section_count, section_entry_bytes, flags, architecture)
        != (9, 224, 12, 48, 0, 3)
        or pos != 224
    ):
        raise ValueError("malformed SYKNNUE9 header")
    hasher = hashlib.sha256()
    hasher.update(data[:172])
    hasher.update(b"\0" * 32)
    hasher.update(data[204:])
    if hasher.digest() != expected_hash:
        raise ValueError("SYKNNUE9 content hash mismatch")

    type_sizes = {1: 1, 3: 2, 4: 4}
    sections = {}
    previous_end = ((224 + section_count * 48 + 63) // 64) * 64
    for index in range(section_count):
        fields = struct.unpack_from("<HBBI4IQQII", data, 224 + index * 48)
        section_id, element_type, rank, section_flags = fields[:4]
        dimensions = tuple(fields[4:8])
        offset, byte_length, crc32, reserved = fields[8:]
        expected_length = type_sizes[element_type]
        for dimension in dimensions[:rank]:
            expected_length *= dimension
        payload = data[offset : offset + byte_length]
        if (
            section_flags != 1
            or reserved != 0
            or offset < previous_end
            or expected_length != byte_length
            or zlib.crc32(payload) & 0xFFFFFFFF != crc32
        ):
            raise ValueError("malformed SYKNNUE9 section")
        sections[section_id] = {
            "type": element_type,
            "shape": dimensions[:rank],
            "payload": payload,
        }
        previous_end = offset + byte_length
    return {
        "architecture": "pairwise-mlp-p3",
        "feature_set": feature_set,
        "input_bucket_count": input_bucket_count,
        "output_bucket_count": output_bucket_count,
        "output_bucket_scheme": OUTPUT_BUCKET_SCHEME_MATERIAL,
        "ft_hidden_size": h,
        "dense1_size": d1,
        "dense2_size": d2,
        "activation_ids": activation_ids,
        "q0": q0,
        "threat_quant": threat_quant,
        "pool_quant": pool_quant,
        "q": q,
        "scale": scale,
        "bucket_layout_64": bucket_layout,
        "threat_feature_count": threat_feature_count,
        "threat_scheme_id": threat_scheme_id,
        "threat_packing_sha256": packing_hash.hex(),
        "psq_abs_bound": psq_abs_bound,
        "threat_abs_bound": threat_abs_bound,
        "p3_rank": p3_rank,
        "p3_pawn_count": p3_pawn_count,
        "p3_context_count": p3_context_count,
        "p3_quant": p3_quant,
        "p3_activation_shift": p3_activation_shift,
        "p3_l1_abs_bound": p3_l1_abs_bound,
        "sections": sections,
    }


def decode_tensors(net: dict) -> dict:
    import numpy as np

    type_dtypes = {1: np.dtype("i1"), 3: np.dtype("<i2"), 4: np.dtype("<i4")}
    result = {}
    for section_id, section in net["sections"].items():
        dtype = type_dtypes.get(section["type"])
        if dtype is None:
            continue
        result[section_id] = (
            np.frombuffer(section["payload"], dtype=dtype)
            .reshape(section["shape"])
            .astype(np.int64)
        )
    return result


def run_engine(engine: str, net: str, fens: str) -> dict:
    proc = subprocess.run(
        [
            engine,
            "nnuecheck",
            "--net",
            net,
            "--fens",
            fens,
            "--verify-incremental",
        ],
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
    args = parse_args()
    net_path = Path(args.net)
    magic = net_path.read_bytes()[:8]
    if magic == MAGIC_V8:
        net = read_syk_nnue_v8(net_path)
    elif magic == MAGIC_V9:
        net = read_syk_nnue_v9_for_parity(net_path)
    else:
        raise SystemExit(f"unsupported network magic: {magic!r}")
    tensors = decode_tensors(net)
    evaluator = lambda fen: reference_eval(net, tensors, fen)

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
        ref = evaluator(fen)
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
