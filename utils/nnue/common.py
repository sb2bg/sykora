#!/usr/bin/env python3
"""Shared helpers for Sykora NNUE tooling."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, List, Tuple

import chess


LEGACY_INPUT_SIZE = 768
QA = 255
QB = 64
SCALE = 400
MAGIC_V3 = b"SYKNNUE3"
FORMAT_VERSION_V3 = 3

FEATURE_SET_LEGACY = 0
FEATURE_SET_KING_BUCKETS_MIRRORED = 1

SYKORA_BUCKET_LAYOUT_32 = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
]


def expand_mirrored_bucket_layout(layout_32: List[int]) -> List[int]:
    if len(layout_32) != 32:
        raise ValueError("layout_32 must contain exactly 32 entries")
    expanded: List[int] = [0] * 64
    mirror = [0, 1, 2, 3, 3, 2, 1, 0]
    for idx in range(64):
        expanded[idx] = int(layout_32[(idx // 8) * 4 + mirror[idx % 8]])
    return expanded


def num_buckets(layout: Iterable[int]) -> int:
    items = [int(v) for v in layout]
    if not items:
        raise ValueError("layout must not be empty")
    return max(items) + 1


def input_size_for_feature_set(feature_set: int, bucket_layout_64: List[int] | None = None) -> int:
    if feature_set == FEATURE_SET_LEGACY:
        return LEGACY_INPUT_SIZE
    if feature_set == FEATURE_SET_KING_BUCKETS_MIRRORED:
        if bucket_layout_64 is None:
            raise ValueError("bucket_layout_64 is required for king-bucketed inputs")
        return LEGACY_INPUT_SIZE * num_buckets(bucket_layout_64)
    raise ValueError(f"unsupported feature_set: {feature_set}")


def flip_vertical(square: int) -> int:
    return square ^ 56


def feature_index(
    perspective_is_white: bool,
    square: int,
    piece_type: chess.PieceType,
    color: chess.Color,
    *,
    feature_set: int = FEATURE_SET_LEGACY,
    perspective_king_square: int | None = None,
    bucket_layout_64: List[int] | None = None,
) -> int:
    sq = square if perspective_is_white else flip_vertical(square)
    king_sq = (
        perspective_king_square
        if perspective_is_white
        else flip_vertical(perspective_king_square)
    )
    side = color if perspective_is_white else (not color)
    side_idx = 0 if side == chess.WHITE else 1
    piece_idx = piece_type - 1  # python-chess piece types are 1..6
    base = side_idx * 6 * 64 + piece_idx * 64 + sq
    if feature_set == FEATURE_SET_LEGACY:
        return base
    if feature_set != FEATURE_SET_KING_BUCKETS_MIRRORED:
        raise ValueError(f"unsupported feature_set: {feature_set}")
    if perspective_king_square is None or bucket_layout_64 is None:
        raise ValueError("perspective_king_square and bucket_layout_64 are required")
    sq = sq ^ 7 if king_sq % 8 > 3 else sq
    bucket_offset = LEGACY_INPUT_SIZE * bucket_layout_64[king_sq]
    return bucket_offset + side_idx * 6 * 64 + piece_idx * 64 + sq


def board_feature_indices(
    board: chess.Board,
    *,
    feature_set: int = FEATURE_SET_LEGACY,
    bucket_layout_64: List[int] | None = None,
) -> Tuple[List[int], List[int], bool]:
    white_features: List[int] = []
    black_features: List[int] = []
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    if white_king_square is None or black_king_square is None:
        raise ValueError("board must contain both kings")
    for square, piece in board.piece_map().items():
        white_features.append(
            feature_index(
                True,
                square,
                piece.piece_type,
                piece.color,
                feature_set=feature_set,
                perspective_king_square=white_king_square,
                bucket_layout_64=bucket_layout_64,
            )
        )
        black_features.append(
            feature_index(
                False,
                square,
                piece.piece_type,
                piece.color,
                feature_set=feature_set,
                perspective_king_square=black_king_square,
                bucket_layout_64=bucket_layout_64,
            )
        )
    return white_features, black_features, board.turn == chess.WHITE


def fen_feature_indices(
    fen: str,
    *,
    feature_set: int = FEATURE_SET_LEGACY,
    bucket_layout_64: List[int] | None = None,
) -> Tuple[List[int], List[int], bool]:
    return board_feature_indices(
        chess.Board(fen),
        feature_set=feature_set,
        bucket_layout_64=bucket_layout_64,
    )


def _pack_i16(values: Iterable[int]) -> bytes:
    return b"".join(struct.pack("<h", int(v)) for v in values)


def write_syk_nnue(
    path: Path,
    *,
    hidden_size: int,
    input_biases_i16: List[int],
    input_weights_i16: List[int],
    output_weights_i16: List[int],
    output_bias_i32: int,
    activation_type: int = 1,
    feature_set: int = FEATURE_SET_LEGACY,
    bucket_layout_64: List[int] | None = None,
) -> None:
    if hidden_size <= 0:
        raise ValueError("hidden_size must be > 0")
    if len(input_biases_i16) != hidden_size:
        raise ValueError("input_biases length mismatch")
    input_size = input_size_for_feature_set(feature_set, bucket_layout_64)
    if len(input_weights_i16) != input_size * hidden_size:
        raise ValueError("input_weights length mismatch")
    if len(output_weights_i16) != 2 * hidden_size:
        raise ValueError("output_weights length mismatch")

    if feature_set == FEATURE_SET_LEGACY:
        bucket_layout_64 = [0] * 64
    elif bucket_layout_64 is None or len(bucket_layout_64) != 64:
        raise ValueError("bucket_layout_64 must contain exactly 64 entries")

    bucket_count = num_buckets(bucket_layout_64)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(MAGIC_V3)
        handle.write(struct.pack("<H", FORMAT_VERSION_V3))
        handle.write(struct.pack("<B", feature_set))
        handle.write(struct.pack("<H", hidden_size))
        handle.write(struct.pack("<B", activation_type))
        handle.write(struct.pack("<H", bucket_count))
        handle.write(bytes(int(v) for v in bucket_layout_64))
        handle.write(struct.pack("<i", int(output_bias_i32)))
        handle.write(_pack_i16(input_biases_i16))
        handle.write(_pack_i16(input_weights_i16))
        handle.write(_pack_i16(output_weights_i16))
