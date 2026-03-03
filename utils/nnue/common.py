#!/usr/bin/env python3
"""Shared helpers for Sykora NNUE tooling."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import chess


INPUT_SIZE = 768
QA = 255
QB = 64
SCALE = 400
MAGIC_V2 = b"SYKNNUE2"
MAGIC_V3 = b"SYKNNUE3"
MAGIC = MAGIC_V2  # default for backward compat
FORMAT_VERSION_V2 = 2
FORMAT_VERSION_V3 = 3
FORMAT_VERSION = FORMAT_VERSION_V2


def flip_vertical(square: int) -> int:
    return square ^ 56


def feature_index(
    perspective_is_white: bool,
    square: int,
    piece_type: chess.PieceType,
    color: chess.Color,
) -> int:
    sq = square if perspective_is_white else flip_vertical(square)
    side = color if perspective_is_white else (not color)
    side_idx = 0 if side == chess.WHITE else 1
    piece_idx = piece_type - 1  # python-chess piece types are 1..6
    return side_idx * 6 * 64 + piece_idx * 64 + sq


def board_feature_indices(board: chess.Board) -> Tuple[List[int], List[int], bool]:
    white_features: List[int] = []
    black_features: List[int] = []
    for square, piece in board.piece_map().items():
        white_features.append(feature_index(True, square, piece.piece_type, piece.color))
        black_features.append(feature_index(False, square, piece.piece_type, piece.color))
    return white_features, black_features, board.turn == chess.WHITE


def fen_feature_indices(fen: str) -> Tuple[List[int], List[int], bool]:
    return board_feature_indices(chess.Board(fen))


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
    l2_size: int = 0,
    l2_weights_i16: Optional[List[int]] = None,
    l2_biases_i16: Optional[List[int]] = None,
) -> None:
    if hidden_size <= 0:
        raise ValueError("hidden_size must be > 0")
    if len(input_biases_i16) != hidden_size:
        raise ValueError("input_biases length mismatch")
    if len(input_weights_i16) != INPUT_SIZE * hidden_size:
        raise ValueError("input_weights length mismatch")

    if l2_size > 0:
        if l2_weights_i16 is None or len(l2_weights_i16) != 2 * hidden_size * l2_size:
            raise ValueError(
                f"l2_weights length mismatch: expected {2 * hidden_size * l2_size}, "
                f"got {len(l2_weights_i16) if l2_weights_i16 else 0}"
            )
        if l2_biases_i16 is None or len(l2_biases_i16) != l2_size:
            raise ValueError(
                f"l2_biases length mismatch: expected {l2_size}, "
                f"got {len(l2_biases_i16) if l2_biases_i16 else 0}"
            )
        if len(output_weights_i16) != l2_size:
            raise ValueError(
                f"output_weights length mismatch for L2 net: expected {l2_size}, "
                f"got {len(output_weights_i16)}"
            )
    else:
        if len(output_weights_i16) != 2 * hidden_size:
            raise ValueError("output_weights length mismatch")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        if l2_size > 0:
            handle.write(MAGIC_V3)
            handle.write(struct.pack("<H", FORMAT_VERSION_V3))
            handle.write(struct.pack("<H", hidden_size))
            handle.write(struct.pack("<H", l2_size))
        else:
            handle.write(MAGIC_V2)
            handle.write(struct.pack("<H", FORMAT_VERSION_V2))
            handle.write(struct.pack("<H", hidden_size))
        handle.write(struct.pack("<B", activation_type))
        handle.write(struct.pack("<i", int(output_bias_i32)))
        handle.write(_pack_i16(input_biases_i16))
        handle.write(_pack_i16(input_weights_i16))
        if l2_size > 0:
            handle.write(_pack_i16(l2_weights_i16))
            handle.write(_pack_i16(l2_biases_i16))
        handle.write(_pack_i16(output_weights_i16))

