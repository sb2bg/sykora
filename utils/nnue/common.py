#!/usr/bin/env python3
"""Shared helpers for Sykora NNUE tooling."""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path
from typing import Iterable, List, Tuple

import chess


LEGACY_INPUT_SIZE = 768
NNUE_Q0 = 255
NNUE_Q = 64
SCALE = 400
MAGIC_V6 = b"SYKNNUE6"
FORMAT_VERSION_V6 = 6
MAGIC_V7 = b"SYKNNUE7"
FORMAT_VERSION_V7 = 7
V7_HEADER_BYTES = 160
V7_SECTION_ENTRY_BYTES = 48
V7_HASH_OFFSET = 114

V7_ARCH_PAIRWISE_MLP = 1
V7_POOL_PAIRWISE_PRODUCT = 1
V7_DENSE1_DUAL_CSCRELU = 1
V7_DENSE2_SCRELU = 1
V7_SELECTOR_MATERIAL = 1

V7_TYPE_I8 = 1
V7_TYPE_U8 = 2
V7_TYPE_I16 = 3
V7_TYPE_I32 = 4
V7_TYPE_UTF8 = 5

V7_SECTION_FT_BIAS = 1
V7_SECTION_FT_WEIGHT = 2
V7_SECTION_L1_BIAS = 10
V7_SECTION_L1_WEIGHT = 11
V7_SECTION_L2_BIAS = 12
V7_SECTION_L2_WEIGHT = 13
V7_SECTION_OUT_BIAS = 14
V7_SECTION_OUT_WEIGHT = 15

FEATURE_SET_LEGACY = 0
FEATURE_SET_KING_BUCKETS_MIRRORED = 1

ACTIVATION_RELU = 0
ACTIVATION_SCRELU = 1

OUTPUT_BUCKET_SCHEME_SINGLE = 0
OUTPUT_BUCKET_SCHEME_MATERIAL = 1

SYKORA16_BUCKET_LAYOUT_32 = [
    0, 0, 1, 1,
    2, 2, 3, 3,
    4, 4, 5, 5,
    6, 6, 7, 7,
    8, 8, 9, 9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    14, 14, 15, 15,
]

# Proven v3 10-bucket layout (32-entry mirrored half). Expands via
# expand_mirrored_bucket_layout() to the spec's baseline 64-entry layout.
V3_BUCKET_LAYOUT_32 = [
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


def _pack_i32(values: Iterable[int]) -> bytes:
    return b"".join(struct.pack("<i", int(v)) for v in values)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def write_syk_nnue_v7(
    path: Path,
    *,
    ft_hidden_size: int,
    dense1_size: int,
    dense2_size: int,
    input_bucket_layout_64: List[int],
    output_bucket_count: int,
    ft_bias_bytes: bytes,
    ft_weight_bytes: bytes,
    l1_bias_bytes: bytes,
    l1_weight_bytes: bytes,
    l2_bias_bytes: bytes,
    l2_weight_bytes: bytes,
    out_bias_bytes: bytes,
    out_weight_bytes: bytes,
    q0: int = NNUE_Q0,
    pool_quant: int = 128,
    q: int = NNUE_Q,
    scale: int = SCALE,
) -> None:
    """Write the registered pairwise-MLP SYKNNUE7 profile."""
    if len(input_bucket_layout_64) != 64:
        raise ValueError("input_bucket_layout_64 must contain exactly 64 entries")
    input_bucket_count = num_buckets(input_bucket_layout_64)
    if input_bucket_count <= 0 or output_bucket_count != 8:
        raise ValueError("SYKNNUE7 pairwise_mlp requires positive input buckets and O=8")
    if ft_hidden_size <= 0 or ft_hidden_size % 2:
        raise ValueError("ft_hidden_size must be positive and even")
    if dense1_size <= 0 or dense2_size <= 0:
        raise ValueError("dense widths must be positive")
    if min(q0, pool_quant, q, scale) <= 0:
        raise ValueError("quantisation and score scales must be positive")

    input_size = LEGACY_INPUT_SIZE * input_bucket_count
    sections = [
        (V7_SECTION_FT_BIAS, V7_TYPE_I16, (ft_hidden_size,), ft_bias_bytes),
        (V7_SECTION_FT_WEIGHT, V7_TYPE_I16, (input_size, ft_hidden_size), ft_weight_bytes),
        (V7_SECTION_L1_BIAS, V7_TYPE_I32, (output_bucket_count, dense1_size), l1_bias_bytes),
        (V7_SECTION_L1_WEIGHT, V7_TYPE_I8, (output_bucket_count, ft_hidden_size, dense1_size), l1_weight_bytes),
        (V7_SECTION_L2_BIAS, V7_TYPE_I32, (output_bucket_count, dense2_size), l2_bias_bytes),
        (V7_SECTION_L2_WEIGHT, V7_TYPE_I8, (output_bucket_count, 2 * dense1_size, dense2_size), l2_weight_bytes),
        (V7_SECTION_OUT_BIAS, V7_TYPE_I32, (output_bucket_count,), out_bias_bytes),
        (V7_SECTION_OUT_WEIGHT, V7_TYPE_I8, (output_bucket_count, dense2_size), out_weight_bytes),
    ]
    type_sizes = {V7_TYPE_I8: 1, V7_TYPE_I16: 2, V7_TYPE_I32: 4}
    for section_id, element_type, dimensions, payload in sections:
        expected = type_sizes[element_type]
        for dimension in dimensions:
            expected *= dimension
        if len(payload) != expected:
            raise ValueError(
                f"section {section_id} has {len(payload)} bytes, expected {expected}"
            )

    header = bytearray(V7_HEADER_BYTES)
    pos = 0

    def put(fmt: str, *values) -> None:
        nonlocal pos
        struct.pack_into(fmt, header, pos, *values)
        pos += struct.calcsize(fmt)

    header[0:8] = MAGIC_V7
    pos = 8
    put("<H", FORMAT_VERSION_V7)
    put("<H", V7_HEADER_BYTES)
    put("<H", len(sections))
    put("<H", V7_SECTION_ENTRY_BYTES)
    put("<I", 0)
    put("<H", V7_ARCH_PAIRWISE_MLP)
    put("<H", FEATURE_SET_KING_BUCKETS_MIRRORED)
    put("<H", input_bucket_count)
    put("<H", output_bucket_count)
    put("<H", ft_hidden_size)
    put("<H", dense1_size)
    put("<H", dense2_size)
    put("<B", ACTIVATION_RELU)
    put("<B", V7_POOL_PAIRWISE_PRODUCT)
    put("<B", V7_DENSE1_DUAL_CSCRELU)
    put("<B", V7_DENSE2_SCRELU)
    put("<B", V7_SELECTOR_MATERIAL)
    pos += 3
    put("<H", q0)
    put("<H", pool_quant)
    put("<H", q)
    put("<H", scale)
    header[pos : pos + 64] = bytes(int(value) for value in input_bucket_layout_64)
    pos += 64
    hash_offset = pos
    if hash_offset != V7_HASH_OFFSET:
        raise AssertionError(f"SYKNNUE7 hash field moved to {hash_offset}")
    pos += 32
    pos += 14
    if pos != V7_HEADER_BYTES:
        raise AssertionError(f"SYKNNUE7 header construction ended at {pos}")

    table = bytearray(len(sections) * V7_SECTION_ENTRY_BYTES)
    payload_start = _align_up(V7_HEADER_BYTES + len(table), 64)
    output = bytearray(payload_start)
    output[0:V7_HEADER_BYTES] = header
    output[V7_HEADER_BYTES : V7_HEADER_BYTES + len(table)] = table

    offset = payload_start
    for index, (section_id, element_type, dimensions, payload) in enumerate(sections):
        offset = _align_up(offset, 64)
        if len(output) < offset:
            output.extend(b"\0" * (offset - len(output)))
        dims = list(dimensions) + [1] * (4 - len(dimensions))
        entry = struct.pack(
            "<HBBI4IQQII",
            section_id,
            element_type,
            len(dimensions),
            1,
            *dims,
            offset,
            len(payload),
            zlib.crc32(payload) & 0xFFFFFFFF,
            0,
        )
        start = V7_HEADER_BYTES + index * V7_SECTION_ENTRY_BYTES
        output[start : start + V7_SECTION_ENTRY_BYTES] = entry
        output.extend(payload)
        offset += len(payload)

    digest = hashlib.sha256(output).digest()
    output[hash_offset : hash_offset + 32] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


def read_syk_nnue_v7(path: Path) -> dict:
    """Read and validate a registered pairwise-MLP SYKNNUE7 net."""
    data = Path(path).read_bytes()
    if len(data) < V7_HEADER_BYTES or data[0:8] != MAGIC_V7:
        raise ValueError("not a SYKNNUE7 net")
    pos = 8

    def take(fmt: str):
        nonlocal pos
        size = struct.calcsize(fmt)
        if pos + size > len(data):
            raise ValueError("truncated SYKNNUE7 header")
        values = struct.unpack_from(fmt, data, pos)
        pos += size
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
    pos += 5
    reserved0 = data[pos : pos + 3]
    pos += 3
    q0 = take("<H")
    pool_quant = take("<H")
    q = take("<H")
    scale = take("<H")
    bucket_layout = list(data[pos : pos + 64])
    pos += 64
    expected_hash = data[pos : pos + 32]
    pos += 32
    reserved1 = data[pos : pos + 14]
    pos += 14
    if (
        version != FORMAT_VERSION_V7
        or header_bytes != V7_HEADER_BYTES
        or section_entry_bytes != V7_SECTION_ENTRY_BYTES
        or flags != 0
        or architecture != V7_ARCH_PAIRWISE_MLP
        or feature_set != FEATURE_SET_KING_BUCKETS_MIRRORED
        or output_bucket_count != 8
        or section_count != 8
        or activation_ids != (0, 1, 1, 1, 1)
        or pos != V7_HEADER_BYTES
        or any(reserved0)
        or any(reserved1)
    ):
        raise ValueError("unsupported or malformed SYKNNUE7 header")
    hasher = hashlib.sha256()
    hasher.update(data[:V7_HASH_OFFSET])
    hasher.update(b"\0" * 32)
    hasher.update(data[V7_HASH_OFFSET + 32 :])
    if hasher.digest() != expected_hash:
        raise ValueError("SYKNNUE7 content hash mismatch")
    if max(bucket_layout) + 1 != input_bucket_count:
        raise ValueError("SYKNNUE7 input bucket layout mismatch")

    type_sizes = {V7_TYPE_I8: 1, V7_TYPE_U8: 1, V7_TYPE_I16: 2, V7_TYPE_I32: 4}
    sections = {}
    for index in range(section_count):
        start = V7_HEADER_BYTES + index * V7_SECTION_ENTRY_BYTES
        end = start + V7_SECTION_ENTRY_BYTES
        if end > len(data):
            raise ValueError("truncated SYKNNUE7 section table")
        fields = struct.unpack_from("<HBBI4IQQII", data, start)
        section_id, element_type, rank, section_flags = fields[:4]
        dimensions = tuple(fields[4:8])
        offset, byte_length, crc32, reserved = fields[8:]
        if section_id in sections or rank > 4 or section_flags != 1 or reserved != 0:
            raise ValueError("malformed SYKNNUE7 section entry")
        if offset % 64 or offset + byte_length > len(data):
            raise ValueError("invalid SYKNNUE7 section bounds")
        if element_type not in type_sizes:
            raise ValueError("unsupported SYKNNUE7 section type")
        expected_length = type_sizes[element_type]
        for dimension in dimensions[:rank]:
            expected_length *= dimension
        if any(dimension != 1 for dimension in dimensions[rank:]) or expected_length != byte_length:
            raise ValueError("SYKNNUE7 section shape mismatch")
        payload = data[offset : offset + byte_length]
        if zlib.crc32(payload) & 0xFFFFFFFF != crc32:
            raise ValueError("SYKNNUE7 section CRC mismatch")
        sections[section_id] = {
            "type": element_type,
            "shape": dimensions[:rank],
            "payload": payload,
        }

    return {
        "architecture": "pairwise-mlp",
        "feature_set": feature_set,
        "input_bucket_count": input_bucket_count,
        "output_bucket_count": output_bucket_count,
        "output_bucket_scheme": OUTPUT_BUCKET_SCHEME_MATERIAL,
        "ft_hidden_size": h,
        "dense1_size": d1,
        "dense2_size": d2,
        "activation_ids": activation_ids,
        "q0": q0,
        "pool_quant": pool_quant,
        "q": q,
        "scale": scale,
        "bucket_layout_64": bucket_layout,
        "sections": sections,
    }


def read_syk_nnue_v6(path: Path) -> dict:
    """Parse a SYKNNUE6 net into a dict of header fields + weight arrays.

    Mirrors the loader/payload contract in src/nnue.zig. Weight arrays are
    returned as Python lists of ints in their on-disk (quantized) domain.
    """
    data = Path(path).read_bytes()
    if data[0:8] != MAGIC_V6:
        raise ValueError(f"not a SYKNNUE6 net (magic={data[0:8]!r})")
    pos = 8

    def take(fmt: str):
        nonlocal pos
        size = struct.calcsize(fmt)
        if pos + size > len(data):
            raise ValueError("SYKNNUE6 truncated header")
        value = struct.unpack_from(fmt, data, pos)[0]
        pos += size
        return value

    version = take("<H")
    if version != FORMAT_VERSION_V6:
        raise ValueError(f"unsupported version: {version}")
    feature_set = take("<B")
    if feature_set != FEATURE_SET_KING_BUCKETS_MIRRORED:
        raise ValueError("only king_buckets_mirrored is supported")
    ft_hidden_size = take("<H")
    activation_type = take("<B")
    input_bucket_count = take("<B")
    output_bucket_count = take("<B")
    output_bucket_scheme = take("<B")
    q0 = take("<H")
    q = take("<H")
    scale = take("<H")
    bucket_layout_64 = list(data[pos : pos + 64])
    pos += 64
    if len(bucket_layout_64) != 64:
        raise ValueError("SYKNNUE6 truncated bucket layout")

    h = ft_hidden_size
    input_size = LEGACY_INPUT_SIZE * input_bucket_count

    def take_ints(fmt_char: str, count: int):
        nonlocal pos
        end = pos + count * struct.calcsize("<" + fmt_char)
        if end > len(data):
            raise ValueError("SYKNNUE6 truncated payload")
        values = list(struct.unpack_from(f"<{count}{fmt_char}", data, pos))
        pos = end
        return values

    out_biases = take_ints("i", output_bucket_count)
    ft_biases = take_ints("h", h)
    ft_weights = take_ints("h", input_size * h)
    out_weights = take_ints("h", output_bucket_count * 2 * h)
    if pos != len(data):
        raise ValueError(f"trailing bytes in SYKNNUE6 net: {len(data) - pos}")

    return {
        "ft_hidden_size": h,
        "activation_type": activation_type,
        "input_bucket_count": input_bucket_count,
        "output_bucket_count": output_bucket_count,
        "output_bucket_scheme": output_bucket_scheme,
        "q0": q0,
        "q": q,
        "scale": scale,
        "bucket_layout_64": bucket_layout_64,
        "out_biases": out_biases,
        "ft_biases": ft_biases,
        "ft_weights": ft_weights,
        "out_weights": out_weights,
    }


def write_syk_nnue_v6(
    path: Path,
    *,
    ft_hidden_size: int,
    ft_biases_i16: List[int],
    ft_weights_i16: List[int],
    out_biases_i32: List[int],
    out_weights_i16: List[int],
    activation_type: int = ACTIVATION_SCRELU,
    feature_set: int = FEATURE_SET_KING_BUCKETS_MIRRORED,
    bucket_layout_64: List[int] | None = None,
    output_bucket_count: int = 8,
    output_bucket_scheme: int = OUTPUT_BUCKET_SCHEME_MATERIAL,
    q0: int = NNUE_Q0,
    q: int = NNUE_Q,
    scale: int = SCALE,
) -> None:
    if feature_set != FEATURE_SET_KING_BUCKETS_MIRRORED:
        raise ValueError("SYKNNUE6 requires king_buckets_mirrored inputs")
    if bucket_layout_64 is None or len(bucket_layout_64) != 64:
        raise ValueError("bucket_layout_64 must contain exactly 64 entries")
    if ft_hidden_size <= 0:
        raise ValueError("ft_hidden_size must be > 0")
    if output_bucket_count <= 0 or output_bucket_count > 255:
        raise ValueError("output_bucket_count must be in 1..255")
    if activation_type not in (ACTIVATION_RELU, ACTIVATION_SCRELU):
        raise ValueError("unsupported activation_type")
    if output_bucket_scheme not in (
        OUTPUT_BUCKET_SCHEME_SINGLE,
        OUTPUT_BUCKET_SCHEME_MATERIAL,
    ):
        raise ValueError("unsupported output_bucket_scheme")
    if output_bucket_scheme == OUTPUT_BUCKET_SCHEME_SINGLE and output_bucket_count != 1:
        raise ValueError("single output-bucket scheme requires output_bucket_count == 1")
    if output_bucket_scheme == OUTPUT_BUCKET_SCHEME_MATERIAL and 32 % output_bucket_count != 0:
        raise ValueError("material output-bucket scheme requires output_bucket_count to divide 32")
    if q0 <= 0 or q <= 0 or scale <= 0:
        raise ValueError("q0/q/scale must be > 0")

    input_size = input_size_for_feature_set(feature_set, bucket_layout_64)
    input_bucket_count = num_buckets(bucket_layout_64)
    h = ft_hidden_size

    if len(ft_biases_i16) != h:
        raise ValueError("ft_biases length mismatch")
    if len(ft_weights_i16) != input_size * h:
        raise ValueError("ft_weights length mismatch")
    if len(out_biases_i32) != output_bucket_count:
        raise ValueError("out_biases length mismatch")
    if len(out_weights_i16) != output_bucket_count * 2 * h:
        raise ValueError("out_weights length mismatch")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        # Header
        handle.write(MAGIC_V6)
        handle.write(struct.pack("<H", FORMAT_VERSION_V6))
        handle.write(struct.pack("<B", feature_set))
        handle.write(struct.pack("<H", h))
        handle.write(struct.pack("<B", activation_type))
        handle.write(struct.pack("<B", input_bucket_count))
        handle.write(struct.pack("<B", output_bucket_count))
        handle.write(struct.pack("<B", output_bucket_scheme))
        handle.write(struct.pack("<H", q0))
        handle.write(struct.pack("<H", q))
        handle.write(struct.pack("<H", scale))
        handle.write(bytes(int(v) for v in bucket_layout_64))
        # Payload: output_biases, ft_biases, ft_weights, output_weights
        handle.write(_pack_i32(out_biases_i32))
        handle.write(_pack_i16(ft_biases_i16))
        handle.write(_pack_i16(ft_weights_i16))
        handle.write(_pack_i16(out_weights_i16))
