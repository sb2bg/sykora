#!/usr/bin/env python3
"""Canonical generator and Python reference for ``full_threats_v1``.

The ID space is deliberately arithmetic and contains holes.  IDs are ordered
by attacker-relative colour, attacker type, directed geometry, then target
colour/type slot.  The generated binary table is the canonical, hashable
decode representation shared by the trainer, exporter, and engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from functools import lru_cache
from pathlib import Path
import chess


SCHEME_NAME = "full_threats_v1"
SCHEME_ID = 1
THREAT_FEATURE_COUNT = 60_720
MAX_ACTIVE_THREATS = 240

OURS = 0
THEIRS = 1

PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5

ATTACKER_TYPES = (PAWN, KNIGHT, BISHOP, ROOK, QUEEN)
GEOMETRY_COUNTS = (132, 336, 560, 896, 1_456)
TARGET_TYPES = (
    (PAWN, KNIGHT, ROOK),
    (PAWN, KNIGHT, BISHOP, ROOK, QUEEN),
    (PAWN, KNIGHT, BISHOP, ROOK),
    (PAWN, KNIGHT, BISHOP, ROOK),
    (PAWN, KNIGHT, BISHOP, ROOK, QUEEN),
)
TARGET_SLOT_COUNTS = tuple(2 * len(types) for types in TARGET_TYPES)

TYPE_BASES: tuple[int, ...]
_bases: list[int] = []
_offset = 0
for _geometries, _slots in zip(GEOMETRY_COUNTS, TARGET_SLOT_COUNTS, strict=True):
    _bases.append(_offset)
    _offset += _geometries * _slots
TYPE_BASES = tuple(_bases)
FEATURES_PER_ATTACKER_COLOUR = _offset
assert FEATURES_PER_ATTACKER_COLOUR == 30_360
assert 2 * FEATURES_PER_ATTACKER_COLOUR == THREAT_FEATURE_COUNT


def _rank(square: int) -> int:
    return square // 8


def _file(square: int) -> int:
    return square % 8


def _valid_geometry(attacker_colour: int, attacker_type: int, source: int, target: int) -> bool:
    if not (0 <= source < 64 and 0 <= target < 64) or source == target:
        return False
    source_rank, source_file = divmod(source, 8)
    target_rank, target_file = divmod(target, 8)
    dr = target_rank - source_rank
    df = target_file - source_file
    adr, adf = abs(dr), abs(df)

    if attacker_type == PAWN:
        if source_rank in (0, 7):
            return False
        forward = 1 if attacker_colour == OURS else -1
        return dr == forward and adf <= 1
    if attacker_type == KNIGHT:
        return (adr, adf) in ((1, 2), (2, 1))
    if attacker_type == BISHOP:
        return adr == adf
    if attacker_type == ROOK:
        return dr == 0 or df == 0
    if attacker_type == QUEEN:
        return adr == adf or dr == 0 or df == 0
    return False


@lru_cache(maxsize=None)
def geometry_pairs(attacker_colour: int, attacker_type: int) -> tuple[tuple[int, int], ...]:
    pairs = tuple(
        (source, target)
        for source in range(64)
        for target in range(64)
        if _valid_geometry(attacker_colour, attacker_type, source, target)
    )
    expected = GEOMETRY_COUNTS[attacker_type]
    if len(pairs) != expected:
        raise AssertionError(
            f"geometry count for colour={attacker_colour} type={attacker_type}: "
            f"{len(pairs)} != {expected}"
        )
    return pairs


@lru_cache(maxsize=None)
def _geometry_indices(attacker_colour: int, attacker_type: int) -> dict[tuple[int, int], int]:
    return {pair: index for index, pair in enumerate(geometry_pairs(attacker_colour, attacker_type))}


def _target_slot(attacker_type: int, target_colour: int, target_type: int) -> int | None:
    try:
        type_index = TARGET_TYPES[attacker_type].index(target_type)
    except ValueError:
        return None
    return target_colour * len(TARGET_TYPES[attacker_type]) + type_index


def _slot_target(attacker_type: int, slot: int) -> tuple[int, int]:
    types = TARGET_TYPES[attacker_type]
    return slot // len(types), types[slot % len(types)]


def structurally_reachable(
    attacker_colour: int,
    attacker_type: int,
    source: int,
    target: int,
    target_colour: int,
    target_type: int,
) -> bool:
    if target_type == PAWN and _rank(target) in (0, 7):
        return False
    if attacker_type == target_type:
        enemy_pair = target_colour != attacker_colour
        friendly_symmetric = target_colour == attacker_colour and attacker_type != PAWN
        if (enemy_pair or friendly_symmetric) and source >= target:
            return False
    return True


def encode(
    attacker_colour: int,
    attacker_type: int,
    source: int,
    target: int,
    target_colour: int,
    target_type: int,
) -> int | None:
    """Return a reachable threat ID, or ``None`` for a dropped relation."""
    if attacker_colour not in (OURS, THEIRS) or target_colour not in (OURS, THEIRS):
        return None
    if attacker_type not in ATTACKER_TYPES or target_type == KING:
        return None
    geometry = _geometry_indices(attacker_colour, attacker_type).get((source, target))
    slot = _target_slot(attacker_type, target_colour, target_type)
    if geometry is None or slot is None:
        return None
    if not structurally_reachable(
        attacker_colour,
        attacker_type,
        source,
        target,
        target_colour,
        target_type,
    ):
        return None
    feature = (
        attacker_colour * FEATURES_PER_ATTACKER_COLOUR
        + TYPE_BASES[attacker_type]
        + geometry * TARGET_SLOT_COUNTS[attacker_type]
        + slot
    )
    if not 0 <= feature < THREAT_FEATURE_COUNT:
        raise AssertionError(feature)
    return feature


def decode(feature: int) -> tuple[int, int, int, int, int, int, bool]:
    if not 0 <= feature < THREAT_FEATURE_COUNT:
        raise ValueError(f"threat feature out of range: {feature}")
    attacker_colour, local = divmod(feature, FEATURES_PER_ATTACKER_COLOUR)
    attacker_type = -1
    geometry = -1
    slot = -1
    for candidate in ATTACKER_TYPES:
        region_size = GEOMETRY_COUNTS[candidate] * TARGET_SLOT_COUNTS[candidate]
        if TYPE_BASES[candidate] <= local < TYPE_BASES[candidate] + region_size:
            attacker_type = candidate
            geometry, slot = divmod(
                local - TYPE_BASES[candidate], TARGET_SLOT_COUNTS[candidate]
            )
            break
    if attacker_type < 0:
        raise AssertionError(feature)
    source, target = geometry_pairs(attacker_colour, attacker_type)[geometry]
    target_colour, target_type = _slot_target(attacker_type, slot)
    reachable = structurally_reachable(
        attacker_colour,
        attacker_type,
        source,
        target,
        target_colour,
        target_type,
    )
    return (
        attacker_colour,
        attacker_type,
        source,
        target,
        target_colour,
        target_type,
        reachable,
    )


def canonical_table() -> bytes:
    records = bytearray()
    for feature in range(THREAT_FEATURE_COUNT):
        attacker_colour, attacker_type, source, target, target_colour, target_type, reachable = decode(feature)
        records += struct.pack(
            "<8B",
            attacker_colour,
            attacker_type,
            source,
            target,
            target_colour,
            target_type,
            int(reachable),
            0,
        )
    return bytes(records)


def packing_hash_bytes() -> bytes:
    return hashlib.sha256(canonical_table()).digest()


def packing_hash_hex() -> str:
    return packing_hash_bytes().hex()


def _oriented_square(square: int, perspective: chess.Color, king_square: int) -> int:
    oriented = square if perspective == chess.WHITE else square ^ 56
    oriented_king = king_square if perspective == chess.WHITE else king_square ^ 56
    if oriented_king % 8 > 3:
        oriented ^= 7
    return oriented


def _piece_index(piece_type: chess.PieceType) -> int:
    return piece_type - 1


def _occupied_targets(board: chess.Board, square: int, piece: chess.Piece) -> int:
    if piece.piece_type == chess.PAWN:
        attacks = board.attacks_mask(square)
        forward = square + (8 if piece.color == chess.WHITE else -8)
        if 0 <= forward < 64 and board.piece_at(forward) is not None:
            attacks |= chess.BB_SQUARES[forward]
        return attacks & board.occupied
    return board.attacks_mask(square) & board.occupied


def enumerate_board(board: chess.Board, perspective: chess.Color) -> list[int]:
    """Enumerate sorted, unique full-threat IDs for a python-chess board."""
    king_square = board.king(perspective)
    if king_square is None:
        raise ValueError("perspective king is missing")
    features: list[int] = []
    for source, attacker in board.piece_map().items():
        attacker_type = _piece_index(attacker.piece_type)
        if attacker_type == KING:
            continue
        targets = _occupied_targets(board, source, attacker)
        while targets:
            target = chess.scan_forward(targets).__next__()
            targets &= targets - 1
            target_piece = board.piece_at(target)
            if target_piece is None or target_piece.piece_type == chess.KING:
                continue
            attacker_colour = OURS if attacker.color == perspective else THEIRS
            target_colour = OURS if target_piece.color == perspective else THEIRS
            feature = encode(
                attacker_colour,
                attacker_type,
                _oriented_square(source, perspective, king_square),
                _oriented_square(target, perspective, king_square),
                target_colour,
                _piece_index(target_piece.piece_type),
            )
            if feature is not None:
                features.append(feature)
    result = sorted(set(features))
    if len(result) > MAX_ACTIVE_THREATS:
        raise AssertionError(f"active threat bound exceeded: {len(result)}")
    return result


def _golden_records() -> list[dict[str, int | bool]]:
    selected: list[int] = []
    for colour in (OURS, THEIRS):
        for attacker_type in ATTACKER_TYPES:
            region_start = colour * FEATURES_PER_ATTACKER_COLOUR + TYPE_BASES[attacker_type]
            region_end = region_start + GEOMETRY_COUNTS[attacker_type] * TARGET_SLOT_COUNTS[attacker_type]
            selected.extend((region_start, region_start + TARGET_SLOT_COUNTS[attacker_type] - 1, region_end - 1))
    records = []
    for feature in sorted(set(selected)):
        attacker_colour, attacker_type, source, target, target_colour, target_type, reachable = decode(feature)
        records.append(
            {
                "id": feature,
                "attacker_colour": attacker_colour,
                "attacker_type": attacker_type,
                "source": source,
                "target": target,
                "target_colour": target_colour,
                "target_type": target_type,
                "reachable": reachable,
            }
        )
    return records


def manifest() -> dict:
    table = canonical_table()
    reachable = sum(table[index + 6] for index in range(0, len(table), 8))
    return {
        "scheme": SCHEME_NAME,
        "scheme_id": SCHEME_ID,
        "feature_count": THREAT_FEATURE_COUNT,
        "record_bytes": 8,
        "canonical_order": "attacker_colour,type,source,target,target_colour,target_type",
        "packing_sha256": hashlib.sha256(table).hexdigest(),
        "reachable_id_count": reachable,
        "features_per_attacker_colour": FEATURES_PER_ATTACKER_COLOUR,
        "geometry_counts": list(GEOMETRY_COUNTS),
        "target_slot_counts": list(TARGET_SLOT_COUNTS),
        "type_bases": list(TYPE_BASES),
        "max_active_threats": MAX_ACTIVE_THREATS,
        "golden": _golden_records(),
    }


def write_generated(root: Path) -> None:
    table_path = root / "utils" / "nnue" / "full_threats_v1.bin"
    manifest_path = root / "utils" / "nnue" / "full_threats_v1_manifest.json"
    table_path.write_bytes(canonical_table())
    manifest_path.write_text(json.dumps(manifest(), indent=2) + "\n")
    print(f"wrote {table_path}")
    print(f"wrote {manifest_path}")


def verify_generated(root: Path) -> None:
    table_path = root / "utils" / "nnue" / "full_threats_v1.bin"
    manifest_path = root / "utils" / "nnue" / "full_threats_v1_manifest.json"
    expected_table = canonical_table()
    expected_manifest = json.dumps(manifest(), indent=2) + "\n"
    if not table_path.is_file() or table_path.read_bytes() != expected_table:
        raise ValueError(f"generated packing table is stale: {table_path}")
    if not manifest_path.is_file() or manifest_path.read_text() != expected_manifest:
        raise ValueError(f"generated packing manifest is stale: {manifest_path}")
    print(f"verified {table_path}")
    print(f"verified {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate/verify full_threats_v1 packing data")
    parser.add_argument("--write", action="store_true", help="write canonical table and manifest")
    parser.add_argument("--check", action="store_true", help="verify generated files are current")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root",
    )
    args = parser.parse_args()
    if not args.write and not args.check:
        print(json.dumps(manifest(), indent=2))
    if args.write:
        write_generated(args.root.resolve())
    if args.check:
        verify_generated(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
