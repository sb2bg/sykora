#!/usr/bin/env python3
"""Split Stockfish .binpack files into shards at chunk boundaries.

A binpack is a sequence of independent blocks, each a "BINP" magic, a
little-endian u32 size, and that many bytes of chains. Cutting between blocks
yields valid binpacks, so a source stored as a few huge files can be split
into many shards that --interleave-sources then spreads across the run.

Shards are written next to each other as <stem>.<index>.binpack in the output
directory; the originals are left untouched.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

MAGIC = b"BINP"
HEADER = struct.Struct("<4sI")


def split(path: Path, output_dir: Path, shard_bytes: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_index = 0
    shard = None
    shard_size = 0
    with path.open("rb") as source:
        while header := source.read(HEADER.size):
            if len(header) != HEADER.size:
                raise ValueError(f"{path}: truncated block header")
            magic, size = HEADER.unpack(header)
            if magic != MAGIC:
                raise ValueError(f"{path}: bad block magic at byte {source.tell() - HEADER.size}")
            body = source.read(size)
            if len(body) != size:
                raise ValueError(f"{path}: truncated block")
            if shard is None or shard_size >= shard_bytes:
                if shard is not None:
                    shard.close()
                shard = (output_dir / f"{path.stem}.{shard_index:05d}.binpack").open("xb")
                shard_index += 1
                shard_size = 0
            shard.write(header)
            shard.write(body)
            shard_size += HEADER.size + size
    if shard is not None:
        shard.close()
    return shard_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, nargs="+", help=".binpack files to split")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-mb", type=int, default=1024, help="Approximate shard size")
    args = parser.parse_args()
    for path in args.input:
        shards = split(path, args.output_dir, args.shard_mb * 1024 * 1024)
        print(f"{path}: {shards} shard(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
