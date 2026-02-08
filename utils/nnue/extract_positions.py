#!/usr/bin/env python3
"""Extract NNUE training positions from PGN files."""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path
from typing import Iterable, List

import chess
import chess.pgn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract training positions from PGN files.")
    parser.add_argument(
        "--pgn-glob",
        default="history/matches/*/pgn/*.pgn",
        help="Glob for source PGN files",
    )
    parser.add_argument(
        "--output",
        default="nnue/data/positions.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--min-ply", type=int, default=8, help="Minimum ply to include")
    parser.add_argument("--max-ply", type=int, default=180, help="Maximum ply to include")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.3,
        help="Sampling probability per eligible position",
    )
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip positions where side to move is in check",
    )
    return parser.parse_args()


def game_white_score(result_header: str) -> float | None:
    result = result_header.strip()
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    if result == "1/2-1/2":
        return 0.5
    return None


def iter_pgn_games(paths: Iterable[Path]) -> Iterable[tuple[Path, chess.pgn.Game]]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                yield path, game


def main() -> int:
    args = parse_args()
    if not (0.0 < args.sample_rate <= 1.0):
        print("--sample-rate must be in (0, 1].", file=sys.stderr)
        return 2
    if args.min_ply < 0 or args.max_ply < args.min_ply:
        print("Invalid ply bounds.", file=sys.stderr)
        return 2

    random.seed(args.seed)

    pgn_paths = [Path(p) for p in sorted(glob.glob(args.pgn_glob))]
    if not pgn_paths:
        print(f"No PGNs matched: {args.pgn_glob}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    games_seen = 0
    positions_written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path, game in iter_pgn_games(pgn_paths):
            white_score = game_white_score(game.headers.get("Result", ""))
            if white_score is None:
                continue

            games_seen += 1
            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                board.push(move)
                ply += 1

                if ply < args.min_ply or ply > args.max_ply:
                    continue
                if args.skip_check and board.is_check():
                    continue
                if random.random() > args.sample_rate:
                    continue

                stm_white = board.turn == chess.WHITE
                stm_score = white_score if stm_white else (1.0 - white_score)
                row = {
                    "fen": board.fen(en_passant="fen"),
                    "game_result_white": white_score,
                    "target_result_stm": stm_score,
                    "stm_white": stm_white,
                    "ply": ply,
                    "source_pgn": str(path),
                    "event": game.headers.get("Event", ""),
                }
                out.write(json.dumps(row) + "\n")
                positions_written += 1

    print(f"PGNs: {len(pgn_paths)} | Games: {games_seen} | Positions: {positions_written}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

