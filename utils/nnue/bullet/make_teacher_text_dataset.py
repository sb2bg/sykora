#!/usr/bin/env python3
"""Build a Bullet text dataset from PGNs using a Stockfish teacher.

Output line format:
  <FEN> | <score_cp_white_relative> | <result_white_relative>
"""

from __future__ import annotations

import argparse
import glob
import gzip
import random
import sys
from pathlib import Path
from typing import Iterable, TextIO

import chess
import chess.engine
import chess.pgn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create high-quality Bullet text training data.")
    parser.add_argument(
        "--pgn-glob",
        action="append",
        default=[],
        help=(
            "Glob for source PGNs (.pgn/.pgn.gz). Repeatable. "
            "Defaults to history/matches/*/pgn/*.pgn when omitted."
        ),
    )
    parser.add_argument(
        "--output",
        default="nnue/data/bullet/train/teacher_text.txt",
        help="Output text file (<FEN> | <cp> | <result>)",
    )
    parser.add_argument(
        "--stockfish",
        default="stockfish",
        help="Stockfish binary path",
    )
    parser.add_argument("--depth", type=int, default=12, help="Teacher depth")
    parser.add_argument("--threads", type=int, default=4, help="Teacher Threads option")
    parser.add_argument("--hash-mb", type=int, default=2048, help="Teacher Hash option (MB)")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument("--sample-rate", type=float, default=0.2, help="Sampling probability")
    parser.add_argument("--min-ply", type=int, default=12, help="Minimum ply")
    parser.add_argument("--max-ply", type=int, default=220, help="Maximum ply")
    parser.add_argument("--cp-clip", type=int, default=2500, help="Clip cp to +/- this value")
    parser.add_argument("--max-positions", type=int, default=0, help="Optional hard cap (0=all)")
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip positions where side to move is in check",
    )
    parser.add_argument(
        "--skip-captures",
        action="store_true",
        help="Skip positions immediately after a capture move",
    )
    parser.add_argument(
        "--dedupe-fen",
        action="store_true",
        help="Drop exact duplicate FENs (uses in-memory set)",
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


def result_token(value: float) -> str:
    if value >= 0.75:
        return "1.0"
    if value <= 0.25:
        return "0.0"
    return "0.5"


def iter_pgn_games(paths: Iterable[Path]) -> Iterable[tuple[Path, chess.pgn.Game]]:
    def open_text(path: Path) -> TextIO:
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        return path.open("r", encoding="utf-8", errors="replace")

    for path in paths:
        with open_text(path) as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                yield path, game


def configure_teacher(engine: chess.engine.SimpleEngine, args: argparse.Namespace) -> None:
    cfg: dict[str, object] = {}
    if "Threads" in engine.options:
        cfg["Threads"] = args.threads
    if "Hash" in engine.options:
        cfg["Hash"] = args.hash_mb
    if cfg:
        engine.configure(cfg)
        engine.ping()


def main() -> int:
    args = parse_args()

    if args.sample_rate <= 0.0 or args.sample_rate > 1.0:
        print("--sample-rate must be in (0, 1].", file=sys.stderr)
        return 2
    if args.min_ply < 0 or args.max_ply < args.min_ply:
        print("Invalid ply bounds.", file=sys.stderr)
        return 2
    if args.depth <= 0:
        print("--depth must be > 0.", file=sys.stderr)
        return 2
    if args.threads <= 0 or args.hash_mb <= 0:
        print("--threads and --hash-mb must be > 0.", file=sys.stderr)
        return 2
    if args.cp_clip <= 0:
        print("--cp-clip must be > 0.", file=sys.stderr)
        return 2

    random.seed(args.seed)

    patterns = args.pgn_glob if args.pgn_glob else ["history/matches/*/pgn/*.pgn"]
    all_paths: list[Path] = []
    for pattern in patterns:
        all_paths.extend(Path(p) for p in glob.glob(pattern, recursive=True))
    pgn_paths = sorted(set(all_paths))
    if not pgn_paths:
        print(f"No PGNs matched: {patterns}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_fens: set[str] | None = set() if args.dedupe_fen else None

    games_seen = 0
    positions_written = 0
    analysed = 0
    skipped_check = 0
    skipped_capture = 0
    skipped_dedupe = 0
    skipped_teacher = 0

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        configure_teacher(engine, args)

        with out_path.open("w", encoding="utf-8") as out:
            for _, game in iter_pgn_games(pgn_paths):
                white_score = game_white_score(game.headers.get("Result", ""))
                if white_score is None:
                    continue

                games_seen += 1
                board = game.board()
                ply = 0
                for move in game.mainline_moves():
                    is_capture = board.is_capture(move)
                    board.push(move)
                    ply += 1

                    if ply < args.min_ply or ply > args.max_ply:
                        continue
                    if args.skip_check and board.is_check():
                        skipped_check += 1
                        continue
                    if args.skip_captures and is_capture:
                        skipped_capture += 1
                        continue
                    if random.random() > args.sample_rate:
                        continue

                    fen = board.fen(en_passant="fen")
                    if seen_fens is not None:
                        if fen in seen_fens:
                            skipped_dedupe += 1
                            continue
                        seen_fens.add(fen)

                    try:
                        info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                    except chess.engine.EngineError:
                        skipped_teacher += 1
                        continue
                    except chess.engine.EngineTerminatedError:
                        print("Stockfish terminated during analyse.", file=sys.stderr)
                        return 1

                    score_obj = info["score"].pov(board.turn)
                    cp_stm = score_obj.score(mate_score=100000)
                    if cp_stm is None:
                        skipped_teacher += 1
                        continue

                    cp_stm = max(-args.cp_clip, min(args.cp_clip, int(cp_stm)))
                    cp_white = cp_stm if board.turn == chess.WHITE else -cp_stm

                    out.write(f"{fen} | {cp_white} | {result_token(white_score)}\n")
                    positions_written += 1
                    analysed += 1

                    if args.max_positions > 0 and positions_written >= args.max_positions:
                        break

                if args.max_positions > 0 and positions_written >= args.max_positions:
                    break
    finally:
        try:
            engine.quit()
        except chess.engine.EngineTerminatedError:
            pass

    print(f"PGNs: {len(pgn_paths)} | Games: {games_seen}")
    print(f"Wrote: {out_path}")
    print(f"Positions written: {positions_written}")
    print(f"Teacher analysed: {analysed}")
    print(f"Skipped (check): {skipped_check}")
    print(f"Skipped (capture): {skipped_capture}")
    print(f"Skipped (dedupe): {skipped_dedupe}")
    print(f"Skipped (teacher): {skipped_teacher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
