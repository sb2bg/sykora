#!/usr/bin/env python3
"""
Generate a Texel tuning dataset from sykora self-play.

Plays N games at a fast time control, extracts quiet positions with
game-result labels, and writes them in Texel format:
  <FEN> | <cp_score> | <result>

where result is 1.0 (white wins), 0.5 (draw), 0.0 (black wins).
cp_score is provided by Stockfish at shallow depth for better signal;
use --no-sf to skip Stockfish annotation and use 0 instead.

Usage:
  python utils/tuning/generate_texel_dataset.py \\
      --engine  zig-out/bin/sykora \\
      --output  datasets/texel_train.txt \\
      --games   5000 \\
      --movetime-ms 50

  # To also annotate with Stockfish eval:
  python utils/tuning/generate_texel_dataset.py \\
      --engine zig-out/bin/sykora --games 5000 \\
      --stockfish /opt/homebrew/bin/stockfish --sf-depth 8
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

# ──────────────────────────────────────────────
# Opening book (4-ply lines to diversify games)
# ──────────────────────────────────────────────
OPENING_LINES = [
    "e2e4 e7e5 g1f3 b8c6",
    "e2e4 c7c5 g1f3 d7d6",
    "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5",
    "d2d4 d7d5 c2c4 e7e6",
    "d2d4 g8f6 c2c4 e7e6",
    "c2c4 e7e5 b1c3 g8f6",
    "g1f3 d7d5 d2d4 g8f6",
    "g1f3 g8f6 c2c4 c7c5",
    "e2e4 e7e5 f1c4 g8f6",
    "e2e4 c7c5 b1c3 b8c6",
    "d2d4 g8f6 c1g5 d7d5",
    "e2e4 d7d6 d2d4 g8f6",
    "e2e4 g8f6 e4e5 f6d5",
    "d2d4 f7f5 g2g3 g8f6",
    "c2c4 c7c6 b1c3 d7d5",
    "e2e4 e7e5 b1c3 g8f6",
    "d2d4 d7d5 c1f4 g8f6",
    "e2e4 d7d5 e4d5 d8d5",
    "g1f3 d7d5 g2g3 g8f6",
]


def parse_opening(line: str) -> list[chess.Move]:
    board = chess.Board()
    moves = []
    for uci in line.split():
        m = chess.Move.from_uci(uci)
        if board.is_legal(m):
            board.push(m)
            moves.append(m)
    return moves


def result_to_float(result_str: str) -> float | None:
    mapping = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
    return mapping.get(result_str)


def play_game(
    engine: chess.engine.SimpleEngine,
    opening_moves: list[chess.Move],
    movetime_ms: int,
    depth: int | None = None,
    max_moves: int = 200,
) -> tuple[chess.Board, list[tuple[str, bool, bool]], str]:
    """
    Play one self-play game, returning:
      (final_board, [(fen, in_check, last_was_capture), ...], result_str)

    in_check and last_was_capture describe the position at the time it was
    recorded, allowing downstream filtering without re-examining the board.
    """
    board = chess.Board()
    # (fen, in_check, last_was_capture)
    positions: list[tuple[str, bool, bool]] = []
    last_was_capture = False

    # Apply opening
    for m in opening_moves:
        if board.is_legal(m):
            last_was_capture = board.is_capture(m)  # checked BEFORE push
            board.push(m)
        else:
            break

    limit = chess.engine.Limit(depth=depth) if depth else chess.engine.Limit(time=movetime_ms / 1000.0)

    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break

        # Record position state BEFORE the engine plays its move
        positions.append((board.fen(), board.is_check(), last_was_capture))

        result_obj = engine.play(board, limit)
        if result_obj.move is None:
            break
        move = result_obj.move
        last_was_capture = board.is_capture(move)  # BEFORE push
        board.push(move)

    result = board.result(claim_draw=True)
    return board, positions, result


def annotate_with_stockfish(
    fens: list[str],
    sf: chess.engine.SimpleEngine,
    depth: int,
    timeout_per_pos: float = 30.0,
) -> list[int]:
    """Return Stockfish cp evals (white-relative) for a batch of FENs.

    Raises on fatal SF crash (caller handles restart).
    Individual position timeouts fall back to cp=0.
    """
    scores = []
    for fen in fens:
        board = chess.Board(fen)
        try:
            info = sf.analyse(
                board,
                chess.engine.Limit(depth=depth, time=timeout_per_pos),
            )
            score = info["score"].white().score(mate_score=10000)
            scores.append(max(-5000, min(5000, score if score is not None else 0)))
        except chess.engine.EngineTerminatedError:
            # SF process died — re-raise so caller can restart
            raise
        except chess.engine.EngineError:
            # SF protocol error on this position — skip it, keep going
            scores.append(0)
    return scores


def start_stockfish(
    stockfish_path: str,
    sf_threads: int,
    sf_hash_mb: int,
) -> chess.engine.SimpleEngine:
    sf = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    cfg: dict[str, object] = {}
    if "Threads" in sf.options:
        cfg["Threads"] = sf_threads
    if "Hash" in sf.options:
        cfg["Hash"] = sf_hash_mb
    if cfg:
        sf.configure(cfg)
        sf.ping()
    return sf


def safe_quit_engine(engine: chess.engine.SimpleEngine | None) -> None:
    if engine is None:
        return
    try:
        engine.quit()
    except chess.engine.EngineTerminatedError:
        pass


def generate_dataset(
    engine_path: str,
    output_path: str,
    num_games: int,
    movetime_ms: int,
    depth: int | None,
    stockfish_path: str | None,
    sf_depth: int,
    sf_threads: int,
    sf_hash_mb: int,
    sf_restart_retries: int,
    min_move: int,
    max_move: int,
    min_pieces: int,
    positions_per_game: int,
    seed: int | None,
) -> None:
    if seed is not None:
        random.seed(seed)

    tc_str = f"depth {depth}" if depth else f"{movetime_ms}ms"
    print(f"Engine:      {engine_path}")
    print(f"Output:      {output_path}")
    print(f"Games:       {num_games}")
    print(f"Time ctrl:   {tc_str}")
    print(f"Stockfish:   {stockfish_path or 'disabled (result-only)'}")
    if stockfish_path:
        print(f"SF config:   depth={sf_depth} threads={sf_threads} hash={sf_hash_mb}MB retries={sf_restart_retries}")
    if seed is not None:
        print(f"Seed:        {seed}")
    print()

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    sf: chess.engine.SimpleEngine | None = None
    if stockfish_path:
        sf = start_stockfish(stockfish_path, sf_threads, sf_hash_mb)

    # Pre-parse openings; duplicate each for both colours
    openings = [parse_opening(line) for line in OPENING_LINES]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_positions = 0
    start = time.time()

    with open(output, 'w') as out_f:
        for game_idx in range(num_games):
            opening = random.choice(openings)
            _, positions, result_str = play_game(
                engine, opening, movetime_ms, depth=depth
            )

            result_val = result_to_float(result_str)
            if result_val is None:
                continue  # skip adjudicated / unknown result

            # Filter positions
            candidates: list[tuple[str, float]] = []
            for move_num, (fen, in_check, last_was_capture) in enumerate(positions):
                # Skip too early or too late
                if move_num < min_move or move_num > max_move:
                    continue
                # Skip captures/checks — noisy for tuning
                if in_check or last_was_capture:
                    continue
                # Skip positions with too few pieces
                board_tmp = chess.Board(fen)
                if chess.popcount(board_tmp.occupied) < min_pieces:
                    continue
                candidates.append((fen, result_val))

            # Sample to limit positions per game
            if len(candidates) > positions_per_game:
                candidates = random.sample(candidates, positions_per_game)

            # Annotate with Stockfish if requested
            if sf and candidates:
                fens_only = [fen for fen, _ in candidates]
                cp_scores = [0] * len(candidates)
                annotation_ok = False
                for attempt in range(sf_restart_retries + 1):
                    try:
                        cp_scores = annotate_with_stockfish(fens_only, sf, sf_depth)
                        annotation_ok = True
                        break
                    except Exception as e:
                        print(
                            f"\n[warn] SF annotation failed (attempt {attempt + 1}/{sf_restart_retries + 1}): {e}",
                            file=sys.stderr,
                        )
                        safe_quit_engine(sf)
                        sf = None

                        if not stockfish_path or attempt >= sf_restart_retries:
                            break
                        try:
                            sf = start_stockfish(stockfish_path, sf_threads, sf_hash_mb)
                            print("[info] Stockfish restarted after failure.", file=sys.stderr)
                        except Exception as restart_err:
                            print(f"[warn] SF restart failed: {restart_err}", file=sys.stderr)
                            sf = None
                            break

                if not annotation_ok and sf is None:
                    print(
                        "[warn] Disabling Stockfish annotation for remainder of this shard.",
                        file=sys.stderr,
                    )
            else:
                cp_scores = [0] * len(candidates)

            for (fen, result_val), cp in zip(candidates, cp_scores):
                out_f.write(f"{fen} | {cp} | {result_val}\n")
                total_positions += 1

            if (game_idx + 1) % 100 == 0:
                elapsed = time.time() - start
                rate = (game_idx + 1) / elapsed
                eta = (num_games - game_idx - 1) / rate
                print(
                    f"  [{game_idx + 1:5d}/{num_games}] "
                    f"{total_positions:,} positions | "
                    f"{rate:.1f} games/s | "
                    f"ETA {eta/60:.1f}min",
                    end='\r',
                    flush=True,
                )

    print(f"\n\nDone. {total_positions:,} positions in {time.time()-start:.0f}s → {output_path}")

    safe_quit_engine(engine)
    safe_quit_engine(sf)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate Texel tuning dataset from sykora self-play'
    )
    parser.add_argument('--engine', default='zig-out/bin/sykora')
    parser.add_argument('--output', default='datasets/texel_train.txt')
    parser.add_argument('--games', type=int, default=5000)
    parser.add_argument('--movetime-ms', type=int, default=50,
                        help='Time per move in milliseconds (ignored if --depth set)')
    parser.add_argument('--depth', type=int, default=None,
                        help='Fixed search depth per move (much faster than movetime)')
    parser.add_argument('--stockfish', default=None,
                        help='Path to Stockfish (for eval annotation)')
    parser.add_argument('--sf-depth', type=int, default=8,
                        help='Stockfish analysis depth per position')
    parser.add_argument('--sf-threads', type=int, default=1,
                        help='Stockfish Threads option (default: 1)')
    parser.add_argument('--sf-hash-mb', type=int, default=32,
                        help='Stockfish Hash option in MB (default: 32)')
    parser.add_argument('--sf-restart-retries', type=int, default=5,
                        help='How many times to restart Stockfish after annotation failure')
    parser.add_argument('--min-move', type=int, default=8,
                        help='Skip positions before this half-move number')
    parser.add_argument('--max-move', type=int, default=80,
                        help='Skip positions after this half-move number')
    parser.add_argument('--min-pieces', type=int, default=10,
                        help='Skip positions with fewer than this many pieces')
    parser.add_argument('--positions-per-game', type=int, default=20,
                        help='Max positions sampled per game')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional RNG seed for reproducible sampling/opening choice')
    args = parser.parse_args()

    generate_dataset(
        engine_path=args.engine,
        output_path=args.output,
        num_games=args.games,
        movetime_ms=args.movetime_ms,
        depth=args.depth,
        stockfish_path=args.stockfish,
        sf_depth=args.sf_depth,
        sf_threads=args.sf_threads,
        sf_hash_mb=args.sf_hash_mb,
        sf_restart_retries=args.sf_restart_retries,
        min_move=args.min_move,
        max_move=args.max_move,
        min_pieces=args.min_pieces,
        positions_per_game=args.positions_per_game,
        seed=args.seed,
    )



if __name__ == '__main__':
    main()
