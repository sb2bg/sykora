#!/usr/bin/env python3
"""Head-to-head self-play runner for engine regression testing.

This script compares two UCI engines with:
- color-balanced matches
- opening diversification (built-in suite or custom file)
- optional UCI options per engine
- Elo-difference estimate with confidence interval

Examples:
  python utils/selfplay.py ./old_sykora ./zig-out/bin/sykora --games 80 --movetime-ms 200
  python utils/selfplay.py ./old_sykora ./zig-out/bin/sykora --games 120 --depth 8 --openings none
  python utils/selfplay.py ./old_sykora ./zig-out/bin/sykora --games 200 --openings openings.txt --shuffle-openings
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import chess
import chess.engine
import chess.pgn

# 4-ply opening lines in UCI notation.
# Matches are played in pairs: same opening twice with colors swapped.
DEFAULT_OPENINGS_UCI: List[str] = [
    "e2e4 e7e5 g1f3 b8c6",
    "e2e4 c7c5 g1f3 d7d6",
    "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5",
    "d2d4 d7d5 c2c4 e7e6",
    "d2d4 g8f6 c2c4 e7e6",
    "c2c4 e7e5 b1c3 g8f6",
    "g1f3 d7d5 d2d4 g8f6",
    "g1f3 g8f6 c2c4 c7c5",
    "b1c3 d7d5 e2e4 e7e6",
    "e2e4 e7e5 f1c4 g8f6",
    "e2e4 c7c5 b1c3 b8c6",
    "d2d4 g8f6 c1g5 d7d5",
    "c2c4 g8f6 b1c3 e7e6",
    "e2e4 d7d6 d2d4 g8f6",
    "d2d4 d7d6 c2c4 g8f6",
    "e2e4 g8f6 e4e5 f6d5",
    "d2d4 f7f5 g2g3 g8f6",
    "c2c4 c7c6 b1c3 d7d5",
    "g1f3 c7c5 c2c4 g8f6",
]


@dataclass
class MatchResult:
    engine1_wins: int = 0
    engine2_wins: int = 0
    draws: int = 0

    @property
    def total_games(self) -> int:
        return self.engine1_wins + self.engine2_wins + self.draws

    @property
    def engine1_score(self) -> float:
        return self.engine1_wins + 0.5 * self.draws

    @property
    def engine2_score(self) -> float:
        return self.engine2_wins + 0.5 * self.draws


@dataclass
class EloEstimate:
    score_rate: float
    elo: float
    elo_lo: float
    elo_hi: float
    p_value_two_sided: float


def parse_option_value(raw: str):
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def parse_uci_options(values: Iterable[str]) -> Dict[str, object]:
    opts: Dict[str, object] = {}
    for entry in values:
        if "=" not in entry:
            raise ValueError(f"Invalid --engine*-opt '{entry}' (expected Key=Value)")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid option key in '{entry}'")
        opts[key] = parse_option_value(raw_value.strip())
    return opts


def parse_opening_line(raw: str) -> List[str]:
    text = raw.strip()
    if not text:
        return []
    if text.startswith("startpos"):
        tokens = text.split()
        if len(tokens) >= 3 and tokens[1] == "moves":
            return tokens[2:]
        return []
    return text.split()


def validate_opening_line(moves: List[str]) -> bool:
    board = chess.Board()
    try:
        for move in moves:
            board.push_uci(move)
    except ValueError:
        return False
    return True


def load_openings(spec: str, shuffle: bool, seed: int) -> List[List[str]]:
    if spec == "none":
        openings = [[]]
    elif spec == "default":
        openings = [parse_opening_line(line) for line in DEFAULT_OPENINGS_UCI]
    else:
        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"Openings file not found: {path}")
        openings = []
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            moves = parse_opening_line(line)
            if moves:
                openings.append(moves)

    valid_openings: List[List[str]] = []
    for moves in openings:
        if validate_opening_line(moves):
            valid_openings.append(moves)

    if not valid_openings:
        raise ValueError("No valid opening lines available")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(valid_openings)

    return valid_openings


def build_limit(movetime_ms: int, depth: Optional[int]) -> chess.engine.Limit:
    if depth is not None:
        return chess.engine.Limit(time=movetime_ms / 1000.0, depth=depth)
    return chess.engine.Limit(time=movetime_ms / 1000.0)


def configure_engine(
    engine: chess.engine.SimpleEngine,
    options: Dict[str, object],
    verbose: bool,
) -> None:
    if not options:
        return

    accepted: Dict[str, object] = {}
    for name, value in options.items():
        if name in engine.options:
            accepted[name] = value
        elif verbose:
            print(f"Warning: engine does not expose option '{name}', skipping")

    if accepted:
        engine.configure(accepted)


def apply_opening_to_game(game: chess.pgn.Game, board: chess.Board, opening_moves: List[str]) -> chess.pgn.ChildNode:
    node: chess.pgn.Game | chess.pgn.ChildNode = game
    for move in opening_moves:
        chess_move = chess.Move.from_uci(move)
        board.push(chess_move)
        node = node.add_variation(chess_move)
    return node


def play_single_game(
    white_engine: chess.engine.SimpleEngine,
    black_engine: chess.engine.SimpleEngine,
    white_name: str,
    black_name: str,
    limit: chess.engine.Limit,
    opening_moves: List[str],
    round_number: int,
    max_plies: int,
    verbose: bool,
) -> tuple[chess.pgn.Game, Optional[chess.Color], str]:
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Engine Self-Play Match"
    game.headers["Site"] = "Localhost"
    game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
    game.headers["Round"] = str(round_number)
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    if opening_moves:
        game.headers["OpeningLine"] = " ".join(opening_moves)

    node = apply_opening_to_game(game, board, opening_moves)

    plies_played = 0
    termination = "Normal"

    while not board.is_game_over(claim_draw=True):
        if plies_played >= max_plies:
            termination = f"Move limit ({max_plies} plies)"
            game.headers["Result"] = "1/2-1/2"
            game.headers["Termination"] = termination
            if verbose:
                print(f"  Terminated as draw at move limit ({max_plies} plies)")
            return game, None, termination

        engine = white_engine if board.turn == chess.WHITE else black_engine
        side = "White" if board.turn == chess.WHITE else "Black"

        try:
            play_result = engine.play(board, limit)
        except chess.engine.EngineTerminatedError:
            termination = f"{side} engine crashed"
            winner = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
            game.headers["Result"] = "0-1" if winner == chess.BLACK else "1-0"
            game.headers["Termination"] = termination
            if verbose:
                print(f"  {termination}")
            return game, winner, termination
        except Exception as exc:
            termination = f"Engine error: {exc}"
            winner = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
            game.headers["Result"] = "0-1" if winner == chess.BLACK else "1-0"
            game.headers["Termination"] = termination
            if verbose:
                print(f"  {termination}")
            return game, winner, termination

        if play_result.move is None:
            termination = f"{side} returned no move"
            winner = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
            game.headers["Result"] = "0-1" if winner == chess.BLACK else "1-0"
            game.headers["Termination"] = termination
            if verbose:
                print(f"  {termination}")
            return game, winner, termination

        board.push(play_result.move)
        node = node.add_variation(play_result.move)
        plies_played += 1

    outcome = board.outcome(claim_draw=True)
    result_text = board.result(claim_draw=True)
    game.headers["Result"] = result_text

    if outcome is None:
        winner: Optional[chess.Color] = None
        termination = "Unknown"
    else:
        winner = outcome.winner
        termination = str(outcome.termination).replace("Termination.", "")
    game.headers["Termination"] = termination

    if verbose:
        print(f"  Plies: {plies_played}, Result: {result_text}, Termination: {termination}")

    return game, winner, termination


def score_to_elo(score_rate: float) -> float:
    eps = 1e-6
    p = min(max(score_rate, eps), 1.0 - eps)
    return -400.0 * math.log10((1.0 - p) / p)


def estimate_elo(result: MatchResult) -> EloEstimate:
    n = result.total_games
    if n <= 0:
        return EloEstimate(
            score_rate=0.5,
            elo=0.0,
            elo_lo=0.0,
            elo_hi=0.0,
            p_value_two_sided=1.0,
        )

    p = result.engine2_score / n
    elo = score_to_elo(p)

    se = math.sqrt(max(1e-9, p * (1.0 - p) / n))
    p_lo = min(max(p - 1.96 * se, 1e-6), 1.0 - 1e-6)
    p_hi = min(max(p + 1.96 * se, 1e-6), 1.0 - 1e-6)

    elo_lo = score_to_elo(p_lo)
    elo_hi = score_to_elo(p_hi)

    # Two-sided p-value against H0: equal strength (p = 0.5).
    sigma0 = math.sqrt(0.25 / n)
    z = (p - 0.5) / sigma0
    p_value = math.erfc(abs(z) / math.sqrt(2.0))

    return EloEstimate(
        score_rate=p,
        elo=elo,
        elo_lo=elo_lo,
        elo_hi=elo_hi,
        p_value_two_sided=p_value,
    )


def print_summary(result: MatchResult, name1: str, name2: str) -> None:
    elo = estimate_elo(result)

    print("\n" + "=" * 68)
    print("MATCH SUMMARY")
    print("=" * 68)
    print(f"{name1}: W {result.engine1_wins}  D {result.draws}  L {result.engine2_wins}  Score {result.engine1_score}/{result.total_games}")
    print(f"{name2}: W {result.engine2_wins}  D {result.draws}  L {result.engine1_wins}  Score {result.engine2_score}/{result.total_games}")
    print("-" * 68)
    print(f"Score rate ({name2}): {elo.score_rate * 100:.2f}%")
    print(f"Estimated Elo ({name2} - {name1}): {elo.elo:+.1f}")
    print(f"95% CI: [{elo.elo_lo:+.1f}, {elo.elo_hi:+.1f}]")
    print(f"p-value vs equal strength: {elo.p_value_two_sided:.4f}")

    if elo.p_value_two_sided < 0.05:
        if result.engine2_score > result.engine1_score:
            print(f"Verdict: {name2} is stronger in this sample.")
        elif result.engine1_score > result.engine2_score:
            print(f"Verdict: {name1} is stronger in this sample.")
        else:
            print("Verdict: statistically significant but tied score (rare case).")
    else:
        print("Verdict: inconclusive at 95% confidence; run more games.")
    print("=" * 68 + "\n")


def make_summary(
    result: MatchResult,
    name1: str,
    name2: str,
    args: argparse.Namespace,
    engine1_path: str,
    engine2_path: str,
) -> dict:
    elo = estimate_elo(result)
    now = datetime.datetime.now(datetime.UTC).isoformat()
    return {
        "generated_at_utc": now,
        "engine1": {
            "name": name1,
            "path": engine1_path,
        },
        "engine2": {
            "name": name2,
            "path": engine2_path,
        },
        "settings": {
            "games": args.games,
            "movetime_ms": args.movetime_ms,
            "depth": args.depth,
            "openings": args.openings,
            "shuffle_openings": bool(args.shuffle_openings),
            "seed": args.seed,
            "max_plies": args.max_plies,
            "threads": args.threads,
            "hash_mb": args.hash_mb,
            "engine1_opt": list(args.engine1_opt),
            "engine2_opt": list(args.engine2_opt),
        },
        "result": {
            "engine1_wins": result.engine1_wins,
            "engine2_wins": result.engine2_wins,
            "draws": result.draws,
            "total_games": result.total_games,
            "engine1_score": result.engine1_score,
            "engine2_score": result.engine2_score,
        },
        "elo": {
            "score_rate_engine2": elo.score_rate,
            "elo_engine2_minus_engine1": elo.elo,
            "elo_95ci_low": elo.elo_lo,
            "elo_95ci_high": elo.elo_hi,
            "p_value_two_sided": elo.p_value_two_sided,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run color-balanced self-play matches between two UCI engines.",
    )

    parser.add_argument("engine1", help="Path to baseline engine binary")
    parser.add_argument("engine2", help="Path to candidate engine binary")

    parser.add_argument("--name1", default="Baseline", help="Label for engine1")
    parser.add_argument("--name2", default="Candidate", help="Label for engine2")

    parser.add_argument("-g", "--games", type=int, default=80, help="Total games (default: 80)")

    parser.add_argument(
        "--movetime-ms",
        type=int,
        default=200,
        help="Per-move time in milliseconds (default: 200)",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=None,
        help="Optional fixed depth limit",
    )

    parser.add_argument(
        "--openings",
        default="default",
        help="Opening source: default, none, or path to file (default: default)",
    )
    parser.add_argument(
        "--shuffle-openings",
        action="store_true",
        help="Shuffle opening order before running",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for opening shuffle")

    parser.add_argument("--max-plies", type=int, default=300, help="Adjudicate draw after this many plies")

    parser.add_argument("--threads", type=int, default=None, help="Set Threads for both engines")
    parser.add_argument("--hash-mb", type=int, default=None, help="Set Hash (MB) for both engines")
    parser.add_argument(
        "--engine1-opt",
        action="append",
        default=[],
        help="Extra UCI option for engine1, format Key=Value (repeatable)",
    )
    parser.add_argument(
        "--engine2-opt",
        action="append",
        default=[],
        help="Extra UCI option for engine2, format Key=Value (repeatable)",
    )

    parser.add_argument("--output-dir", default=None, help="Directory for per-game PGNs")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write machine-readable match summary JSON",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce output")

    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be > 0")
    if args.movetime_ms <= 0:
        parser.error("--movetime-ms must be > 0")
    if args.depth is not None and args.depth <= 0:
        parser.error("--depth must be > 0")
    if args.max_plies <= 0:
        parser.error("--max-plies must be > 0")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be > 0")
    if args.hash_mb is not None and args.hash_mb <= 0:
        parser.error("--hash-mb must be > 0")

    if not os.path.isfile(args.engine1):
        parser.error(f"engine1 not found: {args.engine1}")
    if not os.path.isfile(args.engine2):
        parser.error(f"engine2 not found: {args.engine2}")

    return args


def run_match(args: argparse.Namespace) -> MatchResult:
    verbose = not args.quiet
    openings = load_openings(args.openings, args.shuffle_openings, args.seed)
    limit = build_limit(args.movetime_ms, args.depth)

    engine1_opts = parse_uci_options(args.engine1_opt)
    engine2_opts = parse_uci_options(args.engine2_opt)

    if args.threads is not None:
        engine1_opts.setdefault("Threads", args.threads)
        engine2_opts.setdefault("Threads", args.threads)
    if args.hash_mb is not None:
        engine1_opts.setdefault("Hash", args.hash_mb)
        engine2_opts.setdefault("Hash", args.hash_mb)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 68)
        print(f"Self-play: {args.name1} vs {args.name2}")
        print(f"Games: {args.games} | movetime: {args.movetime_ms}ms" + (f" | depth: {args.depth}" if args.depth else ""))
        print(f"Openings: {args.openings} ({len(openings)} lines)")
        print("Pairing: same opening twice, colors swapped")
        print("=" * 68)

    result = MatchResult()

    try:
        engine1 = chess.engine.SimpleEngine.popen_uci(args.engine1)
        engine2 = chess.engine.SimpleEngine.popen_uci(args.engine2)
    except Exception as exc:
        print(f"Failed to start engines: {exc}", file=sys.stderr)
        raise SystemExit(1)

    try:
        configure_engine(engine1, engine1_opts, verbose)
        configure_engine(engine2, engine2_opts, verbose)

        for game_index in range(args.games):
            round_number = game_index + 1

            opening_idx = (game_index // 2) % len(openings)
            opening_moves = openings[opening_idx]
            swap_colors = (game_index % 2) == 1

            if not swap_colors:
                white_engine, black_engine = engine1, engine2
                white_name, black_name = args.name1, args.name2
            else:
                white_engine, black_engine = engine2, engine1
                white_name, black_name = args.name2, args.name1

            if verbose:
                opening_text = " ".join(opening_moves) if opening_moves else "(none)"
                print(f"Game {round_number}/{args.games}: {white_name} (W) vs {black_name} (B)")
                print(f"  Opening: {opening_text}")

            game, winner, _termination = play_single_game(
                white_engine=white_engine,
                black_engine=black_engine,
                white_name=white_name,
                black_name=black_name,
                limit=limit,
                opening_moves=opening_moves,
                round_number=round_number,
                max_plies=args.max_plies,
                verbose=verbose,
            )

            if winner is None:
                result.draws += 1
            elif winner == chess.WHITE:
                if not swap_colors:
                    result.engine1_wins += 1
                else:
                    result.engine2_wins += 1
            else:
                if not swap_colors:
                    result.engine2_wins += 1
                else:
                    result.engine1_wins += 1

            if output_dir:
                (output_dir / f"game_{round_number:04d}.pgn").write_text(str(game) + "\n\n")

            if verbose:
                print(
                    f"  Running score: {args.name1} {result.engine1_score:.1f} - {result.engine2_score:.1f} {args.name2}"
                )
                print()

    finally:
        engine1.quit()
        engine2.quit()

    return result


def main() -> None:
    args = parse_args()
    result = run_match(args)
    summary = make_summary(
        result=result,
        name1=args.name1,
        name2=args.name2,
        args=args,
        engine1_path=args.engine1,
        engine2_path=args.engine2,
    )
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print_summary(result, args.name1, args.name2)

    # Exit code semantics for CI:
    # 0 -> candidate won, 1 -> baseline won, 2 -> exact tie
    if result.engine2_score > result.engine1_score:
        raise SystemExit(0)
    if result.engine1_score > result.engine2_score:
        raise SystemExit(1)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
