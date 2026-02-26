#!/usr/bin/env python3
"""Self-play test: new engine vs old engine."""

import argparse
import chess
import chess.engine
import chess.pgn
import datetime
import io
import math
import os
import random
import sys
import time


def play_game(engine_w, engine_b, time_limit, opening_moves=None):
    """Play a single game, return (result, pgn_game)."""
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "new"
    game.headers["Black"] = "old"
    game.headers["Date"] = datetime.date.today().isoformat()

    # Play opening moves if provided
    node = game
    if opening_moves:
        for move_uci in opening_moves:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
                node = node.add_variation(move)
            else:
                break

    while not board.is_game_over(claim_draw=True):
        if board.fullmove_number > 200:
            game.headers["Result"] = "1/2-1/2"
            return 0.5, game

        engine = engine_w if board.turn == chess.WHITE else engine_b
        try:
            result = engine.play(board, chess.engine.Limit(time=time_limit))
            move = result.move
        except Exception as e:
            print(f"  Engine error: {e}", file=sys.stderr)
            # Engine that errors loses
            if board.turn == chess.WHITE:
                game.headers["Result"] = "0-1"
                return 0.0, game
            else:
                game.headers["Result"] = "1-0"
                return 1.0, game

        board.push(move)
        node = node.add_variation(move)

    result = board.result(claim_draw=True)
    game.headers["Result"] = result
    if result == "1-0":
        return 1.0, game
    elif result == "0-1":
        return 0.0, game
    else:
        return 0.5, game


# Common openings for variety (first few moves as UCI)
OPENINGS = [
    [],  # startpos
    ["e2e4", "e7e5"],  # open game
    ["d2d4", "d7d5"],  # closed game
    ["e2e4", "c7c5"],  # sicilian
    ["d2d4", "g8f6", "c2c4", "e7e6"],  # nimzo/QID
    ["e2e4", "e7e5", "g1f3", "b8c6"],  # four knights area
    ["d2d4", "d7d5", "c2c4"],  # QGD
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],  # ruy lopez
    ["g1f3", "d7d5", "g2g3"],  # reti
    ["e2e4", "c7c6"],  # caro-kann
    ["e2e4", "d7d6"],  # pirc
    ["d2d4", "g8f6", "c2c4", "g7g6"],  # KID
]


def elo_diff(wins, draws, losses):
    """Estimate Elo difference from match results."""
    total = wins + draws + losses
    if total == 0:
        return 0.0, 0.0, 0.0
    score = (wins + 0.5 * draws) / total
    if score <= 0.0 or score >= 1.0:
        return float("inf") if score >= 1.0 else float("-inf"), 0, 0
    elo = -400 * math.log10(1.0 / score - 1.0)
    # Standard error
    w_pct = wins / total
    d_pct = draws / total
    l_pct = losses / total
    var = w_pct * (1 - score) ** 2 + d_pct * (0.5 - score) ** 2 + l_pct * (0 - score) ** 2
    se = math.sqrt(var / total) if total > 0 else 0
    elo_se = 400 * se / (math.log(10) * score * (1 - score)) if 0 < score < 1 else 0
    return elo, elo_se * 1.96, score


def main():
    parser = argparse.ArgumentParser(description="Self-play: new vs old engine")
    parser.add_argument("--new", default="./zig-out/bin/sykora", help="Path to new engine")
    parser.add_argument("--old", default="/tmp/sykora-old", help="Path to old engine")
    parser.add_argument("--games", type=int, default=100, help="Number of game pairs (each pair = 2 games with color swap)")
    parser.add_argument("--time", type=float, default=0.3, help="Seconds per move")
    parser.add_argument("--pgn", default="selfplay.pgn", help="PGN output file")
    args = parser.parse_args()

    num_pairs = args.games
    total_games = num_pairs * 2

    print(f"Self-play: {total_games} games ({num_pairs} pairs)")
    print(f"  New: {args.new}")
    print(f"  Old: {args.old}")
    print(f"  Time: {args.time}s/move")
    print()

    engine_new = chess.engine.SimpleEngine.popen_uci(args.new)
    engine_old = chess.engine.SimpleEngine.popen_uci(args.old)

    new_wins = 0
    new_draws = 0
    new_losses = 0
    pgn_games = []

    start_time = time.time()

    for pair_idx in range(num_pairs):
        opening = random.choice(OPENINGS)

        # Game 1: new=White, old=Black
        score1, game1 = play_game(engine_new, engine_old, args.time, opening)
        game1.headers["White"] = "sykora-new"
        game1.headers["Black"] = "sykora-old"
        game1.headers["Round"] = str(pair_idx * 2 + 1)
        pgn_games.append(game1)

        if score1 == 1.0:
            new_wins += 1
        elif score1 == 0.0:
            new_losses += 1
        else:
            new_draws += 1

        # Game 2: old=White, new=Black
        score2, game2 = play_game(engine_old, engine_new, args.time, opening)
        game2.headers["White"] = "sykora-old"
        game2.headers["Black"] = "sykora-new"
        game2.headers["Round"] = str(pair_idx * 2 + 2)
        pgn_games.append(game2)

        new_score2 = 1.0 - score2  # invert since new is Black
        if new_score2 == 1.0:
            new_wins += 1
        elif new_score2 == 0.0:
            new_losses += 1
        else:
            new_draws += 1

        games_played = (pair_idx + 1) * 2
        total_score = new_wins + 0.5 * new_draws
        elo, ci, pct = elo_diff(new_wins, new_draws, new_losses)

        elapsed = time.time() - start_time
        rate = games_played / elapsed if elapsed > 0 else 0

        print(f"[{games_played}/{total_games}] new: W{new_wins} D{new_draws} L{new_losses} "
              f"({total_score:.1f}/{games_played}) "
              f"Elo: {elo:+.1f} ±{ci:.1f} "
              f"({rate:.1f} games/min)")

        # Save PGN periodically
        if (pair_idx + 1) % 10 == 0 or pair_idx == num_pairs - 1:
            with open(args.pgn, "w") as f:
                for g in pgn_games:
                    print(g, file=f)
                    print(file=f)

    engine_new.quit()
    engine_old.quit()

    elapsed = time.time() - start_time
    print(f"\nFinal: W{new_wins} D{new_draws} L{new_losses} ({new_wins + 0.5*new_draws:.1f}/{total_games})")
    elo, ci, pct = elo_diff(new_wins, new_draws, new_losses)
    print(f"Elo: {elo:+.1f} ±{ci:.1f} ({pct*100:.1f}%)")
    print(f"Time: {elapsed:.0f}s ({total_games*60/elapsed:.1f} games/min)")


if __name__ == "__main__":
    main()
