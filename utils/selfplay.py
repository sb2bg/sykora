#!/usr/bin/env python3
"""
Self-play script to compare two versions of the Sykora chess engine.

This script plays matches between an old version and a new version of the engine,
tracking results and saving games as PGN files.

Usage:
    python selfplay.py <old_engine_path> <new_engine_path> [options]

Examples:
    # Play 10 games with 1 second per move
    python selfplay.py ./old_sykora ./new_sykora

    # Play 20 games with 0.5 seconds per move
    python selfplay.py ./old_sykora ./new_sykora --games 20 --time 0.5

    # Compare current build against a git commit
    # First, build the old version:
    #   git stash && git checkout <old_commit> && zig build -Doptimize=ReleaseFast
    #   cp zig-out/bin/sykora ./old_sykora
    #   git checkout - && git stash pop && zig build -Doptimize=ReleaseFast
    # Then run:
    python selfplay.py ./old_sykora ./zig-out/bin/sykora
"""

import argparse
import chess
import chess.engine
import chess.pgn
import datetime
import os
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class MatchResult:
    """Stores the result of a match between two engines."""
    engine1_wins: float = 0
    engine2_wins: float = 0
    draws: float = 0
    
    @property
    def total_games(self) -> int:
        return int(self.engine1_wins + self.engine2_wins + self.draws)
    
    @property
    def engine1_score(self) -> float:
        return self.engine1_wins + (self.draws * 0.5)
    
    @property
    def engine2_score(self) -> float:
        return self.engine2_wins + (self.draws * 0.5)


def play_game(
    white_engine: chess.engine.SimpleEngine,
    black_engine: chess.engine.SimpleEngine,
    white_name: str,
    black_name: str,
    time_limit: float,
    depth_limit: Optional[int] = None,
    verbose: bool = True,
) -> tuple[chess.pgn.Game, Optional[bool]]:
    """
    Play a single game between two engines.
    
    Returns:
        A tuple of (pgn_game, winner) where winner is True for white, False for black, None for draw.
    """
    board = chess.Board()
    game = chess.pgn.Game()
    
    # Set up game headers
    game.headers["Event"] = "Engine Self-Play Match"
    game.headers["Site"] = "Localhost"
    game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    
    node = game
    move_count = 0
    
    # Build the limit object
    if depth_limit:
        limit = chess.engine.Limit(time=time_limit, depth=depth_limit)
    else:
        limit = chess.engine.Limit(time=time_limit)
    
    while not board.is_game_over():
        current_engine = white_engine if board.turn == chess.WHITE else black_engine
        
        try:
            result = current_engine.play(board, limit)
            if result.move is None:
                if verbose:
                    print("  Engine returned no move, ending game")
                break
            board.push(result.move)
            node = node.add_variation(result.move)
            move_count += 1
        except chess.engine.EngineTerminatedError:
            if verbose:
                color = "White" if board.turn == chess.WHITE else "Black"
                print(f"  {color} engine crashed!")
            # The crashed engine loses
            game.headers["Result"] = "0-1" if board.turn == chess.WHITE else "1-0"
            game.headers["Termination"] = "Engine crash"
            return game, board.turn == chess.BLACK
        except Exception as e:
            if verbose:
                print(f"  Error during game: {e}")
            break
    
    outcome = board.outcome()
    game.headers["Result"] = board.result()
    
    if outcome:
        if outcome.termination == chess.Termination.CHECKMATE:
            game.headers["Termination"] = "Checkmate"
        elif outcome.termination == chess.Termination.STALEMATE:
            game.headers["Termination"] = "Stalemate"
        elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
            game.headers["Termination"] = "Insufficient material"
        elif outcome.termination == chess.Termination.FIFTY_MOVES:
            game.headers["Termination"] = "50-move rule"
        elif outcome.termination == chess.Termination.THREEFOLD_REPETITION:
            game.headers["Termination"] = "Threefold repetition"
        else:
            game.headers["Termination"] = "Normal"
    
    if verbose:
        print(f"  Moves: {move_count}, Result: {board.result()}")
    
    return game, outcome.winner if outcome else None


def play_match(
    engine1_path: str,
    engine2_path: str,
    engine1_name: str = "Old",
    engine2_name: str = "New",
    num_games: int = 10,
    time_limit: float = 1.0,
    depth_limit: Optional[int] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> MatchResult:
    """
    Play a match between two engines, alternating colors.
    
    Args:
        engine1_path: Path to the first engine executable
        engine2_path: Path to the second engine executable
        engine1_name: Display name for engine 1
        engine2_name: Display name for engine 2
        num_games: Number of games to play
        time_limit: Time limit per move in seconds
        depth_limit: Optional depth limit for search
        output_dir: Directory to save PGN files (None to skip saving)
        verbose: Whether to print progress
    
    Returns:
        MatchResult with the scores
    """
    result = MatchResult()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Self-Play Match: {engine1_name} vs {engine2_name}")
        print(f"Games: {num_games}, Time per move: {time_limit}s")
        if depth_limit:
            print(f"Depth limit: {depth_limit}")
        print(f"{'='*60}\n")
    
    try:
        engine1 = chess.engine.SimpleEngine.popen_uci(engine1_path)
        engine2 = chess.engine.SimpleEngine.popen_uci(engine2_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find engine - {e}")
        print("Make sure both engine paths are correct and the files are executable.")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting engines: {e}")
        sys.exit(1)
    
    try:
        for game_num in range(1, num_games + 1):
            # Alternate colors - engine1 plays white on odd games
            if game_num % 2 == 1:
                white_engine, black_engine = engine1, engine2
                white_name, black_name = engine1_name, engine2_name
            else:
                white_engine, black_engine = engine2, engine1
                white_name, black_name = engine2_name, engine1_name
            
            if verbose:
                print(f"Game {game_num}/{num_games}: {white_name} (White) vs {black_name} (Black)")
            
            game, winner = play_game(
                white_engine=white_engine,
                black_engine=black_engine,
                white_name=white_name,
                black_name=black_name,
                time_limit=time_limit,
                depth_limit=depth_limit,
                verbose=verbose,
            )
            game.headers["Round"] = str(game_num)
            
            # Track results
            if winner is None:
                result.draws += 1
            elif winner == chess.WHITE:
                if white_engine == engine1:
                    result.engine1_wins += 1
                else:
                    result.engine2_wins += 1
            else:  # Black won
                if black_engine == engine1:
                    result.engine1_wins += 1
                else:
                    result.engine2_wins += 1
            
            # Save PGN if output directory specified
            if output_dir:
                pgn_path = os.path.join(output_dir, f"game_{game_num}.pgn")
                with open(pgn_path, "w") as f:
                    print(game, file=f, end="\n\n")
            
            # Print running score
            if verbose:
                print(f"  Running score: {engine1_name} {result.engine1_score} - {result.engine2_score} {engine2_name}")
                print()
    
    finally:
        engine1.quit()
        engine2.quit()
    
    return result


def print_results(result: MatchResult, engine1_name: str, engine2_name: str):
    """Print final match results."""
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\n{engine1_name}:")
    print(f"  Wins:   {int(result.engine1_wins)}")
    print(f"  Losses: {int(result.engine2_wins)}")
    print(f"  Draws:  {int(result.draws)}")
    print(f"  Score:  {result.engine1_score} / {result.total_games}")
    print(f"  Win %:  {result.engine1_score / result.total_games * 100:.1f}%")
    
    print(f"\n{engine2_name}:")
    print(f"  Wins:   {int(result.engine2_wins)}")
    print(f"  Losses: {int(result.engine1_wins)}")
    print(f"  Draws:  {int(result.draws)}")
    print(f"  Score:  {result.engine2_score} / {result.total_games}")
    print(f"  Win %:  {result.engine2_score / result.total_games * 100:.1f}%")
    
    # Determine winner
    print("\n" + "-" * 60)
    if result.engine1_score > result.engine2_score:
        margin = result.engine1_score - result.engine2_score
        print(f"🏆 {engine1_name} wins by +{margin:.1f}!")
    elif result.engine2_score > result.engine1_score:
        margin = result.engine2_score - result.engine1_score
        print(f"🏆 {engine2_name} wins by +{margin:.1f}!")
    else:
        print("🤝 Match drawn!")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Play matches between two versions of a chess engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./old_sykora ./new_sykora
  %(prog)s ./old_sykora ./zig-out/bin/sykora --games 20 --time 0.5
  %(prog)s engine_v1 engine_v2 --output selfplay_games --depth 8
        """,
    )
    
    parser.add_argument(
        "engine1",
        help="Path to the first (old) engine executable",
    )
    parser.add_argument(
        "engine2",
        help="Path to the second (new) engine executable",
    )
    parser.add_argument(
        "--name1",
        default="Old",
        help="Display name for the first engine (default: Old)",
    )
    parser.add_argument(
        "--name2",
        default="New",
        help="Display name for the second engine (default: New)",
    )
    parser.add_argument(
        "-g", "--games",
        type=int,
        default=10,
        help="Number of games to play (default: 10)",
    )
    parser.add_argument(
        "-t", "--time",
        type=float,
        default=1.0,
        help="Time limit per move in seconds (default: 1.0)",
    )
    parser.add_argument(
        "-d", "--depth",
        type=int,
        default=None,
        help="Optional depth limit for search",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Directory to save PGN files (default: no saving)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    args = parser.parse_args()
    
    # Validate engine paths
    if not os.path.isfile(args.engine1):
        print(f"Error: Engine 1 not found: {args.engine1}")
        sys.exit(1)
    if not os.path.isfile(args.engine2):
        print(f"Error: Engine 2 not found: {args.engine2}")
        sys.exit(1)
    
    # Run the match
    result = play_match(
        engine1_path=args.engine1,
        engine2_path=args.engine2,
        engine1_name=args.name1,
        engine2_name=args.name2,
        num_games=args.games,
        time_limit=args.time,
        depth_limit=args.depth,
        output_dir=args.output,
        verbose=not args.quiet,
    )
    
    # Print final results
    print_results(result, args.name1, args.name2)
    
    # Exit with code based on whether new engine won
    if result.engine2_score > result.engine1_score:
        sys.exit(0)  # New engine won
    elif result.engine1_score > result.engine2_score:
        sys.exit(1)  # Old engine won
    else:
        sys.exit(2)  # Draw


if __name__ == "__main__":
    main()
