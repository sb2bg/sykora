import chess
import chess.engine
import chess.pgn
import sys
import os
import datetime

# Configuration
SYKORA_PATH = "./zig-out/bin/sykora"
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # Assumes stockfish is in your PATH
GAMES_TO_PLAY = 10
TIME_LIMIT = 0.5  # seconds per move (fast games)
PGN_DIR = "benchmark_games"


def play_match(target_elo):
    print(f"Starting match: Sykora vs Stockfish (Elo {target_elo})")

    # Create directory for PGNs
    elo_dir = os.path.join(PGN_DIR, str(target_elo))
    os.makedirs(elo_dir, exist_ok=True)

    try:
        sykora = chess.engine.SimpleEngine.popen_uci(SYKORA_PATH)
        stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure both engines are installed and paths are correct.")
        return

    # Configure Stockfish Elo
    try:
        # Check if options exist
        if (
            "UCI_LimitStrength" not in stockfish.options
            or "UCI_Elo" not in stockfish.options
        ):
            print("Warning: Stockfish does not support UCI_LimitStrength or UCI_Elo.")
            print("Available options:", list(stockfish.options.keys()))
        else:
            # Check Elo limits
            elo_opt = stockfish.options["UCI_Elo"]
            min_elo = elo_opt.min
            max_elo = elo_opt.max

            if target_elo < min_elo:
                print(
                    f"Notice: Target Elo {target_elo} is below Stockfish minimum {min_elo}. Using {min_elo}."
                )
                target_elo = min_elo

            stockfish.configure({"UCI_LimitStrength": True, "UCI_Elo": target_elo})
    except Exception as e:
        print(
            f"Warning: Could not set Stockfish Elo. It might be playing at full strength! Error: {e}"
        )

    sykora_score = 0

    for i in range(GAMES_TO_PLAY):
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = f"Benchmark Match Elo {target_elo}"
        game.headers["Site"] = "Localhost"
        game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
        game.headers["Round"] = str(i + 1)

        # Alternate colors
        if i % 2 == 0:
            white = sykora
            black = stockfish
            white_name = "Sykora"
            black_name = f"Stockfish ({target_elo})"
        else:
            white = stockfish
            black = sykora
            white_name = f"Stockfish ({target_elo})"
            black_name = "Sykora"

        game.headers["White"] = white_name
        game.headers["Black"] = black_name

        print(f"Game {i+1}: {white_name} (White) vs {black_name} (Black)")

        node = game
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                result = white.play(board, chess.engine.Limit(time=TIME_LIMIT))
            else:
                result = black.play(board, chess.engine.Limit(time=TIME_LIMIT))

            board.push(result.move)
            node = node.add_variation(result.move)

        outcome = board.outcome()
        winner = outcome.winner
        game.headers["Result"] = board.result()

        # Save PGN
        pgn_path = os.path.join(elo_dir, f"game_{i+1}.pgn")
        with open(pgn_path, "w") as f:
            print(game, file=f, end="\n\n")

        if winner == chess.WHITE:
            print(f"  Winner: {white_name}")
            if white == sykora:
                sykora_score += 1
        elif winner == chess.BLACK:
            print(f"  Winner: {black_name}")
            if black == sykora:
                sykora_score += 1
        else:
            print("  Draw")
            sykora_score += 0.5

    sykora.quit()
    stockfish.quit()

    print("\n===========================")
    print(f"Match Result vs Stockfish {target_elo}")
    print(f"Sykora Score: {sykora_score} / {GAMES_TO_PLAY}")
    print("===========================\n")

    return sykora_score


if __name__ == "__main__":
    # Try a few levels to bracket the rating
    print("Benchmarking Sykora...")

    # List of Elos to test against
    elos = [1320, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]

    for elo in elos:
        score = play_match(elo)

        # If we score less than 50%, we've likely found the ceiling
        if score < GAMES_TO_PLAY * 0.5:
            print(
                f"Sykora scored less than 50% against {elo}. Estimated rating is around {elo}."
            )
            break

        print(f"Passed level {elo}! Moving to next level...")
