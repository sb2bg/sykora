#!/usr/bin/env python3

import argparse
import datetime
import os
import sys
import time
from typing import List, Optional

import chess
import chess.engine
import chess.pgn

try:
    import tkinter as tk
except Exception:
    tk = None

# Configuration defaults
DEFAULT_SYKORA_PATH = "./zig-out/bin/sykora"
DEFAULT_STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
DEFAULT_GAMES_TO_PLAY = 10
DEFAULT_TIME_LIMIT = 0.5
DEFAULT_PGN_DIR = "benchmark_games"
DEFAULT_ELOS = [2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]

LIGHT_SQUARE_COLOR = "#f0d9b5"
DARK_SQUARE_COLOR = "#b58863"
LIGHT_TEXT_COLOR = "#111111"
DARK_TEXT_COLOR = "#ffffff"


def parse_elo_list(raw: str) -> List[int]:
    values: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid Elo value: {item!r}. Use comma-separated integers."
            ) from exc

    if not values:
        raise argparse.ArgumentTypeError("At least one Elo value is required.")

    return values


def render_live_board(
    board: chess.Board,
    game_number: int,
    total_games: int,
    target_elo: int,
    white_name: str,
    black_name: str,
    last_move_san: Optional[str] = None,
) -> None:
    if sys.stdout.isatty():
        if os.name == "nt":
            os.system("cls")
        else:
            print("\033[2J\033[H", end="")

    print(
        f"Live View - Game {game_number}/{total_games} - "
        f"Sykora vs Stockfish ({target_elo})"
    )
    print(f"White: {white_name} | Black: {black_name}")

    if last_move_san is not None:
        moved_color = "Black" if board.turn == chess.WHITE else "White"
        print(f"Last move ({moved_color}): {last_move_san}")

    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    print(f"Side to move: {side_to_move}")
    print(board.unicode(borders=True))
    print(flush=True)


class GuiBoardViewer:
    def __init__(self) -> None:
        if tk is None:
            raise RuntimeError("tkinter is unavailable in this Python environment.")

        try:
            self.root = tk.Tk()
        except Exception as exc:
            raise RuntimeError(f"could not create Tk window: {exc}") from exc

        self.root.title("Sykora Benchmark Viewer")
        self.root.configure(bg="#1c1c1c")
        self.closed = False

        self.header_var = tk.StringVar(value="Sykora Benchmark")
        self.players_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self.last_move_var = tk.StringVar(value="")

        top_frame = tk.Frame(self.root, bg="#1c1c1c", padx=12, pady=10)
        top_frame.pack(fill="x")

        tk.Label(
            top_frame,
            textvariable=self.header_var,
            font=("Helvetica", 18, "bold"),
            fg="#f5f5f5",
            bg="#1c1c1c",
        ).pack(anchor="w")
        tk.Label(
            top_frame,
            textvariable=self.players_var,
            font=("Helvetica", 12),
            fg="#d6d6d6",
            bg="#1c1c1c",
        ).pack(anchor="w")
        tk.Label(
            top_frame,
            textvariable=self.status_var,
            font=("Helvetica", 11),
            fg="#c4e2ff",
            bg="#1c1c1c",
        ).pack(anchor="w")
        tk.Label(
            top_frame,
            textvariable=self.last_move_var,
            font=("Helvetica", 11),
            fg="#ffe7a1",
            bg="#1c1c1c",
        ).pack(anchor="w")

        board_container = tk.Frame(self.root, bg="#1c1c1c", padx=12, pady=12)
        board_container.pack()

        self.squares: List[List[tk.Label]] = []
        for row in range(8):
            row_widgets: List[tk.Label] = []
            for col in range(8):
                is_light_square = (row + col) % 2 == 0
                square_color = (
                    LIGHT_SQUARE_COLOR if is_light_square else DARK_SQUARE_COLOR
                )
                text_color = LIGHT_TEXT_COLOR if is_light_square else DARK_TEXT_COLOR
                label = tk.Label(
                    board_container,
                    width=2,
                    height=1,
                    text="",
                    font=("Menlo", 34),
                    bg=square_color,
                    fg=text_color,
                )
                label.grid(row=row, column=col, sticky="nsew")
                row_widgets.append(label)
            self.squares.append(row_widgets)

        for idx in range(8):
            board_container.grid_rowconfigure(idx, weight=1)
            board_container.grid_columnconfigure(idx, weight=1)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.minsize(540, 700)
        self._pump_events()

    def _on_close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def _pump_events(self) -> None:
        if self.closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self.closed = True

    def render(
        self,
        board: chess.Board,
        game_number: int,
        total_games: int,
        target_elo: int,
        white_name: str,
        black_name: str,
        last_move_san: Optional[str] = None,
    ) -> None:
        if self.closed:
            return

        self.header_var.set(
            f"Sykora vs Stockfish {target_elo} | Game {game_number}/{total_games}"
        )
        self.players_var.set(f"White: {white_name}   Black: {black_name}")
        side_to_move = "White" if board.turn == chess.WHITE else "Black"
        self.status_var.set(f"Side to move: {side_to_move}")
        self.last_move_var.set(
            f"Last move: {last_move_san}"
            if last_move_san is not None
            else "Last move: (none)"
        )

        for rank in range(7, -1, -1):
            row = 7 - rank
            for file in range(8):
                piece = board.piece_at(chess.square(file, rank))
                text = piece.unicode_symbol() if piece else ""
                self.squares[row][file].configure(text=text)

        self._pump_events()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass


def update_viewers(
    board: chess.Board,
    game_number: int,
    total_games: int,
    target_elo: int,
    white_name: str,
    black_name: str,
    show_live: bool,
    gui_viewer: Optional[GuiBoardViewer],
    live_delay: float,
    last_move_san: Optional[str] = None,
) -> None:
    if show_live:
        render_live_board(
            board=board,
            game_number=game_number,
            total_games=total_games,
            target_elo=target_elo,
            white_name=white_name,
            black_name=black_name,
            last_move_san=last_move_san,
        )

    if gui_viewer is not None:
        gui_viewer.render(
            board=board,
            game_number=game_number,
            total_games=total_games,
            target_elo=target_elo,
            white_name=white_name,
            black_name=black_name,
            last_move_san=last_move_san,
        )

    if live_delay > 0 and (show_live or gui_viewer is not None):
        time.sleep(live_delay)


def play_match(
    target_elo: int,
    games_to_play: int,
    time_limit: float,
    pgn_dir: str,
    sykora_path: str,
    stockfish_path: str,
    show_live: bool = False,
    live_delay: float = 0.0,
    gui_viewer: Optional[GuiBoardViewer] = None,
) -> Optional[float]:
    print(f"Starting match: Sykora vs Stockfish (Elo {target_elo})")

    elo_dir = os.path.join(pgn_dir, str(target_elo))
    os.makedirs(elo_dir, exist_ok=True)

    try:
        sykora = chess.engine.SimpleEngine.popen_uci(sykora_path)
        stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure both engines are installed and paths are correct.")
        return None

    # Configure Stockfish Elo
    try:
        if (
            "UCI_LimitStrength" not in stockfish.options
            or "UCI_Elo" not in stockfish.options
        ):
            print("Warning: Stockfish does not support UCI_LimitStrength or UCI_Elo.")
            print("Available options:", list(stockfish.options.keys()))
        else:
            elo_opt = stockfish.options["UCI_Elo"]
            min_elo = elo_opt.min
            max_elo = elo_opt.max

            if target_elo < min_elo:
                print(
                    f"Notice: Target Elo {target_elo} is below Stockfish minimum {min_elo}. Using {min_elo}."
                )
                target_elo = min_elo
            elif target_elo > max_elo:
                print(
                    f"Notice: Target Elo {target_elo} is above Stockfish maximum {max_elo}. Using {max_elo}."
                )
                target_elo = max_elo

            stockfish.configure({"UCI_LimitStrength": True, "UCI_Elo": target_elo})
    except Exception as e:
        print(
            "Warning: Could not set Stockfish Elo. "
            f"It might be playing at full strength! Error: {e}"
        )

    sykora_score = 0.0

    try:
        for i in range(games_to_play):
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = f"Benchmark Match Elo {target_elo}"
            game.headers["Site"] = "Localhost"
            game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
            game.headers["Round"] = str(i + 1)

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

            update_viewers(
                board=board,
                game_number=i + 1,
                total_games=games_to_play,
                target_elo=target_elo,
                white_name=white_name,
                black_name=black_name,
                show_live=show_live,
                gui_viewer=gui_viewer,
                live_delay=live_delay,
            )

            node = game
            while not board.is_game_over():
                current_engine = white if board.turn == chess.WHITE else black
                result = current_engine.play(board, chess.engine.Limit(time=time_limit))

                if result.move is None:
                    print("  Engine returned no move, ending game early")
                    break

                move_san = board.san(result.move)
                board.push(result.move)
                node = node.add_variation(result.move)

                update_viewers(
                    board=board,
                    game_number=i + 1,
                    total_games=games_to_play,
                    target_elo=target_elo,
                    white_name=white_name,
                    black_name=black_name,
                    show_live=show_live,
                    gui_viewer=gui_viewer,
                    live_delay=live_delay,
                    last_move_san=move_san,
                )

            outcome = board.outcome()
            winner = outcome.winner if outcome else None
            game.headers["Result"] = board.result()

            pgn_path = os.path.join(elo_dir, f"game_{i+1}.pgn")
            with open(pgn_path, "w", encoding="utf-8") as f:
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
    finally:
        sykora.quit()
        stockfish.quit()

    print("\n===========================")
    print(f"Match Result vs Stockfish {target_elo}")
    print(f"Sykora Score: {sykora_score} / {games_to_play}")
    print("===========================\n")

    return sykora_score


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Sykora against Stockfish at increasing Elo levels."
    )
    parser.add_argument(
        "--sykora-path",
        default=DEFAULT_SYKORA_PATH,
        help=f"Path to Sykora binary (default: {DEFAULT_SYKORA_PATH})",
    )
    parser.add_argument(
        "--stockfish-path",
        default=DEFAULT_STOCKFISH_PATH,
        help=f"Path to Stockfish binary (default: {DEFAULT_STOCKFISH_PATH})",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES_TO_PLAY,
        help=f"Games per Elo level (default: {DEFAULT_GAMES_TO_PLAY})",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help=f"Seconds per move (default: {DEFAULT_TIME_LIMIT})",
    )
    parser.add_argument(
        "--pgn-dir",
        default=DEFAULT_PGN_DIR,
        help=f"Directory to save PGNs (default: {DEFAULT_PGN_DIR})",
    )
    parser.add_argument(
        "--elos",
        type=parse_elo_list,
        default=DEFAULT_ELOS,
        help=(
            "Comma-separated Elo levels to test "
            f"(default: {','.join(str(elo) for elo in DEFAULT_ELOS)})"
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Show a live board in the terminal while games are being played.",
    )
    parser.add_argument(
        "--live-delay",
        type=float,
        default=0.0,
        help="Extra delay in seconds after each viewer update (default: 0.0).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show a GUI board window that updates live during games.",
    )

    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be greater than 0")
    if args.time <= 0:
        parser.error("--time must be greater than 0")
    if args.live_delay < 0:
        parser.error("--live-delay must be non-negative")

    print("Benchmarking Sykora...")

    gui_viewer: Optional[GuiBoardViewer] = None
    if args.gui:
        try:
            gui_viewer = GuiBoardViewer()
        except RuntimeError as exc:
            print(f"Error: Unable to start GUI viewer: {exc}")
            return 1

    try:
        for elo in args.elos:
            score = play_match(
                target_elo=elo,
                games_to_play=args.games,
                time_limit=args.time,
                pgn_dir=args.pgn_dir,
                sykora_path=args.sykora_path,
                stockfish_path=args.stockfish_path,
                show_live=args.live,
                live_delay=args.live_delay,
                gui_viewer=gui_viewer,
            )

            if score is None:
                return 1

            if score < args.games * 0.5:
                print(
                    f"Sykora scored less than 50% against {elo}. Estimated rating is around {elo}."
                )
                return 0

            print(f"Passed level {elo}! Moving to next level...")
    finally:
        if gui_viewer is not None:
            gui_viewer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
