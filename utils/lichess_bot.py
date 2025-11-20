import berserk
import chess
import chess.engine
import threading
import sys
from dotenv import dotenv_values

# Configuration from .env
dotenv = dotenv_values()
API_TOKEN = dotenv.get("LICHESS_API_TOKEN")
ENGINE_PATH = dotenv.get("ENGINE_PATH")


class GameHandler(threading.Thread):
    def __init__(self, client, game_id, engine_path):
        super().__init__()
        self.client = client
        self.game_id = game_id
        self.engine_path = engine_path
        self.stream = client.bots.stream_game_state(game_id)
        self.engine = None
        self.board = chess.Board()

    def run(self):
        try:
            print(f"Starting game {self.game_id}")
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

            for event in self.stream:
                if event["type"] == "gameFull":
                    self.handle_game_full(event)
                elif event["type"] == "gameState":
                    self.handle_game_state(event)
                elif event["type"] == "chatLine":
                    pass

        except Exception as e:
            print(f"Game {self.game_id} error: {e}")
        finally:
            if self.engine:
                self.engine.quit()
            print(f"Game {self.game_id} finished")

    def handle_game_full(self, event):
        # Initialize board with moves if any
        state = event["state"]
        self.handle_game_state(state)

    def handle_game_state(self, state):
        moves = state["moves"]
        self.board.reset()
        if moves:
            for move in moves.split():
                self.board.push_uci(move)

        # Check if it's our turn
        # We need to know our color. In gameFull, we can find out.
        # But simpler: if it's white's turn and we are white, or black's turn and we are black.
        # Actually, let's just ask the engine to play if the game isn't over.

        # Wait, we need to know who we are.
        # The client.account.get() gives our user info.
        # But for simplicity, we can just check if the last move was NOT ours, or if it's the start.
        # A robust bot checks its color.

        # Let's just try to play. If it's not our turn, the engine might just think?
        # No, we should only invoke engine if it is our turn.

        # We can get our color from the game info, but stream_game_state doesn't always send it in gameState.
        # It is in gameFull.
        pass

    def play_move(self):
        if self.board.is_game_over():
            return

        # We need to know if it is our turn.
        # We can fetch the game info to be sure, or store it from gameFull.
        pass


# Let's rewrite the class to be simpler and more robust using the example patterns
# We will use a simpler approach:
# 1. Listen for challenges
# 2. Accept challenges
# 3. Listen for game starts
# 4. Spawn a thread for each game


def run_bot():
    session = berserk.TokenSession(API_TOKEN)
    client = berserk.Client(session)

    # Get bot profile
    try:
        profile = client.account.get()
        print(f"Logged in as {profile['username']}")
    except berserk.exceptions.ResponseError as e:
        print(f"Error logging in: {e}")
        print("Please make sure your API token is correct.")
        return

    print("Listening for challenges...")
    for event in client.bots.stream_incoming_events():
        if event["type"] == "challenge":
            challenge = event["challenge"]
            print(f"Accepting challenge from {challenge['challenger']['name']}")
            client.bots.accept_challenge(challenge["id"])

        elif event["type"] == "gameStart":
            game_id = event["game"]["gameId"]
            handler = GameSession(client, game_id, ENGINE_PATH, profile["username"])
            handler.start()


class GameSession(threading.Thread):
    def __init__(self, client, game_id, engine_path, bot_username):
        super().__init__()
        self.client = client
        self.game_id = game_id
        self.engine_path = engine_path
        self.bot_username = bot_username
        self.board = chess.Board()
        self.engine = None
        self.is_white = None

    def run(self):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

            # Stream game states
            for event in self.client.bots.stream_game_state(self.game_id):
                if event["type"] == "gameFull":
                    self.handle_state(event["state"], event["white"], event["black"])
                elif event["type"] == "gameState":
                    self.handle_state(event)

        except Exception as e:
            print(f"Error in game {self.game_id}: {e}")
        finally:
            if self.engine:
                self.engine.quit()

    def handle_state(self, state, white_user=None, black_user=None):
        # Update board
        moves = state["moves"]
        self.board.reset()
        if moves:
            for move in moves.split():
                self.board.push_uci(move)

        # Determine our color
        # If white_user is provided (gameFull), we can store it.
        # But simpler:
        # If it's White's turn (board.turn == chess.WHITE) and we are White?
        # We need to know our color.

        # Let's assume we are playing.
        # We can check if the client.account.get()['username'] matches white or black.
        # But we passed bot_username.

        if white_user:
            self.is_white = (
                white_user.get("name") or white_user.get("username")
            ).lower() == self.bot_username.lower()

        # Check if it's our turn
        is_our_turn = (self.board.turn == chess.WHITE and self.is_white) or (
            self.board.turn == chess.BLACK and not self.is_white
        )

        if is_our_turn and not self.board.is_game_over():
            # Get time remaining and increment (can be timedelta or milliseconds)
            wtime = state.get("wtime", 0)
            btime = state.get("btime", 0)
            winc = state.get("winc", 0)
            binc = state.get("binc", 0)

            # Convert to milliseconds (handle both timedelta and numeric types)
            def to_milliseconds(value):
                if hasattr(value, "total_seconds"):
                    return int(value.total_seconds() * 1000)
                return int(value) if value else 0

            wtime_ms = to_milliseconds(wtime)
            btime_ms = to_milliseconds(btime)
            winc_ms = to_milliseconds(winc)
            binc_ms = to_milliseconds(binc)

            print(
                f"Thinking... (Fen: {self.board.fen()}, W: {wtime_ms}ms+{winc_ms}ms, B: {btime_ms}ms+{binc_ms}ms)"
            )

            # Let the engine handle time management
            limit = chess.engine.Limit(
                white_clock=wtime_ms / 1000.0,
                black_clock=btime_ms / 1000.0,
                white_inc=winc_ms / 1000.0,
                black_inc=binc_ms / 1000.0,
            )

            result = self.engine.play(self.board, limit)
            self.client.bots.make_move(self.game_id, result.move.uci())
            print(f"Played {result.move.uci()}")


if __name__ == "__main__":
    run_bot()
