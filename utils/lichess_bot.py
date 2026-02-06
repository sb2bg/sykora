#!/usr/bin/env python3
"""Run a resilient Lichess bot backed by a UCI engine.

Environment variables:
- LICHESS_API_TOKEN: Lichess bot token (required)
- ENGINE_PATH: Engine command or executable path (required)
- ALLOWED_VARIANTS: Comma-separated variant keys to accept (default: standard)
- BOT_MAX_CONCURRENT_GAMES: Maximum active games (default: 1)
- BOT_RECONNECT_BASE_SECONDS: Base reconnect delay (default: 1.0)
- BOT_MAX_BACKOFF_SECONDS: Maximum reconnect delay (default: 60.0)
- BOT_MOVE_OVERHEAD_MS: Safety overhead subtracted from clocks (default: 150)
- BOT_FALLBACK_MOVE_TIME_SECONDS: Fixed think time if no clocks (default: 1.0)
- BOT_LOG_LEVEL: Logging level (default: INFO)
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import berserk
import chess
import chess.engine
from dotenv import dotenv_values

DOTENV = dotenv_values()
LOGGER = logging.getLogger("lichess-bot")


@dataclass(frozen=True)
class BotConfig:
    api_token: str
    engine_command: tuple[str, ...]
    allowed_variants: frozenset[str]
    max_concurrent_games: int
    reconnect_base_seconds: float
    max_backoff_seconds: float
    move_overhead_ms: int
    fallback_move_time_seconds: float


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, DOTENV.get(name, default))


def _parse_int(name: str, default: int, minimum: int) -> int:
    raw = _env(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer (got {raw!r}).") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got {value}).")
    return value


def _parse_float(name: str, default: float, minimum: float) -> float:
    raw = _env(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number (got {raw!r}).") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got {value}).")
    return value


def _parse_engine_command(raw_engine: str) -> tuple[str, ...]:
    parts = shlex.split(raw_engine)
    if not parts:
        raise ValueError("ENGINE_PATH is empty.")

    binary = os.path.expanduser(parts[0])
    resolved: str | None = None

    if Path(binary).exists():
        resolved = str(Path(binary))
    else:
        on_path = shutil.which(binary)
        if on_path:
            resolved = on_path

    if resolved is None:
        raise ValueError(
            "ENGINE_PATH does not point to an existing executable or command on PATH: "
            f"{parts[0]!r}"
        )

    parts[0] = resolved
    return tuple(parts)


def _parse_variants(raw: str | None) -> frozenset[str]:
    if raw is None:
        return frozenset({"standard"})

    values = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not values:
        return frozenset({"standard"})
    return frozenset(values)


def _load_config() -> BotConfig:
    api_token = (_env("LICHESS_API_TOKEN") or "").strip()
    if not api_token:
        raise ValueError("LICHESS_API_TOKEN is required.")

    raw_engine = (_env("ENGINE_PATH") or "").strip()
    if not raw_engine:
        raise ValueError("ENGINE_PATH is required.")

    return BotConfig(
        api_token=api_token,
        engine_command=_parse_engine_command(raw_engine),
        allowed_variants=_parse_variants(_env("ALLOWED_VARIANTS", "standard")),
        max_concurrent_games=_parse_int("BOT_MAX_CONCURRENT_GAMES", default=1, minimum=1),
        reconnect_base_seconds=_parse_float(
            "BOT_RECONNECT_BASE_SECONDS", default=1.0, minimum=0.1
        ),
        max_backoff_seconds=_parse_float(
            "BOT_MAX_BACKOFF_SECONDS", default=60.0, minimum=1.0
        ),
        move_overhead_ms=_parse_int("BOT_MOVE_OVERHEAD_MS", default=150, minimum=0),
        fallback_move_time_seconds=_parse_float(
            "BOT_FALLBACK_MOVE_TIME_SECONDS", default=1.0, minimum=0.05
        ),
    )


def _to_millis(value: object) -> int:
    if value is None:
        return 0
    if hasattr(value, "total_seconds"):
        return max(0, int(getattr(value, "total_seconds")() * 1000))
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


class GameSession(threading.Thread):
    def __init__(
        self,
        client: berserk.Client,
        config: BotConfig,
        bot_username: str,
        game_id: str,
        on_exit: Callable[[str], None],
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"game-{game_id}", daemon=True)
        self.client = client
        self.config = config
        self.bot_username = bot_username.lower()
        self.game_id = game_id
        self.on_exit = on_exit
        self.stop_event = stop_event

        self.engine: chess.engine.SimpleEngine | None = None
        self.board = chess.Board()
        self.initial_fen = "startpos"
        self.is_white: bool | None = None
        self.finished = False

        # Prevent duplicate move submissions while waiting for stream confirmation.
        self.pending_move_ply: int | None = None

    def run(self) -> None:
        backoff = self.config.reconnect_base_seconds

        try:
            while not self.finished and not self.stop_event.is_set():
                try:
                    for event in self.client.bots.stream_game_state(self.game_id):
                        if self.stop_event.is_set() or self.finished:
                            break
                        self._handle_event(event)

                    if self.stop_event.is_set() or self.finished:
                        break

                    LOGGER.warning(
                        "[%s] game stream ended; reconnecting in %.1fs",
                        self.game_id,
                        backoff,
                    )
                except Exception:
                    if self.stop_event.is_set() or self.finished:
                        break
                    LOGGER.exception(
                        "[%s] game stream error; reconnecting in %.1fs",
                        self.game_id,
                        backoff,
                    )

                time.sleep(backoff)
                backoff = min(backoff * 2.0, self.config.max_backoff_seconds)
        finally:
            self._shutdown_engine()
            self.on_exit(self.game_id)
            LOGGER.info("[%s] game session closed", self.game_id)

    def _handle_event(self, event: object) -> None:
        if not isinstance(event, dict):
            return

        event_type = event.get("type")

        if event_type == "gameFull":
            variant_key = ((event.get("variant") or {}).get("key") or "standard").lower()
            if variant_key not in self.config.allowed_variants:
                LOGGER.warning(
                    "[%s] unsupported variant %r in active game; resigning",
                    self.game_id,
                    variant_key,
                )
                self._resign_game()
                self.finished = True
                return

            self.initial_fen = event.get("initialFen") or "startpos"
            self._set_color(event.get("white") or {}, event.get("black") or {})
            self._handle_state(event.get("state") or {})
            return

        if event_type == "gameState":
            self._handle_state(event)
            return

        if event_type == "chatLine":
            return

        if event_type == "opponentGone":
            return

        LOGGER.debug("[%s] ignored game event type=%r", self.game_id, event_type)

    def _set_color(self, white: dict, black: dict) -> None:
        white_name = str(white.get("name") or white.get("username") or "").lower()
        black_name = str(black.get("name") or black.get("username") or "").lower()

        if white_name == self.bot_username:
            self.is_white = True
        elif black_name == self.bot_username:
            self.is_white = False
        else:
            LOGGER.warning(
                "[%s] could not determine bot color (white=%r black=%r bot=%r)",
                self.game_id,
                white_name,
                black_name,
                self.bot_username,
            )

    def _rebuild_board(self, moves: str) -> bool:
        try:
            if self.initial_fen == "startpos":
                board = chess.Board()
            else:
                board = chess.Board(self.initial_fen)

            if moves:
                for uci_move in moves.split():
                    board.push_uci(uci_move)

            self.board = board
            return True
        except ValueError as exc:
            LOGGER.warning(
                "[%s] invalid state from stream (initial_fen=%r): %s",
                self.game_id,
                self.initial_fen,
                exc,
            )
            return False

    def _handle_state(self, state: dict) -> None:
        if not state:
            return

        moves = str(state.get("moves") or "")
        if not self._rebuild_board(moves):
            return

        status = str(state.get("status") or "started")
        if status != "started" or self.board.is_game_over():
            self.finished = True
            LOGGER.info("[%s] game ended with status=%s", self.game_id, status)
            return

        # If the server now reports more plies than when we submitted a move,
        # our move has been acknowledged and we can think again when appropriate.
        if self.pending_move_ply is not None and self.board.ply() > self.pending_move_ply:
            self.pending_move_ply = None

        if self.pending_move_ply is not None:
            return

        if self.is_white is None:
            return

        our_turn = self.board.turn == (chess.WHITE if self.is_white else chess.BLACK)
        if not our_turn:
            return

        move = self._compute_move(state)
        if move is None:
            return

        current_ply = self.board.ply()
        self._submit_move(move, current_ply)

    def _compute_move(self, state: dict) -> chess.Move | None:
        limit = self._build_limit(state)

        for attempt in (1, 2):
            try:
                self._ensure_engine()
                assert self.engine is not None
                result = self.engine.play(self.board, limit)
                if result.move is None:
                    LOGGER.warning("[%s] engine returned no move", self.game_id)
                    return None
                return result.move
            except chess.engine.EngineTerminatedError:
                LOGGER.exception(
                    "[%s] engine crashed during play (attempt %s/2)",
                    self.game_id,
                    attempt,
                )
                self._restart_engine()
            except Exception:
                LOGGER.exception("[%s] engine play failed", self.game_id)
                return None

        return None

    def _build_limit(self, state: dict) -> chess.engine.Limit:
        wtime_ms = max(0, _to_millis(state.get("wtime")) - self.config.move_overhead_ms)
        btime_ms = max(0, _to_millis(state.get("btime")) - self.config.move_overhead_ms)
        winc_ms = _to_millis(state.get("winc"))
        binc_ms = _to_millis(state.get("binc"))

        if wtime_ms > 0 and btime_ms > 0:
            return chess.engine.Limit(
                white_clock=wtime_ms / 1000.0,
                black_clock=btime_ms / 1000.0,
                white_inc=winc_ms / 1000.0,
                black_inc=binc_ms / 1000.0,
            )

        return chess.engine.Limit(time=self.config.fallback_move_time_seconds)

    def _submit_move(self, move: chess.Move, current_ply: int) -> None:
        try:
            self.client.bots.make_move(self.game_id, move.uci())
            self.pending_move_ply = current_ply
            LOGGER.info("[%s] played %s", self.game_id, move.uci())
        except berserk.exceptions.ResponseError as exc:
            message = str(exc).lower()
            if "not your turn" in message:
                # If we raced with the stream update, do not spam additional moves.
                self.pending_move_ply = current_ply
                LOGGER.debug("[%s] move likely already applied", self.game_id)
            else:
                LOGGER.warning("[%s] make_move failed: %s", self.game_id, exc)
        except Exception:
            LOGGER.exception("[%s] unexpected make_move failure", self.game_id)

    def _ensure_engine(self) -> None:
        if self.engine is not None:
            return

        LOGGER.info("[%s] starting engine: %s", self.game_id, " ".join(self.config.engine_command))
        self.engine = chess.engine.SimpleEngine.popen_uci(list(self.config.engine_command))

    def _restart_engine(self) -> None:
        self._shutdown_engine()

    def _shutdown_engine(self) -> None:
        if self.engine is None:
            return

        try:
            self.engine.quit()
        except Exception:
            LOGGER.debug("[%s] engine quit failed", self.game_id, exc_info=True)
            try:
                self.engine.close()
            except Exception:
                LOGGER.debug("[%s] engine close failed", self.game_id, exc_info=True)
        finally:
            self.engine = None

    def _resign_game(self) -> None:
        try:
            self.client.bots.resign_game(self.game_id)
        except Exception:
            LOGGER.debug("[%s] resign attempt failed", self.game_id, exc_info=True)


class BotService:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.stop_event = threading.Event()
        self.games_lock = threading.Lock()
        self.active_games: dict[str, GameSession] = {}

        self.client = berserk.Client(berserk.TokenSession(config.api_token))
        self.bot_username = ""

    def run_forever(self) -> None:
        self._install_signal_handlers()
        profile = self.client.account.get()
        self.bot_username = str(profile["username"])

        LOGGER.info("Logged in as %s", self.bot_username)
        LOGGER.info(
            "Accepted variants: %s | max concurrent games: %s",
            ",".join(sorted(self.config.allowed_variants)),
            self.config.max_concurrent_games,
        )

        backoff = self.config.reconnect_base_seconds

        try:
            while not self.stop_event.is_set():
                try:
                    for event in self.client.bots.stream_incoming_events():
                        if self.stop_event.is_set():
                            break
                        self._handle_incoming_event(event)

                    if self.stop_event.is_set():
                        break

                    LOGGER.warning(
                        "incoming event stream ended; reconnecting in %.1fs", backoff
                    )
                except Exception:
                    if self.stop_event.is_set():
                        break
                    LOGGER.exception(
                        "incoming event stream error; reconnecting in %.1fs", backoff
                    )

                time.sleep(backoff)
                backoff = min(backoff * 2.0, self.config.max_backoff_seconds)
        finally:
            self._shutdown()

    def _handle_incoming_event(self, event: object) -> None:
        if not isinstance(event, dict):
            return

        event_type = event.get("type")

        if event_type == "challenge":
            self._handle_challenge(event.get("challenge") or {})
            return

        if event_type == "gameStart":
            self._handle_game_start(event.get("game") or {})
            return

        if event_type == "challengeCanceled":
            return

        LOGGER.debug("ignored incoming event type=%r", event_type)

    def _handle_challenge(self, challenge: dict) -> None:
        challenge_id = str(challenge.get("id") or "")
        if not challenge_id:
            return

        challenger = (challenge.get("challenger") or {}).get("name") or "unknown"
        variant_key = ((challenge.get("variant") or {}).get("key") or "standard").lower()

        if variant_key not in self.config.allowed_variants:
            LOGGER.info(
                "Declining challenge %s from %s (unsupported variant=%s)",
                challenge_id,
                challenger,
                variant_key,
            )
            self._decline_challenge(challenge_id, reason="variant")
            return

        if self._active_game_count() >= self.config.max_concurrent_games:
            LOGGER.info(
                "Declining challenge %s from %s (at capacity)",
                challenge_id,
                challenger,
            )
            self._decline_challenge(challenge_id, reason="later")
            return

        try:
            self.client.bots.accept_challenge(challenge_id)
            LOGGER.info("Accepted challenge %s from %s", challenge_id, challenger)
        except Exception:
            LOGGER.exception("Failed to accept challenge %s", challenge_id)

    def _decline_challenge(self, challenge_id: str, reason: str) -> None:
        try:
            self.client.bots.decline_challenge(challenge_id, reason=reason)
            return
        except TypeError:
            # Backward compatibility for older berserk signatures.
            try:
                self.client.bots.decline_challenge(challenge_id, reason)
                return
            except Exception:
                LOGGER.exception("Failed to decline challenge %s", challenge_id)
                return
        except Exception:
            if reason != "generic":
                LOGGER.warning(
                    "Decline with reason=%s failed for %s, retrying generic",
                    reason,
                    challenge_id,
                )
                try:
                    self.client.bots.decline_challenge(challenge_id, reason="generic")
                    return
                except Exception:
                    pass
            LOGGER.exception("Failed to decline challenge %s", challenge_id)

    def _handle_game_start(self, game: dict) -> None:
        game_id = str(game.get("gameId") or game.get("id") or "")
        if not game_id:
            return

        with self.games_lock:
            if game_id in self.active_games:
                return

            if len(self.active_games) >= self.config.max_concurrent_games:
                LOGGER.warning(
                    "Game %s started while at capacity; resigning to protect service",
                    game_id,
                )
                try:
                    self.client.bots.resign_game(game_id)
                except Exception:
                    LOGGER.exception("Failed to resign overflow game %s", game_id)
                return

            session = GameSession(
                client=self.client,
                config=self.config,
                bot_username=self.bot_username,
                game_id=game_id,
                on_exit=self._on_game_exit,
                stop_event=self.stop_event,
            )
            self.active_games[game_id] = session

        LOGGER.info("Starting game session for %s", game_id)
        session.start()

    def _on_game_exit(self, game_id: str) -> None:
        with self.games_lock:
            self.active_games.pop(game_id, None)

    def _active_game_count(self) -> int:
        with self.games_lock:
            return len(self.active_games)

    def _shutdown(self) -> None:
        self.stop_event.set()
        with self.games_lock:
            sessions = list(self.active_games.values())

        if sessions:
            LOGGER.info("Waiting for %d active game session(s) to stop", len(sessions))

        for session in sessions:
            session.join(timeout=10.0)

        LOGGER.info("Bot shutdown complete")

    def _install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: object) -> None:
            LOGGER.info("Received signal %s; shutting down", signum)
            self.stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except Exception:
                # Some environments do not allow custom signal handlers.
                pass


def _configure_logging() -> None:
    level_name = (_env("BOT_LOG_LEVEL", "INFO") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    _configure_logging()

    try:
        config = _load_config()
    except ValueError as exc:
        LOGGER.error(str(exc))
        return 1

    service = BotService(config)

    try:
        service.run_forever()
        return 0
    except berserk.exceptions.ResponseError as exc:
        LOGGER.error("Lichess API startup error: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
