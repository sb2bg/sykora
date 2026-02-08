#!/usr/bin/env python3
"""
Run Strategic Test Suite (STS) EPD files against a UCI engine.

This utility supports STS-style scoring metadata:
- c8 + c9 (preferred): numeric scores + matching move list (usually UCI/LAN)
- c0 with move=score pairs (fallback)
- bm (fallback): assigns default full score when no weighted scores are present
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import chess
    import chess.engine
except ImportError:
    print(
        "Error: python-chess is required for STS testing.\n"
        "Install it with: pip install chess",
        file=sys.stderr,
    )
    raise SystemExit(1)


DEFAULT_ENGINE_PATH = "./zig-out/bin/sykora"
DEFAULT_EPD_GLOB = "STS*.epd"
DEFAULT_MOVETIME_MS = 300
DEFAULT_BM_SCORE = 10


@dataclasses.dataclass
class PositionSpec:
    index_in_file: int
    theme: str
    source_file: str
    epd_line: str
    fen: str
    pos_id: str
    bm_moves: set[str]
    scored_moves: Dict[str, int]

    @property
    def max_points(self) -> int:
        return max(self.scored_moves.values(), default=0)


@dataclasses.dataclass
class ScoreTally:
    positions: int = 0
    points: int = 0
    max_points: int = 0
    top_hits: int = 0
    bm_hits: int = 0
    bm_total: int = 0

    def record(self, scored: int, max_scored: int, top_hit: bool, bm_hit: Optional[bool]) -> None:
        self.positions += 1
        self.points += scored
        self.max_points += max_scored
        if top_hit:
            self.top_hits += 1
        if bm_hit is not None:
            self.bm_total += 1
            if bm_hit:
                self.bm_hits += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run STS EPD suites against a UCI chess engine."
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE_PATH,
        help=f"Path to UCI engine binary (default: {DEFAULT_ENGINE_PATH})",
    )
    parser.add_argument(
        "--epd",
        required=True,
        help="Path to an EPD file or a directory containing EPD files.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_EPD_GLOB,
        help=f"Glob pattern when --epd is a directory (default: {DEFAULT_EPD_GLOB})",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=0,
        help="Optional global cap on parsed positions (0 = no cap).",
    )

    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--movetime-ms",
        type=int,
        default=DEFAULT_MOVETIME_MS,
        help=f"Search time per position in milliseconds (default: {DEFAULT_MOVETIME_MS})",
    )
    limit_group.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Fixed search depth per position.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Engine Threads option, if supported.",
    )
    parser.add_argument(
        "--hash-mb",
        type=int,
        default=None,
        help="Engine Hash option in MB, if supported.",
    )
    parser.add_argument(
        "--engine-opt",
        action="append",
        default=[],
        help="Extra UCI option, format Key=Value (repeatable)",
    )
    parser.add_argument(
        "--bm-score",
        type=int,
        default=DEFAULT_BM_SCORE,
        help=f"Score assigned to bm moves if no weighted metadata exists (default: {DEFAULT_BM_SCORE}).",
    )
    parser.add_argument(
        "--show",
        choices=("none", "misses", "all"),
        default="misses",
        help="Per-position logging style (default: misses).",
    )

    args = parser.parse_args()
    if args.max_positions < 0:
        parser.error("--max-positions must be >= 0")
    if args.movetime_ms is not None and args.movetime_ms <= 0:
        parser.error("--movetime-ms must be > 0")
    if args.depth is not None and args.depth <= 0:
        parser.error("--depth must be > 0")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be > 0")
    if args.hash_mb is not None and args.hash_mb <= 0:
        parser.error("--hash-mb must be > 0")
    return args


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


def parse_uci_options(entries: Sequence[str]) -> Dict[str, object]:
    options: Dict[str, object] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --engine-opt '{entry}' (expected Key=Value)")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid option key in '{entry}'")
        options[key] = parse_option_value(raw_value.strip())
    return options


def find_epd_files(path: str, pattern: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Not found: {path}")

    files = sorted(glob.glob(os.path.join(path, pattern)))
    if not files:
        raise FileNotFoundError(f"No EPD files matched pattern '{pattern}' in {path}")
    return files


def theme_name_for_file(file_path: str) -> str:
    base = os.path.basename(file_path)
    stem = os.path.splitext(base)[0]
    return stem


def sort_key_for_theme(theme: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", theme)
    if match:
        return int(match.group(1)), theme
    return (10_000, theme)


def _extract_fen4(epd_line: str) -> str:
    tokens = epd_line.strip().split()
    if len(tokens) < 4:
        raise ValueError("EPD line does not contain 4 FEN fields.")
    return " ".join(tokens[:4])


def _parse_ints(text: str) -> List[int]:
    return [int(value) for value in re.findall(r"-?\d+", text)]


def _parse_move_tokens(text: str) -> List[str]:
    tokens = re.split(r"[,\s]+", text.strip())
    return [token for token in tokens if token]


def _token_to_uci(board: chess.Board, token: str) -> Optional[str]:
    token = token.strip()
    if not token:
        return None

    token = token.rstrip(",")
    token = token.replace("0-0-0", "O-O-O").replace("0-0", "O-O")

    try:
        move = chess.Move.from_uci(token.lower())
        if move in board.legal_moves:
            return move.uci()
    except ValueError:
        pass

    try:
        return board.parse_san(token).uci()
    except ValueError:
        return None


def _scored_moves_from_c0(c0_value: str, board: chess.Board) -> Dict[str, int]:
    scores: Dict[str, int] = {}
    for move_token, score_str in re.findall(r'([^\s,;:=\"]+)\s*=\s*(-?\d+)', c0_value):
        uci = _token_to_uci(board, move_token)
        if uci is None:
            continue
        score = int(score_str)
        previous = scores.get(uci)
        if previous is None or score > previous:
            scores[uci] = score
    return scores


def parse_epd_line(
    line: str,
    source_file: str,
    index_in_file: int,
    theme: str,
    bm_score: int,
) -> Optional[PositionSpec]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    board = chess.Board()
    try:
        operations = board.set_epd(stripped)
    except ValueError:
        return None

    fen4 = _extract_fen4(stripped)
    hmvc = int(operations.get("hmvc", 0))
    fmvn = int(operations.get("fmvn", 1))
    fen = f"{fen4} {hmvc} {fmvn}"

    pos_id = str(operations.get("id", f"{theme}.{index_in_file:03d}"))

    bm_moves: set[str] = set()
    bm_value = operations.get("bm")
    if isinstance(bm_value, list):
        for move in bm_value:
            if isinstance(move, chess.Move):
                bm_moves.add(move.uci())
            else:
                uci = _token_to_uci(board, str(move))
                if uci:
                    bm_moves.add(uci)

    scored_moves: Dict[str, int] = {}

    c8 = operations.get("c8")
    c9 = operations.get("c9")
    if c8 is not None and c9 is not None:
        score_values = _parse_ints(str(c8))
        move_tokens = _parse_move_tokens(str(c9))
        for token, score in zip(move_tokens, score_values):
            uci = _token_to_uci(board, token)
            if uci is None:
                continue
            previous = scored_moves.get(uci)
            if previous is None or score > previous:
                scored_moves[uci] = score

    if not scored_moves:
        c0 = operations.get("c0")
        if isinstance(c0, str):
            scored_moves = _scored_moves_from_c0(c0, board)

    if not scored_moves and bm_moves:
        scored_moves = {move: bm_score for move in bm_moves}

    if not scored_moves and not bm_moves:
        return None

    return PositionSpec(
        index_in_file=index_in_file,
        theme=theme,
        source_file=source_file,
        epd_line=stripped,
        fen=fen,
        pos_id=pos_id,
        bm_moves=bm_moves,
        scored_moves=scored_moves,
    )


def load_positions(epd_files: Sequence[str], max_positions: int, bm_score: int) -> List[PositionSpec]:
    positions: List[PositionSpec] = []
    for file_path in epd_files:
        theme = theme_name_for_file(file_path)
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            local_index = 0
            for raw_line in handle:
                parsed = parse_epd_line(
                    line=raw_line,
                    source_file=file_path,
                    index_in_file=local_index + 1,
                    theme=theme,
                    bm_score=bm_score,
                )
                if parsed is None:
                    continue
                local_index += 1
                positions.append(parsed)
                if max_positions > 0 and len(positions) >= max_positions:
                    return positions
    return positions


def configure_engine(
    engine: chess.engine.SimpleEngine,
    threads: Optional[int],
    hash_mb: Optional[int],
    extra_options: Dict[str, object],
) -> None:
    config: Dict[str, object] = {}
    if threads is not None and "Threads" in engine.options:
        config["Threads"] = threads
    if hash_mb is not None and "Hash" in engine.options:
        config["Hash"] = hash_mb
    for key, value in extra_options.items():
        if key in engine.options:
            config[key] = value
        else:
            print(f"Warning: engine does not expose option '{key}', skipping", file=sys.stderr)
    if config:
        engine.configure(config)


def evaluate_positions(
    engine: chess.engine.SimpleEngine,
    positions: Sequence[PositionSpec],
    movetime_ms: Optional[int],
    depth: Optional[int],
    show_mode: str,
) -> Tuple[Dict[str, ScoreTally], ScoreTally]:
    by_theme: Dict[str, ScoreTally] = {}
    overall = ScoreTally()

    if depth is not None:
        limit = chess.engine.Limit(depth=depth)
    else:
        assert movetime_ms is not None
        limit = chess.engine.Limit(time=movetime_ms / 1000.0)

    for idx, position in enumerate(positions, start=1):
        board = chess.Board(position.fen)
        play_result = engine.play(board, limit)
        played_uci = play_result.move.uci() if play_result.move else "0000"

        scored = position.scored_moves.get(played_uci, 0)
        max_scored = position.max_points
        top_hit = max_scored > 0 and scored == max_scored

        bm_hit: Optional[bool]
        if position.bm_moves:
            bm_hit = played_uci in position.bm_moves
        else:
            bm_hit = None

        tally = by_theme.setdefault(position.theme, ScoreTally())
        tally.record(scored, max_scored, top_hit, bm_hit)
        overall.record(scored, max_scored, top_hit, bm_hit)

        if show_mode == "all" or (show_mode == "misses" and not top_hit):
            print(
                f"[{idx:4d}] {position.theme} #{position.index_in_file:03d} "
                f"id=\"{position.pos_id}\" move={played_uci} score={scored}/{max_scored}"
            )

    return by_theme, overall


def pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def print_summary(by_theme: Dict[str, ScoreTally], overall: ScoreTally) -> None:
    print("\nTheme Summary")
    print(
        f"{'Theme':<14} {'Pos':>5} {'Score':>10} {'Max':>10} "
        f"{'Score%':>8} {'Top%':>8} {'BM%':>8}"
    )
    print("-" * 74)

    for theme in sorted(by_theme.keys(), key=sort_key_for_theme):
        tally = by_theme[theme]
        score_pct = pct(tally.points, tally.max_points)
        top_pct = pct(tally.top_hits, tally.positions)
        bm_pct = pct(tally.bm_hits, tally.bm_total) if tally.bm_total else 0.0
        print(
            f"{theme:<14} {tally.positions:>5d} {tally.points:>10d} {tally.max_points:>10d} "
            f"{score_pct:>7.2f}% {top_pct:>7.2f}% {bm_pct:>7.2f}%"
        )

    print("-" * 74)
    overall_score_pct = pct(overall.points, overall.max_points)
    overall_top_pct = pct(overall.top_hits, overall.positions)
    overall_bm_pct = pct(overall.bm_hits, overall.bm_total) if overall.bm_total else 0.0
    print(
        f"{'TOTAL':<14} {overall.positions:>5d} {overall.points:>10d} {overall.max_points:>10d} "
        f"{overall_score_pct:>7.2f}% {overall_top_pct:>7.2f}% {overall_bm_pct:>7.2f}%"
    )


def main() -> int:
    args = parse_args()
    try:
        extra_options = parse_uci_options(args.engine_opt)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    try:
        epd_files = find_epd_files(args.epd, args.pattern)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    positions = load_positions(epd_files, args.max_positions, args.bm_score)
    if not positions:
        print("Error: no valid STS positions were parsed.", file=sys.stderr)
        return 1

    limit_desc = f"depth={args.depth}" if args.depth is not None else f"movetime={args.movetime_ms}ms"
    print(f"Engine: {args.engine}")
    print(f"Files: {len(epd_files)} | Positions: {len(positions)} | Limit: {limit_desc}")
    if extra_options:
        print(f"Extra engine options: {extra_options}")

    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    except FileNotFoundError:
        print(f"Error: engine not found: {args.engine}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: failed to start engine: {error}", file=sys.stderr)
        return 1

    try:
        configure_engine(engine, args.threads, args.hash_mb, extra_options)
        by_theme, overall = evaluate_positions(
            engine=engine,
            positions=positions,
            movetime_ms=args.movetime_ms if args.depth is None else None,
            depth=args.depth,
            show_mode=args.show,
        )
    finally:
        engine.quit()

    print_summary(by_theme, overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
