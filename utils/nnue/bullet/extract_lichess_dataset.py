#!/usr/bin/env python3
"""Extract pre-labeled positions from Lichess open database PGN dumps.

Downloads a monthly .pgn.zst from database.lichess.org and extracts positions
that have Stockfish %eval comments, outputting in Bullet text format:
  <FEN> | <score_cp_white_relative> | <result_white_relative>

Lichess analyzes ~30-40% of games server-side with Stockfish.  A single recent
month contains hundreds of millions of analyzed positions — enough for large-
scale NNUE training without any local Stockfish teacher step.

Requires: zstandard, python-chess  (pip install zstandard chess)
"""

from __future__ import annotations

import argparse
import io
import math
import multiprocessing as mp
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import TextIO

# ---------------------------------------------------------------------------
# Eval comment regex: [%eval 1.23] or [%eval #5] or [%eval #-3]
# ---------------------------------------------------------------------------
_EVAL_RE = re.compile(r"\[%eval\s+([#\-\d.]+)\]")
_RESULT_RE = re.compile(r'\[Result\s+"([^"]+)"\]')
_ELO_RE = re.compile(r'\[(White|Black)Elo\s+"(\d+)"\]')
_TERMINATION_RE = re.compile(r'\[Termination\s+"([^"]+)"\]')

# Sentinel to tell workers to stop.
_POISON = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Lichess eval-annotated positions for NNUE training."
    )
    p.add_argument(
        "--month",
        default="2026-01",
        help="Month to download (YYYY-MM). Older months are smaller. (default: 2026-01)",
    )
    p.add_argument(
        "--pgn-zst",
        default="",
        help="Path to an already-downloaded .pgn.zst file (skips download)",
    )
    p.add_argument(
        "--download-dir",
        default="datasets/lichess",
        help="Directory to store downloaded .pgn.zst files",
    )
    p.add_argument(
        "--output",
        default="nnue/data/bullet/train/lichess_{month}.txt",
        help="Output text file (supports {month} placeholder)",
    )
    p.add_argument("--workers", type=int, default=0, help="Worker processes (0 = auto, ncpu - 2)")
    p.add_argument("--min-elo", type=int, default=1800, help="Minimum avg Elo of both players")
    p.add_argument("--min-ply", type=int, default=12, help="Skip positions before this ply")
    p.add_argument("--max-ply", type=int, default=300, help="Skip positions after this ply")
    p.add_argument("--cp-clip", type=int, default=2500, help="Clip cp to +/- this value")
    p.add_argument("--sample-rate", type=float, default=1.0, help="Sample rate for positions (1.0 = all)")
    p.add_argument("--skip-check", action="store_true", help="Skip positions where side-to-move is in check")
    p.add_argument("--skip-captures", action="store_true", help="Skip positions after a capture")
    p.add_argument("--max-positions", type=int, default=0, help="Hard cap on total positions (0 = unlimited)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    p.add_argument(
        "--require-normal-termination",
        action="store_true",
        default=True,
        help="Skip games terminated by abandonment/time forfeit (default: on)",
    )
    p.add_argument(
        "--no-require-normal-termination",
        action="store_false",
        dest="require_normal_termination",
    )
    p.add_argument("--chunk-size", type=int, default=500, help="Games per worker chunk")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_pgn_zst(month: str, download_dir: Path) -> Path:
    """Download a Lichess monthly dump if not already present."""
    filename = f"lichess_db_standard_rated_{month}.pgn.zst"
    dest = download_dir / filename
    if dest.exists():
        size_gb = dest.stat().st_size / (1024 ** 3)
        print(f"Already downloaded: {dest} ({size_gb:.1f} GB)")
        return dest

    url = f"https://database.lichess.org/standard/{filename}"
    download_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    print(f"Downloading {url} ...")
    print("(This can be 10-30+ GB for recent months. Ctrl-C to cancel.)")

    # Use curl/wget if available for better progress, else urllib.
    if _has_cmd("curl"):
        subprocess.run(["curl", "-L", "-o", str(tmp), "--progress-bar", url], check=True)
    elif _has_cmd("wget"):
        subprocess.run(["wget", "-O", str(tmp), "--show-progress", url], check=True)
    else:
        _urllib_download(url, tmp)

    os.replace(tmp, dest)
    size_gb = dest.stat().st_size / (1024 ** 3)
    print(f"Downloaded: {dest} ({size_gb:.1f} GB)")
    return dest


def _has_cmd(name: str) -> bool:
    try:
        subprocess.run([name, "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _urllib_download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with dest.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / (1<<30):.2f} / {total / (1<<30):.2f} GB ({pct:.1f}%)", end="", flush=True)
        print()


# ---------------------------------------------------------------------------
# Streaming decompressor → game splitter
# ---------------------------------------------------------------------------

def iter_game_texts(pgn_zst_path: Path):
    """Yield raw game text strings from a .pgn.zst file using zstd CLI."""
    # Prefer zstd CLI — it's faster and uses less memory than the Python binding.
    if _has_cmd("zstd"):
        proc = subprocess.Popen(
            ["zstd", "-d", "-c", str(pgn_zst_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1 << 22,  # 4 MB buffer
        )
        stream: TextIO = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")  # type: ignore[arg-type]
    else:
        try:
            import zstandard as zstd
        except ImportError:
            print(
                "Neither `zstd` CLI nor `zstandard` Python package found.\n"
                "Install one: `brew install zstd` or `pip install zstandard`",
                file=sys.stderr,
            )
            sys.exit(1)
        dctx = zstd.ZstdDecompressor()
        raw = pgn_zst_path.open("rb")
        reader = dctx.stream_reader(raw, read_size=1 << 22)
        stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        proc = None  # type: ignore[assignment]

    lines: list[str] = []
    in_game = False
    try:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("[Event "):
                if in_game and lines:
                    yield "".join(lines)
                lines = [line]
                in_game = True
            elif in_game:
                lines.append(line)

        if in_game and lines:
            yield "".join(lines)
    finally:
        stream.close()
        if proc is not None:
            proc.terminate()
            proc.wait()


# ---------------------------------------------------------------------------
# Worker: parse a chunk of game texts → list of output lines
# ---------------------------------------------------------------------------

def _result_token(value: float) -> str:
    if value >= 0.75:
        return "1.0"
    if value <= 0.25:
        return "0.0"
    return "0.5"


def _parse_eval_cp(raw: str) -> int | None:
    """Parse a Lichess %eval value to centipawns (side-to-move relative)."""
    if raw.startswith("#"):
        sign = -1 if raw[1:].startswith("-") else 1
        return sign * 30000  # Mate score → large cp
    try:
        return int(round(float(raw) * 100))
    except ValueError:
        return None


def process_chunk(args_tuple: tuple) -> list[str]:
    """Process a chunk of raw game texts. Returns output lines."""
    game_texts, cfg = args_tuple
    import chess
    import chess.pgn

    min_elo = cfg["min_elo"]
    min_ply = cfg["min_ply"]
    max_ply = cfg["max_ply"]
    cp_clip = cfg["cp_clip"]
    sample_rate = cfg["sample_rate"]
    skip_check = cfg["skip_check"]
    skip_captures = cfg["skip_captures"]
    require_normal = cfg["require_normal_termination"]
    seed = cfg["seed"]

    rng = random.Random(seed)
    results: list[str] = []

    for game_text in game_texts:
        # Quick header checks before full parse.
        result_m = _RESULT_RE.search(game_text)
        if not result_m:
            continue
        result_str = result_m.group(1)
        if result_str == "1-0":
            white_score = 1.0
        elif result_str == "0-1":
            white_score = 0.0
        elif result_str == "1/2-1/2":
            white_score = 0.5
        else:
            continue

        # Skip if no eval comments at all (fast reject ~60-70% of games).
        if "%eval" not in game_text:
            continue

        # Elo filter.
        elo_matches = _ELO_RE.findall(game_text)
        if len(elo_matches) < 2:
            continue
        elos = [int(m[1]) for m in elo_matches]
        if (elos[0] + elos[1]) / 2 < min_elo:
            continue

        # Termination filter.
        if require_normal:
            term_m = _TERMINATION_RE.search(game_text)
            if term_m:
                term = term_m.group(1).lower()
                if "time" in term and "forfeit" in term:
                    continue
                if "abandon" in term:
                    continue

        # Full parse.
        try:
            game = chess.pgn.read_game(io.StringIO(game_text))
        except Exception:
            continue
        if game is None:
            continue

        board = game.board()
        node = game
        ply = 0
        for node in game.mainline():
            move = node.move
            is_capture = board.is_capture(move)
            board.push(move)
            ply += 1

            if ply < min_ply or ply > max_ply:
                continue

            comment = node.comment
            if not comment:
                continue
            eval_m = _EVAL_RE.search(comment)
            if not eval_m:
                continue

            cp_stm = _parse_eval_cp(eval_m.group(1))
            if cp_stm is None:
                continue

            if skip_check and board.is_check():
                continue
            if skip_captures and is_capture:
                continue
            if sample_rate < 1.0 and rng.random() > sample_rate:
                continue

            cp_stm = max(-cp_clip, min(cp_clip, cp_stm))
            # Lichess eval is from current side-to-move perspective after the move.
            # But %eval is from the perspective of the side that JUST moved.
            # So we need to negate to get current STM, then convert to white-relative.
            cp_after_move_maker = cp_stm
            # The eval comment is from the POV of the player who made the move.
            # After push, board.turn is the OPPONENT. So eval is from opponent's POV? No.
            # Lichess: %eval is from WHITE's perspective always.
            cp_white = cp_stm
            results.append(f"{board.fen(en_passant='fen')} | {cp_white} | {_result_token(white_score)}\n")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Resolve paths.
    if args.pgn_zst:
        pgn_zst = Path(args.pgn_zst)
        if not pgn_zst.exists():
            print(f"File not found: {pgn_zst}", file=sys.stderr)
            return 1
    else:
        pgn_zst = download_pgn_zst(args.month, Path(args.download_dir))

    output = Path(args.output.format(month=args.month))
    output.parent.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers if args.workers > 0 else max(1, os.cpu_count() - 2)  # type: ignore[operator]
    print(f"Source:  {pgn_zst}")
    print(f"Output:  {output}")
    print(f"Workers: {n_workers}")
    print(f"Min Elo: {args.min_elo}")
    print(f"Sample:  {args.sample_rate}")
    print(f"Max pos: {args.max_positions or 'unlimited'}")
    print()

    cfg = {
        "min_elo": args.min_elo,
        "min_ply": args.min_ply,
        "max_ply": args.max_ply,
        "cp_clip": args.cp_clip,
        "sample_rate": args.sample_rate,
        "skip_check": args.skip_check,
        "skip_captures": args.skip_captures,
        "require_normal_termination": args.require_normal_termination,
        "seed": args.seed,
    }

    total_positions = 0
    total_games = 0
    games_with_eval = 0
    chunk: list[str] = []
    chunk_id = 0
    start_time = time.time()

    # Use a process pool for parallel extraction.
    # We ignore SIGINT in workers so the main process can handle Ctrl-C.
    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(n_workers)
    signal.signal(signal.SIGINT, original_sigint)

    pending_futures: list[mp.pool.AsyncResult] = []  # type: ignore[name-defined]
    max_pending = n_workers * 3  # Keep workers fed but don't buffer too much.

    try:
        with output.open("w", encoding="utf-8") as out:
            for game_text in iter_game_texts(pgn_zst):
                total_games += 1
                chunk.append(game_text)

                if len(chunk) >= args.chunk_size:
                    # Vary seed per chunk so workers don't correlate.
                    chunk_cfg = {**cfg, "seed": args.seed + chunk_id}
                    future = pool.apply_async(process_chunk, ((chunk, chunk_cfg),))
                    pending_futures.append(future)
                    chunk = []
                    chunk_id += 1

                    # Drain completed futures to write output and free memory.
                    while len(pending_futures) >= max_pending:
                        result_lines = pending_futures.pop(0).get()
                        if result_lines:
                            games_with_eval += 1  # Approximate.
                            for line in result_lines:
                                out.write(line)
                                total_positions += 1
                                if args.max_positions > 0 and total_positions >= args.max_positions:
                                    break
                        if args.max_positions > 0 and total_positions >= args.max_positions:
                            break

                    if args.max_positions > 0 and total_positions >= args.max_positions:
                        break

                # Progress reporting.
                if total_games % 100_000 == 0:
                    elapsed = time.time() - start_time
                    rate = total_games / elapsed if elapsed > 0 else 0
                    print(
                        f"\r  Games scanned: {total_games:,} | "
                        f"Positions: {total_positions:,} | "
                        f"Rate: {rate:,.0f} games/sec",
                        end="",
                        flush=True,
                    )

            # Final chunk.
            if chunk and not (args.max_positions > 0 and total_positions >= args.max_positions):
                chunk_cfg = {**cfg, "seed": args.seed + chunk_id}
                future = pool.apply_async(process_chunk, ((chunk, chunk_cfg),))
                pending_futures.append(future)

            # Drain remaining.
            for future in pending_futures:
                if args.max_positions > 0 and total_positions >= args.max_positions:
                    break
                result_lines = future.get()
                if result_lines:
                    for line in result_lines:
                        out.write(line)
                        total_positions += 1
                        if args.max_positions > 0 and total_positions >= args.max_positions:
                            break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        pool.terminate()
        pool.join()
        # Still print stats for what we got.
    else:
        pool.close()
        pool.join()

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Games scanned:    {total_games:,}")
    print(f"Positions written: {total_positions:,}")
    print(f"Elapsed:          {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
    if elapsed > 0:
        print(f"Throughput:       {total_games / elapsed:,.0f} games/sec")
        print(f"                  {total_positions / elapsed:,.0f} positions/sec")
    print(f"Output:           {output}")
    size_mb = output.stat().st_size / (1 << 20) if output.exists() else 0
    print(f"Output size:      {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
