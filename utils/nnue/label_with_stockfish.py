#!/usr/bin/env python3
"""Label extracted positions with a Stockfish teacher score."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import chess
import chess.engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate positions with teacher scores.")
    parser.add_argument("--input", default="nnue/data/positions.jsonl", help="Input JSONL")
    parser.add_argument("--output", default="nnue/data/labeled.jsonl", help="Output JSONL")
    parser.add_argument(
        "--stockfish",
        default="/opt/homebrew/bin/stockfish",
        help="Stockfish binary path",
    )
    parser.add_argument(
        "--eval-file",
        default=None,
        help="Optional Stockfish EvalFile (.nnue) to use as teacher network",
    )
    parser.add_argument("--depth", type=int, default=12, help="Fixed depth")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all)")
    parser.add_argument("--threads", type=int, default=None, help="Stockfish Threads option")
    parser.add_argument("--hash-mb", type=int, default=None, help="Stockfish Hash option in MB")
    parser.add_argument(
        "--result-mix",
        type=float,
        default=0.2,
        help="Blend factor with game result target (0..1)",
    )
    parser.add_argument(
        "--eval-scale",
        type=float,
        default=400.0,
        help="Scale used by sigmoid(cp / scale)",
    )
    return parser.parse_args()


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def configure_teacher_engine(engine: chess.engine.SimpleEngine, args: argparse.Namespace) -> int:
    cfg: dict[str, object] = {}

    if args.threads is not None:
        if "Threads" not in engine.options:
            print("Engine does not expose Threads option.", file=sys.stderr)
            return 1
        cfg["Threads"] = args.threads

    if args.hash_mb is not None:
        if "Hash" not in engine.options:
            print("Engine does not expose Hash option.", file=sys.stderr)
            return 1
        cfg["Hash"] = args.hash_mb

    if args.eval_file:
        if "EvalFile" not in engine.options:
            print("Engine does not expose EvalFile option.", file=sys.stderr)
            return 1
        cfg["EvalFile"] = args.eval_file

    if cfg:
        engine.configure(cfg)
        engine.ping()

    return 0


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.result_mix <= 1.0):
        print("--result-mix must be in [0, 1].", file=sys.stderr)
        return 2
    if args.depth <= 0:
        print("--depth must be > 0.", file=sys.stderr)
        return 2
    if args.eval_scale <= 0:
        print("--eval-scale must be > 0.", file=sys.stderr)
        return 2
    if args.threads is not None and args.threads <= 0:
        print("--threads must be > 0.", file=sys.stderr)
        return 2
    if args.hash_mb is not None and args.hash_mb <= 0:
        print("--hash-mb must be > 0.", file=sys.stderr)
        return 2

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        return 1
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.eval_file:
        eval_path = Path(args.eval_file)
        if not eval_path.is_file():
            print(f"EvalFile not found: {eval_path}", file=sys.stderr)
            return 1

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    processed = 0
    try:
        try:
            rc = configure_teacher_engine(engine, args)
            if rc != 0:
                return rc
        except chess.engine.EngineError as exc:
            print(f"Failed to configure Stockfish options: {exc}", file=sys.stderr)
            return 1
        except chess.engine.EngineTerminatedError:
            print("Stockfish terminated while applying options (possibly invalid EvalFile).", file=sys.stderr)
            return 1

        with in_path.open("r", encoding="utf-8") as inp, out_path.open("w", encoding="utf-8") as out:
            for line in inp:
                if not line.strip():
                    continue
                row = json.loads(line)
                fen = row["fen"]
                board = chess.Board(fen)

                try:
                    info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                except chess.engine.EngineTerminatedError:
                    print(
                        "Stockfish terminated during analysis "
                        "(often means incompatible EvalFile with this Stockfish build).",
                        file=sys.stderr,
                    )
                    return 1
                score_obj = info["score"].pov(board.turn)
                cp = score_obj.score(mate_score=100000)
                if cp is None:
                    continue

                teacher_target = sigmoid(float(cp) / args.eval_scale)
                result_target = float(row.get("target_result_stm", 0.5))
                blended_target = args.result_mix * result_target + (1.0 - args.result_mix) * teacher_target

                row["teacher_cp_stm"] = int(cp)
                row["teacher_sigmoid_stm"] = teacher_target
                row["target_blended_stm"] = blended_target
                out.write(json.dumps(row) + "\n")

                processed += 1
                if args.limit > 0 and processed >= args.limit:
                    break
    finally:
        try:
            engine.quit()
        except chess.engine.EngineTerminatedError:
            pass

    print(f"Labeled positions: {processed}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
