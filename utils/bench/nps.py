#!/usr/bin/env python3
"""Measure engine search NPS across a small position suite.

Examples:
  ~/.pyenv/shims/python utils/bench/nps.py --engine ./zig-out/bin/sykora --depth 10
  ~/.pyenv/shims/python utils/bench/nps.py --engine ./zig-out/bin/sykora --movetime-ms 500 --runs 2
  ~/.pyenv/shims/python utils/bench/nps.py --position-file ./fens.txt --engine-opt Threads=8 --engine-opt Hash=4096
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chess
import chess.engine


DEFAULT_POSITIONS: List[Tuple[str, str]] = [
    ("startpos", "startpos"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("middlegame-1", "r1bq1rk1/pppn1ppp/2pbpn2/3p4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 9"),
    ("middlegame-2", "2rq1rk1/1b2bppp/p2ppn2/1pn5/3NP3/1BN1B3/PPQ2PPP/2RR2K1 w - - 2 16"),
    ("endgame-1", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("endgame-2", "8/8/2p5/2P2kp1/p1KP4/1p3PpP/1P4P1/8 w - - 8 66"),
]


@dataclass
class BenchResult:
    label: str
    run_idx: int
    depth: Optional[int]
    nodes: int
    time_s: float
    nps: float
    bestmove: Optional[str]


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
            raise ValueError(f"Invalid --engine-opt '{entry}' (expected Key=Value)")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid option key in '{entry}'")
        opts[key] = parse_option_value(raw_value.strip())
    return opts


def load_positions(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            label, fen = line.split("|", 1)
            label = label.strip()
            fen = fen.strip()
        else:
            label = f"pos-{line_no}"
            fen = line
        if not fen:
            continue
        rows.append((label, fen))
    if not rows:
        raise ValueError(f"No usable positions in {path}")
    return rows


def board_from_spec(spec: str) -> chess.Board:
    if spec == "startpos":
        return chess.Board()
    return chess.Board(spec)


def run_position(
    engine: chess.engine.SimpleEngine,
    label: str,
    fen_or_startpos: str,
    *,
    depth: Optional[int],
    movetime_ms: Optional[int],
) -> BenchResult:
    board = board_from_spec(fen_or_startpos)
    if movetime_ms is not None:
        limit = chess.engine.Limit(time=movetime_ms / 1000.0, depth=depth)
    else:
        assert depth is not None
        limit = chess.engine.Limit(depth=depth)

    info = engine.analyse(board, limit, info=chess.engine.INFO_BASIC | chess.engine.INFO_PV)

    nodes = int(info.get("nodes", 0))
    time_s = float(info.get("time", 0.0))
    nps = float(info.get("nps", 0.0))
    if nps <= 0.0 and nodes > 0 and time_s > 0:
        nps = nodes / time_s

    pv = info.get("pv", [])
    bestmove = pv[0].uci() if pv else None
    depth_info = info.get("depth")
    depth_out = int(depth_info) if depth_info is not None else None

    return BenchResult(
        label=label,
        run_idx=0,
        depth=depth_out,
        nodes=nodes,
        time_s=time_s,
        nps=nps,
        bestmove=bestmove,
    )


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark search NPS over a fixed position suite.")
    parser.add_argument("--engine", default="./zig-out/bin/sykora", help="Path to UCI engine binary")
    parser.add_argument("--depth", type=int, default=10, help="Search depth (ignored only if --movetime-ms is used without depth)")
    parser.add_argument("--movetime-ms", type=int, default=None, help="Per-position movetime in ms")
    parser.add_argument("--runs", type=int, default=1, help="Repeat full suite this many times")
    parser.add_argument("--position-file", default=None, help="Optional text file with 'label|fen' lines")
    parser.add_argument("--engine-opt", action="append", default=[], help="UCI option Key=Value (repeatable)")
    parser.add_argument("--quiet", action="store_true", help="Only print aggregate summary")
    args = parser.parse_args()

    if args.depth is not None and args.depth <= 0:
        parser.error("--depth must be > 0")
    if args.movetime_ms is not None and args.movetime_ms <= 0:
        parser.error("--movetime-ms must be > 0")
    if args.runs <= 0:
        parser.error("--runs must be > 0")

    positions = DEFAULT_POSITIONS if args.position_file is None else load_positions(Path(args.position_file))
    options = parse_uci_options(args.engine_opt)

    all_results: List[BenchResult] = []
    try:
        with chess.engine.SimpleEngine.popen_uci(args.engine) as engine:
            if options:
                accepted: Dict[str, object] = {}
                for name, value in options.items():
                    if name in engine.options:
                        accepted[name] = value
                    elif not args.quiet:
                        print(f"Warning: engine does not expose option '{name}', skipping")
                if accepted:
                    engine.configure(accepted)

            for run_idx in range(1, args.runs + 1):
                for label, fen in positions:
                    result = run_position(
                        engine,
                        label,
                        fen,
                        depth=args.depth,
                        movetime_ms=args.movetime_ms,
                    )
                    result.run_idx = run_idx
                    all_results.append(result)
                    if not args.quiet:
                        depth_str = f"d{result.depth}" if result.depth is not None else "d?"
                        print(
                            f"run={run_idx:02d} pos={label:<14} {depth_str:<4} "
                            f"nodes={result.nodes:>9} time={result.time_s:>6.3f}s "
                            f"nps={result.nps:>10.0f} best={result.bestmove or 'none'}"
                        )
    except FileNotFoundError:
        print(f"Engine not found: {args.engine}", file=sys.stderr)
        return 1
    except chess.engine.EngineError as exc:
        print(f"Engine error: {exc}", file=sys.stderr)
        return 1

    if not all_results:
        print("No results collected.", file=sys.stderr)
        return 1

    nps_values = [r.nps for r in all_results if r.nps > 0]
    total_nodes = sum(r.nodes for r in all_results)
    total_time = sum(r.time_s for r in all_results)
    overall_nps = (total_nodes / total_time) if total_time > 0 else 0.0

    print("\nNPS summary")
    print("-" * 60)
    print(f"engine:          {args.engine}")
    print(f"positions:       {len(positions)}")
    print(f"runs:            {args.runs}")
    print(f"depth:           {args.depth}" if args.movetime_ms is None else f"depth/time:      {args.depth} / {args.movetime_ms}ms")
    print(f"overall NPS:     {overall_nps:.0f}")
    print(f"mean NPS:        {mean(nps_values):.0f}")
    print(f"median NPS:      {median(nps_values):.0f}")
    print(f"total nodes:     {total_nodes}")
    print(f"total time (s):  {total_time:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
