#!/usr/bin/env python3
"""SPRT runner for baseline-vs-candidate UCI engine testing.

This wrapper runs `utils/match/selfplay.py` in batches and performs a
Sequential Probability Ratio Test over cumulative score.

Exit codes:
  0 -> candidate accepted as stronger (H1) or practical stronger threshold met
  1 -> candidate accepted as weaker (H0)
  2 -> inconclusive (max games reached without decision)
  3 -> execution/configuration error
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Totals:
    baseline_wins: int = 0
    candidate_wins: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.baseline_wins + self.candidate_wins + self.draws

    @property
    def candidate_score(self) -> float:
        return self.candidate_wins + 0.5 * self.draws

    @property
    def candidate_rate(self) -> float:
        g = self.games
        return (self.candidate_score / g) if g > 0 else 0.5


def p_from_elo(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def elo_from_rate(rate: float) -> float:
    eps = 1e-6
    p = min(max(rate, eps), 1.0 - eps)
    return -400.0 * math.log10((1.0 - p) / p)


def one_sided_p_value_stronger(rate: float, games: int) -> float:
    if games <= 0:
        return 1.0
    sigma0 = math.sqrt(0.25 / games)
    z = (rate - 0.5) / sigma0
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def compute_llr(candidate_score: float, games: int, p0: float, p1: float) -> float:
    return candidate_score * math.log(p1 / p0) + (games - candidate_score) * math.log((1.0 - p1) / (1.0 - p0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batched self-play SPRT between two engines.")

    parser.add_argument("engine1", help="Path to baseline engine binary")
    parser.add_argument("engine2", help="Path to candidate engine binary")
    parser.add_argument("--name1", default="Baseline", help="Label for engine1")
    parser.add_argument("--name2", default="Candidate", help="Label for engine2")

    parser.add_argument("--elo0", type=float, default=-30.0, help="SPRT H0 Elo (candidate-baseline)")
    parser.add_argument("--elo1", type=float, default=30.0, help="SPRT H1 Elo (candidate-baseline)")
    parser.add_argument("--alpha", type=float, default=0.10, help="Type I error target")
    parser.add_argument("--beta", type=float, default=0.10, help="Type II error target")

    parser.add_argument("--games-per-batch", type=int, default=12, help="Games per self-play batch")
    parser.add_argument("--max-games", type=int, default=360, help="Hard cap on total games")
    parser.add_argument("--movetime-ms", type=int, default=80, help="Per-move time in ms")
    parser.add_argument("--depth", type=int, default=None, help="Optional fixed depth")
    parser.add_argument("--game-time-ms", type=int, default=None, help="Optional per-side game clock")
    parser.add_argument("--inc-ms", type=int, default=0, help="Per-move increment with --game-time-ms")
    parser.add_argument("--openings", default="default", help="Opening source for selfplay.py")
    parser.add_argument("--shuffle-openings", action="store_true", help="Shuffle opening order")
    parser.add_argument("--seed", type=int, default=1, help="Base RNG seed (incremented per batch)")
    parser.add_argument("--max-plies", type=int, default=220, help="Draw adjudication ply cap")
    parser.add_argument("--threads", type=int, default=None, help="Threads UCI option for both engines")
    parser.add_argument("--hash-mb", type=int, default=None, help="Hash UCI option for both engines")

    parser.add_argument(
        "--practical-min-games",
        type=int,
        default=120,
        help="Minimum games before practical stronger check can trigger",
    )
    parser.add_argument(
        "--practical-p-threshold",
        type=float,
        default=0.05,
        help="One-sided p-value threshold for practical stronger early stop",
    )
    parser.add_argument(
        "--allow-inconclusive",
        action="store_true",
        help="Return success (0) on inconclusive max-games stop",
    )
    parser.add_argument("--summary-json", default=None, help="Optional output path for final SPRT summary JSON")

    args = parser.parse_args()

    if args.elo1 <= args.elo0:
        parser.error("--elo1 must be greater than --elo0")
    if not (0.0 < args.alpha < 1.0):
        parser.error("--alpha must be in (0,1)")
    if not (0.0 < args.beta < 1.0):
        parser.error("--beta must be in (0,1)")
    if args.games_per_batch <= 0:
        parser.error("--games-per-batch must be > 0")
    if args.max_games <= 0:
        parser.error("--max-games must be > 0")
    if args.max_games < args.games_per_batch:
        parser.error("--max-games must be >= --games-per-batch")
    if args.movetime_ms <= 0:
        parser.error("--movetime-ms must be > 0")
    if args.depth is not None and args.depth <= 0:
        parser.error("--depth must be > 0")
    if args.game_time_ms is not None and args.game_time_ms <= 0:
        parser.error("--game-time-ms must be > 0")
    if args.inc_ms < 0:
        parser.error("--inc-ms must be >= 0")
    if args.max_plies <= 0:
        parser.error("--max-plies must be > 0")
    if args.practical_min_games < 0:
        parser.error("--practical-min-games must be >= 0")
    if not (0.0 < args.practical_p_threshold < 1.0):
        parser.error("--practical-p-threshold must be in (0,1)")

    return args


def run_batch(args: argparse.Namespace, games: int, seed: int) -> tuple[int, dict]:
    summary_fd, summary_path = tempfile.mkstemp(prefix="sprt_batch_", suffix=".json")
    os.close(summary_fd)
    summary_file = Path(summary_path)

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("selfplay.py")),
        args.engine1,
        args.engine2,
        "--name1",
        args.name1,
        "--name2",
        args.name2,
        "--games",
        str(games),
        "--movetime-ms",
        str(args.movetime_ms),
        "--openings",
        args.openings,
        "--seed",
        str(seed),
        "--max-plies",
        str(args.max_plies),
        "--summary-json",
        str(summary_file),
        "--quiet",
    ]

    if args.shuffle_openings:
        cmd.append("--shuffle-openings")
    if args.depth is not None:
        cmd.extend(["--depth", str(args.depth)])
    if args.game_time_ms is not None:
        cmd.extend(["--game-time-ms", str(args.game_time_ms), "--inc-ms", str(args.inc_ms)])
    if args.threads is not None:
        cmd.extend(["--threads", str(args.threads)])
    if args.hash_mb is not None:
        cmd.extend(["--hash-mb", str(args.hash_mb)])

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode not in (0, 1, 2):
        sys.stderr.write(f"selfplay.py failed with exit code {proc.returncode}\n")
        if proc.stdout:
            sys.stderr.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise RuntimeError("selfplay.py execution failed")

    try:
        summary = json.loads(summary_file.read_text())
    finally:
        summary_file.unlink(missing_ok=True)

    return proc.returncode, summary


def main() -> int:
    args = parse_args()

    p0 = p_from_elo(args.elo0)
    p1 = p_from_elo(args.elo1)
    upper = math.log((1.0 - args.beta) / args.alpha)
    lower = math.log(args.beta / (1.0 - args.alpha))

    totals = Totals()
    decision = "max_games"
    batch_idx = 0

    print("SPRT setup")
    print(f"  H0: {args.elo0:+.1f} Elo, H1: {args.elo1:+.1f} Elo")
    print(f"  alpha={args.alpha:.3f}, beta={args.beta:.3f}")
    print(f"  bounds: lower={lower:+.4f}, upper={upper:+.4f}")
    print(
        "  batch={batch}, max_games={max_games}, movetime_ms={mt}, max_plies={mp}".format(
            batch=args.games_per_batch,
            max_games=args.max_games,
            mt=args.movetime_ms,
            mp=args.max_plies,
        )
    )
    print("")

    while totals.games < args.max_games:
        batch_idx += 1
        remaining = args.max_games - totals.games
        batch_games = min(args.games_per_batch, remaining)
        batch_seed = args.seed + batch_idx

        _, batch_summary = run_batch(args, batch_games, batch_seed)
        r = batch_summary["result"]

        totals.baseline_wins += int(r["engine1_wins"])
        totals.candidate_wins += int(r["engine2_wins"])
        totals.draws += int(r["draws"])

        games = totals.games
        score = totals.candidate_score
        rate = totals.candidate_rate
        llr = compute_llr(score, games, p0, p1)
        p_one = one_sided_p_value_stronger(rate, games)
        elo_est = elo_from_rate(rate)

        print(
            "Batch {batch:02d} | N={games:4d} | WDL(cand)={w}-{d}-{l} | score={score:.1f}/{games} ({pct:.2f}%) | Elo~{elo:+.1f} | LLR={llr:+.4f} | p_one={p_one:.4f}".format(
                batch=batch_idx,
                games=games,
                w=totals.candidate_wins,
                d=totals.draws,
                l=totals.baseline_wins,
                score=score,
                pct=100.0 * rate,
                elo=elo_est,
                llr=llr,
                p_one=p_one,
            )
        )

        if llr >= upper:
            decision = "accept_h1"
            print("")
            print("SPRT decision: ACCEPT H1 (candidate stronger).")
            break
        if llr <= lower:
            decision = "accept_h0"
            print("")
            print("SPRT decision: ACCEPT H0 (candidate weaker).")
            break

        if (
            games >= args.practical_min_games
            and rate > 0.5
            and p_one < args.practical_p_threshold
        ):
            decision = "practical_stronger"
            print("")
            print("Practical decision: candidate stronger by one-sided significance threshold.")
            break

    if decision == "max_games":
        print("")
        print("No SPRT boundary reached before max games.")

    final = {
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "engine1": {"name": args.name1, "path": args.engine1},
        "engine2": {"name": args.name2, "path": args.engine2},
        "settings": {
            "elo0": args.elo0,
            "elo1": args.elo1,
            "alpha": args.alpha,
            "beta": args.beta,
            "games_per_batch": args.games_per_batch,
            "max_games": args.max_games,
            "movetime_ms": args.movetime_ms,
            "depth": args.depth,
            "game_time_ms": args.game_time_ms,
            "inc_ms": args.inc_ms,
            "openings": args.openings,
            "shuffle_openings": bool(args.shuffle_openings),
            "seed": args.seed,
            "max_plies": args.max_plies,
            "threads": args.threads,
            "hash_mb": args.hash_mb,
            "practical_min_games": args.practical_min_games,
            "practical_p_threshold": args.practical_p_threshold,
        },
        "result": {
            "engine1_wins": totals.baseline_wins,
            "engine2_wins": totals.candidate_wins,
            "draws": totals.draws,
            "total_games": totals.games,
            "engine1_score": totals.baseline_wins + 0.5 * totals.draws,
            "engine2_score": totals.candidate_score,
            "engine2_score_rate": totals.candidate_rate,
            "engine2_elo_estimate": elo_from_rate(totals.candidate_rate),
            "engine2_one_sided_p_value_stronger": one_sided_p_value_stronger(totals.candidate_rate, totals.games),
            "llr": compute_llr(totals.candidate_score, totals.games, p0, p1),
            "llr_lower_bound": lower,
            "llr_upper_bound": upper,
            "decision": decision,
        },
    }

    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(final, indent=2) + "\n")

    print("")
    print("Final tally:")
    print(f"  Games={totals.games}, baseline W={totals.baseline_wins}, candidate W={totals.candidate_wins}, draws={totals.draws}")
    print(f"  Candidate score={totals.candidate_score:.1f}/{totals.games} ({100.0 * totals.candidate_rate:.2f}%)")
    print(f"  Decision={decision}")

    if decision in ("accept_h1", "practical_stronger"):
        return 0
    if decision == "accept_h0":
        return 1
    if decision == "max_games" and args.allow_inconclusive:
        return 0
    if decision == "max_games":
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
