#!/usr/bin/env python3
"""Bulk-download .pgn.gz files from finished fishtest runs.

Directory layout:
  PATH/YY-MM-DD/<test-id>/<test-id>.pgn.gz
  PATH/YY-MM-DD/<test-id>/<test-id>.json
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional


def parse_bool_flag(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean flag: {raw!r} (expected true/false)")


def format_large_number(number: int) -> str:
    suffixes = ["", "K", "M", "G", "T", "P"]
    n = float(number)
    for suffix in suffixes:
        if n < 1000.0:
            return f"{n:.0f}{suffix}"
        n /= 1000.0
    return f"{n:.0f}{suffixes[-1]}"


def open_file_rt(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else path.open("rt")


def count_games(path: Path) -> int:
    count = 0
    with open_file_rt(path) as handle:
        for line in handle:
            if "Result" in line and ("1-0" in line or "0-1" in line or "1/2-1/2" in line):
                count += 1
    return count


def find_downloaded_ids(root: Path) -> set[str]:
    pattern = re.compile(r"([a-z0-9]*)(-[0-9]*)?\.pgn(|\.gz)$")
    downloaded: set[str] = set()
    for _, _, files in os.walk(root):
        for name in files:
            match = pattern.match(name)
            if match:
                downloaded.add(match.group(1))
    return downloaded


def tc_base_value(tc: str) -> Optional[float]:
    match = re.search(r"^(\d+(\.\d+)?)", tc)
    if not match:
        return None
    return float(match.group(1))


def should_skip_for_tc(tc_strings: Iterable[str], low: Optional[float], high: Optional[float]) -> bool:
    for tc in tc_strings:
        base = tc_base_value(tc)
        if base is None:
            return True
        if low is not None and base < low:
            return True
        if high is not None and base > high:
            return True
    return False


def build_finished_runs_query(args: argparse.Namespace) -> str:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    lower_bound_utc = now_utc - datetime.timedelta(hours=args.time_delta)
    unix_timestamp = lower_bound_utc.timestamp()

    parts = [f"timestamp={unix_timestamp}"]
    if args.ltc_only:
        parts.append("ltc_only=1")
    if args.success_only:
        parts.append("success_only=1")
    if args.yellow_only:
        parts.append("yellow_only=1")
    if args.username:
        parts.append(f"username={args.username}")
    return "&".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk-download .pgn.gz files from finished tests on fishtest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        default="./pgns",
        help="Downloaded .pgn.gz files will be stored in PATH/YY-MM-DD/test-Id/.",
    )
    parser.add_argument(
        "--time-delta",
        type=float,
        default=168.0,
        help="Delta of hours from now since the desired tests are last updated.",
    )
    parser.add_argument(
        "--ltc-only",
        type=parse_bool_flag,
        default=True,
        help="Download LTC tests only.",
    )
    parser.add_argument(
        "--tc-lower-limit",
        type=float,
        default=None,
        help="Download only tests where base tc for each side is at least this.",
    )
    parser.add_argument(
        "--tc-upper-limit",
        type=float,
        default=None,
        help="Download only tests where base tc for each side is at most this.",
    )
    parser.add_argument(
        "--success-only",
        type=parse_bool_flag,
        default=False,
        help="Download green tests only.",
    )
    parser.add_argument(
        "--yellow-only",
        type=parse_bool_flag,
        default=False,
        help="Download yellow tests only.",
    )
    parser.add_argument(
        "--standard-chess-only",
        type=parse_bool_flag,
        default=False,
        help="Download tests with standard chess books only.",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Download tests by this username only.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output with -v or -vv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.path).resolve()
    root.mkdir(parents=True, exist_ok=True)

    downloaded = find_downloaded_ids(root)
    print(f"Found {len(downloaded)} downloaded tests in {root} already.")

    query = build_finished_runs_query(args)
    page = 1
    had_error = False

    while True:
        url = f"https://tests.stockfishchess.org/api/finished_runs?page={page}&{query}"
        try:
            with urllib.request.urlopen(url) as response:
                payload = response.read().decode("utf-8")
                response_json: Dict[str, dict] = json.loads(payload)
        except urllib.error.HTTPError as exc:
            print(f"HTTP Error: {exc.code} - {exc.reason}")
            had_error = True
            break
        except urllib.error.URLError as exc:
            print(f"URL Error: {exc.reason}")
            had_error = True
            break
        except json.JSONDecodeError as exc:
            print(f"JSON decoding error: {exc}")
            had_error = True
            break
        except Exception as exc:  # pragma: no cover
            print(f"Error: {exc}")
            had_error = True
            break

        if not response_json:
            break

        for test_id, meta in response_json.items():
            if test_id in downloaded:
                continue
            if "spsa" in meta.get("args", {}):
                if args.verbose:
                    print(f"Skipping SPSA test {test_id}")
                continue

            start_time = meta.get("start_time")
            if not start_time:
                continue

            try:
                date_dir = datetime.datetime.fromisoformat(start_time).strftime("%y-%m-%d")
            except ValueError:
                continue

            results = meta.get("results", {})
            wins = int(results.get("wins", 0))
            draws = int(results.get("draws", 0))
            losses = int(results.get("losses", 0))
            games = wins + draws + losses
            if games <= 0:
                if args.verbose:
                    print(f"No games found, skipping test {test_id}")
                continue

            tc_strings: list[str] = []
            run_args = meta.get("args", {})
            tc = run_args.get("tc", "")
            new_tc = run_args.get("new_tc", "")
            if tc and new_tc:
                tc_strings = [tc] if tc == new_tc else [tc, new_tc]

            book = str(run_args.get("book", ""))
            if args.standard_chess_only and "frc" in book.lower():
                if args.verbose:
                    print(f"Skipping FRC test {test_id} with book {book}")
                continue

            if (args.tc_lower_limit is not None or args.tc_upper_limit is not None) and (
                not tc_strings or should_skip_for_tc(tc_strings, args.tc_lower_limit, args.tc_upper_limit)
            ):
                if args.verbose:
                    print(f"Skipping test {test_id} due to tc filter")
                continue

            target_dir = root / date_dir / test_id
            target_dir.mkdir(parents=True, exist_ok=True)
            pgn_gz_path = target_dir / f"{test_id}.pgn.gz"
            tmp_path = target_dir / f"{test_id}.tmp"
            meta_path = target_dir / f"{test_id}.json"
            pgn_url = f"https://tests.stockfishchess.org/api/run_pgns/{test_id}.pgn.gz"

            try:
                with urllib.request.urlopen(pgn_url) as response:
                    content_length = response.getheader("Content-Length")
                size_txt = ""
                if content_length:
                    size_txt = f"{format_large_number(int(content_length))}B "

                msg = f"Downloading {size_txt}.pgn.gz with {games} games"
                if args.verbose:
                    msg += f" (WDL={wins}/{draws}/{losses})"
                    if tc_strings:
                        msg += " at TC " + " vs ".join(tc_strings)
                print(msg + f" to {target_dir} ...")

                urllib.request.urlretrieve(pgn_url, str(tmp_path))
                os.replace(tmp_path, pgn_gz_path)
                meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

                downloaded.add(test_id)

                if args.verbose:
                    actual_games = count_games(pgn_gz_path)
                    print(f"Download completed. The file contains {actual_games} games.")
            except Exception as exc:
                if args.verbose >= 2:
                    print(f"Skipping {test_id}: {exc}")
                continue

        page += 1

    print("Finished downloading PGNs.")
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
