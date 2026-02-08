#!/usr/bin/env python3
"""Create a Lichess challenge against a given username."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _resolve_token(explicit_token: str | None) -> str:
    if explicit_token:
        return explicit_token.strip()

    env_token = os.getenv("LICHESS_API_TOKEN")
    if env_token:
        return env_token.strip()

    dotenv_token = _parse_dotenv(Path(".env")).get("LICHESS_API_TOKEN")
    if dotenv_token:
        return dotenv_token.strip()

    raise ValueError("LICHESS_API_TOKEN not found. Set env var or pass --token.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Challenge a Lichess user/bot account.",
    )
    parser.add_argument("username", help="Target Lichess username to challenge.")
    parser.add_argument(
        "--token",
        default=None,
        help="API token. If omitted, uses LICHESS_API_TOKEN from env or .env.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LICHESS_BASE_URL", "https://lichess.org"),
        help="Lichess base URL (default: https://lichess.org).",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=3.0,
        help="Initial clock in minutes (default: 3.0).",
    )
    parser.add_argument(
        "--increment",
        type=int,
        default=2,
        help="Increment per move in seconds (default: 2).",
    )
    parser.add_argument(
        "--rated",
        action="store_true",
        help="Create a rated challenge (default: casual).",
    )
    parser.add_argument(
        "--color",
        choices=("white", "black", "random"),
        default="random",
        help="Preferred color (default: random).",
    )
    parser.add_argument(
        "--variant",
        default="standard",
        help="Variant key (default: standard).",
    )
    return parser


def _post_challenge(
    *,
    base_url: str,
    token: str,
    username: str,
    minutes: float,
    increment: int,
    rated: bool,
    color: str,
    variant: str,
) -> dict[str, object]:
    if minutes <= 0:
        raise ValueError("--minutes must be > 0")
    if increment < 0:
        raise ValueError("--increment must be >= 0")

    clock_limit_seconds = int(round(minutes * 60.0))
    if clock_limit_seconds < 1:
        raise ValueError("--minutes is too small")

    payload = urllib.parse.urlencode(
        {
            "rated": "true" if rated else "false",
            "variant": variant,
            "color": color,
            "clock.limit": str(clock_limit_seconds),
            "clock.increment": str(increment),
        }
    ).encode("utf-8")

    url = f"{base_url.rstrip('/')}/api/challenge/{username}"
    req = urllib.request.Request(
        url=url,
        method="POST",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc


def _extract_challenge_info(response: dict[str, object]) -> tuple[str | None, str | None]:
    candidate: object = response.get("challenge")
    data = candidate if isinstance(candidate, dict) else response
    challenge_id = data.get("id") if isinstance(data, dict) else None
    challenge_url = data.get("url") if isinstance(data, dict) else None

    cid = str(challenge_id).strip() if challenge_id else None
    curl = str(challenge_url).strip() if challenge_url else None
    return cid, curl


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        token = _resolve_token(args.token)
        response = _post_challenge(
            base_url=args.base_url,
            token=token,
            username=args.username,
            minutes=args.minutes,
            increment=args.increment,
            rated=args.rated,
            color=args.color,
            variant=args.variant,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    challenge_id, challenge_url = _extract_challenge_info(response)
    print(f"Challenged {args.username}.")
    if challenge_id:
        print(f"Challenge ID: {challenge_id}")
    if challenge_url:
        print(f"Challenge URL: {challenge_url}")
    if not challenge_id and not challenge_url:
        print("Response:")
        print(json.dumps(response, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
