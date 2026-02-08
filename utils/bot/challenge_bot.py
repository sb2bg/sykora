#!/usr/bin/env python3
"""Create a Lichess challenge against a given username."""

from __future__ import annotations

import argparse
import json
import os
import random
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
    parser.add_argument(
        "username",
        nargs="?",
        help="Target Lichess username to challenge.",
    )
    parser.add_argument(
        "--random-online-bot",
        action="store_true",
        help="Pick a random online bot and challenge it.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Username to exclude from random selection (repeatable).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic random choice.",
    )
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


def _api_get_json(*, base_url: str, token: str, path: str) -> dict[str, object]:
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}{path}",
        method="GET",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {}
            payload = json.loads(raw)
            return payload if isinstance(payload, dict) else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc


def _fetch_account_username(*, base_url: str, token: str) -> str | None:
    account = _api_get_json(base_url=base_url, token=token, path="/api/account")
    username = account.get("username")
    return str(username).strip() if username else None


def _extract_username(entry: object) -> str | None:
    if isinstance(entry, str):
        value = entry.strip()
        return value if value else None
    if not isinstance(entry, dict):
        return None

    for key in ("username", "name", "id"):
        value = entry.get(key)
        if value:
            text = str(value).strip()
            if text:
                return text
    return None


def _is_busy(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    return bool(entry.get("playing"))


def _iter_online_bot_entries(*, base_url: str, token: str, max_entries: int = 1000) -> list[object]:
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/api/bot/online",
        method="GET",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/x-ndjson, application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            entries: list[object] = []
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(payload, list):
                    for item in payload:
                        entries.append(item)
                        if len(entries) >= max_entries:
                            return entries
                    continue

                entries.append(payload)
                if len(entries) >= max_entries:
                    return entries
            return entries
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc


def _pick_random_online_bot(
    *,
    base_url: str,
    token: str,
    exclude: set[str],
    seed: int | None,
) -> str:
    entries = _iter_online_bot_entries(base_url=base_url, token=token)

    candidates: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        username = _extract_username(entry)
        if not username:
            continue
        key = username.lower()
        if key in exclude or key in seen:
            continue
        if _is_busy(entry):
            continue
        seen.add(key)
        candidates.append(username)

    if not candidates:
        raise RuntimeError("No eligible online bots found.")

    chooser = random.Random(seed)
    return chooser.choice(candidates)


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
        if args.random_online_bot and args.username:
            raise ValueError("Provide a username or --random-online-bot, not both.")
        if not args.random_online_bot and not args.username:
            raise ValueError("Provide a username or pass --random-online-bot.")

        username = args.username
        if args.random_online_bot:
            exclude = {name.strip().lower() for name in args.exclude if name.strip()}
            self_username = _fetch_account_username(base_url=args.base_url, token=token)
            if self_username:
                exclude.add(self_username.lower())
            username = _pick_random_online_bot(
                base_url=args.base_url,
                token=token,
                exclude=exclude,
                seed=args.seed,
            )

        assert username is not None
        response = _post_challenge(
            base_url=args.base_url,
            token=token,
            username=username,
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
    print(f"Challenged {username}.")
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
