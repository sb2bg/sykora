#!/usr/bin/env python3
"""Bootstrap helpers for the vendored-on-demand Bullet repo."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BULLET_REPO = REPO_ROOT / "nnue" / "bullet_repo"
DEFAULT_REMOTE = "https://github.com/jw1912/bullet.git"
PINNED_COMMIT = "4e9317ffb07ee01b3ae5202d083526fc1d90fa2f"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_bullet_repo(
    repo_path: Path,
    *,
    remote: str = DEFAULT_REMOTE,
    pinned_commit: str = PINNED_COMMIT,
    build_utils: bool = False,
) -> Path:
    repo_path = repo_path.resolve()

    if not repo_path.exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(["git", "clone", remote, str(repo_path)])
        run_cmd(["git", "checkout", pinned_commit], cwd=repo_path)

    if build_utils:
        bullet_utils = repo_path / "target" / "release" / "bullet-utils"
        if not bullet_utils.is_file():
            run_cmd(["cargo", "build", "-r", "-p", "bullet-utils"], cwd=repo_path)

    return repo_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the Bullet training repo.")
    parser.add_argument(
        "--bullet-repo",
        default=str(DEFAULT_BULLET_REPO),
        help="Destination path for the Bullet checkout",
    )
    parser.add_argument(
        "--build-utils",
        action="store_true",
        help="Build bullet-utils after ensuring the repo exists",
    )
    args = parser.parse_args()

    path = ensure_bullet_repo(Path(args.bullet_repo), build_utils=args.build_utils)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
