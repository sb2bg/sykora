#!/usr/bin/env python3
"""Reproduce the September 4 audit findings without modifying engine source.

Usage: python3 reproduce.py --zig /path/to/zig-0.15.2
The fixtures assert observed pre-fix behavior, not desired behavior.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zig", default="zig")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    repo = args.repo.resolve()
    fixtures = Path(__file__).resolve().parent

    print("Compiler:", flush=True)
    subprocess.run([args.zig, "version"], check=True)
    print("Repository revision:", flush=True)
    subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True)

    with tempfile.TemporaryDirectory(prefix="sykora-audit-repros-") as directory:
        temporary = Path(directory)
        for source in (repo / "src").rglob("*.zig"):
            target = temporary / source.relative_to(repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (temporary / "src/net.sknnue").symlink_to(repo / "src/net.sknnue")

        for name, fixture in (
            ("search.zig", "search_repros.zig.txt"),
            ("gensfen.zig", "gensfen_repros.zig.txt"),
        ):
            target = temporary / "src" / name
            with target.open("a") as output:
                output.write("\n" + (fixtures / fixture).read_text())
            print(f"\nIsolated diagnostics for {name}:", flush=True)
            subprocess.run(
                [args.zig, "test", str(target), "-O", "ReleaseSafe", "--test-filter", "audit"],
                cwd=temporary,
                check=True,
            )


if __name__ == "__main__":
    main()
