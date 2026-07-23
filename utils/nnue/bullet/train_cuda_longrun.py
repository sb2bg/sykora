#!/usr/bin/env python3
"""Launch reproducible staged Sykora NNUE training runs with Bullet."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from bootstrap import (  # noqa: E402
    DEFAULT_BULLET_REPO,
    PINNED_COMMIT,
    ensure_bullet_repo,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_MANIFEST = REPO_ROOT / "utils" / "nnue" / "bullet_runner" / "Cargo.toml"

V3_BUCKET_LAYOUT_32 = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
]

BUCKET_LAYOUTS_32 = {"v3_10": V3_BUCKET_LAYOUT_32}


def expand_mirrored_bucket_layout(layout_32: list[int]) -> list[int]:
    mirror = [0, 1, 2, 3, 3, 2, 1, 0]
    return [int(layout_32[(idx // 8) * 4 + mirror[idx % 8]]) for idx in range(64)]


def bucket_layout_64(name: str) -> list[int]:
    if name not in BUCKET_LAYOUTS_32:
        raise ValueError(f"unsupported bucket layout: {name!r}")
    return expand_mirrored_bucket_layout(BUCKET_LAYOUTS_32[name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch staged Bullet NNUE training runs.")
    parser.add_argument(
        "--dataset",
        required=True,
        nargs="+",
        help="Training dataset file(s), excluding held-out validation shards",
    )
    parser.add_argument(
        "--validation-dataset",
        nargs="*",
        default=[],
        help="Held-out SF binpack shard(s) used only to build a validation sample",
    )
    parser.add_argument(
        "--validation-sample",
        default="",
        help="Existing Bullet-format validation sample; bypasses sample generation",
    )
    parser.add_argument(
        "--validation-cache",
        default="",
        help=(
            "Persistent Bullet-format validation sample: reuse it when present, or create it "
            "from --validation-dataset when absent"
        ),
    )
    parser.add_argument(
        "--validation-positions",
        type=int,
        default=262_144,
        help="Positions in the held-out validation sample",
    )
    parser.add_argument(
        "--validation-buffer-mb",
        type=int,
        default=512,
        help="Shuffle buffer used while sampling held-out SF binpacks",
    )
    parser.add_argument(
        "--validate-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rank saved checkpoints on the held-out sample when training finishes",
    )
    parser.add_argument(
        "--validate-all-checkpoints",
        action="store_true",
        help="Validate every saved checkpoint instead of only the final checkpoint",
    )
    parser.add_argument(
        "--export-best-validation",
        action="store_true",
        help="Export the lowest-MSE validated checkpoint instead of blindly exporting the final one",
    )

    parser.add_argument(
        "--bullet-repo", default=str(DEFAULT_BULLET_REPO), help="Path to Bullet repository"
    )
    parser.add_argument(
        "--allow-bullet-revision",
        action="store_true",
        help="Allow a Bullet HEAD other than the repository pin (still recorded)",
    )
    parser.add_argument(
        "--require-clean-bullet",
        action="store_true",
        help="Refuse a dirty Bullet checkout instead of snapshotting its diff",
    )
    parser.add_argument(
        "--full-dataset-hash",
        action="store_true",
        help="SHA-256 every complete dataset (slow for large binpacks)",
    )
    parser.add_argument(
        "--output-root", default="nnue/models/bullet", help="Training run root"
    )
    parser.add_argument("--run-id", default="", help="Run identifier")

    parser.add_argument(
        "--architecture",
        choices=["pairwise-linear", "pairwise-mlp"],
        default="pairwise-mlp",
        help="Network graph to train",
    )
    parser.add_argument(
        "--network-format",
        choices=["syk7", "syk8"],
        default="syk7",
        help="Checkpoint architecture family",
    )
    parser.add_argument("--hidden", type=int, default=1024, help="FT width")
    parser.add_argument("--dense1", type=int, default=16, help="First v7 dense width")
    parser.add_argument("--dense2", type=int, default=32, help="Second v7 dense width")
    parser.add_argument(
        "--bucket-layout", choices=["v3_10"], default="v3_10", help="Input bucket layout"
    )
    parser.add_argument(
        "--output-buckets", type=int, default=8, choices=[1, 8], help="Output head count"
    )

    parser.add_argument("--start-superbatch", type=int, default=1)
    parser.add_argument("--end-superbatch", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--batches-per-superbatch", type=int, default=6104)
    parser.add_argument("--lr-start", type=float, default=0.0010)
    parser.add_argument(
        "--lr-origin-superbatch",
        type=int,
        default=1,
        help="Superbatch at which this cosine phase starts (set to fine-tune start to restart LR)",
    )
    parser.add_argument(
        "--lr-final-superbatch",
        type=int,
        default=0,
        help="Cosine horizon; 0 uses end-superbatch (set to 800 for a resumable v8 pilot)",
    )
    parser.add_argument(
        "--lr-final", type=float, default=0.0, help="0 uses lr_start * 0.3^5"
    )
    parser.add_argument("--wdl", type=float, default=0.75)
    parser.add_argument("--save-rate", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--backend",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Bullet execution backend (CUDA is the production default)",
    )

    parser.add_argument(
        "--data-format", choices=["bullet", "binpack"], default="bullet"
    )
    parser.add_argument("--binpack-buffer-mb", type=int, default=1024)
    parser.add_argument("--binpack-threads", type=int, default=4)
    parser.add_argument(
        "--resume",
        default="",
        help="Checkpoint directory whose optimiser_state should be resumed",
    )
    parser.add_argument(
        "--warm-start",
        default="",
        help="Full-precision v7 checkpoint used to initialise a zero-threat T1024 run",
    )
    parser.add_argument(
        "--allow-random-v8-init",
        action="store_true",
        help="Allow v8 without a v7 warm start (smoke diagnostics only)",
    )
    parser.add_argument(
        "--export-after",
        action="store_true",
        help="Convert and export the final checkpoint after training",
    )
    parser.add_argument(
        "--parity-engine",
        default="",
        help="Engine binary used for bit-exact parity after export",
    )
    parser.add_argument(
        "--parity-fens",
        default=str(REPO_ROOT / "utils" / "nnue" / "parity.fens"),
        help="FEN suite used with --parity-engine",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_text(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()


def git_snapshot(repo: Path, output_dir: Path, name: str) -> dict:
    if not (repo / ".git").exists():
        return {"path": str(repo.resolve()), "is_git_repo": False}

    head = run_text(["git", "rev-parse", "HEAD"], repo)
    status = run_text(["git", "status", "--porcelain=v1"], repo)
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    diff_hash = hashlib.sha256(diff).hexdigest()
    patch_path = output_dir / f"{name}.patch"
    if diff:
        patch_path.write_bytes(diff)

    return {
        "path": str(repo.resolve()),
        "is_git_repo": True,
        "head": head,
        "dirty": bool(status),
        "status": status.splitlines(),
        "tracked_diff_sha256": diff_hash,
        "tracked_patch": str(patch_path.resolve()) if diff else "",
    }


def dataset_fingerprint(path: Path, full_hash: bool) -> dict:
    stat = path.stat()
    hasher = hashlib.sha256()
    mode = "full" if full_hash else "head-tail-1MiB"
    with path.open("rb") as handle:
        if full_hash:
            while chunk := handle.read(8 * 1024 * 1024):
                hasher.update(chunk)
        else:
            hasher.update(stat.st_size.to_bytes(8, "little", signed=False))
            hasher.update(handle.read(1024 * 1024))
            if stat.st_size > 1024 * 1024:
                handle.seek(max(0, stat.st_size - 1024 * 1024))
                hasher.update(handle.read(1024 * 1024))
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "fingerprint_mode": mode,
        "sha256": hasher.hexdigest(),
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.validation_sample and args.validation_cache:
        raise ValueError("--validation-sample and --validation-cache are mutually exclusive")
    positive = {
        "--hidden": args.hidden,
        "--dense1": args.dense1,
        "--dense2": args.dense2,
        "--batch-size": args.batch_size,
        "--batches-per-superbatch": args.batches_per_superbatch,
        "--save-rate": args.save_rate,
        "--threads": args.threads,
        "--binpack-buffer-mb": args.binpack_buffer_mb,
        "--binpack-threads": args.binpack_threads,
        "--validation-positions": args.validation_positions,
        "--validation-buffer-mb": args.validation_buffer_mb,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    if args.start_superbatch <= 0 or args.end_superbatch < args.start_superbatch:
        raise ValueError("invalid superbatch bounds")
    if args.lr_origin_superbatch <= 0 or args.lr_origin_superbatch > args.start_superbatch:
        raise ValueError("--lr-origin-superbatch must be in [1, start-superbatch]")
    lr_final_superbatch = args.lr_final_superbatch or args.end_superbatch
    if lr_final_superbatch < args.end_superbatch or lr_final_superbatch < args.lr_origin_superbatch:
        raise ValueError(
            "--lr-final-superbatch must be >= end-superbatch and lr-origin-superbatch"
        )
    if args.lr_start <= 0 or args.lr_final < 0:
        raise ValueError("learning rates must be non-negative and lr_start must be > 0")
    if not 0.0 <= args.wdl <= 1.0:
        raise ValueError("--wdl must be in [0, 1]")
    if args.resume and args.warm_start:
        raise ValueError("--resume and --warm-start are mutually exclusive")
    if args.warm_start and args.network_format != "syk8":
        raise ValueError("--warm-start is only supported for syk8 T1024")
    if args.allow_random_v8_init and args.network_format != "syk8":
        raise ValueError("--allow-random-v8-init is only valid for syk8")
    if args.allow_random_v8_init and (args.resume or args.warm_start):
        raise ValueError("--allow-random-v8-init cannot be combined with resume or warm start")
    if args.architecture == "pairwise-linear" and args.network_format != "syk7":
        raise ValueError("pairwise-linear must use --network-format syk7")
    if args.architecture == "pairwise-mlp" and args.network_format not in {"syk7", "syk8"}:
        raise ValueError("pairwise-mlp must use --network-format syk7 or syk8")
    if args.network_format == "syk8":
        if args.architecture != "pairwise-mlp":
            raise ValueError("syk8 requires --architecture pairwise-mlp")
        if args.hidden not in {768, 1024}:
            raise ValueError("syk8 registered widths are --hidden 1024 (T1024) and 768 (T768)")
        if args.dense1 != 16 or args.dense2 != 32 or args.output_buckets != 8:
            raise ValueError("syk8 requires --dense1 16 --dense2 32 --output-buckets 8")
        if not args.resume and not args.warm_start and not args.allow_random_v8_init:
            raise ValueError(
                "syk8 requires --warm-start/--resume, or --allow-random-v8-init for diagnostics"
            )
        if args.hidden == 768 and args.warm_start:
            raise ValueError("T768 cannot use the exact H=1024 v7 warm start")
    if args.hidden % 2:
        raise ValueError("pairwise architectures require an even --hidden")
    if args.export_after and args.architecture == "pairwise-linear":
        raise ValueError("pairwise-linear is an ablation and has no deployable exporter")
    if args.parity_engine and not args.export_after:
        raise ValueError("--parity-engine requires --export-after")
    if args.export_best_validation and (not args.export_after or not args.validate_after):
        raise ValueError("--export-best-validation requires validation and --export-after")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    datasets = [Path(path).resolve() for path in args.dataset]
    validation_datasets = [Path(path).resolve() for path in args.validation_dataset]
    for dataset in datasets + validation_datasets:
        if not dataset.is_file():
            print(f"Dataset not found: {dataset}", file=sys.stderr)
            return 1
    overlap = set(datasets) & set(validation_datasets)
    if overlap:
        print(
            "Validation shards must not also be training shards: "
            + ", ".join(str(path) for path in sorted(overlap)),
            file=sys.stderr,
        )
        return 2

    data_format = args.data_format
    if data_format == "bullet" and any(path.suffix == ".binpack" for path in datasets):
        data_format = "binpack"
        print("Auto-detected binpack training data")
    if validation_datasets and any(path.suffix != ".binpack" for path in validation_datasets):
        print("--validation-dataset currently requires SF .binpack files", file=sys.stderr)
        return 2

    bullet_repo = ensure_bullet_repo(Path(args.bullet_repo))
    bullet_head = run_text(["git", "rev-parse", "HEAD"], bullet_repo)
    if bullet_head != PINNED_COMMIT and not args.allow_bullet_revision:
        print(
            f"Bullet HEAD is {bullet_head}, expected {PINNED_COMMIT}; "
            "use --allow-bullet-revision only for an intentional experiment",
            file=sys.stderr,
        )
        return 2

    run_id = args.run_id.strip() or datetime.datetime.now(datetime.UTC).strftime(
        "run_%Y%m%dT%H%M%SZ"
    )
    run_dir = Path(args.output_root).resolve() / run_id
    ckpt_dir = run_dir / "checkpoints"
    provenance_dir = run_dir / "provenance"
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"Run directory already exists and is not empty: {run_dir}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    warm_start_cmd: list[str] = []
    warm_start_weights = Path()
    warm_start_report = Path()
    if args.warm_start:
        warm_source = Path(args.warm_start).resolve()
        if not warm_source.exists():
            print(f"Warm-start source not found: {warm_source}", file=sys.stderr)
            return 1
        warm_dir = run_dir / "warm_start"
        warm_start_weights = warm_dir / "weights.bin"
        warm_start_report = warm_dir / "verification.json"
        warm_start_cmd = [
            sys.executable,
            str(THIS_DIR / "warm_start_v8.py"),
            "--source",
            str(warm_source),
            "--output",
            str(warm_start_weights),
            "--report",
            str(warm_start_report),
        ]

    repo_provenance = git_snapshot(REPO_ROOT, provenance_dir, "sykora")
    bullet_provenance = git_snapshot(bullet_repo, provenance_dir, "bullet")
    if args.require_clean_bullet and bullet_provenance.get("dirty"):
        print("Bullet checkout is dirty; see git status or omit --require-clean-bullet", file=sys.stderr)
        return 2
    if bullet_provenance.get("dirty"):
        print("WARNING: Bullet checkout is dirty; the tracked diff is snapshotted in run metadata")

    validation_sample_arg = args.validation_sample or args.validation_cache
    validation_sample = (
        Path(validation_sample_arg).resolve()
        if validation_sample_arg
        else run_dir / "validation.data"
    )
    sample_cmd: list[str] = []
    if args.validation_sample:
        if not validation_sample.is_file():
            print(f"Validation sample not found: {validation_sample}", file=sys.stderr)
            return 1
    elif args.validation_cache and validation_sample.is_file():
        expected_bytes = args.validation_positions * 32
        actual_bytes = validation_sample.stat().st_size
        if actual_bytes != expected_bytes:
            print(
                f"Validation cache has {actual_bytes} bytes, expected {expected_bytes} "
                f"for {args.validation_positions} positions: {validation_sample}",
                file=sys.stderr,
            )
            return 2
        print(f"Reusing validation cache: {validation_sample}")
    elif validation_datasets:
        sample_cmd = [
            "cargo",
            "run",
            "-r",
            "--no-default-features",
            "--features",
            "cpu",
            "--manifest-path",
            str(RUNNER_MANIFEST),
            "--bin",
            "sample_binpack",
            "--",
        ]
        for dataset in validation_datasets:
            sample_cmd.extend(["--input", str(dataset)])
        sample_cmd.extend(
            [
                "--output",
                str(validation_sample),
                "--positions",
                str(args.validation_positions),
                "--buffer-mb",
                str(args.validation_buffer_mb),
                "--threads",
                str(args.binpack_threads),
            ]
        )
    elif args.validation_cache:
        print(
            f"Validation cache does not exist and no --validation-dataset can create it: "
            f"{validation_sample}",
            file=sys.stderr,
        )
        return 2
    else:
        validation_sample = Path()
        print("WARNING: no held-out validation data was supplied")
    if args.export_best_validation and str(validation_sample) == ".":
        print("--export-best-validation requires held-out validation data", file=sys.stderr)
        return 2

    final_lr = args.lr_final if args.lr_final > 0 else args.lr_start * (0.3**5)
    lr_final_superbatch = args.lr_final_superbatch or args.end_superbatch
    dataset_str = ";".join(str(path) for path in datasets)
    env = os.environ.copy()
    env.update(
        {
            "SYK_DATASET": dataset_str,
            "SYK_ARCHITECTURE": args.architecture,
            "SYK_HIDDEN": str(args.hidden),
            "SYK_DENSE1": str(args.dense1),
            "SYK_DENSE2": str(args.dense2),
            "SYK_NETWORK_FORMAT": args.network_format,
            "SYK_BUCKET_LAYOUT": args.bucket_layout,
            "SYK_LR_START": str(args.lr_start),
            "SYK_LR_FINAL": str(final_lr),
            "SYK_LR_ORIGIN_SUPERBATCH": str(args.lr_origin_superbatch),
            "SYK_LR_FINAL_SUPERBATCH": str(lr_final_superbatch),
            "SYK_START_SUPERBATCH": str(args.start_superbatch),
            "SYK_END_SUPERBATCH": str(args.end_superbatch),
            "SYK_BATCH_SIZE": str(args.batch_size),
            "SYK_BATCHES_PER_SUPERBATCH": str(args.batches_per_superbatch),
            "SYK_WDL": str(args.wdl),
            "SYK_SAVE_RATE": str(args.save_rate),
            "SYK_THREADS": str(args.threads),
            "SYK_OUTPUT_BUCKETS": str(args.output_buckets),
            "SYK_OUTPUT_DIR": str(ckpt_dir),
            "SYK_NET_ID": run_id,
            "SYK_DATA_FORMAT": data_format,
            "SYK_BINPACK_BUFFER_MB": str(args.binpack_buffer_mb),
            "SYK_BINPACK_THREADS": str(args.binpack_threads),
            "SYK_VALIDATION_SAMPLE": str(validation_sample) if str(validation_sample) != "." else "",
        }
    )
    if warm_start_cmd:
        env["SYK_WARM_START_WEIGHTS"] = str(warm_start_weights)
    resume_metadata = None
    if args.resume:
        resume = Path(args.resume).resolve()
        if not (resume / "optimiser_state").is_dir():
            print(f"Resume checkpoint lacks optimiser_state: {resume}", file=sys.stderr)
            return 1
        checkpoint_match = re.search(r"-(\d+)$", resume.name)
        if checkpoint_match and args.start_superbatch <= int(checkpoint_match.group(1)):
            print(
                f"--start-superbatch must be greater than resumed checkpoint "
                f"{checkpoint_match.group(1)}",
                file=sys.stderr,
            )
            return 2
        resume_meta_path = resume.parent.parent / "run_meta.json"
        if resume_meta_path.is_file():
            resume_metadata = json.loads(resume_meta_path.read_text())
            previous = resume_metadata.get("network", {})
            expected = {
                "format": args.network_format,
                "architecture": args.architecture,
                "bucket_layout_name": args.bucket_layout,
                "ft_hidden": args.hidden,
                "dense1": args.dense1,
                "dense2": args.dense2,
                "output_bucket_count": args.output_buckets,
            }
            mismatches = {
                key: (previous.get(key), value)
                for key, value in expected.items()
                if previous.get(key) is not None and previous.get(key) != value
            }
            if mismatches:
                print(f"Resume architecture mismatch: {mismatches}", file=sys.stderr)
                return 2
        else:
            print(f"WARNING: no run_meta.json found for resume checkpoint {resume}")
        env["SYK_RESUME"] = str(resume)

    train_cmd = [
        "cargo",
        "run",
        "-r",
    ]
    if args.backend == "cpu":
        train_cmd.extend(["--no-default-features", "--features", "cpu"])
    train_cmd.extend(
        [
            "--manifest-path",
            str(RUNNER_MANIFEST),
            "--bin",
            "sykora-bullet-runner",
        ]
    )
    validation_cmd = [
        sys.executable,
        str(THIS_DIR / "validate_checkpoints.py"),
        "--checkpoints-dir",
        str(ckpt_dir),
        "--validation-data",
        str(validation_sample),
        "--run-meta",
        str(run_dir / "run_meta.json"),
        "--output",
        str(run_dir / "validation_results.json"),
    ]
    if not args.validate_all_checkpoints:
        validation_cmd.append("--final-only")

    final_checkpoint = ckpt_dir / f"{run_id}-{args.end_superbatch}"
    final_npz = run_dir / "final.npz"
    final_net = run_dir / f"{run_id}.sknnue"
    converter_cmd = [
        sys.executable,
        str(THIS_DIR / "checkpoint_raw_to_npz.py"),
        "--input",
        str(final_checkpoint),
        "--run-meta",
        str(run_dir / "run_meta.json"),
        "--output",
        str(final_npz),
    ]
    exporter_name = {
        "syk7": "export_npz_to_syk7.py",
        "syk8": "export_npz_to_syk8.py",
    }[args.network_format]
    export_cmd = [
        sys.executable,
        str(THIS_DIR / exporter_name),
        "--input",
        str(final_npz),
        "--output-net",
        str(final_net),
    ]
    if args.network_format == "syk8" and args.allow_random_v8_init:
        export_cmd.append("--allow-clipping")
    parity_cmd: list[str] = []
    if args.parity_engine:
        parity_engine = Path(args.parity_engine).resolve()
        parity_fens = Path(args.parity_fens).resolve()
        if not parity_engine.is_file() or not parity_fens.is_file():
            print(f"Parity engine or FEN suite missing: {parity_engine}, {parity_fens}", file=sys.stderr)
            return 2
        parity_cmd = [
            sys.executable,
            str(THIS_DIR / "check_net_parity.py"),
            "--net",
            str(final_net),
            "--fens",
            str(parity_fens),
            "--engine",
            str(parity_engine),
        ]

    train_fingerprints = [dataset_fingerprint(path, args.full_dataset_hash) for path in datasets]
    validation_fingerprints = [
        dataset_fingerprint(path, args.full_dataset_hash) for path in validation_datasets
    ]
    tool_sources = [
        REPO_ROOT / "utils" / "nnue" / "bullet_runner" / "src" / "main.rs",
        REPO_ROOT / "utils" / "nnue" / "bullet_runner" / "src" / "full_threats_v1.rs",
        REPO_ROOT / "utils" / "nnue" / "bullet_runner" / "src" / "bin" / "sample_binpack.rs",
        REPO_ROOT / "src" / "nnue.zig",
        REPO_ROOT / "src" / "full_threats_v1.zig",
        REPO_ROOT / "specs" / "syknnue8_spec.md",
        Path(__file__).resolve(),
        REPO_ROOT / "utils" / "nnue" / "common.py",
        THIS_DIR / "checkpoint_raw_to_npz.py",
        THIS_DIR / "validate_checkpoints.py",
        THIS_DIR / "export_npz_to_syk7.py",
        THIS_DIR / "export_npz_to_syk8.py",
        THIS_DIR / "warm_start_v8.py",
        REPO_ROOT / "utils" / "nnue" / "full_threats_v1.py",
        REPO_ROOT / "utils" / "nnue" / "full_threats_v1.bin",
        REPO_ROOT / "utils" / "nnue" / "full_threats_v1_manifest.json",
        THIS_DIR / "check_net_parity.py",
        REPO_ROOT / "launch_training.ps1",
    ]
    meta = {
        "run_id": run_id,
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "prepared",
        "provenance": {
            "sykora": repo_provenance,
            "bullet": bullet_provenance,
            "pinned_bullet_commit": PINNED_COMMIT,
            "tool_sources": [dataset_fingerprint(path, True) for path in tool_sources],
        },
        "datasets": {
            "training": train_fingerprints,
            "validation_sources": validation_fingerprints,
            "validation_sample": str(validation_sample) if str(validation_sample) != "." else "",
            "validation_positions": args.validation_positions if validation_datasets else None,
            "validation_sample_fingerprint": (
                dataset_fingerprint(validation_sample, False)
                if str(validation_sample) != "." and validation_sample.is_file()
                else None
            ),
        },
        "data_format": data_format,
        "output_dir": str(run_dir),
        "checkpoint_dir": str(ckpt_dir),
        "resume": {
            "checkpoint": env.get("SYK_RESUME", ""),
            "source_run_id": resume_metadata.get("run_id") if resume_metadata else None,
        },
        "warm_start": {
            "source": str(Path(args.warm_start).resolve()) if args.warm_start else "",
            "weights": str(warm_start_weights) if warm_start_cmd else "",
            "verification": str(warm_start_report) if warm_start_cmd else "",
            "allow_random_v8_init": args.allow_random_v8_init,
        },
        "commands": {
            "sample_validation": sample_cmd,
            "prepare_warm_start": warm_start_cmd,
            "train": train_cmd,
            "validate": validation_cmd if str(validation_sample) != "." else [],
            "convert_final": converter_cmd if args.export_after else [],
            "export_final": export_cmd if args.export_after else [],
            "parity": parity_cmd,
        },
        "network": {
            "format": args.network_format,
            "architecture": args.architecture,
            "bucket_layout_name": args.bucket_layout,
            "bucket_layout_64": bucket_layout_64(args.bucket_layout),
            "ft_hidden": args.hidden,
            "dense1": args.dense1,
            "dense2": args.dense2,
            "factorised": True,
            "factorisation_mode": "virtual_sparse" if args.network_format == "syk8" else "separate_tensor",
            "output_bucket_count": args.output_buckets,
            "output_bucket_scheme": "single" if args.output_buckets == 1 else "material_popcount",
        },
        "training": {
            "start_superbatch": args.start_superbatch,
            "end_superbatch": args.end_superbatch,
            "batch_size": args.batch_size,
            "batches_per_superbatch": args.batches_per_superbatch,
            "samples_per_superbatch": args.batch_size * args.batches_per_superbatch,
            "planned_samples": args.batch_size
            * args.batches_per_superbatch
            * (args.end_superbatch - args.start_superbatch + 1),
            "lr_start": args.lr_start,
            "lr_final": final_lr,
            "lr_origin_superbatch": args.lr_origin_superbatch,
            "lr_final_superbatch": lr_final_superbatch,
            "wdl": args.wdl,
            "save_rate": args.save_rate,
            "threads": args.threads,
            "backend": args.backend,
            "rng_seed": None,
            "rng_note": "Pinned Bullet uses entropy-seeded initialisation and shuffle RNGs",
        },
        "env": {key: value for key, value in env.items() if key.startswith("SYK_")},
        "artifacts": {
            "final_checkpoint": str(final_checkpoint) if args.export_after else "",
            "export_checkpoint": str(final_checkpoint) if args.export_after else "",
            "final_npz": str(final_npz) if args.export_after else "",
            "final_net": str(final_net) if args.export_after else "",
        },
    }
    if args.network_format == "syk8":
        meta["network"].update(
            {
                "architecture_id": "pairwise_mlp_threats",
                "feature_set": "mirrored_psq_full_threats_v1",
                "psq_feature_count": 768 * (max(meta["network"]["bucket_layout_64"]) + 1),
                "threat_feature_count": 60_720,
                "threat_scheme_id": 1,
                "threat_packing_sha256": "964591edbe856c9f90694dcbfabe42d58b011a469e3275a8aaa9e4249b21988a",
                "threat_storage": "i8",
                "resolved_accumulator": "i32",
            }
        )
    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    print(f"Architecture: {args.architecture}, H={args.hidden}, O={args.output_buckets}")
    print(
        f"Sampled positions: {meta['training']['planned_samples']:,} "
        f"({meta['training']['samples_per_superbatch']:,} per superbatch)"
    )
    print("Training datasets:")
    for dataset in datasets:
        print(f"  {dataset}")
    if validation_datasets:
        print("Held-out validation shards:")
        for dataset in validation_datasets:
            print(f"  {dataset}")
    print(f"Metadata: {meta_path}")

    if sample_cmd:
        print("$", " ".join(sample_cmd))
    if warm_start_cmd:
        print("$", " ".join(warm_start_cmd))
    print("$", " ".join(train_cmd))
    if args.dry_run:
        return 0

    if sample_cmd:
        meta["status"] = "sampling_validation"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        try:
            subprocess.run(sample_cmd, cwd=str(REPO_ROOT), check=True)
        except subprocess.CalledProcessError as exc:
            meta["status"] = "validation_sampling_failed"
            meta["validation_sampling_exit_code"] = exc.returncode
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            return exc.returncode
        meta["datasets"]["validation_sample_fingerprint"] = dataset_fingerprint(
            validation_sample, False
        )
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    if warm_start_cmd:
        meta["status"] = "preparing_warm_start"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        try:
            subprocess.run(warm_start_cmd, cwd=str(REPO_ROOT), check=True)
        except subprocess.CalledProcessError as exc:
            meta["status"] = "warm_start_failed"
            meta["warm_start_exit_code"] = exc.returncode
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            return exc.returncode
        meta["warm_start"]["verification_result"] = json.loads(warm_start_report.read_text())
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    meta["status"] = "training"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    try:
        subprocess.run(train_cmd, cwd=str(REPO_ROOT), env=env, check=True)
    except subprocess.CalledProcessError as exc:
        meta["status"] = "failed"
        meta["training_exit_code"] = exc.returncode
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"Training failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    meta["status"] = "trained"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Training finished. Checkpoints: {ckpt_dir}")

    if args.validate_after and str(validation_sample) != ".":
        print("$", " ".join(validation_cmd))
        try:
            subprocess.run(validation_cmd, cwd=str(REPO_ROOT), check=True)
        except subprocess.CalledProcessError as exc:
            meta["status"] = "validation_failed"
            meta["validation_exit_code"] = exc.returncode
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            return exc.returncode
        meta["status"] = "validated"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    elif str(validation_sample) != ".":
        print("Run validation later with:")
        print(" ".join(validation_cmd))

    if args.export_after:
        export_checkpoint = final_checkpoint
        if args.export_best_validation:
            validation_report = json.loads((run_dir / "validation_results.json").read_text())
            export_checkpoint = Path(validation_report["best"]["checkpoint"]).resolve()
            if export_checkpoint.parent != ckpt_dir.resolve():
                meta["status"] = "export_failed"
                meta["export_error"] = f"validated checkpoint escaped run directory: {export_checkpoint}"
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")
                print(meta["export_error"], file=sys.stderr)
                return 2
            converter_cmd[3] = str(export_checkpoint)
            meta["commands"]["convert_final"] = converter_cmd
            meta["artifacts"]["export_checkpoint"] = str(export_checkpoint)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"Exporting best held-out checkpoint: {export_checkpoint.name}")
        if not (export_checkpoint / "raw.bin").is_file():
            meta["status"] = "export_failed"
            meta["export_error"] = f"export checkpoint not found: {export_checkpoint}"
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            print(meta["export_error"], file=sys.stderr)
            return 1
        meta["status"] = "exporting"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        try:
            subprocess.run(converter_cmd, cwd=str(REPO_ROOT), check=True)
            subprocess.run(export_cmd, cwd=str(REPO_ROOT), check=True)
            if parity_cmd:
                subprocess.run(parity_cmd, cwd=str(REPO_ROOT), check=True)
        except subprocess.CalledProcessError as exc:
            meta["status"] = "export_failed"
            meta["export_exit_code"] = exc.returncode
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            return exc.returncode
        meta["status"] = "verified" if parity_cmd else "exported"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"Ready-to-load net: {final_net}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
