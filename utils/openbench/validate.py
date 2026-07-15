#!/usr/bin/env python3
"""Validate Sykora's OpenBench SPSA, UCI, and bench contracts."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence


DEFAULT_PARAMS = Path(__file__).with_name("spsa.txt")
SAFE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
UCI_SPIN_RE = re.compile(
    r"^option name (?P<name>\S+) type spin default (?P<default>-?\d+) "
    r"min (?P<minimum>-?\d+) max (?P<maximum>-?\d+)$"
)


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: str
    current: float
    minimum: float
    maximum: float
    c_end: float
    r_end: float


def load_parameters(path: Path) -> tuple[Parameter, ...]:
    parameters = []
    names: set[str] = set()
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc

    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            raise ValidationError(f"{path}:{line_number}: expected seven comma-separated fields")
        name, kind = fields[:2]
        if not SAFE_NAME_RE.fullmatch(name):
            raise ValidationError(f"{path}:{line_number}: unsafe OpenBench parameter name {name!r}")
        if name in names:
            raise ValidationError(f"{path}:{line_number}: duplicate parameter {name}")
        if kind not in ("int", "float"):
            raise ValidationError(f"{path}:{line_number}: type must be int or float")
        try:
            current, minimum, maximum, c_end, r_end = map(float, fields[2:])
        except ValueError as exc:
            raise ValidationError(f"{path}:{line_number}: parameter values must be numeric") from exc
        if not minimum <= current <= maximum or minimum >= maximum:
            raise ValidationError(f"{path}:{line_number}: invalid current/min/max range")
        if c_end <= 0 or r_end <= 0:
            raise ValidationError(f"{path}:{line_number}: C_end and R_end must be positive")
        if kind == "int" and any(not value.is_integer() for value in (current, minimum, maximum)):
            raise ValidationError(f"{path}:{line_number}: integer current/min/max must be integral")
        names.add(name)
        parameters.append(Parameter(name, kind, current, minimum, maximum, c_end, r_end))

    if not parameters:
        raise ValidationError(f"{path}: no SPSA parameters found")
    return tuple(parameters)


def parse_uci_spins(output: str) -> Dict[str, tuple[int, int, int]]:
    options: Dict[str, tuple[int, int, int]] = {}
    for raw in output.splitlines():
        match = UCI_SPIN_RE.fullmatch(raw.strip())
        if match:
            options[match.group("name")] = (
                int(match.group("default")),
                int(match.group("minimum")),
                int(match.group("maximum")),
            )
    return options


def validate_uci_output(output: str, parameters: Iterable[Parameter]) -> None:
    options = parse_uci_spins(output)
    for required in ("Threads", "Hash"):
        if required not in options:
            raise ValidationError(f"engine does not expose required UCI spin option {required}")
    for parameter in parameters:
        if parameter.kind != "int":
            raise ValidationError(
                f"{parameter.name}: float OpenBench parameters require a UCI string option; "
                "Sykora's scaffold intentionally uses fixed-point integers"
            )
        actual = options.get(parameter.name)
        expected = (int(parameter.current), int(parameter.minimum), int(parameter.maximum))
        if actual is None:
            raise ValidationError(f"engine does not expose UCI spin option {parameter.name}")
        if actual != expected:
            raise ValidationError(f"{parameter.name}: UCI default/min/max {actual} != SPSA input {expected}")


def parse_bench_output(output: str) -> tuple[int, int]:
    nodes = re.findall(r"(\d+)\s+nodes", output, flags=re.IGNORECASE)
    nps = re.findall(r"(\d+)\s+nps", output, flags=re.IGNORECASE)
    if not nodes or not nps:
        raise ValidationError("bench output must contain '<N> nodes <N> nps'")
    parsed = int(nodes[-1]), int(nps[-1])
    if parsed[0] <= 0 or parsed[1] <= 0:
        raise ValidationError("bench nodes and NPS must be positive")
    return parsed


def run_engine(engine: Path, input_text: str | None, args: Sequence[str] = ()) -> str:
    try:
        proc = subprocess.run(
            [str(engine), *args],
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValidationError(f"cannot run {engine}: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise ValidationError(f"{engine} exited with {proc.returncode}: {detail}")
    return proc.stdout


def validate_engine(engine: Path, parameters: tuple[Parameter, ...], skip_bench: bool) -> None:
    validate_uci_output(run_engine(engine, "uci\nquit\n"), parameters)
    if skip_bench:
        return
    first_nodes, _ = parse_bench_output(run_engine(engine, None, ("bench",)))
    second_nodes, _ = parse_bench_output(run_engine(engine, None, ("bench",)))
    if first_nodes != second_nodes:
        raise ValidationError(f"non-deterministic bench nodes: {first_nodes} != {second_nodes}")
    print(f"OpenBench validation passed: {len(parameters)} parameters, deterministic bench {first_nodes} nodes")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path, help="Built Sykora executable")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS, help="OpenBench SPSA input file")
    parser.add_argument("--skip-bench", action="store_true", help="Only validate parameter and UCI contracts")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    engine = args.engine.expanduser().resolve()
    if not engine.is_file():
        raise ValidationError(f"engine not found: {engine}")
    parameters = load_parameters(args.params.expanduser().resolve())
    validate_engine(engine, parameters, args.skip_bench)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"openbench validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
