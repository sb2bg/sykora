# OpenBench integration

This directory contains the Sykora-side scaffolding for an OpenBench instance.
OpenBench owns game pairing, distributed scheduling, SPSA updates, reporting,
and resume state; Sykora only provides a reproducible build, deterministic
benchmark, and UCI tuning surface.

## Build and validate

OpenBench invokes `make -j EXE=<name>`. The same contract can be tested locally:

```bash
make EXE=sykora-openbench
./sykora-openbench bench
python utils/openbench/validate.py ./sykora-openbench
make clean EXE=sykora-openbench
```

Zig `0.15.2` and GNU Make must be available to every worker. The benchmark
searches six fixed positions to depth 10 with one thread, a cleared 16 MB hash,
the embedded NNUE, and default tuning values. OpenBench requires the final node
count to remain identical across builds and repeated executions; NPS may vary.

## Instance configuration

Copy `Sykora.json.example` to the OpenBench instance's `Engines/Sykora.json`.
Before enabling it:

1. Ensure the configured opening book exists on that instance.
2. Replace the provisional `nps` value with a measurement from the instance's
   reference machine.
3. Confirm workers advertise Zig `>=0.15.2` and a supported operating system.
4. Validate the source URL and branch names for the repository being tested.

The example presets follow OpenBench's usual `BATCHED` reporting, `SINGLE`
distribution, eight pairs per SPSA point, and standard alpha/gamma/A-ratio
defaults. They are starting points, not promotion criteria.

## SPSA input

Paste `spsa.txt` into OpenBench's SPSA Input field. Every parameter is declared
as `int` because UCI `spin` options are integral. The three scale parameters use
fixed point: `100` means logical `1.00`, `105` means `1.05`.
The checked-in starting values are the latest candidate accepted by SPRT.

The initial tune intentionally contains only relatively smooth search-shape
parameters:

- `LMRScale`
- `LMRHistoryScale`
- `LMPMoveScale`
- `HistoryMaxBonus`

Raw reductions, shallow-depth gates, and null-move depth thresholds are omitted.
Their tiny discrete domains can dominate paired outcomes and contaminate every
simultaneous SPSA gradient. Test those separately with focused SPRTs or first
reparameterize them into a genuinely smoother search formula.

The supplied `C_end` values are approximately one twentieth of each allowed
range and `R_end` is `0.002`, matching OpenBench's documented starting guidance.
Review the resulting perturbation digest before committing distributed compute.
