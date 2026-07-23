# Sykora agent instructions

## Toolchain

- Build and test with Zig 0.15.2. On this machine use
  `/Users/sullivanbognar/.zvm/0.15.2/zig` explicitly.
- Preserve unrelated working-tree changes. In particular, do not add generated
  PGO data or local design notes to search experiment commits.

## Search experiments

- Put each independently testable search idea on its own `feat/<name>` branch.
- Start experiment branches from the designated shared foundation commit, not
  from another experiment branch.
- Keep each experiment to one conceptual change. Run `zig build test`, a
  deterministic bench twice, and a before/after fixed-depth search comparison.
- Commit and push the branch before requesting an OpenBench workload. Report
  the immutable 40-character commit SHA and both deterministic bench results.
- Do not combine individually accepted ideas without a sequential OpenBench
  confirmation test of the combined branch.

## OpenBench

OpenBench is the canonical SPRT and tuning system for this repository. The
instance is at `https://bench.sbognar.com`; workload links use `/test/<id>/`.

### Access and safety

- Prefer `ssh zekrom` and the OpenBench Docker/Django tooling. Do not use browser
  automation unless SSH is unavailable and the user explicitly approves the
  fallback.
- Confirm the relevant containers with `docker ps`. Expected names include
  `openbench-coordinator` and `openbench-worker`.
- Never read, print, or expose secrets, environment files, tokens, or passwords.
- Use Django's workload validation and creation paths. Never insert a workload
  row directly into the database.

### Read-only inspection

Use `docker exec openbench-coordinator python manage.py shell` and inspect the
test's games, pentanomial results, LLR/bounds, priority, approval/finished state,
and machine assignment. For Elo, use `OpenBench.stats.Elo(test.results())`.

### Workload creation and modification

- Create tests through `verify_workload()` followed by `create_workload()` with
  a Django `RequestFactory`; validation errors must be empty.
- Modify tests through `modify_workload()` and validate the result.
- Use immutable 40-character commit SHAs and deterministic bench node counts.
- Default Sykora SPRT settings unless the task says otherwise:
  - engine: Sykora
  - book: UHO
  - time control: `10+0.1`
  - options: `Threads=1 Hash=16`
  - workload size: 32
  - bounds: `[0, 3]`
  - confidence: `[0.05, 0.05]`
  - no WDL tablebases; adjudication tablebases are optional
  - enable win/draw adjudication
- Queue new tests below the priority of currently active tests. Before leaving a
  test queued, verify it has zero games and that an active machine still owns the
  currently running workload.

The repository-side engine definition and validator live in `utils/openbench/`.
