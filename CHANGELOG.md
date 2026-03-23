# Changelog

All notable changes to Sykora are documented in this file.

This changelog was reconstructed from the tagged release history and the commits between tags.

## [Unreleased]

### Changed

- Promoted the embedded NNUE to the stronger `v3_512` checkpoint `run_20260323T063759Z-600`.
- Switched NNUE tooling to emit `SYKNNUE3` only. `SYKNNUE2` remains readable for backward compatibility with older external nets.
- Added transposition-table probing/storage inside quiescence search.
- Reworked search move ordering with continuation-history heuristics for quieter move scoring and better follow-up awareness.
- Added singular extension logic and TT-aware multicut handling to sharpen tactical search around strong transposition-table moves.
- Simplified repetition detection with a cheap halfmove fast path and early exit on first match.
- Retired Texel/HCE tuning from the default workflow and documentation.
- Fixed the release SPRT workflow and bumped the engine version string to `0.2.3`.

### Removed

- Removed the old Texel-tuning binaries and scripts (`sykora-tune`, `sykora-texel`, and the Texel helpers under `utils/tuning`).

## [0.2.2] - 2026-03-23

### Added

- Added Bullet bootstrap and runner utilities to make NNUE training setup more repeatable.

### Changed

- Moved the engine to a stronger embedded NNUE setup with `SYKNNUE3` support, retained `SYKNNUE2` compatibility, and shipped a much larger network.
- Reworked search internals with a static evaluation stack, pruning cleanups, SIMD-oriented NNUE updates, and harder SMP synchronization.
- Refreshed the README and release/testing documentation around NNUE workflows and regression checks.

### Fixed

- Fixed a pruning regression after an Internal Iterative Reduction experiment was reverted.
- Fixed king-bucket perspective handling in NNUE evaluation.
- Fixed CI tag resolution and release-test workflow behavior.

## [0.2.1] - 2026-03-02

### Added

- First release with NNUE support in the shipped engine.
- Added a full Texel tuning toolchain, including a native tuner, dataset builders, large-run orchestration scripts, and parameter application helpers.
- Added `gensfen`, Lichess PGN extraction, and parallel self-play tooling for generating NNUE training data.
- Added incremental NNUE accumulator support in search to make embedded NNUE practical during play.

### Changed

- Made embedded NNUE the default evaluator and shipped the first successful `SYKNNUE2` network with updated training/export scripts.
- Retuned the classical evaluator with Texel-generated parameters, updated mobility tables, and cleanup across evaluation utilities.
- Expanded the README to describe engine features, tuning, and NNUE workflows in more operational detail.

### Fixed

- Fixed search history handling so the current position is tracked correctly for repetition logic.
- Hardened dataset generation, Stockfish annotation retries, and cross-platform script behavior.
- Tightened option handling and parameter extraction in evaluation/tuning utilities.

## [0.1.0] - 2026-02-17

### Added

- First tagged release of the Zig-based Sykora UCI engine.
- Added the core engine stack: bitboards, magic move generation, legal move validation, perft/divide tooling, alpha-beta search, transposition tables, and clock-aware search control.
- Added a classical evaluation pipeline with pawn-structure, mobility, king-safety, and endgame terms.
- Added developer tooling for STS, self-play, benchmark/NPS runs, history tracking, automated tuning, and Lichess bot play/challenges.
- Added multi-target GitHub release automation for shipping binaries across Linux, macOS, and Windows.

### Changed

- Improved move generation, hashing, undo support, null-move handling, repetition detection, and search heuristics throughout the initial development cycle.
- Expanded project documentation to cover engine usage, testing, tuning, and experiment history.
- Broadened build support for additional CPU architectures and optimized release build commands.

### Fixed

- Fixed castling, en passant, capture handling, move legality edge cases, transposition-table scoring issues, and early search stability problems.
- Fixed cleanup and shutdown behavior around search threads, bot sessions, and Zig `0.16` compatibility.

[Unreleased]: https://github.com/sb2bg/sykora/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/sb2bg/sykora/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/sb2bg/sykora/compare/v0.1.0...v0.2.1
[0.1.0]: https://github.com/sb2bg/sykora/commits/v0.1.0
