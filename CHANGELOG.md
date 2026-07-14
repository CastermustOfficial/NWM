# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-14

A modernization release focused on engineering quality, reproducibility, an
extended scientific benchmark, and a formal write-up of the method. The public
Python API (`NWM`, `NWMAgent`, `NWMConfig`, `PersistentCentroid`,
`PersistentPotentialField`) remains backward compatible.

### Added
- **`src/` layout** packaging for import isolation and cleaner builds.
- **Reproducibility**: `nwm.seeding.set_global_seed()` and a `seed` argument on
  `NWMAgent`/`NWMConfig` that seeds a private `numpy.random.Generator` per agent
  (no reliance on global RNG state).
- **Extended benchmark suite** (`benchmarks/`) comparing NWM against Random,
  tabular Q-learning (tile coding), and a PyTorch DQN across multiple Gymnasium
  environments and seeds, with JSON/CSV logging, aggregated statistics, and
  learning-curve / bar plots.
- **Scientific paper** (`paper/`) in LaTeX with the formalized method, related
  work, experimental protocol, and results generated from the benchmark suite.
- **Tooling**: Ruff (lint + format), mypy in `strict` mode, pytest coverage,
  and pre-commit hooks.
- **Typing**: `py.typed` marker so downstream users get type information.
- New tests covering seeding, determinism, and end-to-end reproducibility.

### Changed
- `PersistentPotentialField` now caches the stacked centroid-state matrix and
  invalidates it on mutation, removing the per-step `O(N·d)` rebuild in
  `add()`/`query_forces()`.
- Public exports now include `NWMAgent` (previously only the `NWM` alias was
  exported, which broke `from nwm import NWMAgent` and the shipped test suite).
- `requires-python` raised to `>=3.9`.
- Corrected `project.urls` to point at the real repository
  (`CastermustOfficial/NWM`) instead of the non-existent `nwm-research/nwm`.

### Fixed
- **Broken test collection**: `tests/test_agent.py` imported `NWMAgent`, which
  was not exported by `nwm/__init__.py`; the published test suite could not be
  collected. Now fixed and green.
- Removed committed `__pycache__/*.pyc` byte-compiled artifacts from version
  control.

## [1.0.1] - Prior release
- Initial public release on PyPI as `nwm-rl`: potential-field agent, centroid
  memory, Dynamic Smart Lock, adaptive exploration, examples, and a basic test
  suite.

[2.0.0]: https://github.com/CastermustOfficial/NWM/releases/tag/v2.0.0
