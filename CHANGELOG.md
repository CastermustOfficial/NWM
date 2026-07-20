# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.5.0] - 2026-07-19

The research release: NWM's field gets a long-horizon mechanism. Plus real
checkpoint/restore, honestly measured.

### Added
- **Credit propagation on the centroid graph**
  (`NWMConfig.credit_propagation`, default `0.0`; `0.3` in the benchmark):
  the field records successor links (centroid, action) -> centroid along
  observed trajectories and, at each episode end, runs 3 sweeps of value
  propagation in score space: `V(c) = (1-beta)*recent_score(c) +
  beta*E[V(successors)]`. Forces blend each action's own score with the
  propagated value of that action's successors, so good outcomes flow
  backwards across episodes (trajectory stitching) while scores stay on the
  absolute 0-1 scale with 0.5 neutral -- the decision rule is untouched,
  which is what the 2.2.0 negative results demanded. Selection seeds 10-19:
  Acrobot -233.1 -> **-183.8**; held-out seeds 5-9: -250.4 -> **-211.9**
  (median seed -167), with a CartPole cost (270.7 -> 211.2, still above the
  corrected DQN's 205.7). beta swept: 0.2 too weak, 0.5 too strong.
- **Stagnation revival** (`NWMConfig.stagnation_revival`, default `False`):
  snapshots the field on new recent-average bests, restores it after a
  sustained collapse well below the best, and re-inflates exploration when
  progress stalls. `checkpoint_if_best` / `restore_best_model` are now real
  implementations instead of no-op stubs. Measured honestly: on selection
  seeds it did not improve benchmark means (CartPole 298.2 -> 273.1, Acrobot
  -233.1 -> -265.4), so it ships **off** in the shared config -- kept as an
  opt-in for long-running custom setups.
- Unit tests for both mechanisms.

### Changed
- `NWM_SHARED_CONFIG` adds `credit_propagation: 0.3` (winner on 2 of 3
  environments on selection seeds vs the 2.4.0 config as control; validated
  once on held-out seeds 5-9). Held-out results vs corrected baselines:
  CartPole **211.2 +- 131.3** (DQN 205.7 +- 137.6), Acrobot -211.9 +- 113.8
  (DQN **-123.9 +- 46.1**), MountainCar **-142.3 +- 17.0** (DQN -200.0).
- `PersistentPotentialField.add` now returns the uid of the absorbing
  centroid (previously returned None).
- With `credit_propagation=0` and `stagnation_revival=False`, behavior is
  bit-identical to 2.4.0 (verified seed-for-seed).

## [2.4.0] - 2026-07-19

Memory, credit, and baseline-correctness release. NWM gains recency-aware
memory and truncation-aware credit; the value-based baselines get a
significant bug fix that -- honestly -- reverses one headline result.

### Added
- **`NWMConfig.truncation_credit`** (default `False`, on in the benchmark):
  temporal credit now uses the terminated/truncated distinction. A bad episode
  ending in a true terminal event keeps late-step blame; a bad truncated
  episode (aimless until timeout) gets flat mild credit; a good truncated
  episode (survived the full limit) gets full flat credit instead of
  under-crediting its early steps. `NWMAgent.step`'s `terminated` parameter is
  un-deprecated to carry this signal.
- **`NWMConfig.eval_sticky`** (default `False`, on in the benchmark): the
  random fallback used during greedy evaluation on unknown states now honors
  sticky exploration, preserving momentum (Acrobot selection seeds:
  -275.9 -> -233.1).
- **`NWMConfig.score_ema`** (default `0.0`): per-action centroid scores become
  exponential moving averages instead of all-time sums, so stale judgments
  fade as the percentile scale shifts. Helps long-horizon Acrobot
  (-277.6 -> -201.6 with alpha 0.3) but costs CartPole/MountainCar, so it is
  off in the shared benchmark config.
- **`NWMConfig.dynamic_unlock`** (default `False`): locked centroids whose
  recent merged scores collapse (EMA < 0.45) lose their lock instead of
  staying protected forever.
- **`NWMConfig.relative_explore`** (default `False`): percentile-based
  exploration collapse replacing the CartPole-specific 450/300 thresholds.
  Measured and documented: it hurt Acrobot (-369.7) in the shared config, so
  it ships off but replaces magic constants for custom setups.
- Test suite for all new mechanisms (`tests/test_v24_features.py`).

### Fixed
- **Baselines: truncation-aware bootstrapping.** DQN and tabular Q-learning
  treated time-limit truncation as a true terminal state, zeroing the
  bootstrap target -- a classic bug that poisons value estimates on
  long-horizon tasks. The runner now passes `terminated` separately and both
  baselines bootstrap through truncations. Effect on held-out seeds: DQN
  Acrobot -335.6 -> -123.9 (now the strongest method there, overtaking NWM;
  the README reports this honestly), DQN MountainCar -194.0 -> -200.0,
  CartPole within noise.

### Changed
- `NWM_SHARED_CONFIG` adds `truncation_credit` and `eval_sticky` (selected on
  seeds 10-19 against the 2.3.0 config as control, validated once on held-out
  seeds 5-9). Held-out results vs corrected baselines: CartPole
  **270.7 +- 148.9** (DQN 205.7 +- 137.6), Acrobot -250.4 +- 145.6 (DQN
  **-123.9 +- 46.1**), MountainCar **-143.5 +- 15.1** (DQN -200.0 +- 0.0).
- With all new flags off and `terminated` not supplied, agent behavior is
  bit-identical to 2.3.0 (verified seed-for-seed).

## [2.3.0] - 2026-07-19

A fairness-and-credit release: two new opt-in credit-assignment fixes, and a
benchmark protocol with **zero per-environment tuning** -- NWM now uses one
shared config on every task, exactly like the DQN baseline, and beats or
matches it everywhere on held-out seeds.

### Added
- **`NWMConfig.credit_blend`** (default `False`): the temporal weight now
  blends the episode score toward the neutral value 0.5
  (`0.5 + (score - 0.5) * t_weight`) instead of scaling it toward 0. Early
  steps of good episodes stay mildly attractive instead of turning repulsive.
- **`NWMConfig.relative_gate`** (default `False`): sign-aware dynamic quality
  gate. The legacy gate `max(40, avg*1.25)` is constant at 40 whenever recent
  returns are negative, silently capping every episode at score 0.6 -- which
  disabled attraction (needs > 0.6) and the Dynamic Smart Lock (needs > 0.8)
  on Acrobot and MountainCar. With the flag on and negative averages, the
  threshold becomes `avg + 0.25*|avg|` ("25% better than the recent average").

### Changed
- **Benchmark: single shared NWM config** (`NWM_SHARED_CONFIG`: credit_blend,
  relative_gate, adaptive_repeat, warmup 20, merge_threshold 0.5) replaces all
  per-environment overrides, removing NWM's tuning advantage over DQN. The
  config was selected on seeds 10-19 and validated once on held-out seeds 5-9.
- README results table re-measured with every agent run in the same
  environment on the held-out seeds: NWM 278.6 +- 159.5 vs DQN 279.4 +- 172.4
  on CartPole (tie), **-271.2 +- 77.3 vs -335.6 +- 201.4 on Acrobot** and
  **-141.5 +- 15.2 vs -194.0 +- 11.9 on MountainCar** (NWM wins both).

### Fixed / honesty notes
- The previously published Acrobot numbers were not reproducible from the
  released code in a fresh environment (measured NWM baseline: -453.5 +- 92.9
  vs the reported -265.7 +- 161.0; CartPole and MountainCar reproduced to the
  decimal). Reported DQN numbers also shift across library versions; the new
  table measures all agents under identical conditions.

## [2.2.0] - 2026-07-14

A small, honest release: one new self-tuning exploration feature, plus documented
negative results from a credit-assignment research push. Public API unchanged;
benchmark configurations and headline numbers are identical to 2.1.0.

### Added
- **Adaptive stickiness** (`NWMConfig.adaptive_repeat`, default `False`): while
  recent returns are perfectly flat -- the signature of a sparse-reward task
  whose goal has never been reached -- `exploration_repeat` auto-ramps
  (+0.1 per episode, up to 0.95, active during warmup too) and stops as soon as
  returns vary. On MountainCar this reaches **-162.2 ± 28.8** over 5 seeds with
  *zero* task-specific tuning, beating DQN (-173.2 ± 27.5); the hand-tuned
  `exploration_repeat=0.9` remains stronger (-130.4 ± 9.7) and stays the
  benchmark configuration. Keep the flag off for tasks that give a learning
  signal from the start (on Acrobot the early flat window makes it harmful).

### Negative results (tried, measured, rejected)
- **Return-to-go percentile credit** (per-step G_t ranked against a rolling
  history, with truncation-aware bootstrapping): neutral on MountainCar,
  clearly worse on Acrobot (-378 to -400 vs. -247 baseline over 3 seeds).
  Diagnosis: absolute per-step scores conflate distance-from-goal with action
  quality, turning early states repulsive.
- **Advantage force model** (per-action score vs. the other actions in the same
  centroid, argmax-style): worse still (-375 to -443), including with the
  proven episode credit. NWM's decision rule (Fear & Greed thresholds, veto,
  confidence weights) is co-designed around absolute scores with 0.5 neutral;
  changing force semantics requires redesigning the rule, not swapping a
  formula. Both experiments were removed from the codebase; only the paper
  documents them.

## [2.1.0] - 2026-07-14

A performance release. The headline is **temporally-correlated (sticky)
exploration**, which turns NWM's previously reported sparse-reward failure into a
win: NWM now solves MountainCar and outperforms DQN there. The public API is
unchanged and fully backward compatible.

### Added
- **Sticky exploration** (`NWMConfig.exploration_repeat`, default `0.0`): during
  exploratory steps the previous action is repeated with probability
  `exploration_repeat`, otherwise a uniform action is drawn. This builds the
  sustained momentum that uniform noise cannot, and is what lets NWM reach the
  goal on sparse-reward tasks. `0.0` preserves the classic uniform behavior.

### Changed
- Benchmark per-env tuning: MountainCar now uses `exploration_repeat=0.9`
  (−200 → **−130.4 ± 9.7**, beating DQN's −173.2 ± 27.5); Acrobot uses a coarser
  `merge_threshold=0.5` (−307.9 → −265.7).
- Refreshed the full 5-seed benchmark, `results/` artifacts, the paper's results
  table/figures, and the README table. DQN is re-reported from a fresh run; its
  high across-seed variance (e.g. CartPole 193.7 ± 158.5) contrasts with NWM's
  markedly lower variance.
- Paper (`paper/nwm.tex`): formalized sticky exploration (new equation and
  algorithm step) and rewrote the abstract, results, and limitations to reflect
  that the MountainCar barrier was **exploration, not credit assignment**.

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

[2.5.0]: https://github.com/CastermustOfficial/NWM/releases/tag/v2.5.0
[2.4.0]: https://github.com/CastermustOfficial/NWM/releases/tag/v2.4.0
[2.3.0]: https://github.com/CastermustOfficial/NWM/releases/tag/v2.3.0
[2.2.0]: https://github.com/CastermustOfficial/NWM/releases/tag/v2.2.0
[2.1.0]: https://github.com/CastermustOfficial/NWM/releases/tag/v2.1.0
[2.0.0]: https://github.com/CastermustOfficial/NWM/releases/tag/v2.0.0
