# NWM — Negative Weight Mapping

**A non-parametric reinforcement learning framework built on persistent potential fields.**

[![PyPI](https://img.shields.io/pypi/v/nwm-rl.svg)](https://pypi.org/project/nwm-rl/)
[![Python](https://img.shields.io/pypi/pyversions/nwm-rl.svg)](https://pypi.org/project/nwm-rl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typed: mypy strict](https://img.shields.io/badge/typing-mypy%20strict-blue.svg)](http://mypy-lang.org/)

NWM turns an agent's past experience into a **potential force field** over the
observation space. Instead of training a neural network by gradient descent, it
remembers *where* things went well or badly and acts by following forces:

- **Attractive forces** pull the agent toward actions that succeeded before.
- **Repulsive forces** push it away from actions that led to failure.

The result is a transparent, reproducible, dependency-light agent (NumPy +
Gymnasium) that starts behaving sensibly from very few episodes.

> **What's new in 2.6** — the **honesty release**. The library is unchanged; the
> benchmark moves from 5 seeds to **20**, and claims that don't survive that
> move are **retracted** — including the `credit_propagation` trade-off shipped
> in 2.5 and the CartPole win over DQN. The same ablation reversed sign between
> two different 5-seed samples, which is what prompted the audit. Results are
> now reported with paired per-seed significance tests. (2.5 added credit
> propagation on the centroid graph; 2.4 added recency-aware memory and fixed a
> truncation-bootstrap bug *in the baselines*; 2.3 moved the benchmark to a
> single untuned shared config; 2.2 added adaptive stickiness; 2.1 added sticky
> exploration.) See the [CHANGELOG](CHANGELOG.md).

## Key ideas

| Mechanism            | What it does                                                   |
| -------------------- | ------------------------------------------------------------- |
| **Potential field**  | Maps states to per-action attractive/repulsive forces.        |
| **Persistent memory**| Experiences merge into bounded *centroids* with progressive stiffness. |
| **Dynamic Smart Lock** | Protects high-confidence memories from being overwritten.   |
| **Fear & Greed**     | Rejects dangerous actions *before* maximizing reward.         |
| **Adaptive exploration** | Collapses exploration once performance is high.           |
| **Credit propagation** | Propagates value backwards along the centroid graph, stitching outcomes across episodes. |

## Installation

```bash
pip install nwm-rl
```

From source (with development and benchmark extras):

```bash
git clone https://github.com/CastermustOfficial/NWM.git
cd NWM
pip install -e ".[dev,benchmark]"
```

Extras: `plots` (matplotlib), `baselines` (torch, for the DQN baseline),
`benchmark` (both + pandas), `dev` (ruff, mypy, pytest, pre-commit).

## Quick start

```python
import gymnasium as gym
from nwm import NWM

env = gym.make("CartPole-v1")
agent = NWM(
    state_dim=env.observation_space.shape[0],
    num_actions=env.action_space.n,
    seed=0,  # reproducible: no global RNG state touched
)

for episode in range(200):
    state, _ = env.reset(seed=episode)
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
    print(f"Episode {episode + 1}: best={agent.best_reward:.0f}")

env.close()
```

More examples live in [`examples/`](examples/): `quickstart.py`,
`cartpole_training.py` (with plotting/demo), and `custom_environment.py`.

## API overview

```python
from nwm import NWM, NWMConfig, set_global_seed

set_global_seed(42)  # seeds Python / NumPy / torch for the whole experiment

config = NWMConfig(
    max_centroids=500,      # memory capacity
    warmup_episodes=50,     # pure-exploration episodes before learning
    exploration_rate=1.0,   # initial epsilon
    exploration_decay=0.99,
    min_exploration=0.05,
    merge_threshold=0.3,    # distance below which experiences merge
    distance_cutoff=2.5,    # max influence radius
    seed=42,
)

agent = NWM(state_dim=4, num_actions=2, config=config)
action = agent.select_action(state, training=True)
agent.step(state, action, reward, next_state, done)
stats = agent.get_stats()
agent.save("agent.pkl")
agent = NWM.load("agent.pkl")   # restores the RNG stream too
```

## Benchmarks

A reproducible harness compares NWM against **Random**, **tabular Q-learning**,
and **DQN** across a difficulty gradient of Gymnasium tasks (CartPole, Acrobot,
MountainCar), over **20 seeds** with a fixed greedy-evaluation protocol.

**Fairness note.** NWM uses **one shared configuration for every environment**
(`NWM_SHARED_CONFIG` in `benchmarks/config.py`) — no per-task tuning — exactly
like the DQN baseline. Seeds are separated by role so nothing is reported on
seeds that influenced a choice: config selected on **10–19**, mechanism
ablations on **20–39**, and the table below on **40–59**, used once.

```bash
python -m benchmarks.run_benchmark --quick               # fast smoke run
python -m benchmarks.run_benchmark --seeds $(seq 40 59)  # full protocol
```

Outputs land in `results/`: per-run JSON, an aggregated `summary.csv`, a
Markdown table, and learning-curve / comparison plots. The accompanying paper in
[`paper/`](paper/) is built from exactly these numbers.

**Final greedy evaluation** (mean ± std over 20 seeds; higher is better —
Acrobot and MountainCar returns are negative). All agents measured in the same
environment (CPU torch 2.6, gymnasium 1.2), **after fixing the
truncation-bootstrap bug in the value-based baselines** (see below). Bold marks
the best *mean* — see the significance column, because two of these three gaps
are not statistically supported:

| Environment    | Random       | TabularQ      | DQN                | NWM                 | NWM vs. best baseline |
| -------------- | ------------ | ------------- | ------------------ | ------------------- | --------------------- |
| CartPole-v1    | 22.9 ± 2.2   | 139.9 ± 30.5  | 162.1 ± 97.0       | **210.6 ± 105.9**   | +48.5, *p*=0.064 — n.s. |
| Acrobot-v1     | −499.0 ± 2.6 | −420.6 ± 45.9 | **−216.6 ± 178.9** | −317.7 ± 145.8      | −101.1, *p*=0.071 — n.s. |
| MountainCar-v0 | −200.0 ± 0.0 | −200.0 ± 0.0  | −200.0 ± 0.0       | **−135.3 ± 25.8**   | **+64.7, *p*=0.0002** |

Comparisons are paired per seed (same seeds for every agent), Wilcoxon
signed-rank.

**Takeaways.** One result is unambiguous: on **sparse-reward MountainCar NWM
wins decisively** (+64.7 on 18 of 20 seeds, *p*=0.0002) — every baseline,
DQN included, finishes at exactly −200.0, never reaching the goal once. On
**CartPole NWM has the best mean and the best sample efficiency** (AUC 135.1
vs. DQN's 109.3), but the +48.5 gap over DQN spans zero (*p*=0.064): the
defensible claim is *parity*, not superiority. On **Acrobot DQN is better** —
NWM wins only 5 of 20 seeds, and the medians separate much further than the
means (−87.5 vs. −272.3).

**On statistical power (why the numbers moved).** Earlier versions of this
table used 5 seeds. With across-seed σ ≈ 130–140 on CartPole and Acrobot, that
gives a standard error near 60 — differences below ~120 points are
indistinguishable from noise, which covered essentially every mechanism claim
we made. We verified this the hard way: an ablation measured on one set of 5
seeds **reversed sign** on another. Two prior claims did not survive the move
to 20 seeds, and both are retracted below. Baselines were also affected: DQN on
Acrobot read −86.1 ± 10.2 at n=5 and −216.6 ± 178.9 at n=20 — the tight early
figure was a lucky sample.

**On the baseline bug.** DQN and tabular Q-learning were zeroing the bootstrap
target on time-limit truncation, poisoning value estimates precisely on
long-horizon tasks. Both now bootstrap through truncations. This **reversed a
result in DQN's favour** on Acrobot, where we had previously claimed a lead for
NWM. We report the corrected comparison because beating a handicapped baseline
is not a result.

**Credit propagation** (`credit_propagation=0.3`, new in 2.5): the field records
which centroid follows which along observed trajectories and runs a few sweeps
of value propagation in score space, so good outcomes flow backwards across
episodes (trajectory stitching). Ablated paired over 20 seeds:

| Environment    | β=0    | β=0.3  | Paired Δ | Wins  | *p*   |
| -------------- | ------ | ------ | -------- | ----- | ----- |
| CartPole-v1    | 260.8  | 269.4  | +8.6     | 10/20 | 0.93  |
| Acrobot-v1     | −276.6 | −219.2 | +57.5    | 12/20 | 0.37  |
| MountainCar-v0 | −159.7 | −146.6 | +13.1    | 13/20 | **0.017** |

*Retraction:* we previously described this as an Acrobot gain bought at a
CartPole cost. **Neither half survives at 20 seeds** — the Acrobot gain is not
significant and the CartPole cost does not exist; both were artifacts of a ±60
standard error. The only effect that holds up is on MountainCar, which we had
not claimed. It stays on by default (neutral-to-positive everywhere,
significant on one task), but not for the reason originally given.

The other NWM mechanisms: `credit_blend` (temporal credit blends toward the
neutral score), `relative_gate` (sign-aware quality gate), `adaptive_repeat`
(self-tuning sticky exploration), `truncation_credit` (late-step blame only
when a true terminal event exists), and `eval_sticky` (momentum-preserving
eval fallback). Opt-in extras: `score_ema` (recency-weighted memories),
`dynamic_unlock` (locks that release when recent scores collapse), and
`stagnation_revival` (best-field snapshot/restore on collapse). These were
evaluated under the old 5-seed protocol, so treat their reported magnitudes as
indicative only — "did not help" there is a statement about the evidence, not
about the mechanism. Absolute numbers shift with library versions; regenerate
with `python -m benchmarks.run_benchmark --seeds $(seq 40 59)`.

## Paper

The method is formalized and evaluated in a short paper under [`paper/`](paper/).
Build it with `cd paper && latexmk -pdf nwm.tex` (see
[`paper/README.md`](paper/README.md)). Tables and figures are regenerated from
the benchmark via `python paper/make_paper_assets.py`.

## Project structure

```
NWM/
├── src/nwm/            # library (installed package)
│   ├── agents/         # NWMAgent
│   ├── core/           # centroid + potential field
│   ├── utils/          # configuration
│   └── seeding.py      # reproducibility helpers
├── benchmarks/         # reproducible benchmark suite + baselines
├── examples/           # runnable usage examples
├── paper/              # LaTeX paper + asset generation
├── tests/              # pytest suite (unit + integration)
└── results/            # benchmark outputs (generated)
```

## Development

```bash
pip install -e ".[dev,benchmark]"
pre-commit install
ruff check . && ruff format --check .   # lint + format
mypy                                    # strict static typing
pytest --cov=nwm                        # tests + coverage
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Citation

```bibtex
@software{nwm2026,
  title   = {NWM: Negative Weight Mapping — A Non-Parametric Potential-Field
             Framework for Reinforcement Learning},
  author  = {CastermustOfficial},
  year    = {2026},
  url     = {https://github.com/CastermustOfficial/NWM},
  version = {2.6.0}
}
```

## License

MIT — see [LICENSE](LICENSE).
