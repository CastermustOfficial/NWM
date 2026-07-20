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

> **What's new in 2.2** — **adaptive stickiness** (`adaptive_repeat`): exploration
> auto-ramps when the task gives no learning signal, matching DQN on MountainCar
> with *zero* hand-tuning; plus documented negative results on bootstrapped
> credit assignment. (2.1 added sticky exploration itself — NWM solves
> MountainCar and beats DQN there; 2.0 added the `src/` layout, seeding, the
> benchmark suite, and the LaTeX paper.) See the [CHANGELOG](CHANGELOG.md).

## Key ideas

| Mechanism            | What it does                                                   |
| -------------------- | ------------------------------------------------------------- |
| **Potential field**  | Maps states to per-action attractive/repulsive forces.        |
| **Persistent memory**| Experiences merge into bounded *centroids* with progressive stiffness. |
| **Dynamic Smart Lock** | Protects high-confidence memories from being overwritten.   |
| **Fear & Greed**     | Rejects dangerous actions *before* maximizing reward.         |
| **Adaptive exploration** | Collapses exploration once performance is high.           |

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
MountainCar), over 5 seeds with a fixed greedy-evaluation protocol.

**Fairness note.** NWM uses **one shared configuration for every environment**
(`NWM_SHARED_CONFIG` in `benchmarks/config.py`) — no per-task tuning — exactly
like the DQN baseline. The config was selected on seeds 10–19 and validated
once on held-out seeds 5–9; the table below reports the held-out seeds.

```bash
python -m benchmarks.run_benchmark --quick            # fast smoke run
python -m benchmarks.run_benchmark --seeds 0 1 2 3 4  # full protocol
```

Outputs land in `results/`: per-run JSON, an aggregated `summary.csv`, a
Markdown table, and learning-curve / comparison plots. The accompanying paper in
[`paper/`](paper/) is built from exactly these numbers.

**Final greedy evaluation** (mean ± std over held-out seeds 5–9; higher is
better — Acrobot and MountainCar returns are negative). All agents measured in
the same environment (CPU torch 2.13, gymnasium 1.3), **after fixing the
truncation-bootstrap bug in the value-based baselines** (see below). Best per
environment in **bold**:

| Environment    | Random        | TabularQ      | DQN               | NWM                |
| -------------- | ------------- | ------------- | ----------------- | ------------------ |
| CartPole-v1    | 22.0 ± 2.2    | 154.4 ± 19.7  | 205.7 ± 137.6     | **270.7 ± 148.9**  |
| Acrobot-v1     | −498.9 ± 1.4  | −431.7 ± 32.5 | **−123.9 ± 46.1** | −250.4 ± 145.6     |
| MountainCar-v0 | −200.0 ± 0.0  | −200.0 ± 0.0  | −200.0 ± 0.0      | **−143.5 ± 15.1**  |

**Takeaways.** With a single untuned configuration, NWM **leads on dense
CartPole** and **wins clearly on sparse-reward MountainCar**, where the
(corrected) DQN never reaches the goal within the episode budget. On
**Acrobot the corrected DQN is the strongest method**: earlier versions of
this table showed NWM ahead there, but that advantage evaporated once we
fixed a bug *in the baselines* — DQN and tabular Q-learning were zeroing the
bootstrap target on time-limit truncation, which poisons value estimates
precisely on long-horizon tasks (DQN on Acrobot: −335.6 → −123.9 after the
fix). We report the corrected comparison because beating a handicapped
baseline is not a result. NWM's remaining Acrobot gap is consistent with the
paper's negative results on bootstrapped credit assignment: value propagation
over long horizons is exactly what a memory-based method lacks.

The NWM mechanisms behind the wins: `credit_blend` (temporal credit blends
toward the neutral score, so early steps of good episodes stay attractive),
`relative_gate` (sign-aware quality gate; the legacy gate silently disabled
attraction and the Smart Lock under negative returns), `adaptive_repeat`
(self-tuning sticky exploration for sparse rewards), `truncation_credit`
(late-step blame only when a true terminal event exists), and `eval_sticky`
(momentum-preserving fallback on unknown states during greedy evaluation).
Opt-in extras for non-stationary or long-horizon settings: `score_ema`
(recency-weighted memories) and `dynamic_unlock` (locks that release when
their recent scores collapse) — they trade CartPole/MountainCar performance
for Acrobot gains, so they stay off in the shared benchmark config. Absolute
numbers shift with library versions; regenerate with
`python -m benchmarks.run_benchmark --seeds 5 6 7 8 9`.

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
  version = {2.0.0}
}
```

## License

MIT — see [LICENSE](LICENSE).
