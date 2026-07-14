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

```bash
python -m benchmarks.run_benchmark --quick            # fast smoke run
python -m benchmarks.run_benchmark --seeds 0 1 2 3 4  # full protocol
```

Outputs land in `results/`: per-run JSON, an aggregated `summary.csv`, a
Markdown table, and learning-curve / comparison plots. The accompanying paper in
[`paper/`](paper/) is built from exactly these numbers.

**Final greedy evaluation** (mean ± std over 5 seeds; higher is better — Acrobot
and MountainCar returns are negative). Best per environment in **bold**:

| Environment    | Random        | TabularQ      | DQN                | NWM                |
| -------------- | ------------- | ------------- | ------------------ | ------------------ |
| CartPole-v1    | 25.1 ± 3.2    | 98.4 ± 20.2   | **193.7 ± 158.5**  | 159.8 ± 94.0       |
| Acrobot-v1     | −499.9 ± 0.3  | −423.1 ± 53.5 | **−210.2 ± 166.0** | −265.7 ± 161.0     |
| MountainCar-v0 | −200.0 ± 0.0  | −200.0 ± 0.0  | −173.2 ± 27.5      | **−130.4 ± 9.7**   |

**Takeaways.** NWM is competitive with DQN on dense CartPole at markedly lower
variance; it learns on Acrobot but trails DQN's value bootstrapping; and on
sparse-reward **MountainCar it is now the strongest and most stable method,
beating DQN** — thanks to *temporally-correlated (sticky) exploration*, which
builds the momentum uniform noise cannot. That last result shows the earlier
MountainCar failure was an *exploration* problem, not a credit-assignment one.
With `adaptive_repeat=True` the stickiness self-tunes: MountainCar reaches
−162.2 ± 28.8 (still ahead of DQN) with **no task-specific tuning at all**.
See [`paper/`](paper/) for the full analysis, including negative results on
bootstrapped credit assignment.

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
