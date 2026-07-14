"""Plotting utilities: learning curves and summary bar charts.

Matplotlib is imported lazily so the rest of the benchmark runs without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.metrics import mean_eval_curve
from benchmarks.runner import RunResult

# A stable, colorblind-friendly color per agent.
_AGENT_COLORS = {
    "Random": "#9E9E9E",
    "TabularQ": "#4C78A8",
    "DQN": "#F58518",
    "NWM": "#54A24B",
}


def plot_learning_curves(
    env_name: str, runs_by_agent: dict[str, list[RunResult]], out_path: Path
) -> None:
    """Plot mean +/- std evaluation curves for every agent on one environment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for agent, runs in runs_by_agent.items():
        if not runs or not runs[0].eval_points:
            continue
        points, mean, std = mean_eval_curve(runs)
        color = _AGENT_COLORS.get(agent)
        ax.plot(points, mean, label=agent, color=color, linewidth=2)
        ax.fill_between(points, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_title(f"Learning curves - {env_name}")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Greedy evaluation return")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_final_comparison(
    summaries_by_env: dict[str, dict[str, tuple[float, float]]], out_path: Path
) -> None:
    """Grouped bar chart of final-evaluation mean +/- std per env and agent."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    envs = list(summaries_by_env.keys())
    agents = sorted({a for env in summaries_by_env.values() for a in env})
    n_envs = len(envs)
    n_agents = len(agents)
    width = 0.8 / max(1, n_agents)

    fig, ax = plt.subplots(figsize=(1.6 * n_envs + 3, 4.5))
    x = np.arange(n_envs)
    for j, agent in enumerate(agents):
        means = [summaries_by_env[e].get(agent, (np.nan, 0.0))[0] for e in envs]
        stds = [summaries_by_env[e].get(agent, (np.nan, 0.0))[1] for e in envs]
        ax.bar(
            x + j * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=agent,
            color=_AGENT_COLORS.get(agent),
        )

    ax.set_title("Final greedy evaluation (mean +/- std across seeds)")
    ax.set_ylabel("Return")
    ax.set_xticks(x + width * (n_agents - 1) / 2)
    ax.set_xticklabels(envs, rotation=15)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
