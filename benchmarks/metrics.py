"""Aggregation of per-run results into per-(env, agent) summary statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.runner import RunResult


@dataclass
class AgentSummary:
    """Aggregated statistics for one agent on one environment across seeds."""

    env: str
    agent: str
    n_seeds: int
    final_eval_mean: float
    final_eval_std: float
    best_eval_mean: float
    best_eval_std: float
    mean_train_reward: float
    auc: float  # area under the eval curve (sample-efficiency proxy)
    wall_time_s_mean: float

    def to_dict(self) -> dict:
        return {
            "env": self.env,
            "agent": self.agent,
            "n_seeds": self.n_seeds,
            "final_eval_mean": self.final_eval_mean,
            "final_eval_std": self.final_eval_std,
            "best_eval_mean": self.best_eval_mean,
            "best_eval_std": self.best_eval_std,
            "mean_train_reward": self.mean_train_reward,
            "auc": self.auc,
            "wall_time_s_mean": self.wall_time_s_mean,
        }


# ``np.trapezoid`` (NumPy >= 2.0) supersedes the deprecated ``np.trapz``.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201


def _auc(points: list[int], means: list[float]) -> float:
    """Normalized area under the evaluation curve (mean reward over training)."""
    if len(points) < 2:
        return float(means[0]) if means else float("nan")
    x = np.asarray(points, dtype=float)
    y = np.asarray(means, dtype=float)
    return float(_trapezoid(y, x) / (x[-1] - x[0]))


def summarize(runs: list[RunResult]) -> AgentSummary:
    """Aggregate a list of same-(env, agent) runs over seeds."""
    finals = np.array([r.final_eval for r in runs], dtype=float)
    bests = np.array([r.best_eval for r in runs], dtype=float)
    means = np.array([r.mean_train_reward for r in runs], dtype=float)
    aucs = np.array([_auc(r.eval_points, r.eval_means) for r in runs], dtype=float)
    times = np.array([r.wall_time_s for r in runs], dtype=float)

    return AgentSummary(
        env=runs[0].env,
        agent=runs[0].agent,
        n_seeds=len(runs),
        final_eval_mean=float(np.mean(finals)),
        final_eval_std=float(np.std(finals)),
        best_eval_mean=float(np.mean(bests)),
        best_eval_std=float(np.std(bests)),
        mean_train_reward=float(np.mean(means)),
        auc=float(np.mean(aucs)),
        wall_time_s_mean=float(np.mean(times)),
    )


def mean_eval_curve(runs: list[RunResult]) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return (eval_points, mean_curve, std_curve) averaged across seeds."""
    points = runs[0].eval_points
    matrix = np.array([r.eval_means for r in runs], dtype=float)
    return points, matrix.mean(axis=0), matrix.std(axis=0)
