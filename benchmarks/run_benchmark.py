"""Benchmark entry point.

Trains every selected agent on every selected environment across multiple seeds,
then writes per-run JSON, an aggregated CSV, a Markdown summary table, and plots
under the output directory (default ``results/``).

Examples
--------
    python -m benchmarks.run_benchmark --quick
    python -m benchmarks.run_benchmark --seeds 0 1 2 3 4
    python -m benchmarks.run_benchmark --envs cartpole acrobot --agents NWM DQN
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from benchmarks.config import (
    AGENT_BUILDERS,
    DEFAULT_SEEDS,
    ENVIRONMENTS,
    EVAL_EPISODES,
    EVAL_INTERVAL,
)
from benchmarks.metrics import AgentSummary, summarize
from benchmarks.runner import RunResult, train_one_run


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the NWM benchmark suite.")
    p.add_argument(
        "--envs",
        nargs="+",
        default=list(ENVIRONMENTS.keys()),
        choices=list(ENVIRONMENTS.keys()),
        help="Environments to run.",
    )
    p.add_argument(
        "--agents",
        nargs="+",
        default=list(AGENT_BUILDERS.keys()),
        choices=list(AGENT_BUILDERS.keys()),
        help="Agents to run.",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), help="Random seeds."
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override episodes per env (default: per-env value from config).",
    )
    p.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    p.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke profile: 2 seeds, 40 episodes, small eval.",
    )
    p.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    seeds = args.seeds
    episodes_override = args.episodes
    eval_interval = args.eval_interval
    eval_episodes = args.eval_episodes

    if args.quick:
        seeds = seeds[:2] if len(seeds) >= 2 else seeds
        episodes_override = 40
        eval_interval = 10
        eval_episodes = 5

    agents = list(args.agents)
    if "DQN" in agents and not _torch_available():
        print("[warn] torch not installed; skipping DQN. Install with '.[baselines]'.")
        agents = [a for a in agents if a != "DQN"]

    out = args.output
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    all_runs: list[RunResult] = []
    summaries: list[AgentSummary] = []
    # env -> agent -> list[RunResult]
    grouped: dict[str, dict[str, list[RunResult]]] = {}

    total_jobs = len(args.envs) * len(agents) * len(seeds)
    job = 0
    t0 = time.perf_counter()

    for env_key in args.envs:
        spec = ENVIRONMENTS[env_key]
        grouped[spec.name] = {}
        for agent_label in agents:
            builder = AGENT_BUILDERS[agent_label]
            runs: list[RunResult] = []
            for seed in seeds:
                job += 1
                tag = f"{spec.name}/{agent_label}/seed{seed}"
                print(f"[{job:>3}/{total_jobs}] {tag} ...", flush=True)
                result = train_one_run(
                    spec=spec,
                    agent_label=agent_label,
                    agent_builder=builder,
                    seed=seed,
                    eval_interval=eval_interval,
                    eval_episodes=eval_episodes,
                    episodes_override=episodes_override,
                )
                runs.append(result)
                all_runs.append(result)
                fname = f"{env_key}__{agent_label}__seed{seed}.json"
                (runs_dir / fname).write_text(json.dumps(result.to_dict(), indent=2))
                print(
                    f"        final_eval={result.final_eval:.1f} "
                    f"best={result.best_eval:.1f} "
                    f"({result.wall_time_s:.1f}s)",
                    flush=True,
                )
            grouped[spec.name][agent_label] = runs
            summaries.append(summarize(runs))

    _write_summary_csv(out / "summary.csv", summaries)
    _write_summary_markdown(out / "summary.md", summaries, seeds)

    if not args.no_plots:
        _make_plots(out, grouped, summaries)

    print(f"\nDone in {time.perf_counter() - t0:.1f}s. Results in '{out}/'.")
    _print_summary(summaries)
    return 0


def _write_summary_csv(path: Path, summaries: list[AgentSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].to_dict().keys()))
        writer.writeheader()
        for s in summaries:
            writer.writerow(s.to_dict())


def _write_summary_markdown(path: Path, summaries: list[AgentSummary], seeds: list[int]) -> None:
    lines = [
        "# NWM Benchmark Results",
        "",
        f"Seeds: {seeds} | Eval: greedy rollouts on disjoint seeds.",
        "",
        "| Environment | Agent | Final eval (mean +/- std) | Best eval | AUC | Time (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        lines.append(
            f"| {s.env} | {s.agent} | "
            f"{s.final_eval_mean:.1f} +/- {s.final_eval_std:.1f} | "
            f"{s.best_eval_mean:.1f} | {s.auc:.1f} | {s.wall_time_s_mean:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _make_plots(
    out: Path,
    grouped: dict[str, dict[str, list[RunResult]]],
    summaries: list[AgentSummary],
) -> None:
    try:
        from benchmarks.plots import plot_final_comparison, plot_learning_curves
    except ImportError:
        print("[warn] matplotlib not installed; skipping plots ('.[plots]').")
        return

    plots_dir = out / "plots"
    for env_name, runs_by_agent in grouped.items():
        slug = env_name.lower().replace("-", "").replace("_", "")
        plot_learning_curves(env_name, runs_by_agent, plots_dir / f"learning_{slug}.png")

    summaries_by_env: dict[str, dict[str, tuple[float, float]]] = {}
    for s in summaries:
        summaries_by_env.setdefault(s.env, {})[s.agent] = (
            s.final_eval_mean,
            s.final_eval_std,
        )
    plot_final_comparison(summaries_by_env, plots_dir / "final_comparison.png")
    print(f"[ok] plots written to '{plots_dir}/'.")


def _print_summary(summaries: list[AgentSummary]) -> None:
    print("\n=== Final evaluation (mean +/- std across seeds) ===")
    current_env = None
    for s in summaries:
        if s.env != current_env:
            current_env = s.env
            print(f"\n{s.env}:")
        print(
            f"  {s.agent:<10} {s.final_eval_mean:8.1f} +/- {s.final_eval_std:5.1f}"
            f"   (best {s.best_eval_mean:7.1f})"
        )


if __name__ == "__main__":
    sys.exit(main())
