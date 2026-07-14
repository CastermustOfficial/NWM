# Contributing to NWM

Thanks for your interest in improving NWM! This guide covers the local
development workflow.

## Development setup

```bash
git clone https://github.com/CastermustOfficial/NWM.git
cd NWM
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,benchmark]"
pre-commit install
```

## Quality gates

All of the following must pass before a PR is merged (they also run in CI):

```bash
ruff check .            # lint
ruff format --check .   # formatting
mypy                    # strict static typing (src/nwm)
pytest --cov=nwm        # tests + coverage
```

`pre-commit run --all-files` runs the same lint/format/type hooks locally.

## Project layout

| Path            | Purpose                                                      |
| --------------- | ----------------------------------------------------------- |
| `src/nwm/`      | Library source (installed package).                         |
| `tests/`        | Pytest suite (unit + integration).                          |
| `benchmarks/`   | Reproducible benchmark harness and baseline agents.         |
| `examples/`     | Runnable usage examples.                                    |
| `paper/`        | LaTeX source for the accompanying paper.                    |
| `results/`      | Benchmark outputs (JSON/CSV/plots); not committed.          |

## Conventions

- Public functions and methods carry type hints and NumPy-style docstrings.
- Keep the public API backward compatible; deprecate before removing.
- Add or update tests for any behavioral change.
- New algorithmic behavior should come with a benchmark or a test that
  demonstrates it.
- Update `CHANGELOG.md` under an `Unreleased` section.

## Releasing

1. Bump `version` in `pyproject.toml` and `__version__` in `src/nwm/__init__.py`.
2. Update `CHANGELOG.md`.
3. Tag `vX.Y.Z` and push; the `release.yml` workflow builds and publishes to
   PyPI via Trusted Publishing.
