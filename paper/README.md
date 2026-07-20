# NWM Paper

LaTeX source for *"NWM: Negative Weight Mapping — A Non-Parametric
Potential-Field Framework for Reinforcement Learning"*.

## Regenerating results, tables, and figures

The tables and figures are generated from the benchmark outputs, so the paper
always reflects the released numbers:

```bash
# 1. Run the benchmark on the held-out seeds the paper reports
#    (writes results/summary.csv and results/plots/*.png)
python -m benchmarks.run_benchmark --seeds 5 6 7 8 9

# 2. Turn the results into paper assets (tables/results_table.tex, figures/*.png)
python paper/make_paper_assets.py
```

## Building the PDF

A prebuilt **[`nwm.pdf`](nwm.pdf)** is committed alongside the source, so you can
read the paper without a LaTeX toolchain. To rebuild it you need a TeX
distribution (TeX Live / MiKTeX) with `latexmk`:

```bash
cd paper
latexmk -pdf nwm.tex        # or: make
```

Manual build:

```bash
pdflatex nwm && bibtex nwm && pdflatex nwm && pdflatex nwm
```

## Files

| File                     | Purpose                                            |
| ------------------------ | -------------------------------------------------- |
| `nwm.tex`                | Main manuscript.                                   |
| `references.bib`         | Bibliography.                                      |
| `make_paper_assets.py`   | Builds `tables/` and `figures/` from `results/`.   |
| `tables/results_table.tex` | Auto-generated results table (do not edit).      |
| `figures/`               | Auto-generated figures copied from `results/plots`.|
