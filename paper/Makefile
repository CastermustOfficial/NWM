# Build the NWM paper. Requires latexmk + a TeX distribution.
PAPER = nwm

.PHONY: all clean assets

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib tables/results_table.tex
	latexmk -pdf -interaction=nonstopmode $(PAPER).tex

# Regenerate tables/figures from the benchmark outputs in ../results
assets:
	cd .. && python paper/make_paper_assets.py

clean:
	latexmk -C $(PAPER).tex 2>/dev/null || true
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out
