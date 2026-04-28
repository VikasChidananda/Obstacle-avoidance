# Overleaf NeurIPS Package

This folder is a NeurIPS-style manuscript package based on current experiment logs.

## Files
- main.tex
- references.bib
- figures/training_reward_curves.png
- figures/training_reward_curves_Cn.png

## Notes
- The manuscript compiles as a standard article locally.
- For strict NeurIPS styling on Overleaf, enable the `neurips_2024` package lines in `main.tex` and ensure `neurips_2024.sty` is present (Overleaf NeurIPS templates already include it).

## Compile locally
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
