# Overleaf IEEE Package

This folder is an Overleaf-ready IEEE conference manuscript package.

## Contents
- `main.tex` : manuscript source
- `references.bib` : BibTeX database (includes placeholders)
- `figures/` : bundled result plots

## Compile (local)
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Overleaf usage
1. Create a new project on Overleaf.
2. Upload all files from this folder.
3. Set compiler to `pdfLaTeX`.
4. Replace placeholder citation entries in `references.bib` as needed.
