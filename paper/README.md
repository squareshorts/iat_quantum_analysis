# Paper Folder

This folder contains the manuscript scaffold corresponding to the current repo state.

Current status:

- `main.tex` has been added from the supplied manuscript draft
- figure and table paths point to the repo-level `figures/` and `tables/` directories
- the bibliography file is still missing and must be added as `paper/references.bib`

Compile from this directory after adding the bibliography:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
