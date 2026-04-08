#!/bin/bash
# Build script for LaTeX manuscript
# Compiles main.tex to PDF and cleans up intermediate files

set -e

cd "$(dirname "$0")"

for doc in main revision_notes; do
  echo "Compiling ${doc}.tex..."
  pdflatex -interaction=nonstopmode "${doc}.tex" > /dev/null
  bibtex "$doc" > /dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode "${doc}.tex" > /dev/null
  pdflatex -interaction=nonstopmode "${doc}.tex" > /dev/null

  echo "Cleaning up intermediate files for ${doc}..."
  rm -f "${doc}".{aux,bbl,blg,log,out,toc,lof,lot,nav,snm,vrb,loc,soc}
  echo "Done! Output: ${doc}.pdf"
  echo ""
done
