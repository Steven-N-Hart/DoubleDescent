#!/bin/bash
# Build script for LaTeX manuscript
# Compiles main.tex to PDF and cleans up intermediate files

set -e

cd "$(dirname "$0")"

echo "Compiling LaTeX document..."
pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null

echo "Cleaning up intermediate files..."
rm -f main.aux main.bbl main.blg main.log main.out main.toc main.lof main.lot main.nav main.snm main.vrb

echo "Done! Output: main.pdf"
