#!/usr/bin/env bash
# Build the manuscript. Run from anywhere:  bash paper/build.sh [--clean]
#
# Intermediates go to paper/build/; the deliverable is copied back out to
# paper/recycling_the_living.pdf, which is the one path .gitignore lets through.
set -euo pipefail
cd "$(dirname "$0")"

[ "${1:-}" = "--clean" ] && latexmk -C -outdir=build main.tex >/dev/null 2>&1 || true

# bibtex runs with build/ as its cwd, so it needs to be told where refs.bib and
# tmlr.bst live. Without this it reports "I couldn't open database file".
export BIBINPUTS=".:$PWD:"
export BSTINPUTS=".:$PWD:"
export TEXINPUTS=".:$PWD:"

latexmk -pdf -interaction=nonstopmode -outdir=build main.tex
cp build/main.pdf recycling_the_living.pdf
echo
echo "-> paper/recycling_the_living.pdf ($(du -h recycling_the_living.pdf | cut -f1), $(pdfinfo build/main.pdf 2>/dev/null | awk '/^Pages/{print $2" pages"}'))"
