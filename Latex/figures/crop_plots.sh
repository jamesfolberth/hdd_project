#! /usr/bin/env bash
# Crop the margins off of matlab's PDF plots using texlive's `pdfcrop'

find . -name wch_box\* -type f -execdir pdfcrop -margin 10 '{}' '{}' \;

find . -name optim\* -type f -execdir pdfcrop -margin 10 '{}' '{}' \;

