#!/bin/bash

# You will need to have Graphviz installed to run this script
# You can install it with `sudo apt-get install graphviz` on Debian-based systems

# Make figures from .dot files
for f in *.dot; do
    dot -Tsvg $f -o ${f%.dot}.svg
done
