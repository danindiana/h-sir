#!/usr/bin/env bash
# Re-render every diagram: .dot -> .png (160 dpi) + .svg. Requires graphviz.
#   sudo apt-get install -y graphviz && bash docs/diagrams/render.sh
set -euo pipefail
cd "$(dirname "$0")"
status=0
for f in *.dot; do
  b="${f%.dot}"
  if dot -Tpng -Gdpi=160 "$f" -o "$b.png" && dot -Tsvg "$f" -o "$b.svg"; then
    echo "rendered $b"
  else
    echo "FAILED $f" >&2; status=1
  fi
done
exit "$status"
