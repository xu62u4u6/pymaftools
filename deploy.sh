#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
    echo "Usage: $0" >&2
    echo "Publishing is handled exclusively by the protected GitHub release workflow." >&2
    exit 2
fi

rm -rf -- build dist
python -m build
python -m twine check dist/*
echo "Build verified. Push an exact vX.Y.Z tag from main to start the release workflow."
