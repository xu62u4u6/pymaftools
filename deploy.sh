#!/usr/bin/env bash
set -euo pipefail

upload=false
case "${1:-}" in
    "") ;;
    --upload) upload=true ;;
    *)
        echo "Usage: $0 [--upload]" >&2
        exit 2
        ;;
esac

if [[ "$upload" == true ]]; then
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "Refusing to upload from a dirty working tree." >&2
        exit 1
    fi

    tag="$(git describe --tags --exact-match 2>/dev/null || true)"
    if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Refusing to upload: HEAD must have an exact vX.Y.Z tag." >&2
        exit 1
    fi

    expected_version="${tag#v}"
    actual_version="$(python -m setuptools_scm)"
    if [[ "$actual_version" != "$expected_version" ]]; then
        echo "Tag/version mismatch: tag=$expected_version setuptools-scm=$actual_version" >&2
        exit 1
    fi
fi

rm -rf -- build dist
python -m build
python -m twine check dist/*

if [[ "$upload" == true ]]; then
    python -m twine upload dist/*
else
    echo "Build verified. Run '$0 --upload' from the clean release tag to publish."
fi
