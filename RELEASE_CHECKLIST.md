# Release Checklist

Every version release should go through this checklist.

## 1. Code Quality

- [ ] All tests pass: `uv run --extra test pytest tests/ -v --tb=short`
- [ ] Lint clean: `uvx ruff check pymaftools/`
- [ ] Format clean: `uvx ruff format --check pymaftools/`
- [ ] Coverage meets CI threshold: `pytest --cov=pymaftools --cov-fail-under=60`

## 2. Version Bump

The package version is derived from the Git tag by ``setuptools-scm``. Update
release-facing metadata, then create the matching tag:

- [ ] `CITATION.cff` → `version: X.Y.Z` and `date-released: YYYY-MM-DD`
- [ ] `CHANGELOG.md` → Add new version section at top
- [ ] `.claude/skills/pymaftools/SKILL.md` → Update if API changed
- [ ] Confirm `python -m setuptools_scm` reports `X.Y.Z` at tag `vX.Y.Z`

## 3. Documentation

- [ ] `README.md` — update if API changed
- [ ] `CHANGELOG.md` — write release notes (breaking changes, new features, fixes)
- [ ] `docs/getting_started.rst` — update if usage changed
- [ ] Sphinx builds without errors: `sphinx-build docs docs/_build/html`

## 4. Build & Verify

```bash
# Build and check locally (does not upload)
bash deploy.sh

# Upload only from the clean, exact release tag
bash deploy.sh --upload
```

Or manually:

```bash
rm -rf dist/ build/ *.egg-info
python -m build
twine check dist/*

# Test install in clean env
uv venv /tmp/test-install && \
  uv pip install dist/*.whl --python /tmp/test-install/bin/python && \
  /tmp/test-install/bin/python -c "import pymaftools; print('OK')"
```

- [ ] Build succeeds
- [ ] `twine check dist/*` no warnings
- [ ] Clean install works

## 5. Git & Publish

- [ ] All changes committed
- [ ] Push to `main`
- [ ] CI passes on `main`
- [ ] Create git tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- [ ] Push tag: `git push origin vX.Y.Z`
- [ ] Create GitHub Release with CHANGELOG content
- [ ] Publish to PyPI: `twine upload dist/*`

## 6. Post-Release

- [ ] Verify on PyPI: `pip install pymaftools==X.Y.Z`
- [ ] Verify docs deployed: `https://dionic.xyz/pymaftools/`
- [ ] Confirm post-tag development builds receive the next setuptools-scm dev version
