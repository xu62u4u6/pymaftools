# Release Checklist

Every release must use `.github/workflows/publish.yml`. Local commands build and
validate artifacts only; they must never upload to TestPyPI or PyPI.
Known risks, recovery procedures, and deferred improvements are documented in
`RELEASE_AUDIT.md`.

## 0. One-Time Repository Configuration

- [ ] Create GitHub environments named `testpypi` and `pypi`.
- [ ] Require a reviewer for the `pypi` environment.
- [ ] Configure a TestPyPI Trusted Publisher for this repository, workflow
      `publish.yml`, and environment `testpypi`.
- [ ] Configure a PyPI Trusted Publisher for this repository, workflow
      `publish.yml`, and environment `pypi`.
- [ ] Protect `main`: require a pull request and the Tests and Docs checks.
- [ ] Do not store `PYPI_TOKEN` or `TEST_PYPI_TOKEN` repository secrets.

## 1. Release Metadata

The package version comes exclusively from the Git tag through setuptools-scm.
Before opening the release pull request:

- [ ] `CITATION.cff` has `version: X.Y.Z` and the correct release date.
- [ ] `CHANGELOG.md` has a `Version X.Y.Z` section at the top.
- [ ] Release-note links in that changelog section work outside the repository
      file view; prefer absolute URLs.
- [ ] `.claude/skills/pymaftools/SKILL.md` reflects public API changes.
- [ ] `README.md` and `docs/` reflect public API changes.
- [ ] `DATA_SOURCES.md` provenance and checksums cover bundled data changes.

Do not add the release tag on a development branch. The release workflow rejects
tags whose commit is not reachable from `origin/main`.

## 2. Pull Request Gate

- [ ] Open a `dev` to `main` pull request.
- [ ] Tests pass on supported Python 3.10, 3.11, and 3.12.
- [ ] Experimental Python 3.13 and 3.14 jobs have been reviewed.
- [ ] The minimum pandas 2.2 job passes.
- [ ] Coverage is at least 60%.
- [ ] Ruff lint and format checks pass.
- [ ] Sphinx builds with warnings treated as errors.
- [ ] Merge only after all required checks pass.

Optional local equivalents:

```bash
pytest tests/ -v --tb=short --cov=pymaftools --cov-fail-under=60
ruff check pymaftools/
ruff format --check pymaftools/
sphinx-build -W --keep-going docs docs/_build/html
python scripts/extract_release_notes.py X.Y.Z --output /tmp/release-notes.md
```

## 3. Local Artifact Verification

From a clean checkout, run:

```bash
bash deploy.sh
```

This removes old local artifacts, builds wheel and sdist, and runs Twine's
metadata check. It never uploads. Optionally install the resulting wheel into a
temporary environment:

```bash
uv venv /tmp/pymaftools-release
uv pip install dist/*.whl --python /tmp/pymaftools-release/bin/python
/tmp/pymaftools-release/bin/python -c \
  "import pymaftools; print(pymaftools.__version__)"
```

## 4. Start the Release

After the release pull request is merged and `main` CI is green:

```bash
git switch main
git pull --ff-only origin main
git status --short
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

Before pushing, confirm `git status` is empty, `git rev-parse HEAD` is the commit
intended for release, and the exact version does not already exist on TestPyPI or
PyPI. Do not push another release tag while a release workflow is active.

The tag starts one protected pipeline that:

1. Verifies the exact `vX.Y.Z` tag, release metadata, and `main` ancestry.
2. Re-runs Ruff, tests, coverage, and warning-free Sphinx documentation.
3. Builds wheel and sdist once, runs `twine check`, and records SHA256 checksums.
4. Publishes that artifact to TestPyPI with OIDC Trusted Publishing.
5. Downloads the TestPyPI wheel without dependencies, compares it byte-for-byte
   with the build artifact, installs dependencies only from PyPI, and smoke-tests
   the bundled HDF5 table.
6. Waits for approval on the protected `pypi` environment.
7. Verifies checksums and publishes the same artifacts to PyPI with OIDC Trusted
   Publishing.
8. Creates the GitHub Release from the matching changelog section and attaches
   wheel, sdist, and `SHA256SUMS`.

Never run `twine upload` manually. Published package files are immutable. Follow
the failure recovery matrix in `RELEASE_AUDIT.md`; whether a tag can be recreated
depends on whether TestPyPI or PyPI accepted any file.

## 5. Post-Release

- [ ] `pip install pymaftools==X.Y.Z` succeeds from PyPI.
- [ ] The bundled HDF5 example loads from the installed wheel.
- [ ] PyPI contains both wheel and sdist, and their hashes match `SHA256SUMS`.
- [ ] The GitHub Release contains wheel, sdist, `SHA256SUMS`, and the complete
      changelog section for this version.
- [ ] Documentation is deployed at `https://dionic.xyz/pymaftools/`.
- [ ] Commits after the tag receive the next setuptools-scm development version.
