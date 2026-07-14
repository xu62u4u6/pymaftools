# Release Process Audit

Last reviewed: 2026-07-15

This document audits the release path for `pymaftools`. It separates controls in
the repository from settings that live in GitHub, TestPyPI, and PyPI. The
operational checklist is in `RELEASE_CHECKLIST.md`.

## Current Release Path

1. A pull request merges release metadata and code into `main` after required
   Tests and Docs checks pass.
2. An annotated `vX.Y.Z` tag starts `.github/workflows/publish.yml`.
3. The workflow validates the tag syntax, `main` ancestry, setuptools-scm
   version, `CITATION.cff`, and the matching `CHANGELOG.md` section.
4. Ruff, the test suite, coverage, and warning-free Sphinx documentation run
   again from the tagged commit.
5. The workflow builds one wheel and one sdist, runs `twine check`, records their
   SHA256 hashes, and stores them as Actions artifacts.
6. The exact artifacts are published to TestPyPI with OIDC. The wheel is then
   downloaded without dependencies, compared byte-for-byte with the build
   artifact, and installed with dependencies obtained only from PyPI.
7. The HDF5 example workflow is smoke-tested from the installed wheel.
8. The protected `pypi` environment waits for approval, verifies the saved
   checksums, and publishes the same artifacts to PyPI with OIDC.
9. A GitHub Release is created from the matching changelog section with the
   wheel, sdist, and `SHA256SUMS` attached.

Local `deploy.sh` only builds and validates. It has no upload credentials or
upload command.

## Controls Verified

Repository controls verified on 2026-07-15:

- `main` requires a pull request, strict required checks, resolved review
  conversations, and applies protection to administrators.
- Force pushes and branch deletion are disabled on `main`.
- Required checks cover Ruff, Python 3.10-3.12, minimum pandas, package build,
  and documentation build.
- `testpypi` and `pypi` environments accept deployments only from `v*` tags.
- `pypi` requires approval by `xu62u4u6`; self-approval is allowed. This is an
  intentional confirmation gate, not independent two-person review.
- TestPyPI and PyPI Trusted Publishers are proven operational by the successful
  OIDC release of 0.5.0. Their configuration is external to this repository.
- Release 0.5.0 completed the full quality, build, TestPyPI, installed HDF5
  smoke, PyPI, and GitHub Release sequence.

## Findings

### Addressed

| ID | Risk | Control |
| --- | --- | --- |
| A1 | GitHub-generated notes reduced a large release to PR titles. | Release notes are extracted from the exact `CHANGELOG.md` version section; missing, duplicate, or empty sections fail validation. |
| A2 | TestPyPI installation used two indexes, so package and dependency origin was ambiguous. | The target wheel is downloaded from TestPyPI with `--no-deps`, compared to the build artifact, then installed with dependencies from PyPI only. |
| A3 | Consumers could not easily confirm that GitHub assets matched the built distributions. | The pipeline verifies and publishes `SHA256SUMS`; the TestPyPI wheel is also compared byte-for-byte. |
| A4 | Release metadata verification depended on an undeclared setuptools-scm CLI. | The quality job checks the installed distribution version through `importlib.metadata`; the build job installs its build tools explicitly. |
| A5 | A local publishing script could bypass protected environments. | `deploy.sh` is build-only, and publishing is restricted to OIDC environments. |

### Open Operational Risks

| ID | Priority | Risk and current decision |
| --- | --- | --- |
| O1 | High | **Partial publication is not automatically recoverable.** Package index files are immutable. Use the recovery matrix below; never rebuild or replace an artifact after PyPI accepted it. |
| O2 | Medium | **Final PyPI retrieval is manual.** The pipeline validates TestPyPI before promotion, but does not download the final PyPI files and compare them to `SHA256SUMS`. Keep the post-release check mandatory; automate it if releases become frequent. |
| O3 | Medium | **Any ancestor of `main` can be tagged.** This supports deliberate maintenance releases but also permits an accidentally stale commit. The checklist requires updating `main` and reviewing the tag target. Enforce equality with `origin/main` if maintenance tags are never needed. |
| O4 | Medium | **Release runs have no global concurrency group.** Two different version tags can progress concurrently. Do not push a second release tag while one is active; add a non-cancelling global release concurrency group if release frequency increases. |
| O5 | Medium | **GitHub and package-index settings can drift outside Git.** Environment reviewers, tag filters, branch protection, and Trusted Publishers require a manual audit before each minor release. |
| O6 | Medium | **Third-party Actions use mutable major-version tags.** This is normal and maintainable, but a compromised upstream tag is a supply-chain risk. Pin Actions to full commit SHAs and enable Dependabot updates if stronger assurance is required. |
| O7 | Low | **Build inputs are bounded but not fully locked.** Runner images, apt packages, pip, and build dependencies can change. A constraints file or locked release environment would improve reproducibility at maintenance cost. |
| O8 | Low | **Only the wheel is installation-smoke-tested.** The sdist receives metadata validation but is not built and imported in a fresh environment. Add an sdist install test if downstream source builds become important. |
| O9 | Low | **Documentation follows `main`, not release versions.** The public docs may describe unreleased changes after development resumes. Versioned documentation is useful only when API release cadence justifies its upkeep. |
| O10 | Low | **The changelog can contain relative links that do not resolve correctly in a GitHub Release body.** Review links before tagging and prefer absolute repository URLs in release sections. |
| O11 | Low | **Experimental Python 3.13 and 3.14 jobs may fail without blocking merge.** Their output must be reviewed, and a supported-version change must update the matrix and metadata together. |
| O12 | Low | **Tags are not protected by a repository ruleset and commits are not required to be signed.** Branch protection, tag validation, and the PyPI approval gate are sufficient for the current single-maintainer model. Revisit for multiple maintainers. |

### Deferred by Design

- GitHub artifact attestations and SBOM publication are not enabled. OIDC Trusted
  Publishing, build-once promotion, exact wheel comparison, and checksums are
  proportionate today. Attestations become useful when consumers will actually
  verify them or when the project has stronger supply-chain requirements.
- Independent reviewer approval is not required. The current `pypi` reviewer is
  the maintainer and self-review is allowed, so the gate prevents accidental
  publication but does not protect against a compromised maintainer account.

## Failure Recovery Matrix

First determine whether the version exists on TestPyPI or PyPI. Do not move or
recreate a tag until this is known.

| Failure point | Recovery |
| --- | --- |
| Before any TestPyPI upload | Fix through a pull request. If the old tag never published anywhere, delete and recreate it only after confirming both indexes are empty. A new patch version is still the safer option. |
| After TestPyPI, before PyPI | Use **Re-run failed jobs** when no workflow change is required. If code or workflow changes are required, release a new patch version because TestPyPI already owns the old filename. |
| During PyPI upload | Check the PyPI project page and JSON API first. If any file exists, treat the version as published and never reuse it. Diagnose whether all expected files arrived before deciding the next action. |
| After PyPI, before GitHub Release | Do not rebuild. Recover the exact wheel and sdist from the retained Actions artifact or PyPI, verify `SHA256SUMS`, and create or repair the GitHub Release manually. |
| GitHub Release exists but notes/assets are wrong | Edit the release notes or use `gh release upload --clobber` only with files whose hashes match the published PyPI artifacts. The tag and PyPI version remain unchanged. |
| Documentation deployment fails | Fix Docs through a pull request and redeploy `main`. Do not republish the package. |

## External Configuration Audit

Before each minor release, verify:

```bash
gh api repos/xu62u4u6/pymaftools/branches/main/protection
gh api repos/xu62u4u6/pymaftools/environments
gh api repos/xu62u4u6/pymaftools/environments/testpypi/deployment-branch-policies
gh api repos/xu62u4u6/pymaftools/environments/pypi/deployment-branch-policies
```

GitHub cannot confirm the package-index Trusted Publisher configuration. Check
the TestPyPI and PyPI project publishing settings directly. They must name:

- owner: `xu62u4u6`
- repository: `pymaftools`
- workflow: `publish.yml`
- environments: `testpypi` and `pypi`, respectively

Also confirm that no long-lived PyPI tokens remain in GitHub secrets or local
automation.

## Improvement Trigger Points

Avoid adding controls without a concrete need:

- Add final-PyPI automated hash verification after another release requires
  manual recovery or when releases become routine.
- Add global release concurrency before multiple release branches or maintainers
  can publish tags concurrently.
- Pin Actions by SHA when Dependabot is configured to keep those pins current.
- Add attestations when a consumer or policy will verify them.
- Add versioned docs when `main` regularly gets ahead of the latest public API.
