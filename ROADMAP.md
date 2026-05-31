# pymaftools Optimization Roadmap

> Last updated: 2026-05-31

## Phase 1: Foundation ✅ Complete

### 1.1 Fix Critical Code Quality Issues

- [x] Replace bare `except:` with specific exceptions in `MAF.py`
- [x] Replace wildcard imports in `__init__.py` with explicit imports
- [x] PivotTable `False`/`"WT"` — confirmed as intentional SQLite serialization workaround

### 1.2 Dependency Version Constraints

- [x] Add upper bounds to all dependencies in `pyproject.toml`
- [ ] Consider making `beautifulsoup4`, `statsmodels` optional dependencies

### 1.3 CI/CD Improvements

- [x] Add test coverage reporting and threshold enforcement (`--cov-fail-under=40`)
- [ ] Add `ruff` linting to CI workflow
- [ ] Add `mypy` type checking to CI workflow

---

## Phase 2: Type Safety & Testing (In Progress)

### 2.1 Add Type Hints & Docstrings ✅ Complete

- [x] Core modules: `MAF.py`, `PivotTable.py`, `Cohort.py`
- [x] Plot modules: `OncoPlot.py`, `LollipopPlot.py`
- [x] Model modules: `StackingModel.py`
- [x] Utils modules: `geneset.py`, `reduction.py`
- [x] Standardize docstrings to NumPy style
- [x] Translate comments to English
- [x] Add `requests.get` timeout in `geneset.py`

### 2.2 Increase Test Coverage (current: 31%, target: 60%+)

- [ ] Add tests for `MAF.py` (currently 31%, core parsing untested)
- [ ] Add tests for `CopyNumberVariationTable.py` (currently 13%)
- [ ] Add tests for `model/StackingModel.py` (currently 28%)
- [ ] Add tests for `utils/geneset.py` (currently 19%)
- [ ] Add edge case tests: empty DataFrames, invalid inputs, unicode gene names
- [ ] Add integration tests for Cohort + PivotTable workflows

---

## Phase 3: Architecture & Performance

### 3.1 Refactor PivotTable (1,692 lines -> smaller modules)

- [ ] Extract statistical methods to `core/statistics.py`
- [ ] Extract PCA/clustering logic to `core/analysis.py`
- [ ] Extract persistence (SQLite/HDF5) to `core/persistence.py`
- [ ] Keep data operations in `PivotTable.py`

### 3.2 Performance Optimization

- [ ] Add memoization/caching for `calculate_feature_frequency()`
- [ ] Reduce unnecessary `.copy()` calls across codebase
- [ ] Consider lazy loading for `protein_domains.csv` (24MB)

### 3.3 Move Large Data Files

- [ ] Move `protein_domains.csv` out of git repository
- [ ] Implement download-on-first-use with local caching

---

## Phase 4: Documentation & DX

### 4.1 Documentation

- [ ] Set up Sphinx with autodoc
- [ ] Add architecture overview diagram
- [ ] Deploy docs to GitHub Pages

### 4.2 Developer Experience

- [ ] Add `CONTRIBUTING.md`
- [ ] Add `.pre-commit-config.yaml` (ruff, mypy)
- [ ] Add automatic PyPI publishing workflow
- [ ] Add changelog generation (e.g. `git-cliff`)

---

## Phase 5: Feature Enhancements

### 5.1 Export & Compatibility

- [ ] Add Parquet export support
- [ ] Add Excel export support
- [ ] Add compression options for exports

### 5.2 Advanced Analytics

- [ ] Add survival analysis support
- [ ] Add cross-validation utilities for ML models
- [x] Add more statistical tests (filter_by_variance, filter_by_statistical_test)

### 5.3 User Experience

- [ ] Add CLI tool for common operations
- [ ] Consistent progress bar usage (tqdm)
- [ ] Add structured logging (replace print statements)
- [ ] Consider interactive plots (plotly) as optional feature

---

## Known Issues & API Gaps (from figure reproduction, 2026-05)

Surfaced while reproducing Fig3A/B/C and Supp Fig5A/B. Severity: 🔴 silently
wrong output · 🟠 forces a workaround · 🟡 ergonomics. A1① shipped in v0.4.1
(`add_freq` now fails loud instead of writing all-NaN); the rest are pending.

### Data model (PivotTable / MAF)
- [ ] 🟠 **A1②③** Add `PivotTable.relabel_features(mapper)` to sync data index *and* `feature_metadata.index`; auto-validate metadata before plotting. (A1① done in v0.4.1.)
- [ ] 🟠 **A2** `to_mutation_table()` drops `Chromosome / Variant_Type / Variant_Classification / HGVSp_Short`; add them to `feature_metadata` and provide `format_feature_labels(style=...)` for human-readable labels (`EGFR p.L858R`). Unify the 6- vs 7-segment mutation key schema.
- [ ] 🟡 **A3** No feature-level set/group operations (shared / private); `set_operations(by=..., group_col=...)` + clean pandas 2.2 `fillna` downcasting.
- [ ] 🟡 **A4** `MAF.read_csv` uses CWD-relative paths → easy `FileNotFoundError`; resolve against a base/project dir and report attempted absolute path.
- [ ] 🟡 **A5** (env, not code) Multiple on-disk copies + editable install cause intermittent `cannot import name 'MAF'`; ensure a single installable `pymaftools`.

### Plotting (OncoPlot)
- [ ] 🟠 **B1** `feature_groups` support: stack groups with equal row height, gaps, separators, and section titles (currently needs manual gridspec).
- [ ] 🟠 **B2** `add_xticklabel(fontsize=None, rotation=90)` — fontsize is currently unsettable, rotation hardcoded.
- [ ] 🟡 **B3** Custom y-axis side label; `numeric_heatmap` force-clears ylabel.
- [ ] 🟠 **B4** `numeric_heatmap` reuses `ax_freq` as colorbar → freq column and colorbar can't coexist; give colorbar its own axis.
- [ ] 🟡 **B5** Rigid 4-column layout requires width-0 dummy columns; use named optional components.
- [ ] 🟠 **B6** Method chain has implicit ordering dependencies and silently toggles axes; make axis on/off declarative.
- [ ] 🟡 **B7** `set_config()` calls `plt.close("all")` and rebuilds the figure (global side effect); accept an existing `fig`/subfigure.
- [ ] 🟡 **B8** Color mapping is exact-match; unknown categories silently turn white. Warn + add "Unknown" to legend; normalize case/aliases.

> Detailed write-up (symptoms, root cause with file:line, workarounds) in the analysis repo's `docs/pymaftools_limitations.md`.

## Metrics

| Metric | Current (v0.4.0) | Phase 2 Target | Phase 4 Target |
|--------|-------------------|----------------|----------------|
| Test Coverage | 31% | 60% | 85% |
| Type Hints | ~90% | 95% | 95% |
| Largest File | 1,844 lines | < 800 lines | < 500 lines |
| CI Checks | tests + coverage | + lint + types | + docs |
