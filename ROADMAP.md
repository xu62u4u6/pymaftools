# pymaftools Optimization Roadmap

> Last updated: 2026-07-14

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
- [x] Add `ruff` linting and formatting checks to CI workflow
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

### 2.2 Increase Test Coverage (current: 65%, target: 85%+)

- [ ] Add tests for `MAF.py` (currently 31%, core parsing untested)
- [ ] Add tests for `CopyNumberVariationTable.py` (currently 13%)
- [ ] Add tests for `model/StackingModel.py` (currently 28%)
- [ ] Add tests for `utils/geneset.py` (currently 19%)
- [ ] Add edge case tests: empty DataFrames, invalid inputs, unicode gene names
- [ ] Add integration tests for Cohort + PivotTable workflows

---

## Phase 3: Architecture & Performance

### 3.1 Refactor PivotTable (1,692 lines -> smaller modules)

- [x] Extract IO/persistence helpers to `core/pivot_io.py`
- [x] Extract frequency helpers to `core/pivot_frequency.py`
- [x] Extract sorting helpers to `core/pivot_sorting.py`
- [x] Extract filtering helpers to `core/pivot_filtering.py`
- [x] Extract stats helpers to `core/pivot_stats.py`
- [ ] Extract PCA/clustering logic to `core/analysis.py`
- [x] Keep `PivotTable.py` as the API owner and thin delegation layer

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

- [x] Set up Sphinx with autodoc and enforce a warning-free build
- [ ] Add architecture overview diagram
- [ ] Deploy docs to GitHub Pages

### 4.2 Developer Experience

- [ ] Add `CONTRIBUTING.md`
- [ ] Add `.pre-commit-config.yaml` (ruff, mypy)
- [x] Add tag-gated PyPI publishing workflow
- [ ] Add changelog generation (e.g. `git-cliff`)

---

## Phase 5: Feature Enhancements

### 5.1 Export & Compatibility

- [x] Make HDF5 the canonical persistence format for PivotTable and Cohort data
- [x] Deprecate SQLite persistence APIs in favor of HDF5
- [x] Update README, Skill, and API docs to use HDF5 examples
- [ ] Add Parquet export support
- [ ] Add Excel export support
- [ ] Add compression options for exports
- [x] Add lightweight VCF parser / harmonized VCF table (`VCF.read_vcf`)
- [x] Add same-sample multi-caller consensus merge (`VCF.merge_callers`)
- [ ] Document VCF scope: caller consensus only, not full cross-vendor VCF normalization
- [ ] Add caller adapters for Strelka2 / DRAGEN / DeepVariant / Sentieon if real inputs are available
- [ ] Delegate production VCF normalization to external tools (`bcftools norm` / annotation tools), not hand-rolled Python

### 5.2 Advanced Analytics

- [ ] Add survival analysis support
- [ ] Add cross-validation utilities for ML models
- [x] Add more statistical tests (filter_by_variance, filter_by_statistical_test)
- [x] Add MAF-level WES summaries: `plot_overview`, `plot_titv`, `plot_rainfall`, `plot_vaf`
- [x] Add somatic interaction statistics and heatmap
- [x] Add cohort mutation-frequency comparison and forest-style plot
- [ ] Add `mafbarplot` / top mutated gene barplot convenience API
- [ ] Add `coBarplot` for two-cohort mutation-frequency comparison
- [ ] Add `coOncoplot` for side-by-side cohort oncoplots
- [ ] Add `oncostrip` for compact single/few-gene mutation strips
- [ ] Add `vafCompare` for paired / longitudinal / multi-region VAF comparison
- [ ] Add `plot_somatic_interaction_network`
- [ ] Add `plot_similarity_panel`
- [ ] Add `plot_signature_96` and polished signature heatmap helpers
- [ ] Add `plot_tmb` and multi-metric WES panel helper
- [ ] Add shared/private mutation plot (`UpSet` / Venn / table-style panel)
- [ ] Add CNV frequency / GISTIC-like plots if CNV API scope is accepted

### 5.3 User Experience

- [ ] Add CLI tool for common operations
- [ ] Consistent progress bar usage (tqdm)
- [ ] Add structured logging (replace print statements)
- [ ] Consider interactive plots (plotly) as optional feature
- [x] Isolate optional LLM annotation helpers from numeric clustering code

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
- [x] 🟠 **B1** Feature/sample group support — done in v0.5.0: `group_features`/`group_samples` section the matrix + aligned tracks with real whitespace gaps, equal row heights, and bidirectional group titles (alignable).
- [x] 🟠 **B2** `add_xticklabel(fontsize=None, rotation=90)` — done in v0.5.0.
- [ ] 🟡 **B3** Custom y-axis side label; `numeric_heatmap` force-clears ylabel. (Partial: `show_ylabel` exists.)
- [x] 🟠 **B4** `numeric_heatmap` reuses `ax_freq` as colorbar → freq column and colorbar can't coexist. — done in v0.5.0 (CNV colorbar renders in the legend area, not `ax_freq`).
- [x] 🟡 **B5** Rigid 4-column layout requires width-0 dummy columns; use named optional components. — done in v0.5.0 (`render()` derives the GridSpec from registered tracks; `legend_pad` is the named spacer).
- [x] 🟠 **B6** Method chain has implicit ordering dependencies and silently toggles axes; make axis on/off declarative. — done in v0.5.0 (convenience methods register-only; single declarative `render()`).
- [x] 🟡 **B7** `set_config()` calls `plt.close("all")` and rebuilds the figure (global side effect); accept an existing `fig`/subfigure. — done in v0.5.0 (`set_config` no longer touches pyplot; `render(fig=...)` accepts an existing figure).
- [ ] 🟡 **B8** Color mapping is exact-match; unknown categories silently turn white. Warn + add "Unknown" to legend; normalize case/aliases.

### WES / MAF plotting status
- [x] 🟢 `MAF.plot.overview()` / `plot_overview` generates a maftools-like MAF overview dashboard.
- [x] 🟢 `MAF.plot.titv()` / `summarize_titv` covers six-class Ti/Tv summaries.
- [x] 🟢 `MAF.plot.rainfall()` draws inter-mutation distance from MAF genomic coordinates.
- [x] 🟢 `MAF.plot.vaf()` infers VAF from common columns or `t_alt_count` / `t_depth`.
- [x] 🟢 `PivotTable.plot.somatic_interactions()` returns `(fig, stats)` for co-occurrence / mutual exclusivity.
- [x] 🟢 `MAF.plot.compare_cohorts()` + `MAF.plot.forest()` provide first-pass cohort comparison.
- [ ] 🟠 Forest plot should add confidence intervals / pseudo-count policy instead of only capped display odds ratios.
- [ ] 🟡 Rainfall plot should add chromosome tick labels / boundary marks.
- [ ] 🟡 MAF summary should support maftools-style outlier handling and optional raw Ti/Tv inset.

### Similarity panel design
- [ ] 🟠 Add `plot_similarity_panel` as a `SimilarityMatrix` / `PivotTable.plot` public API.

Proposed API:

```python
fig, result = table.plot.similarity_panel(
    method="jaccard",
    group_col="subtype",
    group_order=["LUAD", "ASC", "LUSC"],
    compare_pairs=[("LUAD", "ASC"), ("ASC", "LUSC")],
)
```

Design:
- Input may be a `PivotTable` plus `method`, or a precomputed `SimilarityMatrix`.
- Main panel: sample x sample similarity heatmap, optionally sample-sorted by `group_col`.
- Bottom/top annotation: group color strip aligned to samples.
- Side panels: group mean similarity heatmap and pairwise p-value heatmap.
- Return a structured result object / dict containing the similarity matrix, group means, p-values, and figure.
- Do not recompute similarity inside every subplot; compute once, then render.
- Keep statistics deterministic and explicit: Mann-Whitney or permutation test must be named by parameter.

### Somatic interaction network design
- [ ] 🟠 Add `plot_somatic_interaction_network` as the graph companion to `somatic_interactions`.

Proposed API:

```python
fig, graph, stats = table.plot.somatic_interaction_network(
    top=25,
    alpha=0.05,
    interaction="both",  # "cooccur" | "exclusive" | "both"
    layout="spring",
)
```

Design:
- Reuse `somatic_interactions()` output; the network must not recompute Fisher tests differently from the heatmap.
- Nodes: genes; node size maps to mutation frequency.
- Edges: significant pairs after FDR; color encodes co-occurrence vs mutual exclusivity; width maps to `-log10(FDR)` or `abs(log2 odds ratio)`.
- Return `(fig, graph, stats)` so downstream users can inspect / export the NetworkX graph.
- Keep NetworkX optional if dependency weight is a concern; otherwise add it explicitly.
- Fail loud when no significant edges are found; optionally allow `show_all=True` for exploratory dense networks.

### Recent design review notes (2026-07-14)
- WES plot naming must avoid notebook shorthand. Use domain names such as `cohort1_sample_col`, `cohort1_sample_count`, `cohort1_gene_counts`, `both_mutated`, not `s1`, `n1`, `m1`, `a/b/c/d`.
- `MAF.plot` owns raw MAF-level views; `PivotTable.plot` owns matrix-level views. Do not make `PivotTable` depend on raw MAF columns.
- VCF merge support is currently caller-consensus over already harmonized rows. Cross-vendor VCF normalization belongs in `io.vcf` adapters plus external tools, not in plotting or MAF core.
- Generated demo plots are under `outputs/wes_plots/`; they are verification artifacts, not package data.

> Detailed write-up (symptoms, root cause with file:line, workarounds) in the analysis repo's `docs/pymaftools_limitations.md`.

## Metrics

| Metric | Current (v0.5.0) | Phase 2 Target | Phase 4 Target |
|--------|-------------------|----------------|----------------|
| Test Coverage | 65% | 60% | 85% |
| Type Hints | ~90% | 95% | 95% |
| Largest File | 1,692 lines | < 800 lines | < 500 lines |
| CI Checks | tests + coverage + Ruff + docs/build | + types | + deploy |
