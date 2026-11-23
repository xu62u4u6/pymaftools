# 🧬 Patch Note

---

## 🔖 Version 0.6.0 (unreleased)

Scientific-validity release adding explicit sample-universe and observation-state
contracts, auditable mutation analysis, stricter TCGA specimen alignment,
producer-aware MAF input, and errors that say how to fix the call.

### ⚠️ Breaking changes

* **NumPy 2 is now required** (`numpy>=2.0`). The bundled example HDF5 table
  stores NumPy-2 pickles, and the compatible stack raises the other floors:
  pandas ≥ 2.2.2, matplotlib ≥ 3.8.4, SciPy ≥ 1.13, scikit-learn ≥ 1.4.2,
  statsmodels ≥ 0.14.2, PyTables ≥ 3.10, typing-extensions ≥ 4.7. CI tests
  these floors directly.
* **Strict MAF validation:** blank `Hugo_Symbol`, duplicate events,
  mixed genome builds or `chr1`/`1` naming and the legacy allele layout raise
  `MAFValidationError` by default; `validate="warn"`/`"off"` and
  `deduplicate=True` opt out. `merge_mafs()` rejects mixed builds unless
  `allow_mixed_builds=True`.
* `read_maf()` keeps `Entrez_Gene_Id` as text; scripts that compare it numerically
  must cast it.
* **Removed** `utils.geneset` (`read_GMT`, `read_gmt`, `fetch_msigdb_geneset`) and
  `ASCStackingModel`: unused inside the package and project-specific. Use
  `gseapy.read_gmt` / `gseapy.Msigdb().get_gmt()`, and
  `model.predict_proba(X)[:, model.class_order.index("LUSC")]` instead of
  `ASCStackingModel.soft_score()`. Accessing the old top-level names raises an
  error naming the replacement. `beautifulsoup4` is no longer a dependency.

### 🏷️ Name unification and deprecations

Old names keep working and return identical results; they only emit a
`DeprecationWarning` that names the replacement and the removal release,
**pymaftools 1.0**. Removals are listed under Breaking changes.

| Deprecated | Use instead |
|---|---|
| `Cohort.to_hdf5()` / `Cohort.read_hdf5()` | `Cohort.to_h5()` / `Cohort.read_h5()` (same file format; matches `PivotTable`) |
| `PivotTable.PCA()` | `PivotTable.pca()` |
| `MAF.vaild_variant_classfication` | `MAF.valid_variant_classification` |
| `utils.geneinfo.get_ncbi_gene_ID(s)` | `get_ncbi_gene_id(s)` |
| `TCGA_sample_type`, `get_target_sample_ID`, `read_TCGA_ASCAT3_CNV_file_sheet` | `tcga_sample_type`, `get_target_sample_id`, `read_tcga_ascat3_cnv_file_sheet` |
| `pymaftools.PCA_CCA` | `pymaftools.utils.reduction.PCA_CCA` |

Project-specific names left the package root (`__all__`); they are still
importable from it with a warning. Earlier deprecations (`to_MAF`, `write_maf`,
`calculate_TMB`, SQLite I/O, `VCF.from_file`, ...) now also state the 1.0
removal release.

### 🧪 Scientific data contracts

* **`SampleManifest`:** declares eligible and excluded samples, patient identity,
  assay, genome build, and sample-specific callable territory; zero-event samples
  remain part of the analysis universe.
* **`ObservationMask`:** distinguishes observed absence from an unobserved
  gene--sample cell and persists with supported table operations and HDF5 files.
* **Audited TMB:** `MAF.calculate_tmb_audit()` returns sample summaries plus an
  event-level inclusion/exclusion ledger under explicit filters, duplicate policy,
  build, and callable-megabase inputs.
* **Audited enrichment:** `PivotTable.mutation_enrichment_test()` reports explicit
  group denominators, odds ratios and intervals, Fisher p-values, and BH-adjusted
  p-values for a declared tested family; repeated patients are rejected by the
  patient-level independence guard.

### 🫁 Scientific validation

* Added exact primary-tumor sample/vial alignment across GDC modalities.
* Added fail-closed download verification and unrelated-normal-file rejection.
* Added adversarial regression coverage for sample universes, observation masks,
  TMB event filtering, patient independence, and enrichment denominators.
* Added a documentation-only clean-wheel user journey for the public quick start.

### 📥 MAF input compatibility

* **Producer-aware `read_maf()`:** GDC, cBioPortal, vcf2maf, Funcotator/Oncotator
  and maftools-style files load; case variants of standard columns are renamed
  (`End_position`), a missing `Tumor_Seq_Allele1` is copied from
  `Reference_Allele`, and identifier columns stay text (sample `"001"`, one
  `Chromosome` dtype). Coordinates are nullable `Int64`.
* **`MAF.ingest_report`:** records the producer guess, comment lines skipped,
  columns renamed or filled, duplicates dropped and every problem found.
* **Denominators:** `to_gene_table(samples=...)` takes the full sample list;
  samples whose events have no gene are kept as all-False columns with a
  warning.
* **Lollipop:** parses cBioPortal bare positions, VEP `?-12/393` ranges and
  `HGVSp_Short`/`Protein_Change`; a missing protein length is taken from the
  domain table with a warning.
* GDC batch readers no longer truncate fields containing `#`.
* Added a five-minute quickstart (`docs/quickstart.rst`,
  `examples/quickstart.py`), producer-style regression fixtures, and a CI
  matrix that installs the declared minimum and the latest dependency versions.

### 🐛 Correctness and usability

* **ROC-AUC follows the model's class order.** `evaluate_model()` (and so
  `cross_validate_importance` / `run_model_evaluation`) scored
  `predict_proba` columns as if they were in sorted label order. With a
  non-alphabetical `OmicsStackingModel.class_order` a perfect multiclass model
  scored 0.5, and a reversed binary order reported 1 - AUC. Benchmarks run with
  such class orders should be re-run. `OmicsStackingModel.classes_` now names
  the `predict_proba` columns (classes seen in training); when `y_test` has a
  class the training fold lacked, AUC is NaN with a warning instead of a
  silently mis-scored value.
* `OmicsStackingModel.plot_final_coefficients()` draws the per-omic `abs_mean`
  track; it was all-NaN.
* Lung-project defaults removed: `CopyNumberVariationTable.plot.band_ratio()`
  shows the cohort's own subtypes (it showed nothing for non-LUAD/ASC/LUSC
  cohorts) and `ModelPlot.plot_metric_comparison_with_annotation()` orders
  groups from the data instead of `SNV/CNV-gene/CNV-cluster/STACK`.
* **Group similarity p-values no longer report NaN as significant.**
  `SimilarityMatrix.calculate_group_similarity_pvalues()` returned the minimum
  p-value `1/(N+1)` whenever the observed group similarity was NaN (e.g. a
  one-sample group); it now returns NaN and drops non-finite permutations from
  the denominator.
* `SimilarityMatrix` group methods align a sample-ID-indexed `groups` Series to
  the matrix rows instead of using its order (reordered metadata silently
  mislabelled samples); a Series missing matrix samples raises.
* `align_cluster_label_dict()` uses the full Hungarian assignment, so a cluster
  absent from the fold's common samples is no longer merged into another one.
* `PCA_CCA.get_weights()` back-projects with CCA's rotations and score scaling;
  the returned feature weights now reproduce the canonical scores (they were
  off by the per-component standard deviation and could rank genes wrongly).
* `MAF.read_maf()` now detects comma- and tab-delimited inputs and reports
  incompatible columns without exposing user-specific paths.
* `MAF.read_csv()` preserves a leading `Hugo_Symbol` as event data and can copy
  a declared source column into `sample_ID`; table conversion also accepts
  named indexes without assuming they are called `index`.
* MAF-to-gene-table conversion scales without repeated frame concatenation.
* TMB duplicate removal occurs after scientific event filters, making results
  independent of input-row order.
* Plot saving accepts `str` and `os.PathLike` destinations.
* Complete sample universes are enforced before patient-level enrichment.
* Somatic interaction FDR has an explicit undirected hypothesis family: each
  unique gene pair is tested once, with self-pairs and mirrored matrix entries
  excluded.
* `somatic_interactions()` respects an attached `ObservationMask`: each pair is
  tested only on jointly observed samples, pairs without any are reported as
  untested and excluded from FDR, and present events marked unobserved are
  rejected (#18).
* `mutation_enrichment_test()` returns an empty result with the full column
  schema for zero-feature tables instead of raising `KeyError` (#19).
* With `method="chi2"`, enrichment marks zero-margin contingency tables as
  untested (`not_tested_reason="degenerate_contingency_table"`) instead of
  aborting the whole analysis; untested rows are excluded from FDR (#20).

* `read_maf()` builds its composite row index column-wise instead of row by
  row: about 2x faster on large files (600k rows: 38 s → 19 s), identical labels.

### 🗣️ LLM usability: fail loud, hint the fix

Calls that used to succeed silently or crash with a bare pandas error now say
what went wrong and how to fix it. Statistical results are unchanged.

* **Large-cohort oncoplots no longer render blank.** Cell gridlines now thin
  with cell size (capped at 20% of a cell, dropped below 2 pt), so a 958-sample
  matrix is visible instead of being painted over by 1 pt white lines. Small
  cohorts keep their previous look.
* `BasePlot.save()` before a figure exists raises `RuntimeError` (it used to
  print "Figure is not exist." and write nothing).
* `PivotTable.add_sample_metadata()` / `add_feature_metadata()` work with the
  default `fill_value=None` (previously a pandas `ValueError`), accept a named
  `Series`, warn on unmatched IDs, and raise when no ID matches at all.
* `OncoPlot.add_freq()` / `plot_freq()` / sample and feature annotations name the
  missing column, list available ones, and point to `table.add_freq()` /
  `add_sample_metadata()`.
* `table.plot.oncoplot()`, `somatic_interactions()` and
  `somatic_interactions_stats()` have explicit signatures and full docstrings
  instead of `**kwargs`; `oncoplot(top=...)` explains that gene selection is a
  table operation.
* `mutation_enrichment_test()` lists available columns / observed group values
  on a bad `group_col` or group name; `maf.plot.lollipop()` says when a gene has
  no mutations in the MAF and lists the most mutated genes.
* New scientific reminders (warnings only): `to_gene_table()` when synonymous /
  non-coding events are kept; somatic interactions on fewer than 10 samples, or
  when most pairs are significant co-occurrences (typical mutation-burden
  confounding).

### 🧾 Machine-readable results, provenance and CLI

Additive only: no return type or computed number changes (statistics and
rendered PNG bytes verified identical to the previous `dev`).

* **`pymaftools.to_json()` / `to_jsonable()`** serialise any result to strict
  JSON (`NaN` → `null`, `±inf` → `"inf"`/`"-inf"`). `IngestReport`,
  `TMBAudit` gain `to_dict()` / `to_json()`.
* **`schema_version`** (`"1.0"`) in every JSON record (provenance, ingest
  report, TMB audit, figure summaries, CLI) and `pymaftools --version`; CLI
  records also start with `pymaftools_version`.
* **Provenance:** `somatic_interactions()`, `mutation_enrichment_test()` and
  `calculate_tmb_audit()` attach analysis name, parameters, input-file SHA-256
  and the sample universe (`pymaftools.get_provenance(result)`; DataFrames carry
  it in `df.attrs`). `read_maf()` records the input's size and SHA-256 in
  `ingest_report`, and derived tables keep that fingerprint.
* **`pymaftools.session_info()`** returns package / Python / dependency versions.
* **Figure summaries:** `plot.summary()` and `plot.save(path, summary=True)`
  (writes `path` as `.json`) describe what was drawn: empty axes, and for
  `OncoPlot` the matrix shape, fill fraction, top features, auto-hidden sample
  labels and cell size in pixels, with warnings for a blank matrix.
  `pymaftools.figure_summary(fig)` works on any matplotlib figure.
* **CLI:** `pymaftools inspect | oncoplot | interactions | session-info`
  (`--json` for machine output, exit code 2 for strict-validation failures);
  also `python -m pymaftools`.
* `BasePlot.save()` now raises backend errors instead of printing them and
  returning as if the file had been written.
* Fixes from a documentation-only agent trial on TCGA ACC (92 samples):
  * `add_sample_metadata()` / `add_feature_metadata()` realign the metadata to
    the table's current order (values were always matched by ID, but the
    *row order* was sorted, so a sorted table failed later with an opaque
    index error). Index-mismatch errors now say whether the IDs differ or
    only their order, and give the fixing call. Docstrings state that both
    return a new table.
  * **Interaction heatmap colour scale changed** to `sign(log OR) × −log10(FDR)`
    (what its docstring always claimed). Previously colour was log2 OR, so
    sparse pairs with OR = 0 / ∞ but no significance were drawn as the
    strongest effects. Untested pairs are grey. The statistics are unchanged;
    the figure's `figure_summary()` reports `n_tested`, `n_significant`,
    `n_infinite_or_zero_odds_ratio` and warns about sparse or
    burden-confounded results.
  * Oncoplot annotation / bar tracks record their column, missing count and
    distinct values in `summary()["tracks"]`; an all-missing column warns at
    registration and in the summary (also > 50 % missing or a single value).
  * CLI: `--samples FILE` (plain list or a cBioPortal clinical-sample table;
    `--sample-col`) declares the full sample universe so zero-event samples
    count in denominators; without it every command warns. `pymaftools
    oncoplot` sorts samples into a waterfall; `inspect` ranks genes like
    `oncoplot`; interaction JSON no longer includes a positional `index`
    column; CLI provenance records `validate`, `all_classes` and
    `samples_file` for every command.

### 🎨 Plot fixes

* **OncoPlot sample labels (#16):** sample-label gutters now share the heatmap's
  `(0, n)` data limits, so labels sit on their column centres instead of half a
  cell to the right (ungrouped and `group_samples` layouts).
* **Legend spacing (#17):** `LegendManager.plot_legends` floors title/item
  spacing at the physical text height, so legend titles no longer overlap the
  first entry on short figures; a `UserWarning` is raised when legends overflow
  the legend axis.

### 🤖 Reusable AI assets

* Added package-distributed researcher, user-tester, biostatistics-reviewer,
  bioinformatics-validator, reproducibility-reviewer, and manuscript-writer
  agents plus a Pymaftools skill.
* Added `pymaftools-ai` installation and dry-run commands for project or user
  scope in Codex and Claude environments.
* The documentation site now publishes `llms.txt` and `llms-full.txt`
  (https://llmstxt.org), generated from the Sphinx sources, so coding agents
  can find the docs without installing anything.
* `AGENTS.md` is now the single contributor/agent guide for Codex, Cursor,
  Copilot and similar tools; `CLAUDE.md` imports it instead of keeping a copy.
* Removed the stale Chinese-language 0.5.0 `AUDIT.md`; release state lives in
  `RELEASE_AUDIT.md` and this changelog. The AI agent test report is now in
  English like the rest of the documentation.

### ✅ Validation

* The `dev` suite after merging #21-#25 passes 409 tests (7 skipped, 10
  deselected), on the declared minimum and the latest dependency versions.
* Warning-free Sphinx, Ruff, and wheel/sdist metadata checks are release gates.

---

## 🔖 Version 0.5.0 (July 2026)

Track-based redesign of the plotting module (`OncoPlot`). The visualization is now
composed of declarative **tracks** (a main matrix plus annotation strips), laid out
by a single `render()` call. See `PLOTTING_REVIEW.md` for the full design.

### ⚠️ Breaking Changes
* **Python data stack baseline:** pandas 2.2 or newer is now required because the public table operations use APIs introduced in pandas 2.2.
* **Continuous event conversion:** ``PivotTable.to_binary_table()`` now requires an explicit threshold for continuous numeric matrices. Boolean, categorical, and 0/1 event tables remain automatic; discrete GISTIC CNV calls retain non-zero event semantics.
* **`render()` is now required.** The eager immediate-draw layout (`update_layout` and the fixed named axes) was removed. The convenience methods (`mutation_heatmap`, `numeric_heatmap`, `plot_freq`, `plot_bar`, `plot_categorical_metadata`, `plot_numeric_metadata`) now only *register* tracks; call `.render()` once before `save()` / `add_xticklabel()`. Migration: add `.render()` to existing chains.
* **`PivotTablePlot` → `PivotStatsPlot`.** The class was renamed (it provides one-shot statistical plots — PCA, annotated boxplots — not the base for all pivot-table plotting). The old import path `from pymaftools.plot.PivotTablePlot import PivotTablePlot` still works via a shim.
* **Removed** `OncoPlot.heatmap_rectangle` and `OncoPlot.plot_color_heatmap` (orphaned; relied on the eager axes).
* **`OncoPlot.default_oncoplot`**: dropped the `width_ratios` argument; it now registers tracks and renders.

### 🆕 New Features
* **HDF5-first persistence:** single tables use ``to_h5`` / ``read_h5`` and cohorts use ``to_hdf5`` / ``read_hdf5``. SQLite persistence is deprecated because its practical column limit is unsuitable for expression and other high-dimensional matrices.
* **Installed example-table API:** ``load_example_table()`` and ``example_table_path()`` expose the bundled HDF5 fixture without relying on the current working directory.
* **VCF public API:** ``pymaftools.VCF`` reads plain or gzip VCF files, expands multi-allelic records with allele-specific AD/AF values, filters calls, writes harmonized VCF, and merges caller consensus.
* **Optional analysis extras:** install ``pymaftools[expression]`` for PyDESeq2 and ``pymaftools[anndata]`` for AnnData conversion; missing dependencies now report the exact install command.
* **Multi-omics feature preparation:** ``OmicsStackingModel.prepare_features()`` aligns common samples and namespaces feature names that overlap across omics layers.
* **Feature-side annotations** (`add_feature_annotation`): `feature_metadata` columns (e.g. `pathway`) can finally be drawn as row-side strips — impossible under the old layout.
* **Declarative track API**: `main()`, `add_bar()`, `add_freq()`, `add_sample_annotation()`, `add_feature_annotation()` register tracks; `render()` derives the GridSpec from each track's `side`/`size`. Numeric vs categorical is inferred from column dtype.
* **Axis grouping** (`group_features`, `group_samples`): partition rows (by a `feature_metadata` column, e.g. `pathway`) and/or columns (by a `sample_metadata` column, e.g. `subtype`) into sections separated by real whitespace gaps, with group titles on the left (features) and top (samples). Every aligned track (bar/freq/strips) is sectioned to match, with value ranges pinned so the sections share one scale. Tunable `gap`, `title_align` (start/center/end).
* **Unified entry point**: oncoplots are reachable from the same accessor as the stats plots — `table.plot.oncoplot(...)`.
* **`main(kind="cnv")`**: continuous (CNV) main matrices go through `render()` too.
* **Numeric metadata colorbars** (`plot_numeric_metadata`, P1#3): numeric strips carry a colorbar so the value scale is interpretable. Placement is a mode — `colorbar="legend"` (default; stacked in the legend area, readable with several numeric columns), `"inset"` (small bar beside the strip), or `"off"`. Per-column cmaps via `cmap_dict=`.
* **`render()` layout knobs**: `main_width`, `main_height`, `legend_width`, `legend_pad` (a named spacer replacing the old phantom column), `colorbar_width`, `wspace`, `hspace`.
* **`add_xticklabel(fontsize=, rotation=)`**: was hardcoded to `rotation=90`.
* **Per-section frequency strips** (`group_samples(..., freq=True)`): each sample group draws its own frequency strip (read from `feature_metadata["{label}_freq"]`, e.g. `LUAD_freq`) right of that group's heatmap section, alongside the overall `add_freq(...)` bar. Tunable `freq_suffix`, `freq_size`, `freq_annot`.
* **Auto sample labels** (`render(show_sample_labels=None)`): per-column sample tick labels auto-hide above 50 samples; pass `True`/`False` to force.
* **Exon size from Ensembl BioMart** (`PivotTable.add_exon_size(col="exon_size", metric="transcript_length"|"cds_length", force_download=False)`): annotates each feature (gene) with its canonical-transcript exon size for size-based gene grouping. Returns a copy; network-dependent (BioMart). Backed by `get_exon_size(genes, metric=...)` / `load_gene_sizes(force=)` in `pymaftools.utils.geneinfo`, re-exported at top level.
* **`add_freq(group_col="subtype")`**: convenience that auto-splits samples by a `sample_metadata` column into per-group `{value}_freq` columns — equivalent to building the `groups={...}` dict manually (mutually exclusive with it).
* **Multi-key `sort_features(by=[...], ascending=[...])`**: `by`/`ascending` now accept lists for hierarchical sort (e.g. `by=["size_group", "freq"]`), keeping each group contiguous; the single-column form still works.
* **Scripts**: `scripts/download_gene_sizes.py` builds the bundled gene-size cache; `scripts/demo_exon_size_grouping.py` demonstrates an exon-size grouped oncoplot.

### 🔧 Behaviour Changes
* **Comparable frequency colour scale**: `FreqTrack` now defaults `vmin=0` / `vmax=1`, so every frequency strip (overall and per-group sections) shares one comparable 0–1 scale. Pass `vmin`/`vmax` to override.

### 🐛 Fixes
* Python 3.10 now imports ``Self`` from ``typing_extensions`` so subclass-aware return types are preserved, and uses ``tomli`` as the configuration parser fallback when ``tomllib`` is unavailable.
* HDF5 cohort registries now separate logical table names from physical keys, so empty cohorts and names containing slashes, quotes, or reserved metadata names round-trip safely.
* SQLite compatibility readers and writers close connections deterministically and write atomically; failed writes no longer replace an existing database.
* Missing mutation values are treated as absent, unlisted sample groups are preserved during sorting, and TCGA case-level builders select one deterministic tumor file before matrix construction.
* TCGA expression QC tolerates missing STAR summary rows; mutation builders report an empty tumor selection clearly; GDC, NCBI, and BioMart requests have bounded timeouts.
* Pairwise similarity summaries exclude self/mirrored pairs, permutation p-values cannot be zero, arbitrary group labels receive colors automatically, and invalid independent-pair tests were replaced by label permutation.
* Clustering folds retain sample identity and use stratification only when an explicit metadata group column is requested.
* Stacking probabilities, multiclass omics weights, and shared gene names across layers are handled consistently.
* Expression clustering no longer creates a fake sample column and DESeq2 inputs are validated as non-negative integer counts.
* Compressed MAF headers, gzip VCF input, and multi-allelic VCF depths are parsed correctly.
* **`default_oncoplot` crash** (P0#1): the convenience entry no longer raises `ValueError` from a width_ratios / column-count mismatch.
* **Legend filtering** (P1#4): the mutation legend now lists only categories present in the data and non-wild-type, instead of the whole colormap plus `False`. Restore the full legend with `mutation_heatmap(show_all_categories=True)`.
* **CNV colorbar** (P2#6): a continuous main matrix's colorbar no longer hijacks the freq column — it renders in the legend area.
* Fixed a latent crash in the `plot_bar` value annotation (formatted the whole array).

### ✅ Tests
* Added regression coverage for HDF5/SQLite persistence, TCGA builders, VCF parsing, expression aggregation, pairwise permutation tests, clustering stability, stacking models, and the OncoPlot track API.

### 🚀 Release Process
* Release tags must match `vX.Y.Z` and point to a commit reachable from `main`; the tag workflow re-runs lint, tests, coverage, documentation, build, and metadata checks.
* Wheel and sdist are built once, published to TestPyPI, installed for an HDF5 smoke test, and then promoted unchanged to PyPI through protected environments.
* TestPyPI and PyPI use OIDC Trusted Publishing. Local `deploy.sh` is build-only, and the workflow creates the GitHub Release after PyPI succeeds.

---

## 🔖 Version 0.4.1 (May 2026)

Bug-fix release targeting cases where pymaftools silently produced wrong output
on valid input. All changes are backward compatible.

### 🐛 Fixes
* **`MAF.read_maf` — comment lines**: leading `#` comment lines are now detected and skipped automatically (0, 1, or many). Previously a hardcoded `skiprows=1` caused `ParserError` on files with multiple comment lines and silently dropped the header on files with none.
* **`MAF.read_maf` — sample identity**: `sample_ID` is now optional and defaults to the per-row `Tumor_Sample_Barcode` column, so a standard multi-sample MAF keeps its samples distinct instead of collapsing into one. Passing `sample_ID` explicitly preserves the previous one-file-one-sample behavior. Raises `ValueError` when neither is available. Added `sample_col` parameter.
* **`PivotTable.add_freq` — silent NaN**: now validates that `feature_metadata.index` matches the data index and raises `ValueError` instead of writing an all-NaN frequency column when the indices have drifted.
* **`PivotTable.calculate_TMB`**: removed a stray debug `print`.

---

## 🔖 Version 0.4.0 (March 2026)

### 🆕 New Features
* **Advanced Filtering**: `filter_by_variance()` supports `var` and `mad` methods with quantile or absolute threshold
* **Statistical Filtering**: `filter_by_statistical_test()` with FDR correction (t-test, Mann-Whitney, Kruskal-Wallis, ANOVA)
* **Sphinx Documentation**: Auto-generated API docs from docstrings, deployed to GitHub Pages

### 🔧 Code Quality
* **Type Hints**: ~90% coverage across all modules with `from __future__ import annotations`
* **NumPy Docstrings**: Standardized all docstrings to NumPy style
* **English Comments**: Translated all Chinese comments to English
* **Ruff Linting**: Added ruff to CI, fixed all lint errors (unused imports, ambiguous variables, formatting)
* **Specific Exceptions**: Replaced bare `except:` with specific exception types
* **Explicit Imports**: Replaced wildcard `import *` with explicit imports in `__init__.py`

### 🛠 Enhancements
* **Dependency Management**: All dependencies now have upper bounds to prevent breaking updates
* **CI/CD**: Added coverage threshold (`--cov-fail-under=40`), ruff lint job, docs auto-deploy
* **Build**: Modernized `deploy.sh` to use `python -m build` instead of deprecated `setup.py`

### 📦 Miscellaneous
* Added `RELEASE_CHECKLIST.md` for standardized release process
* Updated `.gitignore` for `.coverage`, `uv.lock`, `.venv`

---

## 🔖 Version 0.3.0 (October 2025)

### 🔥 Breaking Changes
* **Minimum Python Version**: Now requires Python 3.10+ due to modern type hint syntax (union types with `|`)
* **Dependency Management**: Migrated from requirements.txt files to pyproject.toml for unified dependency management

### 🆕 New Features
* **Modern Dependency Management**: All dependencies now managed through pyproject.toml with optional dependency groups
  - `pip install -e .[test]` for test dependencies
  - `pip install -e .[dev]` for development dependencies
* **Enhanced CI/CD**: GitHub Actions now tests Python 3.10, 3.11, 3.12 (stable) and 3.13, 3.14 (experimental)
* **Improved Testing Framework**: Comprehensive test suite with 47/48 tests passing
* **Cohort Class**: Supports merging multiple `PivotTable` sample_metadata, and provides `subset()` and `order()` operations
* **CNV Class**: New `CNV` object for handling GISTIC outputs via `read_gistic()`
* **ASCAT3 Support**: Updated TCGA CNV processing with ASCAT3 compatibility
* **Multi-Omics Stacking Model**: New `OmicsStackingModel` for integrative analysis of multi-omics data
* **Expression Table**: New `ExpressionTable` class with clustering capabilities via `to_cluster_table()`
* **Cancer Cell Fraction Table**: `CancerCellFractionTable` for processing PyClone data
* **PCA/CCA Analysis**: Added PCA_CCA methods for SNV and CNV data analysis
* **Model Plotting**: New `ModelPlot` class for machine learning model visualization
* **SQLite Persistence Support**: Both `Cohort` and `PivotTable` now support `.to_sqlite()` and `.read_sqlite()` for persistent storage
* **PairwiseMatrix Abstract Base**: Shared logic of `CooccurMatrix` and `SimilarityMatrix` refactored into `PairwiseMatrix` base class for extensibility
* **Cytoband File Support**: Added cytoband file handling for genomic visualization

### 🔧 Refactoring

* **License Format**: Updated license specification to proper TOML format
* **Project Structure**: Modernized package structure following PEP 621 standards
* **CI Configuration**: Streamlined GitHub Actions workflow with proper error handling
* **PivotTable `__getitem__` and `subset()` Refactor**: Rewrote the `subset()` method to delegate its core logic to `__getitem__`, with outer-layer metadata alignment handled explicitly via `reindex()`
* **SimilarityMatrix Refactor**: Merged similarity calculation methods into `compute_similarity()` with improved plot configurability

### 🛠 Enhancements

* **Ordering & Grouping Utilities**:
  * `Cohort` adds `order_cohorts()` for custom group sorting
  * `CNV` adds `sort_by_chromosome()` to improve genomic ordering in visualization
* **Enhanced Plot Configurability**: Improved `symmetric`, `fontsize`, and `yticklabel_size` options
* **Advanced Plotting Features**:
  * Enhanced `plot_boxplot_with_annot` with xlabel/ylabel parameters and rotation support
  * Updated `mutation_heatmap` and `numeric_heatmap` with `ytick_fontsize` and `annot_fontsize` parameters
  * SVG and other plot format support improvements
  * Multi-cohort plotting capabilities
* **Font Management**: New `FontManager` class for consistent typography across plots
* **Color Management**: Separate `ColorManager` for advanced colormap handling with method chaining
* **Clustering Functions**: Added comprehensive clustering analysis tools
* **Method Plots**: `MethodsPlot` for creating 3D cohort demonstration plots

### 🐛 Bug Fixes

* Fixed path/format bug in `write_SigProfilerMatrixGenerator_format`
* Resolved import dependency issues for proper module loading

### ⚠️ Deprecations

* **Deprecated** `to_pickle()` and `read_pickle()` in favor of SQLite persistence for better long-term compatibility

### 📦 Miscellaneous

* Added `deploy.sh` script to automate PyPI packaging and publishing
* Added `savefig` and `geneinfo` utility functions
* Updated `README` with instructions and examples

---

## 🔖 Version 0.2.2 (April 2025)

### 🛠️ What's Fixed

* Fixed critical packaging issue where essential modules (`core`, `plot`, `utils`) were not included in the distribution.
* Added missing `__init__.py` files to ensure proper module discovery.
* Included necessary files via `MANIFEST.in` to ensure complete source distribution (`.tar.gz`) and wheel (`.whl`) builds.

### 📦 Installation

```bash
pip install pymaftools==0.2.2
```

### ⚠️ Note

If you previously installed v0.2, v0.2.1 we recommend upgrading to this version due to incomplete packaging in earlier releases.

```bash
pip install --upgrade pymaftools
```

---

## 🔖 Version 0.2.1 (April 2025)

### What's Changed

* Fixed: broken import issue on `pymaftools.core`
* Improved: packaging structure and updated `setup.py`

---

## 🔖 Version 0.2.0 (April 2025)

### 🆕 New Features

* **Optimized Categorical Heatmap**
  Implemented an optimized version of the `categorical_heatmap` function that enables 10x faster heatmap rendering.

* **LollipopPlot Class**
  Added a new `LollipopPlot` class for improved visualization.

* **PCA Method**
  Added a dedicated method for Principal Component Analysis (PCA).

* **Cosine Similarity Method**
  Introduced a method to compute cosine similarity between datasets.

* **Boxplot for Metadata**
  Added a boxplot visualization for metadata.

* **Custom Color Map (cmap) and Alpha**
  Added support for custom colormap (cmap) and alpha values in visualizations.

---

### 🛠 Improvements

* **Readme Updates**
  Updated README files with clearer instructions, feature descriptions, and additional examples.

* **Rename Attributes and Methods**
  Refined naming conventions (e.g., `gene_metadata` to `feature_metadata` and `genes` to `features`).

* **Performance Enhancements**
  Significant improvements to speed, particularly with heatmap rendering and metadata filtering.

* **Method Optimization**
  Optimized several methods, such as the `subset` method using `reindex` and improved `calculate TMB` functionality.

---

### 🐛 Bug Fixes

* **Circular Import Prevention**
  Resolved issues related to circular imports.

* **Feature Metadata Fixes**
  Fixed issues around changes in feature metadata and related attributes.

* **Bug with Sample Metadata**
  Addressed a bug that caused sample metadata to be lost.

---

### 📦 Other Changes

* **Showframe Addition**
  Added the option to show frames in visualizations.

* **PivotTable, MAF, and CooccurMatrix Refactor**
  Moved critical components to the core for better modularity and reusability.

* **Image Saving**
  Introduced new methods like `save_figure` for exporting visualizations.

---
