# 🧬 Patch Note

---

## 🔖 Version 0.5.0 (June 2026)

Track-based redesign of the plotting module (`OncoPlot`). The visualization is now
composed of declarative **tracks** (a main matrix plus annotation strips), laid out
by a single `render()` call. See `PLOTTING_REVIEW.md` for the full design.

### ⚠️ Breaking Changes
* **`render()` is now required.** The eager immediate-draw layout (`update_layout` and the fixed named axes) was removed. The convenience methods (`mutation_heatmap`, `numeric_heatmap`, `plot_freq`, `plot_bar`, `plot_categorical_metadata`, `plot_numeric_metadata`) now only *register* tracks; call `.render()` once before `save()` / `add_xticklabel()`. Migration: add `.render()` to existing chains.
* **`PivotTablePlot` → `PivotStatsPlot`.** The class was renamed (it provides one-shot statistical plots — PCA, annotated boxplots — not the base for all pivot-table plotting). The old import path `from pymaftools.plot.PivotTablePlot import PivotTablePlot` still works via a shim.
* **Removed** `OncoPlot.heatmap_rectangle` and `OncoPlot.plot_color_heatmap` (orphaned; relied on the eager axes).
* **`OncoPlot.default_oncoplot`**: dropped the `width_ratios` argument; it now registers tracks and renders.

### 🆕 New Features
* **Feature-side annotations** (`add_feature_annotation`): `feature_metadata` columns (e.g. `pathway`) can finally be drawn as row-side strips — impossible under the old layout.
* **Declarative track API**: `main()`, `add_bar()`, `add_freq()`, `add_sample_annotation()`, `add_feature_annotation()` register tracks; `render()` derives the GridSpec from each track's `side`/`size`. Numeric vs categorical is inferred from column dtype.
* **Axis grouping** (`group_features`, `group_samples`): partition rows (by a `feature_metadata` column, e.g. `pathway`) and/or columns (by a `sample_metadata` column, e.g. `subtype`) into labelled sections with separator lines drawn across the matrix and aligned tracks, plus group titles on the left (features) and top (samples).
* **Unified entry point**: oncoplots are reachable from the same accessor as the stats plots — `table.plot.oncoplot(...)`.
* **`main(kind="cnv")`**: continuous (CNV) main matrices go through `render()` too.
* **Numeric metadata colorbars** (`plot_numeric_metadata`, P1#3): numeric strips carry a colorbar so the value scale is interpretable. Placement is a mode — `colorbar="legend"` (default; stacked in the legend area, readable with several numeric columns), `"inset"` (small bar beside the strip), or `"off"`. Per-column cmaps via `cmap_dict=`.
* **`render()` layout knobs**: `main_width`, `main_height`, `legend_width`, `legend_pad` (a named spacer replacing the old phantom column), `colorbar_width`, `wspace`, `hspace`.
* **`add_xticklabel(fontsize=, rotation=)`**: was hardcoded to `rotation=90`.

### 🐛 Fixes
* **`default_oncoplot` crash** (P0#1): the convenience entry no longer raises `ValueError` from a width_ratios / column-count mismatch.
* **Legend filtering** (P1#4): the mutation legend now lists only categories present in the data and non-wild-type, instead of the whole colormap plus `False`. Restore the full legend with `mutation_heatmap(show_all_categories=True)`.
* **CNV colorbar** (P2#6): a continuous main matrix's colorbar no longer hijacks the freq column — it renders in the legend area.
* Fixed a latent crash in the `plot_bar` value annotation (formatted the whole array).

### ✅ Tests
* First test coverage for `OncoPlot` (was previously untested): 23 smoke tests covering the track API, render layout derivation, legend filtering, and the convenience wrappers.

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


