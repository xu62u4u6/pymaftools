# 🧬 Patch Note

---

## 🔖 Version 0.2.3 (June 2025)

### 🆕 New Features

* **Cohort Class**
  Supports merging multiple `PivotTable` sample\_metadata, and provides `subset()` and `order()` operations.

* **CNV Class**
  New `CNV` object for handling GISTIC outputs via `read_gistic()`.

* **Ordering & Grouping Utilities**

  * `Cohort` adds `order_cohorts()` for custom group sorting.
  * `CNV` adds `sort_by_chromosome()` to improve genomic ordering in visualization.

* **SQLite Persistence Support**

  * Both `Cohort` and `PivotTable` now support `.to_sqlite()` and `.read_sqlite()` for persistent storage.
  * ⚠️ **Deprecated** `to_pickle()` and `read_pickle()` in favor of better long-term compatibility.

* **PivotTable `__getitem__` and `subset()` Refactor**  
  Rewrote the `subset()` method to delegate its core logic to `__getitem__`,  
  with outer-layer metadata alignment (e.g., `sample_metadata`) handled explicitly via `reindex()`.

* **PairwiseMatrix Abstract Base**
  Shared logic of `CooccurMatrix` and `SimilarityMatrix` refactored into `PairwiseMatrix` base class for extensibility.

---

### 🛠 Improvements

* **SimilarityMatrix Refactor**

  * Merged similarity calculation methods into `compute_similarity()`.
  * Improved plot configurability with `symmetric`, `fontsize`, and `yticklabel_size`.

---

### 🐛 Bug Fixes

* Fixed path/format bug in `write_SigProfilerMatrixGenerator_format`.
* Resolved import dependency issues for proper module loading.

---

### 📦 Miscellaneous

* Added `deploy.sh` script to automate PyPI packaging and publishing.
* Added `savefig` and `geneinfo` utility functions.
* Updated `README` with instructions and examples.

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


