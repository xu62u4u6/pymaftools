# Version 2.0.0 (April 2025)

## New Features:

- **Optimized Categorical Heatmap**: Implemented an optimized version of the `categorical_heatmap` function that enables 10x faster heatmap rendering.
- **LollipopPlot Class**: Added a new `LollipopPlot` class for improved visualization.
- **PCA Method**: Added a dedicated method for Principal Component Analysis (PCA).
- **Cosine Similarity Method**: Introduced a method to compute cosine similarity between datasets.
- **Boxplot for Metadata**: Added a boxplot visualization for metadata.
- **Custom Color Map (cmap) and Alpha**: Added support for custom colormap (cmap) and alpha values in visualizations.

## Improvements:

- **Readme Updates**: Updated README files with clearer instructions, feature descriptions, and additional examples.
- **Rename Attributes and Methods**: Refined naming conventions (e.g., `gene_metadata` to `feature_metadata` and `genes` to `features`).
- **Performance Enhancements**: Significant improvements to speed, particularly with heatmap rendering and metadata filtering.
- **Method Optimization**: Optimized several methods, such as the `subset` method using `reindex` and improved `calculate TMB` functionality.

## Bug Fixes:

- **Circular Import Prevention**: Resolved issues related to circular imports.
- **Feature Metadata Fixes**: Fixed issues around changes in feature metadata and related attributes.
- **Bug with Sample Metadata**: Addressed a bug that caused sample metadata to be lost.

## Other Changes:

- **Showframe Addition**: Added the option to show frames in visualizations.
- **PivotTable, MAF, and CooccurMatrix Refactor**: Moved critical components to the core for better modularity and reusability.
- **Image Saving**: Introduced new methods like `save_figure` for exporting visualizations.
