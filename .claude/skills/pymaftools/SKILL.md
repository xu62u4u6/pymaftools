---
name: pymaftools
description: Use when writing code that imports pymaftools, or when the user asks about MAF files, oncoplots, mutation analysis, multi-omics integration, copy number variation, gene expression analysis, lollipop plots, or any bioinformatics genomic analysis task.
---

# pymaftools — Python MAF Analysis Toolkit

You are an expert in using the `pymaftools` package for genomic and multi-omics analysis. The source code is at `./pymaftools/pymaftools/`. Always refer to the actual source when uncertain about API details.

## Package Overview

pymaftools provides tools for loading, analyzing, and visualizing Mutation Annotation Format (MAF) files and multi-omics cancer genomics data.

## Core Classes & Usage

### MAF — Load & Filter Mutation Files

```python
from pymaftools import MAF

maf = MAF.read_maf("path/to/file.maf", sample_id="SampleA")
# Filter to nonsynonymous mutations
filtered = maf.filter_maf(filter_type="nonsynonymous")
# Convert to PivotTable (gene x sample matrix)
pt = maf.to_pivot_table()
# Merge multiple MAFs
merged = MAF.merge_mafs([maf1, maf2])
# Protein info for lollipop plots
AA_length, mutations_data = maf.get_protein_info("EGFR")
domains_data, refseq_ID = MAF.get_domain_info("EGFR", AA_length)
```

### PivotTable — Core Analysis Data Structure

A gene/feature x sample matrix with synchronized metadata.

```python
from pymaftools import PivotTable

pt = maf.to_pivot_table()
pt = pt.add_freq()                          # Add mutation frequency to feature_metadata
pt = pt.filter_by_freq(min_freq=0.05)       # Keep genes mutated in >=5% samples
pt = pt.calculate_TMB()                     # Add TMB to sample_metadata
similarity = pt.compute_similarity(method="jaccard")

# Sorting
pt = pt.sort_features(by="freq")
pt = pt.sort_samples_by_mutations()
pt = pt.sort_samples_by_group(group_col="subtype", group_order=["A", "B"], top=10)

# Subsetting
pt_sub = pt.subset(features=["TP53", "KRAS"], samples=sample_list)

# Persistence
pt.to_sqlite("data.db")
pt = PivotTable.read_sqlite("data.db")

# Visualization (lazy-loaded accessor)
pt.plot.plot_pca_samples(group_col="subtype")
pt.plot.plot_boxplot_with_annot(group_col="subtype", value_col="TMB")
pt.plot.plot_heatmap()
```

**Key attributes:**
- `pt.feature_metadata` — DataFrame indexed by features (genes)
- `pt.sample_metadata` — DataFrame indexed by samples

### PivotTable — Advanced Filtering

```python
# Filter by variance (keep top 25% most variable features)
pt = pt.filter_by_variance(quantile=0.75, method="var")   # method: "var" or "mad"
pt = pt.filter_by_variance(threshold=0.5, method="mad")   # absolute threshold

# Filter by statistical test with FDR correction
pt = pt.filter_by_statistical_test(
    group_col="subtype",
    method="kruskal",   # "ttest", "mann_whitney", "kruskal", "anova"
    alpha=0.05
)
# Results in feature_metadata: "p_value", "adjusted_p_value"

# Group frequencies
pt = pt.add_freq(
    groups={"LUAD": pt.subset(samples=pt.sample_metadata.subtype == "LUAD"),
            "LUSC": pt.subset(samples=pt.sample_metadata.subtype == "LUSC")}
)
```

### Cohort — Multi-Omics Container

```python
from pymaftools import Cohort

cohort = Cohort(name="TCGA-Lung", description="...")
# Clinical sets sample_IDs; subsequent add_table auto-subsets to these IDs
cohort.add_sample_metadata(clinical_df)
cohort.add_table(mutation_pt, "mutations")   # auto calls table.subset(samples=cohort.sample_IDs)
cohort.add_table(cnv_table, "cnv")
cohort.add_table(expr_table, "expression")

# Access tables
cohort.tables["mutations"]

# Subset entire cohort
sub = cohort.subset(samples=["TCGA-05-4244", "TCGA-05-4249"])

# Persistence — HDF5 recommended (no column limit, preserves all metadata)
cohort.to_hdf5("cohort.h5")
cohort = Cohort.read_hdf5("cohort.h5")
# HDF5 stores per table: data (matrix), feature_metadata, sample_metadata

# SQLite (deprecated, ~2000 column limit)
cohort.to_sqlite("cohort.db")
```

**Key behavior:** `add_table()` automatically subsets the table to `cohort.sample_IDs`. Set sample_IDs first via `add_sample_metadata()`, then add tables without pre-filtering.

### Specialized Table Types

```python
from pymaftools import (
    CopyNumberVariationTable, ExpressionTable, SignatureTable,
    CancerCellFractionTable, SmallVariationTable
)

# GISTIC results
cnv = CopyNumberVariationTable.read_gistic_arm_level("arm_level.txt")
cnv = CopyNumberVariationTable.read_gistic_gene_level("gene_level.txt")

# Expression data
expr = ExpressionTable(expression_df)
cluster_expr = expr.to_cluster_table()

# COSMIC signatures
sig = SignatureTable.read_signature("signature_file.txt")

# Cancer cell fraction (PyClone output)
ccf_table = CancerCellFractionTable.pyclone_to_sorted_table("pyclone_results.tsv")

# SmallVariationTable — PivotTable subclass for SNV/INDEL data
svt = SmallVariationTable(snv_data)
```

### Pairwise Analysis

```python
from pymaftools import SimilarityMatrix

sim = pt.compute_similarity(method="jaccard")  # Returns SimilarityMatrix
sim.get_mean_group_similarity(group_series)
sim.calculate_group_similarity_pvalues(group_series, n_permutations=1000)
sim.plot_group_heatmap(group_series)
```

## Visualization

### OncoPlot — Mutation Landscape

OncoPlot takes a PivotTable in the constructor and uses method chaining.

```python
from pymaftools import OncoPlot, ColorManager

oncoplot = (OncoPlot(pt)
    .set_config(figsize=(15, 10),
                width_ratios=[20, 2, 2],           # heatmap, freq bar, legend
                categorical_columns=["subtype"])    # optional metadata columns
    .mutation_heatmap()                             # categorical mutation heatmap
    .plot_freq()                                    # frequency bar chart
    .plot_bar()                                     # sample bar chart
    .plot_categorical_metadata(cmap_dict=cmap_dict) # sample metadata rows
    .plot_all_legends()
    .save("oncoplot.png", dpi=300)
)

# Numeric heatmap (e.g., CNV data)
oncoplot = (OncoPlot(cnv_table)
    .set_config(figsize=(30, 10), width_ratios=[25, 1, 0, 3])
    .numeric_heatmap(cmap="coolwarm", vmin=-2, vmax=2)
    .plot_bar()
    .plot_categorical_metadata(cmap_dict=cmap_dict)
    .plot_all_legends()
    .save("cnv_oncoplot.tiff", dpi=600)
)
```

### LollipopPlot — Protein Mutations

```python
from pymaftools import LollipopPlot

AA_length, mutations_data = maf.get_protein_info("TP53")
domains_data, refseq_ID = MAF.get_domain_info("TP53", AA_length)

plot = LollipopPlot(
    protein_name="TP53",
    protein_length=AA_length,
    domains=domains_data,
    mutations=mutations_data
)
plot.plot()
```

### Color & Font Management

```python
from pymaftools import ColorManager, FontManager

cm = ColorManager()
cm.get_cmap("nonsynonymous")   # Mutation type colors
cm.get_cmap("cnv")             # CNV colors
cm.register_cmap("custom", {"A": "red", "B": "blue"})

fm = FontManager()
fm.setup_matplotlib_fonts(family="Arial", size=10)
```

### ModelPlot & MethodsPlot

```python
from pymaftools import ModelPlot, MethodsPlot

# ModelPlot — model performance visualizations (inherits BasePlot)
model_plot = ModelPlot()

# MethodsPlot — 3D cohort demonstration plots
methods_plot = MethodsPlot()
```

## Machine Learning

### Stacking Model for Multi-Omics

```python
from pymaftools import OmicsStackingModel
from pymaftools.model.modelUtils import (
    evaluate_model, cross_validate_importance, get_importance,
    to_importance_table, plot_top_feature_importance_heatmap,
    run_rfecv_feature_selection, run_model_evaluation,
    plot_metric_comparison_with_annotation
)

model = OmicsStackingModel()
model.fit(cohort, labels)
preds = model.predict(cohort)
proba = model.predict_proba(cohort)
importance = model.get_omics_feature_importance()
weights = model.get_omics_weights()

# Evaluation
metrics = evaluate_model(model, X_test, y_test)  # Returns Accuracy, F1, AUC
results = cross_validate_importance(model, X, y, n_seeds=10)

# Feature selection
selected = run_rfecv_feature_selection(model, X, y)

# Importance visualization
plot_top_feature_importance_heatmap(importance_table, top_n=20)
```

## TCGA Data Readers (`pymaftools.io`)

Read GDC-downloaded raw files into PivotTable objects. Each reader scans
gdc-client UUID directory structure and resolves file_uuid → case_id via GDC API.

```python
from pymaftools.io import (
    read_star_counts,         # → ExpressionTable (gene × sample)
    read_maf_files,           # → MAF (flat DataFrame, use .to_pivot_table() for matrix)
    read_seg_files,           # → CopyNumberVariationTable (segment-level)
    read_gene_level_cnv,      # → CopyNumberVariationTable (gene × sample, ASCAT3)
    read_methylation_betas,   # → PivotTable (probe × sample)
    GDCClient,                # API client for queries
)
from pymaftools.io.tcga_readers import read_clinical  # → DataFrame (bcr_patient_barcode index)

# All readers take (data_dir, manifest_path)
# data_dir: gdc-client download dir with uuid subdirs
# manifest_path: GDC manifest TSV (id, filename, md5, size, state)
expr = read_star_counts("data/raw/expression", "data/manifests/manifest_expression.tsv")
cnv = read_gene_level_cnv("data/raw/cnv_gene", "data/manifests/manifest_cnv_gene.tsv",
                           value_column="copy_number")  # or min_copy_number, max_copy_number

# Clinical from local BCR TXT files
clin = read_clinical("data/raw/clinical", file_type="patient")  # patient, drug, radiation, follow_up, nte

# Clinical from GDC API (alternative)
client = GDCClient()
clin = client.fetch_clinical_table(case_ids)
```

**GDC data types available:**
- `expression`: Gene Expression Quantification (STAR - Counts)
- `mutation`: Masked Somatic Mutation (MAF)
- `cnv`: Masked Copy Number Segment
- `cnv_gene`: Gene Level Copy Number (ASCAT3)
- `methylation`: Methylation Beta Value
- `clinical`: Clinical Supplement (BCR XML/TXT)

## Utilities

```python
from pymaftools import read_GMT, fetch_msigdb_geneset, PCA_CCA
from pymaftools.utils.geneinfo import get_ncbi_gene_IDs, parse_gene_info

# Gene sets
gmt = read_GMT("pathways.gmt")
geneset = fetch_msigdb_geneset("HALLMARK_TP53_PATHWAY")

# Gene info from NCBI
gene_ids = get_ncbi_gene_IDs(["TP53", "KRAS", "BRCA1"])

# Dimensionality reduction
pca_cca = PCA_CCA()
```

## Important Notes

- PivotTable extends pandas DataFrame — all pandas operations work
- Always call `add_freq()` before `filter_by_freq()` or `sort_features(by="freq")`
- OncoPlot takes PivotTable in constructor and uses method chaining — most methods return `self`
- `filter_by_variance` and `filter_by_statistical_test` add results to `feature_metadata`
- When unsure about a method's signature, read the source at `./pymaftools/pymaftools/`
- Use `.venv` with `uv` for package management
