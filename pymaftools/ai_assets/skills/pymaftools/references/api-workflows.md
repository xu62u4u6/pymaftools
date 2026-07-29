# API workflows

## Mutation workflow

```python
from pymaftools import MAF

maf = MAF.read_maf("cohort.maf")
table = (
    maf.filter_maf(MAF.nonsynonymous_types)
    .to_gene_table()
    .add_freq()
    .sort_features(by="freq")
    .sort_samples_by_mutations()
)
```

`MAF` owns raw mutation events. Convert it to a matrix before matrix-level
filtering, statistics, metadata operations, or oncoplots.

## Matrix metadata

`PivotTable` and its specialized subclasses contain:

- the feature-by-sample matrix;
- `feature_metadata`, indexed like the matrix rows;
- `sample_metadata`, indexed like the matrix columns.

Use table methods such as `subset()` when changing samples or features so the
metadata remains aligned.

## Oncoplot

Register tracks, render once, then save:

```python
plot = (
    table.plot.oncoplot(figsize=(18, 10))
    .main()
    .add_freq(side="right")
    .add_sample_annotation(["subtype", "sex"], side="bottom")
    .render()
)
plot.save("mutation-landscape.png", dpi=300)
```

Use `.main(kind="cnv")` for continuous CNV matrices.

## Cohort and persistence

```python
from pymaftools import Cohort

cohort = Cohort(name="example")
cohort.add_sample_metadata(clinical)
cohort.add_table(mutation_table, "mutations")
cohort.add_table(expression_table, "expression")
cohort.to_hdf5("cohort.h5")
```

Set cohort sample metadata before adding omics tables. New high-dimensional
workflows should use HDF5; SQLite compatibility is retained only for legacy
reads during its deprecation window.

## TCGA/GDC inputs

Readers under `pymaftools.io` cover mutation, STAR expression, segment and
gene-level CNV, methylation, and clinical exports. Treat GDC network access as
an integration boundary. Unit tests should use fixtures or recorded/mocked
responses rather than depend on live availability.
