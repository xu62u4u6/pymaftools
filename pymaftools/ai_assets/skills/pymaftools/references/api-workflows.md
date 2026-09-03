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

For scientific denominators, pass the complete sample universe rather than
deriving it from event-bearing rows:

```python
import pandas as pd
from pymaftools import MAF, ObservationMask, SampleManifest

manifest = SampleManifest(
    pd.DataFrame(
        {"patient_id": ["P1", "P2"], "eligible": [True, True]},
        index=["S1", "S2"],
    )
)
table = maf.to_gene_table(sample_manifest=manifest)
table = table.with_scientific_context(
    observation_mask=ObservationMask.fully_observed(table)
)
frequency = table.calculate_feature_frequency()
```

Use `ObservationMask.fully_observed()` only when the upstream assay and calling
policy justify that assumption. Otherwise construct a feature-by-sample mask
from callability or QC evidence. Present events marked unobserved fail
validation.

## Audited TMB and enrichment

Use `MAF.calculate_tmb_audit()` for scientific TMB. Supply a `SampleManifest`
with positive, sample-specific `callable_mb` and explicitly name available PASS,
somatic-status, variant-classification, and genome-build rules. Inspect both
`audit.summary` and the event-level `audit.events` exclusion ledger. Do not
report `PivotTable.calculate_tmb(default_capture_size=40)` as scientific TMB.

Use `PivotTable.mutation_enrichment_test()` with an explicit group column,
groups, minimum-mutation rule, method, analysis unit, and confidence level.
The tested family receives BH correction. Patient-level analysis requires an
attached manifest and rejects repeated eligible patients; the method does not
adjust for confounding or paired/repeated observations.

For mutation co-occurrence or mutual exclusivity, use the matrix-level
interaction workflow:

```python
stats = table.plot.somatic_interactions_stats(top=25)
figure, stats = table.plot.somatic_interactions(top=25)
```

Each unordered gene pair is tested once and the BH-FDR correction covers only
that unique pair family. Self-pairs and mirrored matrix entries are excluded;
this is an intentional semantic difference from the matrix-cell correction in
`maftools::somaticInteractions`, so adjusted p-values are not expected to be
numerically identical between the two tools.

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
