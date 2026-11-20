# API workflows

## Mutation workflow

```python
from pymaftools import MAF

maf = MAF.read_maf("cohort.maf.gz")  # GDC, cBioPortal, vcf2maf, Funcotator, maftools
print(maf.ingest_report)  # what the reader detected, renamed, filled, dropped
samples = maf["sample_ID"].unique()  # better: the full sequenced-sample list
table = (
    maf.filter_maf(MAF.nonsynonymous_types)
    .to_gene_table(samples=samples)
    .add_freq()
    .sort_features(by="freq")
    .sort_samples_by_mutations()
)
```

`MAF` owns raw mutation events. Convert it to a matrix before matrix-level
filtering, statistics, metadata operations, or oncoplots.

`read_maf()` validates strictly by default: blank `Hugo_Symbol`, duplicate
events, mixed genome builds or `chr1`/`1` naming, and the legacy allele layout
raise `MAFValidationError` with the fix in the message. Fix the input, or pass
`validate="warn"` / `deduplicate=True` deliberately and say so in the analysis
record. `merge_mafs()` rejects mixed builds unless `allow_mixed_builds=True`.

Pass `samples=` (or a `SampleManifest`) to `to_gene_table()` so samples with no
retained events stay in the denominator; filtering before conversion otherwise
drops them silently. Errors and `UserWarning`s from pymaftools name the missing
input and the fixing call; read them rather than working around them.

Raw MAF-level plots live on `maf.plot`: `overview()` (summary header plus sample,
variant, nucleotide and gene panels) and `lollipop(gene)`, which accepts GDC `HGVSp_Short`,
cBioPortal bare positions and VEP `?-12/393` ranges.

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
The tested family receives BH correction; rows that cannot be tested carry a
`not_tested_reason` (`below_minimum_mutations`, `no_observed_samples`,
`degenerate_contingency_table` for chi-squared) and NaN statistics. Patient-level analysis requires an
attached manifest and rejects repeated eligible patients; the method does not
adjust for confounding or paired/repeated observations.

For mutation co-occurrence or mutual exclusivity, use the matrix-level
interaction workflow:

```python
stats = table.plot.somatic_interactions_stats(top=25)
figure, stats = table.plot.somatic_interactions(top=25)
```

The heatmap colours `sign(log OR) × −log10(FDR)` (evidence, not effect size),
so pairs with OR 0/∞ from sparse counts stay pale unless significant; read
`pm.figure_summary(fig)` for `n_significant` and warnings.

Each unordered gene pair is tested once and the BH-FDR correction covers only
that unique pair family. With an attached `ObservationMask`, each pair's 2x2
table uses only jointly observed samples (`jointly_observed`); pairs with none
are `tested=False`, `interaction="Not tested"`, and are left out of the FDR
family (`tested_family_size`). The function warns on fewer than 10 samples and
when most tested pairs are significant co-occurrences, which usually reflects
mutation-burden confounding rather than gene-gene interaction. Self-pairs and mirrored matrix entries are excluded;
this is an intentional semantic difference from the matrix-cell correction in
`maftools::somaticInteractions`, so adjusted p-values are not expected to be
numerically identical between the two tools.

## Machine-readable results and CLI

Serialise results instead of printing them; the JSON keeps provenance
(analysis, parameters, input SHA-256, sample universe, package version):

```python
import pymaftools as pm

stats = table.plot.somatic_interactions_stats(top=25)
pm.to_json(stats, "interactions.json")
pm.get_provenance(stats)  # also on mutation_enrichment_test() results
maf.ingest_report.to_json("ingest.json")
audit.to_json("tmb.json")  # TMBAudit; include_events=True for the ledger
pm.session_info()  # versions for the methods section
```

`NaN` is written as `null` and `±inf` as `"inf"`/`"-inf"`.

Check a figure without looking at it: `plot.save("x.png", summary=True)` writes
`x.json` (or call `plot.summary()`); read its `warnings`, `n_empty_axes`, and for
oncoplots `fill_fraction`, `n_features`, `n_samples`, `top_features`,
`sample_labels_auto_hidden` and `cell_size_px`. `pm.figure_summary(fig)` works on
any matplotlib figure.

Shell / MCP-friendly commands (stdout is one JSON document with `--json`;
exit code 2 means `--validate strict` rejected the input):

Always pass the full list of sequenced samples with `--samples` (a plain list
or a cBioPortal `data_clinical_sample.txt`); otherwise samples without events
are left out of every denominator.

```bash
pymaftools inspect cohort.maf --samples samples.txt --json
pymaftools oncoplot cohort.maf -o oncoplot.png --top 20   # + oncoplot.json
pymaftools interactions cohort.maf --top 25 --json -o stats.csv
pymaftools session-info --json
```

Every JSON record carries `schema_version` (currently `"1.0"`); check it before
parsing. `pymaftools --version` prints the installed version. Use the current
names (`Cohort.to_h5`/`read_h5`, `PivotTable.pca`, `read_gmt`); the old ones
still work until 1.0 but emit `DeprecationWarning`.

## Matrix metadata

`PivotTable` and its specialized subclasses contain:

- the feature-by-sample matrix;
- `feature_metadata`, indexed like the matrix rows;
- `sample_metadata`, indexed like the matrix columns.

Use table methods such as `subset()` when changing samples or features so the
metadata remains aligned. `add_sample_metadata()` / `add_feature_metadata()`
return a new table (assign it: `table = table.add_sample_metadata(clinical)`)
and match rows by ID, so they work before or after sorting.

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
cohort.to_h5("cohort.h5")
```

Set cohort sample metadata before adding omics tables. New high-dimensional
workflows should use HDF5; SQLite compatibility is retained only for legacy
reads during its deprecation window.

## TCGA/GDC inputs

Readers under `pymaftools.io` cover mutation, STAR expression, segment and
gene-level CNV, methylation, and clinical exports. Treat GDC network access as
an integration boundary. Unit tests should use fixtures or recorded/mocked
responses rather than depend on live availability.
