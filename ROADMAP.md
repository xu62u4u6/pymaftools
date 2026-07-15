# pymaftools Roadmap

> Last reviewed: 2026-07-15

This file describes work that is still relevant after 0.5.0. Completed release
details belong in `CHANGELOG.md`; release operations and risks belong in
`RELEASE_CHECKLIST.md` and `RELEASE_AUDIT.md`.

Items under **Next** have a concrete problem and acceptance condition. Items in
the **Backlog** require a real use case or design decision before implementation.

## Current Baseline

| Area | Current state | Near-term target |
| --- | --- | --- |
| Release | 0.5.0 on PyPI | Patch releases use the protected tag workflow |
| Test coverage | 65.18%; CI fails below 60% | Raise coverage in high-risk modules before raising the gate |
| Python | 3.10-3.12 required; 3.13-3.14 experimental | Review experimental failures on every PR |
| Persistence | HDF5 canonical; SQLite deprecated | Preserve SQLite read compatibility during the deprecation period |
| Documentation | Warning-free Sphinx build deployed to GitHub Pages | Add contributor and architecture guidance |
| Largest module | `core/PivotTable.py`, 2,032 lines | Reduce responsibilities with tested extractions |
| Bundled data | `protein_domains.csv`, about 24 MB | Decide whether network-independent lookup justifies the package cost |

The coverage values above come from the 2026-07-15 full local suite. Useful
module-level baselines include `MAF.py` 89%, `StackingModel.py` 76%,
`geneset.py` 100%, and `CopyNumberVariationTable.py` 23%.

## Shipped in 0.5.0

The full list is in `CHANGELOG.md`. The main roadmap milestones delivered were:

- HDF5-first `PivotTable` and `Cohort` persistence with SQLite deprecation.
- Declarative OncoPlot tracks, feature/sample grouping, aligned annotations,
  frequency strips, legends, and colorbars.
- Public VCF parsing and same-sample caller-consensus merge.
- Optional AnnData and expression integrations.
- Multi-omics feature preparation and exon-size annotation.
- MAF/WES overview, Ti/Tv, rainfall, VAF, somatic interaction, cohort
  comparison, and forest plots.
- Python 3.10 compatibility, expanded regression coverage, and a protected
  OIDC release workflow with TestPyPI verification and release checksums.

## Next

### 1. Correctness and Data Model

- [ ] Add `PivotTable.relabel_features(mapper)` so the data index and
      `feature_metadata.index` change atomically; validate metadata alignment
      before plotting.
- [ ] Preserve mutation annotation fields such as `Chromosome`, `Variant_Type`,
      `Variant_Classification`, and `HGVSp_Short` when converting to a mutation
      table. Unify the mutation-key schema and provide a public label formatter.
- [ ] Add feature set/group operations for shared and private events with clear
      missing-value semantics on pandas 2.2+.
- [ ] Define and test the SQLite removal window. Reading existing files must
      remain possible until the documented removal release.

Acceptance for these items requires public API documentation, migration notes
where applicable, and regression tests for metadata alignment and empty input.

### 2. Test Coverage

- [ ] Increase `CopyNumberVariationTable.py` coverage from 23%, prioritizing
      parsing, coordinate validation, and failure paths.
- [ ] Cover low-tested TCGA client/build paths with recorded or mocked responses;
      network availability must not determine unit-test results.
- [ ] Add focused tests for empty tables, invalid metadata, Unicode feature
      names, and persistence corruption.
- [ ] Add end-to-end tests covering `MAF -> PivotTable -> Cohort -> HDF5`;
      extend the existing installed-wheel HDF5 smoke test when another critical
      public workflow needs packaging-level validation.
- [ ] Raise the CI threshold only after the new tests keep the main branch
      comfortably above the proposed gate.

Already well-covered modules should not receive low-value tests solely to chase
a global percentage.

### 3. Architecture and Performance

- [ ] Continue decomposing `PivotTable.py`. Extract PCA/clustering and other
      self-contained analysis logic while keeping `PivotTable` as the public API
      owner.
- [ ] Reassess whether every remaining `PivotTable` method belongs on the core
      data object or on an accessor/helper. A line-count reduction alone is not
      sufficient justification.
- [ ] Profile feature-frequency calculation and large-table copies before adding
      caching or removing defensive `.copy()` calls.
- [ ] Decide how to handle the bundled 24 MB `protein_domains.csv`: keep it for
      offline reproducibility, compress it, or implement a versioned download
      cache with checksum and offline failure behavior.

### 4. Plotting Gaps

- [ ] Support an explicit custom side label for the main y axis. `show_ylabel`
      currently controls visibility but is not a complete labeling API.
- [ ] Finish unknown-category handling: an `Unknown` legend entry exists, but
      warnings and documented case/alias normalization policy are still needed.
- [ ] Add confidence intervals and an explicit pseudo-count policy to the cohort
      forest plot instead of relying only on capped display odds ratios.
- [ ] Add chromosome ticks/boundaries to rainfall plots.
- [ ] Add optional outlier handling and a raw Ti/Tv inset to the MAF overview.

### 5. Documentation and Developer Experience

- [ ] Add `CONTRIBUTING.md` with environment setup, test selection, style, and
      pull-request expectations.
- [ ] Add an architecture page to Sphinx. The README overview image is useful,
      but does not explain module ownership and data flow.
- [ ] Decide whether to add mypy and `py.typed`. Do not enable a non-blocking type
      check that is never brought to green.
- [ ] Add pre-commit only if it mirrors CI exactly and has a documented update
      process.
- [ ] Reduce duplicate Tests runs caused by simultaneous `dev` push and pull
      request events if runner usage becomes material.

## Feature Backlog

These are not committed to a release. Each needs representative user data and a
public API proposal before implementation.

### Export and Interoperability

- [ ] Parquet export with an explicit strategy for sample/feature metadata.
- [ ] Excel export for small review tables, with documented sheet and size limits.
- [ ] Caller adapters for Strelka2, DRAGEN, DeepVariant, or Sentieon when real
      fixtures and expected harmonization behavior are available.

### Analysis

- [ ] Survival analysis with explicit censoring and time-unit validation.
- [ ] `mafbarplot` / top-mutated-gene convenience API.
- [ ] `coBarplot`, `coOncoplot`, and compact `oncostrip` views.
- [ ] Paired, longitudinal, or multi-region VAF comparison.
- [ ] Signature 96 and polished signature heatmap helpers.
- [ ] TMB and multi-metric WES panel helpers.
- [ ] Shared/private mutation visualization using UpSet, Venn, or a table panel.
- [ ] CNV frequency and GISTIC-like plots after the CNV API scope is agreed.

### User Experience

- [ ] A CLI for repeatable, non-interactive conversion and validation tasks.
- [ ] Structured logging for long-running IO; retain progress bars only where a
      measurable loop exists.
- [ ] Optional interactive plots only after a concrete notebook workflow shows
      that static Matplotlib output is insufficient.

Cross-validation utilities are already public through
`cross_validate_importance()` and clustering/model helpers; they are not a
backlog item.

## Candidate API Designs

These sketches record intent, not an implementation commitment.

### Similarity Panel

```python
fig, result = table.plot.similarity_panel(
    method="jaccard",
    group_col="subtype",
    group_order=["LUAD", "ASC", "LUSC"],
    compare_pairs=[("LUAD", "ASC"), ("ASC", "LUSC")],
)
```

- Accept either a `PivotTable` plus a method or a precomputed
  `SimilarityMatrix`.
- Compute similarity once, then render the sample heatmap, group annotation,
  group means, and named statistical comparisons.
- Return the matrix, summary statistics, p-values, and figure in a structured
  result.
- Require an explicit statistical test and deterministic random state where
  applicable.

### Somatic Interaction Network

```python
fig, graph, stats = table.plot.somatic_interaction_network(
    top=25,
    alpha=0.05,
    interaction="both",
    layout="spring",
)
```

- Reuse `somatic_interactions()` results; do not recompute Fisher tests with a
  different implementation.
- Map node size to mutation frequency and edges to FDR-controlled co-occurrence
  or mutual exclusivity.
- Return the NetworkX graph and statistics with the figure.
- Fail clearly when no significant edges remain; an exploratory `show_all`
  mode must be explicit.

## Deferred and Non-Goals

- Production cross-vendor VCF normalization is delegated to tools such as
  `bcftools norm` and annotation pipelines. `pymaftools.VCF` focuses on parsing
  and caller consensus over harmonized records.
- Automatic changelog generation is deferred. Release notes intentionally come
  from the manually reviewed version section in `CHANGELOG.md`.
- Moving all bundled reference data behind a network request is not acceptable
  without a versioned cache, checksums, and a documented offline path.
- Editable-install conflicts and multiple source checkouts are environment
  management issues, not package roadmap features.
- Artifact attestations, versioned documentation, action SHA pinning, and other
  release-hardening options are tracked in `RELEASE_AUDIT.md` with adoption
  triggers.
