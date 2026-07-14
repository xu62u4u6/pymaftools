# pymaftools — Engineering Guide

This package is **not** "a bag of functions lifted from analysis notebooks." It is
a layered genomics-analysis library. Read this before adding features — it
encodes decisions that are expensive to re-derive.

## 1. The mental model: three layers (the core insight)

The core of pymaftools is **not MAF** — it's "matrix + metadata". Everything is
organised into three layers; **do not mix them**:

| Layer | Objects | Role |
|-------|---------|------|
| **Raw event** | `MAF`, `VCF` | per-mutation event tables (raw, pre-analysis) |
| **Matrix** | `PivotTable`, `SmallVariationTable`, `CopyNumberVariationTable`, `ExpressionTable`, `SignatureTable` | feature × sample matrix **+ feature_metadata + sample_metadata** — the analysis subject |
| **Derived relationship** | `SimilarityMatrix`, `CooccurrenceMatrix`, cohort comparison, interaction stats | relationships *derived* from a matrix |

`MAF` is a raw event table; `PivotTable` is the analysis subject. New features
should land in the layer they belong to, and the **plot API grows along these
three layers** (see §2). Keeping the layers clean is what stops the package
becoming "一鍋粥".

## 2. Plot API: layered `.plot` accessors

The accessor pattern is the right abstraction — it makes this a *library*, not a
grab-bag of `pymaftools.plot_xxx(df)` free functions:

- `maf.plot.summary()` / `maf.plot.lollipop("TP53")` — raw MAF-level
- `pivot_table.plot.somatic_interactions()` / `.oncoplot()` — matrix-level
- (future) `similarity_matrix.plot.panel()` — relationship-level

Think **toolbox drawers**: MAF tools in the MAF drawer, matrix tools in the
matrix drawer. Conventions:

- **Accessor = thin namespace.** Methods delegate to module-level functions
  (e.g. `plot/wes.py`) or table methods; they don't hold the implementation.
- **Route every plot through `BasePlot`** (→ `ColorManager` + `LegendManager`).
  Bypassing it is how colours and legends drift. `MafPlot` historically did not
  inherit `BasePlot` and hard-coded colours — that was the bug, not a pattern to
  copy.
- **`ColorManager` is the single colour source** (`ALL_MUTATION_CMAP`,
  `TITV_CMAP`, `CNV_CMAP`, ...); `plot/style.py` is the house look
  (`style_axes`, accent colours, `SEQUENTIAL_CMAP`/`DIVERGING_CMAP`). Pull from
  these — never hard-code hex values per plot.

## 3. Stats and plot are separate

Compute once, visualise many times, from **one** stats source:

- `somatic_interactions()` produces the stats DataFrame.
- `plot_somatic_interactions()` only *draws* that DataFrame; a future network
  plot must reuse the **same** stats.

Otherwise a heatmap and a network can silently report two different conclusions.

## 4. A plot is not "done" when it renders

Bioinformatics plots fail **silently** — they look fine but mean the wrong
thing. Acceptance for any plot feature = all three:

1. **smoke test** (runs without error),
2. **an actual rendered artifact** (saved to `outputs/`, viewable by the user —
   not `/tmp`), and
3. **human eye check**.

Real defects this caught: legend overlap, an unreadable interaction colour
scale, a forest plot silently dropping infinite odds ratios.

**Test plots across cohort-size regimes.** Plot bugs are *scale-dependent*: what
looks fine on the bundled 6-sample MAF breaks at 1000 (label overflow, legend
crowding, auto-hide thresholds, performance), and distributions degenerate at
the low end (a recurrence curve or Ti/Tv is meaningless with <10 samples). So a
plot is only "verified" once checked at several sizes, roughly:

| regime | samples | what it stresses |
|--------|---------|------------------|
| few | < ~10 | degenerate distributions; does it still render/say something honest? |
| small | ~10–100 | the common case |
| regular | ~100–300 | label density, legend fit |
| many | ~300+ | label auto-hide, overflow, render time |

(Buckets are approximate — adjust to the real data.) The bundled example MAF
covers only the *few* regime; use a real cohort (e.g. an external WES MAF) for
the larger regimes during visual QC.

## 5. Naming quality = maintainability

`s1`/`s2`/`n1`/`m1` is fine in a notebook, **not** in package code. Use domain
names — `cohort1_sample_count`, `cohort1_gene_counts`, `both_mutated` — because
in six months the reader is the next maintainer, not the author.

## 6. VCF scope boundary

VCF must not become a black hole. Cross-caller / cross-vendor VCF merging is a
whole pipeline; **delegate** normalization, left-alignment, multi-allelic split,
and reference checks to `bcftools` / `GATK` / `vcf2maf`. pymaftools does:

- harmonized VCF table, caller adapters, consensus summary,
- and MAF / PivotTable downstream analysis.

It is **not** a full variant-normalization engine — don't pretend otherwise.

## 7. Porting from real projects (e.g. ASC_0217): grammar, not deliverables

Do **not** copy report figures into the package. Port the **reusable plot
grammar**: similarity panel, interaction network, shared/private mutations,
multi-metric panel. Move "reusable syntax", not "one project's finished output".

## 8. Engineering assets

`ROADMAP.md` and `PLOTTING_REVIEW.md` are required, not optional. Work spans API
/ naming / plot quality / VCF scope / port candidates simultaneously; if
decisions aren't written down, the next round re-asks, re-thinks, and re-breaks
them. Record decisions there.

## 9. Recurring gotchas (learned the hard way)

- **`PivotTable.plot` / `MAF.plot` shadow pandas' `.plot`.** Calling
  `pivot.plot(kind="bar")` returns the *accessor object*, not a pandas plot.
  Wrap in a plain `pd.DataFrame(...)` before using pandas plotting.
- **napoleon `Attributes` docstring section + autodoc'd class attribute =
  duplicate-object Sphinx warning.** Drop the `Attributes` section; use inline
  attribute docstrings (`X = ...` followed by `"""..."""`).
- **Versioning: `setuptools-scm` (version from the git tag).** Its file-finder
  adds *every* VCS-tracked file to the sdist, so `MANIFEST.in` must `prune`
  heavy/local dirs (`img`, `.claude`, ...) and CI checkout needs
  `fetch-depth: 0`. Tag `vX.Y.Z` → version `X.Y.Z`; never hard-code a version.
- **Docs drift from bundled data.** Example snippets must be calibrated to the
  dataset they claim to run on; the bundled example MAF lacks `subtype` /
  `pathway` and the genes `TP53`/`KRAS`/`EGFR`. Re-run the `doc-only-user` agent
  after API changes.

## Dev basics

- Run things with `uv run` (env-isolated); tests: `uv run pytest -q` (integration
  tests are deselected by default via `pytest.ini`).
- The package decomposition is ongoing: `PivotTable` logic is being split into
  `pivot_io` / `pivot_filtering` / `pivot_frequency` / `pivot_sorting` /
  `pivot_stats`. Follow that modularity for new core logic.
