# Architecture and contribution rules

## Data model

Keep the three layers separate:

| Layer | Main objects | Responsibility |
| --- | --- | --- |
| Raw events | `MAF`, `VCF` | Per-variant input before analysis |
| Matrix | `PivotTable` and specialized tables | Feature-by-sample matrix plus synchronized metadata |
| Derived relationships | Similarity, co-occurrence, cohort comparisons | Results computed from a matrix |

Place new behavior in the layer that owns it. Do not turn `MAF` into the
analysis container or bypass table metadata alignment.

## Plotting

Use `.plot` accessors as thin namespaces. Put calculations in table or
module-level functions and route plot styling through `BasePlot`,
`ColorManager`, and `LegendManager`.

For oncoplots, use the declarative track API:

```python
plot = (
    table.plot.oncoplot(figsize=(15, 10))
    .main()
    .add_freq(side="right")
    .add_bar("TMB", side="top")
    .render()
)
plot.save("oncoplot.png", dpi=300)
```

Reuse one statistics result across alternative visualizations. A heatmap and a
network view must not silently compute different tests.

## Change discipline

- Preserve public API compatibility unless the requested change explicitly
  includes a migration.
- Add regression tests with behavior changes.
- Keep refactors separate from feature changes when they are independently
  reviewable.
- Check empty inputs, invalid metadata, persistence round trips, and supported
  Python/pandas versions when relevant.
- For plot work, create a rendered artifact and inspect multiple cohort-size
  regimes when layout or interpretation is scale-dependent.

Read `CLAUDE.md`, `ROADMAP.md`, and `PLOTTING_REVIEW.md` in the repository for
current project decisions before changing source.
