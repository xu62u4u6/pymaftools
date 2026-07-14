"""WES/MAF plot helpers expose maftools-like workflows."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pymaftools.core.MAF import MAF
from pymaftools.plot.wes import compare_cohorts, infer_vaf, summarize_titv


def _maf_frame():
    return pd.DataFrame(
        {
            "Hugo_Symbol": ["TP53", "TP53", "KRAS", "EGFR", "PIK3CA", "EGFR"],
            "Tumor_Sample_Barcode": ["S1", "S1", "S2", "S3", "S4", "S4"],
            "sample_ID": ["S1", "S1", "S2", "S3", "S4", "S4"],
            "Variant_Classification": [
                "Missense_Mutation",
                "Nonsense_Mutation",
                "Missense_Mutation",
                "Frame_Shift_Del",
                "Silent",
                "Missense_Mutation",
            ],
            "Variant_Type": ["SNP", "SNP", "SNP", "DEL", "SNP", "SNP"],
            "Chromosome": ["17", "17", "12", "7", "3", "7"],
            "Start_Position": [100, 250, 1200, 2000, 300, 2600],
            "End_Position": [100, 250, 1200, 2000, 300, 2600],
            "Reference_Allele": ["C", "C", "G", "A", "T", "C"],
            "Tumor_Seq_Allele1": ["C", "C", "G", "A", "T", "C"],
            "Tumor_Seq_Allele2": ["T", "A", "A", "-", "C", "T"],
            "t_alt_count": [20, 12, 15, 8, 25, 18],
            "t_depth": [50, 40, 60, 20, 50, 45],
        }
    )


def _maf():
    maf = MAF(_maf_frame())
    maf.index = maf.loc[:, MAF.index_col].apply(
        lambda row: "|".join(row.astype(str)),
        axis=1,
    )
    return maf


def test_maf_plot_accessor_draws_overview_titv_vaf_and_rainfall():
    """MAF.plot covers common raw-MAF WES views without needing a PivotTable."""
    maf = _maf()

    figures = [
        maf.plot.overview(figsize=(12, 8)),
        maf.plot.titv(),
        maf.plot.vaf(),
        maf.plot.rainfall(sample="S1"),
    ]

    assert all(fig.axes for fig in figures)
    for fig in figures:
        plt.close(fig)


def test_maf_plot_summary_accessor_removed():
    """The old raw classification dashboard is no longer a public accessor."""
    assert not hasattr(_maf().plot, "summary")


def test_overview_dashboard_and_primitives_render():
    """The MAF overview + each standalone primitive render without error."""
    maf = _maf()

    overview = maf.plot.overview(figsize=(12, 8))
    # dashboard composes >= 4 panels (burden, composition, snv, top genes) + legend
    assert len(overview.axes) >= 5
    plt.close(overview)

    primitives = [
        maf.plot.sample_burden(),
        maf.plot.mutation_composition(),
        maf.plot.snv_spectrum(),
        maf.plot.gene_recurrence(),
        maf.plot.top_genes(top=3),
    ]
    assert all(fig.axes for fig in primitives)
    for fig in primitives:
        plt.close(fig)


def test_sample_label_auto_hide_threshold():
    """Per-sample x labels auto-hide above the shared limit; force overrides it."""
    from pymaftools.plot import style

    fig, ax = plt.subplots()
    few = [f"S{i}" for i in range(5)]
    many = [f"S{i}" for i in range(style.SAMPLE_LABEL_LIMIT + 1)]

    assert style.apply_sample_xticklabels(ax, few) is True       # auto: few -> show
    assert style.apply_sample_xticklabels(ax, many) is False     # auto: many -> hide
    assert style.apply_sample_xticklabels(ax, many, show=True) is True   # force show
    assert style.apply_sample_xticklabels(ax, few, show=False) is False  # force hide
    plt.close(fig)


def test_summary_stats_reports_cohort_level_numbers():
    """summary_stats counts samples/genes/variants from the raw MAF."""
    stats = _maf().plot.summary_stats()

    assert stats["samples"] == 4  # S1..S4
    assert stats["variants"] == 6
    assert stats["genes_mutated"] == 4  # TP53, KRAS, EGFR, PIK3CA
    assert 0.0 <= stats["missense_fraction"] <= 1.0


def test_titv_summary_uses_six_pyrimidine_context_classes():
    """Ti/Tv summary normalizes reverse-complement SNPs into maftools classes."""
    counts = summarize_titv(_maf())

    assert list(counts.columns) == ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
    assert counts.loc["S2", "C>T"] == 1


def test_vaf_infers_from_alt_count_and_depth():
    """VAF works for standard t_alt_count / t_depth MAF fields."""
    vaf = infer_vaf(_maf())

    assert vaf.iloc[0] == 0.4


def test_lollipop_single_gene_draws_legend_axis():
    """Single-gene lollipop plots should render the collected legends by default."""
    from pymaftools.plot import LollipopPlot

    plot = LollipopPlot(
        protein_name="EGFR",
        protein_length=300,
        domains=[{"Start": 25, "End": 90, "Label": "Domain A"}],
        mutations=[
            {"position": 45, "type": "Missense_Mutation", "count": 2},
            {"position": 120, "type": "Frame_Shift_Del", "count": 1},
        ],
    ).plot()

    assert plot.ax_main is not None
    assert plot.ax_legend is not None
    assert plot.has_legend("Domains")
    assert plot.has_legend("Mutation Types")
    assert len(plot.fig.axes) == 2
    plt.close(plot.fig)


def test_pivot_plot_exposes_somatic_interactions():
    """Somatic interactions belong on the gene x sample PivotTable accessor."""
    table = _maf().to_gene_table()

    fig, stats = table.plot.somatic_interactions(top=4)

    assert {"gene1", "gene2", "odds_ratio", "adjusted_p_value"}.issubset(stats.columns)
    assert fig.axes
    plt.close(fig)


def test_somatic_interactions_displays_upper_triangle_only():
    """The symmetric interaction matrix should not duplicate lower-triangle cells."""
    table = _maf().to_gene_table()

    fig, _stats = table.plot.somatic_interactions(top=4)
    heatmap = fig.axes[0].collections[0]
    mask = heatmap.get_array().mask.reshape((4, 4))

    assert mask[0, 0]
    assert mask[1, 0]
    assert not mask[0, 1]
    plt.close(fig)


def test_compare_cohorts_uses_explicit_cohort_column_names():
    """Cohort comparison output names counts/frequencies by cohort labels."""
    cohort1 = MAF(_maf_frame().loc[lambda df: df["sample_ID"].isin(["S1", "S2"])])
    cohort2 = MAF(_maf_frame().loc[lambda df: df["sample_ID"].isin(["S3", "S4"])])

    result = compare_cohorts(
        cohort1,
        cohort2,
        cohort1_name="LUAD",
        cohort2_name="LUSC",
    )
    fig = cohort1.plot.forest(result, top=3)

    assert {"LUAD_mutated", "LUSC_mutated", "LUAD_freq", "LUSC_freq"}.issubset(
        result.columns
    )
    assert fig.axes
    plt.close(fig)
