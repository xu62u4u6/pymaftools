"""WES/MAF plot helpers expose maftools-like workflows."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from statsmodels.stats.multitest import multipletests

from pymaftools.core.MAF import MAF
from pymaftools.plot.wes import (
    compare_cohorts,
    infer_vaf,
    plot_cohort_comparison_forest,
    plot_rainfall,
    somatic_interactions,
    summarize_titv,
)


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


def test_rainfall_defaults_to_canonical_pyrimidine_substitution_colors():
    from pymaftools.plot.ColorManager import ColorManager

    frame = pd.DataFrame(
        {
            "sample_ID": ["S1"] * 3,
            "Chromosome": ["1"] * 3,
            "Start_Position": [10, 20, 30],
            "Variant_Type": ["SNP"] * 3,
            "Reference_Allele": ["C", "G", "A"],
            "Tumor_Seq_Allele2": ["A", "A", "C"],
            "Variant_Classification": ["Missense_Mutation"] * 3,
        }
    )

    fig = plot_rainfall(frame)

    colors = {
        to_hex(collection.get_facecolors()[0]) for collection in fig.axes[0].collections
    }
    assert colors == {
        to_hex(ColorManager.TITV_CMAP["C>T"]),  # reverse complement G>A
        to_hex(ColorManager.TITV_CMAP["T>G"]),  # reverse complement A>C
    }
    legend_text = {text.get_text() for text in fig.axes[1].texts}
    assert {"Substitution", "C>T", "T>G"} <= legend_text
    plt.close(fig)


def test_rainfall_reference_lengths_determine_chromosome_offsets():
    frame = pd.DataFrame(
        {
            "sample_ID": ["S1"] * 4,
            "Chromosome": ["1", "1", "2", "2"],
            "Start_Position": [10, 20, 5, 15],
            "Variant_Type": ["SNP"] * 4,
            "Reference_Allele": ["C"] * 4,
            "Tumor_Seq_Allele2": ["T"] * 4,
        }
    )

    fig = plot_rainfall(frame, chrom_lengths={"chr1": 100, "2": 200})

    x_positions = np.concatenate(
        [collection.get_offsets()[:, 0] for collection in fig.axes[0].collections]
    )
    np.testing.assert_allclose(np.sort(x_positions), [20, 115])
    plt.close(fig)


def test_rainfall_variant_classification_mode_is_preserved():
    frame = pd.DataFrame(
        {
            "sample_ID": ["S1"] * 3,
            "Chromosome": ["1"] * 3,
            "Start_Position": [10, 20, 30],
            "Variant_Classification": [
                "Missense_Mutation",
                "Nonsense_Mutation",
                "Frame_Shift_Del",
            ],
        }
    )

    fig = plot_rainfall(frame, color_by="variant_classification")

    legend_text = {text.get_text() for text in fig.axes[1].texts}
    assert {"Mutation", "Nonsense_Mutation", "Frame_Shift_Del"} <= legend_text
    plt.close(fig)


def test_rainfall_reference_lengths_must_cover_all_plotted_chromosomes():
    frame = pd.DataFrame(
        {
            "sample_ID": ["S1"] * 4,
            "Chromosome": ["1", "1", "2", "2"],
            "Start_Position": [10, 20, 5, 15],
            "Variant_Type": ["SNP"] * 4,
            "Reference_Allele": ["C"] * 4,
            "Tumor_Seq_Allele2": ["T"] * 4,
        }
    )

    with pytest.raises(ValueError, match="missing plotted chromosome.*2"):
        plot_rainfall(frame, chrom_lengths={"1": 100})


def _kataegis_maf_frame() -> pd.DataFrame:
    """Build a synthetic multi-chromosome MAF containing one APOBEC-like
    kataegis focus on chr17 (8 C>T/C>G SNVs within 1.4 kb), one flanking
    distant SNV on chr17 (4.5 kb away that must be trimmed), and sparse
    background SNVs on chr1 and chr17."""
    rows = []
    # Sparse chr1 background (5 SNVs spaced 200 kb apart)
    for idx, pos in enumerate([100_000, 300_000, 500_000, 700_000, 900_000]):
        rows.append(
            {
                "sample_ID": "S1",
                "Hugo_Symbol": f"BG1_{idx}",
                "Chromosome": "chr1",
                "Start_Position": pos,
                "Variant_Type": "SNP",
                "Reference_Allele": "T",
                "Tumor_Seq_Allele2": "C",
                "Variant_Classification": "Missense_Mutation",
            }
        )
    # Dense APOBEC-like kataegis cluster on chr17 (8 SNVs: 7 C>T/G + 1 C>A)
    cluster_positions = [500_000, 500_120, 500_250, 500_410, 500_620, 500_890, 501_100, 501_400]
    alleles = [
        ("C", "T"),
        ("G", "A"),  # C>T reverse complement
        ("C", "G"),
        ("G", "C"),  # C>G reverse complement
        ("C", "T"),
        ("C", "T"),
        ("C", "G"),
        ("C", "A"),
    ]
    for pos, (ref, alt) in zip(cluster_positions, alleles):
        rows.append(
            {
                "sample_ID": "S1",
                "Hugo_Symbol": "TP53",
                "Chromosome": "chr17",
                "Start_Position": pos,
                "Variant_Type": "SNP",
                "Reference_Allele": ref,
                "Tumor_Seq_Allele2": alt,
                "Variant_Classification": "Missense_Mutation",
            }
        )
    # Flanking unrelated SNV 4,500 bp away (must NOT be absorbed into cluster)
    rows.append(
        {
            "sample_ID": "S1",
            "Hugo_Symbol": "FLANK1",
            "Chromosome": "chr17",
            "Start_Position": 505_900,
            "Variant_Type": "SNP",
            "Reference_Allele": "A",
            "Tumor_Seq_Allele2": "G",
            "Variant_Classification": "Silent",
        }
    )
    # Second sample S2 with only sparse variants (no kataegis)
    for pos in [10_000, 250_000, 600_000]:
        rows.append(
            {
                "sample_ID": "S2",
                "Hugo_Symbol": "KRAS",
                "Chromosome": "chr12",
                "Start_Position": pos,
                "Variant_Type": "SNP",
                "Reference_Allele": "C",
                "Tumor_Seq_Allele2": "A",
                "Variant_Classification": "Missense_Mutation",
            }
        )
    return pd.DataFrame(rows)


def test_detect_kataegis_identifies_apobec_cluster_trims_flank_and_records_provenance():
    from pymaftools.plot.wes import detect_kataegis
    from pymaftools.provenance import get_provenance

    maf = MAF(_kataegis_maf_frame())
    clusters = maf.detect_kataegis()

    assert len(clusters) == 1
    row = clusters.iloc[0]
    assert row["cluster_id"] == "K1"
    assert row["sample"] == "S1"
    assert row["chrom"] == "chr17"
    assert row["start"] == 500_000
    assert row["end"] == 501_400  # 505_900 flank was trimmed, not absorbed
    assert row["n_mutations"] == 8
    assert row["span_bp"] == 1_400
    assert row["mean_inter_mutation_distance"] == pytest.approx(200.0)
    assert row["dominant_substitution"] == "C>T"
    assert row["c_to_t_g_fraction"] == pytest.approx(7 / 8)
    assert bool(row["apobec_like"]) is True
    assert row["genes"] == "TP53"
    assert get_provenance(clusters)["analysis"] == "detect_kataegis"

    # Sparse sample returns empty canonical schema with provenance
    empty = detect_kataegis(maf, sample="S2")
    assert empty.empty
    assert list(empty.columns) == list(clusters.columns)
    assert get_provenance(empty)["parameters"]["sample"] == "S2"


def test_detect_kataegis_validates_thresholds_and_sample():
    from pymaftools.plot.wes import detect_kataegis

    maf = MAF(_kataegis_maf_frame())
    with pytest.raises(ValueError, match="min_mutations"):
        detect_kataegis(maf, min_mutations=1)
    with pytest.raises(ValueError, match="max_mean_distance"):
        detect_kataegis(maf, max_mean_distance=-5)
    with pytest.raises(ValueError, match="not found in MAF"):
        detect_kataegis(maf, sample="UNKNOWN_SAMPLE")


def test_rainfall_highlights_kataegis_and_matches_detect_kataegis():
    from pymaftools.plot.wes import detect_kataegis

    maf = MAF(_kataegis_maf_frame())
    fig, clusters = plot_rainfall(maf, sample="S1", return_clusters=True)

    pd.testing.assert_frame_equal(clusters, detect_kataegis(maf, sample="S1"))
    pd.testing.assert_frame_equal(fig.kataegis_clusters, clusters)
    ax = fig.axes[0]
    assert "1 kataegis cluster detected" in ax.get_title()
    assert any("K1: chr17" in t.get_text() and "APOBEC-like" in t.get_text() for t in ax.texts)
    # Collections must remain strictly the mutation scatter series (no extra scatter collections)
    n_scatter_points = sum(len(c.get_offsets()) for c in ax.collections)
    assert n_scatter_points == 12  # 4 chr1 diffs + 8 chr17 diffs
    plt.close(fig)

    # Sample without kataegis reports an honest negative status in title
    fig_none = plot_rainfall(maf, sample="S2")
    assert fig_none.kataegis_clusters.empty
    assert "No kataegis clusters detected" in fig_none.axes[0].get_title()
    plt.close(fig_none)


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

    assert style.apply_sample_xticklabels(ax, few) is True  # auto: few -> show
    assert style.apply_sample_xticklabels(ax, many) is False  # auto: many -> hide
    assert style.apply_sample_xticklabels(ax, many, show=True) is True  # force show
    assert style.apply_sample_xticklabels(ax, few, show=False) is False  # force hide
    plt.close(fig)


def test_summary_stats_reports_cohort_level_numbers():
    """summary_stats counts samples/genes/variants from the raw MAF."""
    stats = _maf().summary_stats()

    assert stats["samples"] == 4  # S1..S4
    assert stats["variants"] == 6
    assert stats["genes_mutated"] == 4  # TP53, KRAS, EGFR, PIK3CA
    assert stats["median_per_sample"] == pytest.approx(1.5)
    assert stats["missense_fraction"] == pytest.approx(0.5)  # 3 / 6
    assert stats["snv_fraction"] == pytest.approx(5 / 6)
    assert stats["titv_ratio"] == pytest.approx(4.0)  # 4 Ti / 1 Tv


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


def test_lollipop_length_mismatch_warns_and_draws_without_domains(tmp_path):
    """Issue #53: a transcript-length mismatch must not discard otherwise
    usable mutation positions."""
    domain_path = tmp_path / "domains.csv"
    pd.DataFrame(
        {
            "HGNC": ["TP53", "TP53"],
            "aa.length": [391, 393],
            "refseq.ID": ["NM_short", "NM_long"],
            "Start": [10, 10],
            "End": [100, 100],
            "Label": ["Domain", "Domain"],
        }
    ).to_csv(domain_path)
    maf = MAF(
        pd.DataFrame(
            {
                "Hugo_Symbol": ["TP53"],
                "sample_ID": ["S1"],
                "Variant_Classification": ["Missense_Mutation"],
                "Protein_position": ["10/392"],
            }
        )
    )

    with pytest.warns(
        UserWarning,
        match=r"requested protein length 392.*391, 393.*without domains",
    ):
        plot = maf.plot.lollipop("TP53", protein_domains_path=domain_path)

    assert plot.protein_length == 392
    assert plot.domains == []
    plot.plot()
    assert plot.ax_main is not None
    assert plot.ax_main.lines or plot.ax_main.patches
    plt.close(plot.fig)


def test_resolve_lollipop_domains_handles_custom_csv_missing_columns(tmp_path):
    """Fallback length lookup must never mask a missing-domain warning with KeyError."""
    from pymaftools.plot.LollipopPlot import resolve_lollipop_domains

    bad_csv = tmp_path / "custom_domains.csv"
    pd.DataFrame({"gene": ["TP53"], "length": [393]}).to_csv(bad_csv)

    class _StubMaf:
        def get_domain_info(self, gene, aa_length, path=None):
            raise ValueError(
                f"No domain info found for gene {gene} with AA length {aa_length}"
            )

    with pytest.warns(UserWarning, match=r"missing column\(s\).*aa\.length"):
        domains = resolve_lollipop_domains(
            _StubMaf(), "TP53", 392, protein_domains_path=bad_csv
        )

    assert domains == []


def test_multi_cohort_lollipop_saves_extensionless_path_in_dotted_directory(
    tmp_path,
):
    from pymaftools.plot import LollipopPlot

    output_dir = tmp_path / "qc.figures"
    output_dir.mkdir()
    output_path = output_dir / "tp53_lollipop"
    cohorts = {
        "cohort": (
            100,
            [{"position": 20, "type": "Missense_Mutation", "count": 1}],
            [],
            "NM_test",
        )
    }

    plot = LollipopPlot.plot_multi_cohort(
        "TP53",
        cohorts,
        figsize=(6, 3),
        save_path=output_path,
        dpi=72,
    )

    assert output_path.with_suffix(".png").is_file()
    plt.close(plot.fig)


def test_somatic_interactions_displays_upper_triangle_only():
    """Somatic interactions return required columns and plot upper triangle only."""
    table = _maf().to_gene_table()

    plotter = table.plot
    fig, stats = plotter.somatic_interactions(top=4)
    assert plotter.fig is fig
    assert {"gene1", "gene2", "odds_ratio", "adjusted_p_value"}.issubset(stats.columns)
    heatmap = fig.axes[0].collections[0]
    mask = heatmap.get_array().mask.reshape((4, 4))

    assert mask[0, 0]
    assert mask[1, 0]
    assert not mask[0, 1]
    plt.close(fig)


def test_somatic_interactions_corrects_unique_unordered_pair_family():
    """FDR uses one hypothesis per gene pair, without diagonal cells."""
    stats = somatic_interactions(_maf().to_gene_table(), top=4)

    pairs = [frozenset((row.gene1, row.gene2)) for row in stats.itertuples()]
    assert len(stats) == 6  # 4 * 3 / 2, not a 4 x 4 matrix family
    assert len(set(pairs)) == 6
    assert all(row.gene1 != row.gene2 for row in stats.itertuples())

    expected = multipletests(stats["p_value"], method="fdr_bh")[1]
    assert (stats["adjusted_p_value"].to_numpy() == expected).all()


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
    by_gene = result.set_index("gene")
    # TP53 has two rows in S1; compare_cohorts must deduplicate to 1 mutated sample
    assert by_gene.loc["TP53", "LUAD_mutated"] == 1
    assert by_gene.loc["TP53", "LUSC_mutated"] == 0
    assert by_gene.loc["TP53", "LUAD_freq"] == pytest.approx(0.5)
    assert by_gene.loc["TP53", "LUSC_freq"] == pytest.approx(0.0)
    assert by_gene.loc["EGFR", "LUAD_mutated"] == 0
    assert by_gene.loc["EGFR", "LUSC_mutated"] == 2
    assert by_gene.loc["EGFR", "LUSC_freq"] == pytest.approx(1.0)
    assert fig.axes
    assert any(ax.collections or ax.lines or ax.patches for ax in fig.axes)
    plt.close(fig)


def test_forest_plot_rejects_non_comparison_input_with_actionable_hint():
    with pytest.raises(ValueError) as exc_info:
        plot_cohort_comparison_forest(pd.DataFrame())

    message = str(exc_info.value)
    assert all(
        column in message for column in ("gene", "odds_ratio", "adjusted_p_value")
    )
    assert "maf.compare_cohorts(other)" in message


def test_compare_cohorts_applies_bh_fdr_and_attaches_provenance():
    """compare_cohorts computes exact Benjamini-Hochberg FDR and attaches provenance."""
    from pymaftools.provenance import get_provenance

    def _cohort(prefix, n_samples, gene_counts):
        rows = []
        # Ensure every sample appears in the MAF so cohort denominator is exact.
        for i in range(n_samples):
            rows.append(
                {
                    "Hugo_Symbol": "BASELINE",
                    "Tumor_Sample_Barcode": f"{prefix}{i}",
                    "sample_ID": f"{prefix}{i}",
                    "Variant_Classification": "Missense_Mutation",
                }
            )
        for gene, count in gene_counts.items():
            for i in range(count):
                rows.append(
                    {
                        "Hugo_Symbol": gene,
                        "Tumor_Sample_Barcode": f"{prefix}{i}",
                        "sample_ID": f"{prefix}{i}",
                        "Variant_Classification": "Missense_Mutation",
                    }
                )
        return MAF(pd.DataFrame(rows))

    cohort1 = _cohort("A", 20, {"G1": 16, "G2": 10, "G3": 6, "G4": 4, "G5": 2})
    cohort2 = _cohort("B", 20, {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 1})

    result = cohort1.compare_cohorts(cohort2, cohort1_name="LUAD", cohort2_name="LUSC")
    expected_fdr = multipletests(result["p_value"], method="fdr_bh")[1]

    # Raw and BH-adjusted p-values must genuinely differ so the test cannot pass
    # if adjustment is skipped or replaced with raw p-values.
    assert not (
        result["adjusted_p_value"].to_numpy() == result["p_value"].to_numpy()
    ).all()
    assert result["adjusted_p_value"].to_numpy() == pytest.approx(expected_fdr)
    assert result["adjusted_p_value"].is_monotonic_increasing

    prov = get_provenance(result)
    assert prov is not None
    assert prov["analysis"] == "compare_cohorts"
    assert prov["parameters"] == {
        "cohort1_name": "LUAD",
        "cohort2_name": "LUSC",
        "min_mutated": 1,
    }
    assert prov["sample_universe"]["n_samples"] == 40
