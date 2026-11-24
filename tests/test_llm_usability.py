"""Fail-loud and hint behaviour (0.6.0 "LLM usability" pass).

Each test pins a spot where a call used to succeed silently, crash with an
unhelpful pandas error, or hide a likely misinterpretation. None of these
changes may alter computed statistics; those are covered elsewhere.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pymaftools import MAF, load_example_maf, load_example_table
from pymaftools.core.SmallVariationTable import SmallVariationTable
from pymaftools.plot.Track import _fit_linewidths


@pytest.fixture
def table():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maf = load_example_maf()
    return maf.filter_maf(MAF.nonsynonymous_types).to_gene_table()


# --- metadata --------------------------------------------------------------


def test_add_sample_metadata_default_fill_value_does_not_crash(table):
    groups = pd.DataFrame(
        {"subtype": ["A", "A", "A", "B", "B"]}, index=table.columns[:5]
    )
    out = table.add_sample_metadata(groups)
    assert out.sample_metadata.loc[table.columns[0], "subtype"] == "A"
    # sample without metadata stays missing rather than being invented
    assert pd.isna(out.sample_metadata.loc[table.columns[5], "subtype"])


def test_add_sample_metadata_accepts_named_series(table):
    series = pd.Series(["A", "B"] * 3, index=table.columns, name="subtype")
    out = table.add_sample_metadata(series)
    assert list(out.sample_metadata["subtype"]) == ["A", "B"] * 3


def test_add_sample_metadata_unnamed_series_explains_fix(table):
    with pytest.raises(ValueError, match="rename"):
        table.add_sample_metadata(pd.Series(["A"] * 6, index=table.columns))


def test_add_sample_metadata_without_any_matching_sample_raises(table):
    wrong_ids = pd.DataFrame({"subtype": ["A", "B"]}, index=["P1", "P2"])
    with pytest.raises(ValueError, match="None of the 2"):
        table.add_sample_metadata(wrong_ids)


def test_add_feature_metadata_default_fill_value_does_not_crash(table):
    anno = pd.DataFrame({"role": ["x"]}, index=table.index[:1])
    out = table.add_feature_metadata(anno)
    assert out.feature_metadata.loc[table.index[0], "role"] == "x"


# --- plotting --------------------------------------------------------------


@pytest.mark.plot
def test_save_before_render_raises_and_writes_nothing(table, tmp_path):
    target = tmp_path / "oncoplot.png"
    op = table.plot.oncoplot()
    with pytest.raises(RuntimeError, match="render"):
        op.save(target)
    assert not target.exists()


@pytest.mark.plot
def test_oncoplot_add_freq_without_table_freq_explains_fix(table):
    with pytest.raises(ValueError, match=r"table\.add_freq\(\)"):
        table.plot.oncoplot().main().add_freq()


@pytest.mark.plot
def test_oncoplot_missing_annotation_lists_available_columns(table):
    with pytest.raises(ValueError, match="mutations_count"):
        table.plot.oncoplot().main().add_sample_annotation(["subtype"])


@pytest.mark.plot
def test_oncoplot_top_keyword_points_to_table_subsetting(table):
    with pytest.raises(TypeError, match="head"):
        table.plot.oncoplot(top=20)


def test_fit_linewidths_keeps_small_cohorts_and_thins_large_ones():
    fig, ax = plt.subplots(figsize=(10, 5))
    assert _fit_linewidths(ax, 1, (20, 6)) == 1  # wide cells: unchanged
    assert _fit_linewidths(ax, 1, (20, 2000)) == 0.0  # sub-2pt cells: no lines
    plt.close(fig)


@pytest.mark.plot
@pytest.mark.slow
def test_large_cohort_oncoplot_main_matrix_is_not_blank():
    """958 samples used to render an all-white matrix (gridlines covered cells)."""
    big = load_example_table().add_freq().sort_features(by="freq").head(30)
    op = big.plot.oncoplot(figsize=(15, 8)).main().render()
    fig = op.fig
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    main_ax = max(
        fig.axes, key=lambda a: a.get_position().width * a.get_position().height
    )
    x0, y0, x1, y1 = main_ax.get_window_extent().extents.astype(int)
    height = rgb.shape[0]
    region = rgb[height - y1 : height - y0, x0:x1]
    coloured = (region < 240).any(axis=-1).mean()
    op.close()
    # ~35% of cells in this slice are mutated; require a clearly visible matrix
    assert coloured > 0.15


# --- statistics: messages and warnings only, numbers untouched -------------


def test_enrichment_unknown_group_lists_observed_values(table):
    groups = pd.Series(["LUAD", "LUSC"] * 3, index=table.columns, name="subtype")
    labelled = table.add_sample_metadata(groups)
    with pytest.raises(ValueError, match="LUAD"):
        labelled.mutation_enrichment_test("subtype", "LUAD", "luad")


def test_enrichment_missing_group_col_lists_columns(table):
    with pytest.raises(ValueError, match="add_sample_metadata"):
        table.mutation_enrichment_test("subtype", "A", "B")


def test_lollipop_absent_gene_says_so():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maf = load_example_maf()
    with pytest.raises(ValueError, match="no mutations in this MAF"):
        maf.plot.lollipop("TP53")


def test_to_gene_table_warns_when_silent_events_are_kept():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maf = load_example_maf()
    with pytest.warns(UserWarning, match="filter_maf"):
        maf.to_gene_table()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        maf.filter_maf(MAF.nonsynonymous_types).to_gene_table()


def test_somatic_interactions_warns_on_few_samples_without_changing_stats(table):
    with pytest.warns(UserWarning, match="only 6 samples"):
        stats = table.somatic_interactions(top=5)
    assert len(stats) == 10


def test_somatic_interactions_warns_on_pervasive_cooccurrence():
    rng = np.random.default_rng(0)
    burden = rng.random(200) < 0.5  # half the cohort is "hypermutated"
    data = {}
    for i in range(6):
        mutated = burden & (rng.random(200) < 0.9)
        data[f"G{i}"] = pd.Series(mutated).map(
            {True: "Missense_Mutation", False: False}
        )
    data = {k: v.to_numpy(dtype=object) for k, v in data.items()}
    frame = pd.DataFrame(data, index=[f"S{i}" for i in range(200)]).T
    with pytest.warns(UserWarning, match="mutation burden"):
        SmallVariationTable(frame).somatic_interactions(top=6)


@pytest.mark.parametrize("bad_top", [0, -3, True])
def test_somatic_interactions_rejects_invalid_top(table, bad_top):
    with pytest.raises(ValueError, match="top"):
        table.somatic_interactions(top=bad_top)


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 5.0, True])
def test_somatic_interactions_rejects_invalid_alpha(table, bad_alpha):
    with pytest.raises(ValueError, match="alpha"):
        table.somatic_interactions(alpha=bad_alpha)


def test_somatic_interactions_single_gene_preserves_schema_and_provenance(table):
    from pymaftools.provenance import get_provenance

    single = table.head(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = single.somatic_interactions(top=5)
    assert stats.empty
    assert {
        "gene1",
        "gene2",
        "odds_ratio",
        "p_value",
        "adjusted_p_value",
        "interaction",
    }.issubset(stats.columns)
    prov = get_provenance(stats)
    assert prov is not None
    assert prov["analysis"] == "somatic_interactions"


def test_somatic_interactions_warning_points_to_caller(table):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table.somatic_interactions(top=4)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings
    assert user_warnings[0].filename == __file__

    with warnings.catch_warnings(record=True) as caught_plot:
        warnings.simplefilter("always")
        fig, _ = table.plot.somatic_interactions(top=4)
    plt.close(fig)
    plot_warnings = [w for w in caught_plot if issubclass(w.category, UserWarning)]
    assert plot_warnings
    assert plot_warnings[0].filename == __file__


def test_compare_cohorts_rejects_conflicting_aliases_and_bad_min_mutated():
    from pymaftools.provenance import get_provenance

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maf = load_example_maf()

    with pytest.raises(ValueError, match="cohort1_name"):
        maf.compare_cohorts(maf, cohort1_name="A", name1="X")
    with pytest.raises(ValueError, match="cohort2_name"):
        maf.compare_cohorts(maf, cohort2_name="B", name2="Y")
    with pytest.raises(ValueError, match="min_mutated"):
        maf.compare_cohorts(maf, min_mutated=0)

    empty = maf.compare_cohorts(
        maf, cohort1_name="A", cohort2_name="B", min_mutated=999
    )
    assert empty.empty
    assert list(empty.columns) == [
        "gene",
        "A_mutated",
        "B_mutated",
        "A_freq",
        "B_freq",
        "odds_ratio",
        "p_value",
        "adjusted_p_value",
    ]
    assert get_provenance(empty) is not None


def test_pca_on_non_numeric_table_hints_to_binary(table):
    with pytest.raises(ValueError, match="to_binary=True"):
        table.pca()
    with pytest.raises(ValueError, match="to_binary=True"):
        table.plot.pca(color_col="mutations_count")


@pytest.mark.plot
def test_pivot_stats_plot_validates_metadata_columns_upfront(table):
    # Typo in color_col must list available columns before touching PCA matrix
    with pytest.raises(ValueError, match="mutations_count"):
        table.plot.pca(color_col="nonexistent_col")
    with pytest.raises(ValueError, match="mutations_count"):
        table.plot.pca(
            color_col="mutations_count", shape_col="nonexistent_shape", to_binary=True
        )
    with pytest.raises(ValueError, match="mutations_count"):
        table.plot.boxplot(group_col="nonexistent_group")

    groups = pd.Series(["A", "B"] * 3, index=table.columns, name="subtype")
    labelled = table.add_sample_metadata(groups)
    with pytest.raises(ValueError, match="mutations_count"):
        labelled.plot.boxplot(group_col="subtype", test_col="nonexistent_metric")


@pytest.mark.plot
def test_pivot_stats_plot_save_summary_reports_domain_metadata(table, tmp_path):
    import json

    groups = pd.Series(["A", "B"] * 3, index=table.columns, name="subtype")
    labelled = table.add_sample_metadata(groups)

    pca_plotter = labelled.plot
    pca_plotter.pca(color_col="subtype", to_binary=True)
    pca_out = tmp_path / "pca.png"
    pca_plotter.save(pca_out, summary=True)
    pca_summary = pca_plotter.summary()
    pca_plotter.close()
    assert pca_summary["plot_type"] == "pca"
    assert pca_summary["color_col"] == "subtype"
    assert len(pca_summary["explained_variance_ratio"]) == 2
    assert json.loads(pca_out.with_suffix(".json").read_text())["plot_type"] == "pca"

    box_plotter = labelled.plot
    box_plotter.boxplot(group_col="subtype", test_col="mutations_count")
    box_out = tmp_path / "box.png"
    box_plotter.save(box_out, summary=True)
    box_summary = box_plotter.summary()
    box_plotter.close()
    assert box_summary["plot_type"] == "boxplot"
    assert box_summary["groups"] == {"A": 3, "B": 3}
    assert (
        json.loads(box_out.with_suffix(".json").read_text())["plot_type"] == "boxplot"
    )

    int_plotter = labelled.plot
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        int_plotter.somatic_interactions(top=4)
    int_out = tmp_path / "interactions.png"
    int_plotter.save(int_out, summary=True)
    int_summary = int_plotter.summary()
    int_plotter.close()
    assert int_summary["plot_type"] == "somatic_interactions"
    assert int_summary["n_pairs"] == 6
    assert (
        json.loads(int_out.with_suffix(".json").read_text())["plot_type"]
        == "somatic_interactions"
    )
