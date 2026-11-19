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
        stats = table.plot.somatic_interactions_stats(top=5)
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
        SmallVariationTable(frame).plot.somatic_interactions_stats(top=6)
