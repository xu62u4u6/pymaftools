"""Machine-readable results: serialisation, provenance, figure summaries, CLI.

These features are additive metadata. The numeric-equality tests pin that
attaching provenance never changes a computed statistic.
"""

import json
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import pymaftools
from pymaftools import MAF, example_maf_path, get_provenance, to_json, to_jsonable
from pymaftools.cli import EXIT_INVALID_INPUT, EXIT_OK, main as cli_main
from pymaftools.core.SampleManifest import SampleManifest
from pymaftools.plot.summary import figure_summary
from pymaftools.plot.wes import somatic_interactions
from pymaftools.provenance import PROVENANCE_ATTR, file_fingerprint


@pytest.fixture
def maf():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MAF.read_maf(example_maf_path())


@pytest.fixture
def table(maf):
    return maf.filter_maf(MAF.nonsynonymous_types).to_gene_table()


def _write_blank_gene_maf(path):
    """Copy the example MAF with one blank Hugo_Symbol (a strict-mode issue)."""
    with open(example_maf_path()) as handle:
        lines = handle.read().splitlines()
    gene = lines[0].split("\t").index("Hugo_Symbol")
    row = lines[1].split("\t")
    row[gene] = ""
    path.write_text("\n".join([lines[0], "\t".join(row)] + lines[2:]) + "\n")
    return path


# --- serialisation ---------------------------------------------------------


def test_to_jsonable_handles_nan_inf_numpy_and_pandas():
    frame = pd.DataFrame(
        {"odds_ratio": [np.inf, -np.inf, np.nan, 1.5], "n": np.arange(4)},
    )
    record = to_jsonable(frame)
    assert record["columns"] == ["odds_ratio", "n"]
    assert [row["odds_ratio"] for row in record["records"]] == [
        "inf",
        "-inf",
        None,
        1.5,
    ]
    assert all(isinstance(row["n"], int) for row in record["records"])
    # strict JSON: no NaN / Infinity literals
    json.loads(to_json(frame), parse_constant=lambda c: pytest.fail(c))


def test_session_info_reports_versions():
    info = pymaftools.session_info()
    assert info["pymaftools"] == pymaftools.__version__
    assert "pandas" in info["dependencies"]
    json.dumps(info)


# --- ingest report ---------------------------------------------------------


def test_ingest_report_to_dict_has_checksum(maf):
    record = maf.ingest_report.to_dict()
    assert record["sha256"] == file_fingerprint(example_maf_path())["sha256"]
    assert record["n_samples"] == 6
    assert record["ok"] is True
    assert json.loads(maf.ingest_report.to_json())["n_rows"] == record["n_rows"]


def test_validation_error_report_is_serialisable(tmp_path):
    bad = _write_blank_gene_maf(tmp_path / "bad.maf")
    with pytest.raises(pymaftools.core.maf_io.MAFValidationError) as error:
        MAF.read_maf(bad)
    record = error.value.report.to_dict()
    assert record["ok"] is False and record["issues"]


# --- provenance on statistics ----------------------------------------------


def test_somatic_interactions_provenance_and_unchanged_numbers(table):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = somatic_interactions(table, top=8)
    record = get_provenance(stats)
    assert record["analysis"] == "somatic_interactions"
    assert record["parameters"] == {"top": 8, "alpha": 0.05}
    assert record["sample_universe"]["n_samples"] == table.shape[1]
    assert (
        record["inputs"][0]["sha256"] == file_fingerprint(example_maf_path())["sha256"]
    )
    # The statistics themselves are exactly what an attrs-free copy holds.
    plain = pd.DataFrame(stats.to_dict())
    pd.testing.assert_frame_equal(
        pd.DataFrame(stats).reset_index(drop=True),
        plain.reset_index(drop=True),
    )
    assert to_jsonable(stats)["provenance"]["analysis"] == "somatic_interactions"


def test_input_fingerprint_survives_table_operations(table):
    derived = table.add_freq().sort_features(by="freq").head(5).to_binary_table()
    assert derived.attrs["pymaftools_inputs"][0]["path"].endswith(".maf")


def test_enrichment_provenance_records_groups(table):
    table = table.copy()
    table.sample_metadata["group"] = ["A", "A", "A", "B", "B", "B"]
    result = table.mutation_enrichment_test("group", "A", "B")
    record = get_provenance(result)
    assert record["analysis"] == "mutation_enrichment_test"
    assert record["parameters"]["group1_sample_count"] == 3
    assert record["sample_universe"]["n_samples"] == 6
    assert PROVENANCE_ATTR not in result.columns


def test_tmb_audit_to_dict(maf):
    samples = sorted(maf["sample_ID"].unique())
    manifest = SampleManifest(
        pd.DataFrame(
            {
                "patient_id": samples,
                "eligible": True,
                "callable_mb": 30.0,
            },
            index=samples,
        )
    )
    audit = maf.calculate_tmb_audit(
        manifest, variant_classifications=MAF.nonsynonymous_types
    )
    record = json.loads(audit.to_json())
    assert record["provenance"]["analysis"] == "MAF.calculate_tmb_audit"
    assert record["provenance"]["sample_universe"]["n_samples"] == len(samples)
    assert record["n_included_events"] == int(audit.events["included"].sum())
    tmb = {
        row["sample_ID"] if "sample_ID" in row else row["index"]: row["TMB"]
        for row in record["summary"]["records"]
    }
    assert tmb == pytest.approx(audit.summary["TMB"].to_dict())
    assert "events" not in record
    assert "events" in audit.to_dict(include_events=True)


# --- figure summaries ------------------------------------------------------


def test_figure_summary_flags_blank_figure():
    fig, ax = plt.subplots()
    record = figure_summary(fig)
    assert record["n_empty_axes"] == 1
    assert any("blank" in w for w in record["warnings"])
    plt.close(fig)


def test_oncoplot_save_writes_summary_sidecar(table, tmp_path):
    table = table.add_freq().sort_features(by="freq").head(10)
    oncoplot = table.plot.oncoplot(figsize=(8, 5)).main().add_freq().render()
    target = tmp_path / "onco.png"
    oncoplot.save(target, dpi=80, summary=True)
    record = json.loads((tmp_path / "onco.json").read_text())
    oncoplot.close()
    assert record["plot_type"] == "OncoPlot"
    assert record["n_features"] == 10 and record["n_samples"] == 6
    assert record["fill_fraction"] > 0
    assert record["top_features"][0]["feature"] in table.index
    assert record["warnings"] == []


def test_oncoplot_summary_warns_about_blank_matrix(table):
    empty = table.head(5).copy()
    empty.loc[:, :] = False
    oncoplot = empty.plot.oncoplot(figsize=(6, 4)).main().render()
    record = oncoplot.summary()
    oncoplot.close()
    assert record["n_nonempty_cells"] == 0
    assert any("blank" in w for w in record["warnings"])


def test_oncoplot_summary_reports_hidden_labels_and_tiny_cells():
    table = pymaftools.load_example_table()
    table = table.add_freq().sort_features(by="freq").head(10)
    oncoplot = table.plot.oncoplot(figsize=(4, 3)).main().render()
    record = oncoplot.summary(dpi=50)
    oncoplot.close()
    assert record["sample_labels_auto_hidden"] is True
    assert any("auto-hidden" in w for w in record["warnings"])
    assert any("px wide" in w for w in record["warnings"])


def test_save_without_summary_writes_only_figure(table, tmp_path):
    oncoplot = table.head(3).plot.oncoplot(figsize=(5, 3)).main().render()
    oncoplot.save(tmp_path / "plain.png", dpi=50)
    oncoplot.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plain.png"]


def test_save_errors_are_raised(table, tmp_path):
    oncoplot = table.head(3).plot.oncoplot(figsize=(5, 3)).main().render()
    with pytest.raises(Exception):
        oncoplot.save(tmp_path / "missing_dir" / "x.png", dpi=50)
    oncoplot.close()


# --- CLI -------------------------------------------------------------------


def _run_cli(capsys, *argv):
    code = cli_main([str(arg) for arg in argv])
    out = capsys.readouterr().out
    return code, out


def test_cli_inspect_json(capsys):
    code, out = _run_cli(capsys, "inspect", example_maf_path(), "--json")
    record = json.loads(out)
    assert code == EXIT_OK
    assert record["n_samples"] == 6
    assert record["ingest_report"]["sha256"]
    assert record["top_genes_nonsynonymous"][0]["mutated_samples"] >= 1


def test_cli_global_json_flag(capsys):
    code, out = _run_cli(capsys, "--json", "session-info")
    assert code == EXIT_OK
    assert json.loads(out)["pymaftools"] == pymaftools.__version__


def test_cli_oncoplot_writes_figure_and_summary(capsys, tmp_path):
    target = tmp_path / "onco.png"
    code, out = _run_cli(
        capsys,
        "oncoplot",
        example_maf_path(),
        "-o",
        str(target),
        "--top",
        "8",
        "--dpi",
        "60",
        "--json",
    )
    record = json.loads(out)
    assert code == EXIT_OK
    assert target.exists()
    sidecar = json.loads((tmp_path / "onco.json").read_text())
    assert sidecar["n_features"] == 8
    assert sidecar["provenance"]["parameters"]["top"] == 8
    assert record["summary_path"].endswith("onco.json")


def test_cli_interactions_matches_python_api(capsys, table, tmp_path):
    target = tmp_path / "stats.csv"
    code, out = _run_cli(
        capsys,
        "interactions",
        example_maf_path(),
        "--top",
        "6",
        "-o",
        str(target),
        "--json",
    )
    record = json.loads(out)
    assert code == EXIT_OK
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = somatic_interactions(table, top=6)
    assert record["n_rows"] == len(expected)
    assert [r["p_value"] for r in record["records"]] == pytest.approx(
        expected["p_value"].tolist()
    )
    assert any("power" in w for w in record["python_warnings"])
    assert len(pd.read_csv(target)) == len(expected)


def test_cli_strict_validation_exit_code(capsys, tmp_path):
    bad = _write_blank_gene_maf(tmp_path / "bad.maf")
    code, out = _run_cli(capsys, "interactions", str(bad), "--json")
    record = json.loads(out)
    assert code == EXIT_INVALID_INPUT
    assert record["error"] == "MAFValidationError"
    assert record["ingest_report"]["issues"]


def test_cli_missing_file_is_json_error(capsys, tmp_path):
    code, out = _run_cli(capsys, "inspect", str(tmp_path / "nope.maf"), "--json")
    assert code == 1
    assert json.loads(out)["error"] == "FileNotFoundError"


# --- fixes from the 92-sample TCGA ACC agent trial -------------------------


def test_add_sample_metadata_on_sorted_table_aligns_by_id(table):
    sorted_table = table.add_freq().sort_features(by="freq").sort_samples_by_mutations()
    labels = pd.Series(["A", "B"] * 3, index=sorted(sorted_table.columns), name="grp")
    out = sorted_table.add_sample_metadata(labels)
    assert list(out.sample_metadata.index) == list(out.columns)
    for sample in out.columns:
        assert out.sample_metadata.loc[sample, "grp"] == labels[sample]
    # downstream statistics accept it (used to raise an index-mismatch error)
    assert len(out.mutation_enrichment_test("grp", "A", "B")) == len(out)


def test_add_feature_metadata_keeps_feature_order(table):
    sorted_table = table.add_freq().sort_features(by="freq")
    note = pd.Series("x", index=sorted(sorted_table.index), name="note")
    out = sorted_table.add_feature_metadata(note)
    assert list(out.feature_metadata.index) == list(out.index)


def test_metadata_order_mismatch_message_names_fix(table):
    broken = table.copy()
    broken.sample_metadata = broken.sample_metadata.iloc[::-1]
    with pytest.raises(ValueError, match="different order.*reindex"):
        broken._validate_metadata()


def test_interaction_heatmap_colours_evidence_not_zero_odds():
    from pymaftools import PivotTable

    # GENE_A and GENE_B are never co-mutated (odds ratio 0) in a tiny cohort:
    # far from significant, so the cell must stay near the neutral colour.
    table = PivotTable(
        [
            [True, True, False, False, False, False],
            [False, False, True, False, False, False],
            [True, False, True, True, False, False],
        ],
        index=["GENE_A", "GENE_B", "GENE_C"],
        columns=[f"S{i}" for i in range(6)],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, stats = table.plot.somatic_interactions(top=3)
    zero = stats[
        (stats.gene1.isin(["GENE_A", "GENE_B"]))
        & (stats.gene2.isin(["GENE_A", "GENE_B"]))
    ]
    assert float(zero["odds_ratio"].iloc[0]) == 0
    values = fig.axes[0].collections[0].get_array()
    assert np.nanmax(np.abs(values)) < -np.log10(0.05)
    record = figure_summary(fig)
    assert record["n_significant"] == 0
    assert record["n_infinite_or_zero_odds_ratio"] >= 1
    assert any("odds ratio of 0 or infinity" in w for w in record["warnings"])
    assert any("no gene pair reaches" in w for w in record["warnings"])
    plt.close(fig)


def test_all_missing_annotation_warns_and_is_named_in_summary(table):
    table = table.head(5).copy()
    table.sample_metadata["stage_group"] = pd.Series(
        pd.NA, index=table.columns, dtype="object"
    )
    with pytest.warns(UserWarning, match="entirely missing"):
        oncoplot = (
            table.plot.oncoplot(figsize=(6, 4))
            .main()
            .add_sample_annotation(["stage_group"])
        )
    record = oncoplot.render().summary()
    oncoplot.close()
    annotation = [t for t in record["tracks"] if t.get("column") == "stage_group"][0]
    assert annotation["n_missing"] == annotation["n_values"] == 6
    assert any("'stage_group' is entirely missing" in w for w in record["warnings"])


def test_cli_samples_file_includes_zero_event_samples(capsys, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maf = MAF.read_maf(example_maf_path())
    samples = sorted(maf["sample_ID"].unique()) + ["ZERO-EVENT-SAMPLE"]
    sample_file = tmp_path / "clinical.txt"
    sample_file.write_text(
        "#comment line\nPATIENT_ID\tSAMPLE_ID\n"
        + "".join(f"P{i}\t{s}\n" for i, s in enumerate(samples))
    )
    code, out = _run_cli(
        capsys,
        "interactions",
        example_maf_path(),
        "--top",
        "5",
        "--samples",
        sample_file,
        "--json",
    )
    record = json.loads(out)
    assert code == EXIT_OK
    assert record["provenance"]["sample_universe"]["n_samples"] == 7
    assert record["provenance"]["parameters"]["samples_file"] == str(sample_file)
    assert "index" not in record["columns"]
    assert {r["jointly_observed"] for r in record["records"]} == {7}
    assert not any("--samples" in w for w in record.get("python_warnings", []))


def test_cli_without_samples_warns_about_denominator(capsys):
    code, out = _run_cli(
        capsys, "interactions", example_maf_path(), "--top", "4", "--json"
    )
    record = json.loads(out)
    assert any("--samples" in w for w in record["python_warnings"])


def test_cli_oncoplot_sorts_samples(capsys, tmp_path):
    target = tmp_path / "onco.png"
    code, out = _run_cli(
        capsys,
        "oncoplot",
        example_maf_path(),
        "-o",
        target,
        "--top",
        "8",
        "--dpi",
        "50",
        "--json",
    )
    record = json.loads(out)
    assert code == EXIT_OK
    assert record["samples_sorted_by"].startswith("sort_samples_by_mutations")
    assert any("--samples" in w or "no --samples" in w for w in record["warnings"])


def test_read_sample_list_plain_and_explicit_column(tmp_path):
    from pymaftools.cli import read_sample_list

    plain = tmp_path / "plain.txt"
    plain.write_text("S1\nS2\n\n# note\nS3\n")
    assert read_sample_list(str(plain)) == ["S1", "S2", "S3"]
    table = tmp_path / "t.csv"
    table.write_text("id,barcode\n1,B1\n2,B2\n")
    assert read_sample_list(str(table), "barcode") == ["B1", "B2"]
    with pytest.raises(ValueError, match="not in the header"):
        read_sample_list(str(table), "missing")
