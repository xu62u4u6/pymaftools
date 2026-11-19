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
