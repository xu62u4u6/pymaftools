"""Regression tests for MAF inputs from different producers.

Fixtures in ``tests/data/maf_sources`` are small slices of public files
(cBioPortal datahub, maftools extdata) or files derived from them in the
layout of another producer (GDC, Funcotator); see the README there.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from pymaftools import MAF
from pymaftools.core import maf_io
from pymaftools.core.maf_io import MAFValidationError
from pymaftools.core.PivotTable import PivotTable

SOURCES = Path(__file__).resolve().parents[1] / "data" / "maf_sources"
CBIO = SOURCES / "cbioportal_acc_tcga_pan_can_atlas_2018.maf"
GDC = SOURCES / "gdc_style_grch38.maf.gz"
FUNCOTATOR = SOURCES / "funcotator_style.maf"
LAML = SOURCES / "maftools_tcga_laml_head.maf.gz"
BRCA = SOURCES / "maftools_brca_head.maf"


def _cbio_frame() -> pd.DataFrame:
    return pd.read_csv(CBIO, sep="\t", dtype=str, keep_default_na=False)


def _write(frame: pd.DataFrame, path: Path, prefix: str = "") -> Path:
    path.write_text(prefix + frame.to_csv(sep="\t", index=False))
    return path


# --------------------------------------------------------------------------
# public / producer-style fixtures
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (CBIO, "maf"),
        (GDC, "gdc"),
        (FUNCOTATOR, "funcotator"),
        (LAML, "legacy-tcga"),
        (BRCA, "minimal"),
    ],
)
def test_producer_fixtures_load_with_report(path, source):
    maf = MAF.read_maf(path)

    report = maf.ingest_report
    assert report.source == source
    assert report.n_rows == len(maf) > 0
    assert report.n_samples == maf["sample_ID"].nunique()
    assert report.issues == []
    assert str(report).startswith("MAF ingest report")
    assert {type(value) for value in maf["Chromosome"]} == {str}


def test_gdc_comment_block_and_gzip_are_skipped():
    maf = MAF.read_maf(GDC)
    assert maf.ingest_report.comment_lines == 3
    assert maf["Chromosome"].str.startswith("chr").all()
    assert maf.ingest_report.genome_builds == ["GRCh38"]


def test_case_variant_column_is_renamed():
    maf = MAF.read_maf(LAML)  # maftools spells it End_position
    assert "End_Position" in maf.columns
    assert maf.ingest_report.renamed_columns == {"End_position": "End_Position"}
    assert "renamed column" in str(maf.ingest_report)


def test_missing_tumor_seq_allele1_is_filled_from_reference():
    maf = MAF.read_maf(BRCA)
    assert (maf["Tumor_Seq_Allele1"] == maf["Reference_Allele"]).all()
    assert "Tumor_Seq_Allele1" in maf.ingest_report.filled_columns


def test_unknown_classifications_are_reported_not_dropped():
    maf = MAF.read_maf(FUNCOTATOR)
    unknown = maf.ingest_report.unknown_classifications
    assert set(unknown) == {
        "lincRNA",
        "De_novo_Start_InFrame",
        "Start_Codon_SNP",
        "COULD_NOT_DETERMINE",
    }
    assert len(maf) == maf.ingest_report.n_rows
    # Splice_Region (GDC/cBioPortal/vcf2maf) is part of the vocabulary.
    assert (
        "Splice_Region" not in MAF.read_maf(CBIO).ingest_report.unknown_classifications
    )


# --------------------------------------------------------------------------
# typing: identifiers stay text
# --------------------------------------------------------------------------


def test_numeric_looking_sample_ids_are_preserved(tmp_path):
    frame = _cbio_frame()
    ids = {
        old: f"{i:03d}" for i, old in enumerate(frame["Tumor_Sample_Barcode"].unique())
    }
    frame["Tumor_Sample_Barcode"] = frame["Tumor_Sample_Barcode"].map(ids)
    maf = MAF.read_maf(_write(frame, tmp_path / "ids.maf"))
    assert set(maf["sample_ID"]) == set(ids.values())
    assert "000" in set(maf["sample_ID"])


def test_digit_only_chromosomes_stay_strings(tmp_path):
    frame = _cbio_frame()
    frame = frame[frame["Chromosome"].str.isdigit()]
    maf = MAF.read_maf(_write(frame, tmp_path / "digits.maf"))
    assert {type(value) for value in maf["Chromosome"]} == {str}


def test_missing_coordinates_keep_integer_positions(tmp_path):
    frame = _cbio_frame()
    frame.loc[frame.index[:2], ["Start_Position", "End_Position"]] = ""
    frame.loc[frame.index[:2], "Variant_Classification"] = "Fusion"
    maf = MAF.read_maf(_write(frame, tmp_path / "fusion.maf"))
    assert str(maf["Start_Position"].dtype) == "Int64"
    assert not any(".0|" in label for label in maf.index)


# --------------------------------------------------------------------------
# audited problems: strict raises, warn loads, deduplicate fixes
# --------------------------------------------------------------------------


def test_blank_hugo_symbol_is_loud(tmp_path):
    frame = _cbio_frame()
    frame.loc[frame.index[:3], "Hugo_Symbol"] = ""
    frame.loc[frame.index[3], "Hugo_Symbol"] = "NA"
    path = _write(frame, tmp_path / "blank.maf")

    with pytest.raises(MAFValidationError, match="4 event\\(s\\) have a blank"):
        MAF.read_maf(path)
    with pytest.warns(UserWarning, match="blank Hugo_Symbol"):
        maf = MAF.read_maf(path, validate="warn")
    assert maf["Hugo_Symbol"].isna().sum() == 4


def test_strict_error_carries_report(tmp_path):
    frame = _cbio_frame()
    frame.loc[frame.index[0], "Hugo_Symbol"] = ""
    with pytest.raises(MAFValidationError) as excinfo:
        MAF.read_maf(_write(frame, tmp_path / "blank.maf"))
    assert excinfo.value.report.blank_gene_rows == 1
    assert "validate='warn'" in str(excinfo.value)


def test_duplicate_events_raise_or_are_dropped(tmp_path):
    frame = _cbio_frame()
    path = _write(pd.concat([frame, frame.head(5)]), tmp_path / "dup.maf")

    with pytest.raises(MAFValidationError, match="5 duplicate event"):
        MAF.read_maf(path)

    maf = MAF.read_maf(path, deduplicate=True)
    assert len(maf) == len(frame)
    assert maf.ingest_report.dropped_duplicate_rows == 5
    assert (maf.to_gene_table() == "Multi_Hit").values.sum() == (
        MAF.read_maf(CBIO).to_gene_table() == "Multi_Hit"
    ).values.sum()


def test_mixed_builds_and_chromosome_naming_raise(tmp_path):
    frame = _cbio_frame()
    frame.loc[frame.index[:5], "NCBI_Build"] = "GRCh38"
    frame.loc[frame.index[:5], "Chromosome"] = (
        "chr" + frame.loc[frame.index[:5], "Chromosome"]
    )
    with pytest.raises(MAFValidationError) as excinfo:
        MAF.read_maf(_write(frame, tmp_path / "mixed.maf"))
    message = str(excinfo.value)
    assert "mixed genome builds" in message
    assert "mixed chromosome naming" in message


def test_legacy_allele_layout_is_detected(tmp_path):
    frame = _cbio_frame()
    snp = frame["Variant_Type"].eq("SNP")
    frame.loc[snp, "Tumor_Seq_Allele1"] = frame.loc[snp, "Tumor_Seq_Allele2"]
    frame.loc[snp, "Tumor_Seq_Allele2"] = frame.loc[snp, "Reference_Allele"]
    with pytest.raises(MAFValidationError, match="legacy TCGA layout"):
        MAF.read_maf(_write(frame, tmp_path / "legacy.maf"))


def test_placeholder_barcode_is_noted(tmp_path):
    frame = _cbio_frame()
    first = frame["Tumor_Sample_Barcode"].iloc[0]
    frame = frame[frame["Tumor_Sample_Barcode"].eq(first)].copy()
    frame["Tumor_Sample_Barcode"] = "TUMOR"
    maf = MAF.read_maf(_write(frame, tmp_path / "vcf2maf.maf"))
    assert maf.ingest_report.placeholder_sample_ids == ["TUMOR"]
    assert "sample_ID=" in str(maf.ingest_report)


def test_validate_off_records_without_warning(tmp_path):
    frame = _cbio_frame()
    frame.loc[frame.index[0], "Hugo_Symbol"] = ""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        maf = MAF.read_maf(_write(frame, tmp_path / "blank.maf"), validate="off")
    assert maf.ingest_report.issues


def test_invalid_validate_mode():
    with pytest.raises(ValueError, match="validate must be one of"):
        MAF.read_maf(CBIO, validate="loud")


def test_merge_mafs_rejects_mixed_builds():
    grch37 = MAF.read_maf(CBIO)
    grch38 = MAF.read_maf(GDC)
    with pytest.raises(ValueError, match="genome builds"):
        MAF.merge_mafs([grch37, grch38])
    merged = MAF.merge_mafs([grch37, grch38], allow_mixed_builds=True)
    assert len(merged) == len(grch37) + len(grch38)


def test_report_survives_filtering():
    maf = MAF.read_maf(CBIO)
    filtered = maf.filter_maf(MAF.nonsynonymous_types)
    assert filtered.ingest_report is maf.ingest_report


# --------------------------------------------------------------------------
# denominators and IDs
# --------------------------------------------------------------------------


def test_samples_argument_adds_zero_event_samples():
    maf = MAF.read_maf(CBIO)
    universe = [*maf["sample_ID"].unique(), "NO-EVENTS-1", "NO-EVENTS-2"]
    table = maf.to_gene_table(samples=universe)
    assert list(table.columns) == universe
    assert not pd.DataFrame(table)["NO-EVENTS-1"].astype(bool).any()
    assert table.sample_metadata.loc["NO-EVENTS-1", "mutations_count"] == 0


def test_samples_argument_rejects_unlisted_event_samples():
    maf = MAF.read_maf(CBIO)
    with pytest.raises(ValueError, match="barcode level"):
        maf.to_gene_table(samples=maf["sample_ID"].unique()[:3])


def test_samples_and_manifest_are_exclusive():
    maf = MAF.read_maf(CBIO)
    with pytest.raises(ValueError, match="either"):
        maf.to_gene_table(sample_manifest=object(), samples=["x"])


def test_blank_gene_samples_stay_in_denominator(tmp_path):
    frame = _cbio_frame()
    only_sample = frame["Tumor_Sample_Barcode"].value_counts().index[-1]
    frame.loc[frame["Tumor_Sample_Barcode"].eq(only_sample), "Hugo_Symbol"] = ""
    maf = MAF.read_maf(_write(frame, tmp_path / "blank.maf"), validate="off")
    with pytest.warns(UserWarning, match="none in a gene row"):
        table = maf.to_gene_table()
    assert only_sample in table.columns
    assert table.shape[1] == maf["sample_ID"].nunique()


def test_add_sample_metadata_loud_on_id_mismatch():
    table = MAF.read_maf(CBIO).to_gene_table()
    patients = sorted({sample[:12] for sample in table.columns})
    patient_level = pd.DataFrame({"stage": ["I"] * len(patients)}, index=patients)
    with pytest.raises(ValueError, match="barcode level"):
        table.add_sample_metadata(patient_level)

    partial = pd.DataFrame(
        {"stage": ["I", "II"]}, index=[table.columns[0], "NOT-IN-TABLE"]
    )
    with pytest.warns(UserWarning, match="1 matched"):
        result = table.add_sample_metadata(partial, fill_value="NA")
    assert isinstance(result, PivotTable)


# --------------------------------------------------------------------------
# lollipop positions across producers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "start", "length"),
    [
        ("858/1210", 858, 1210),
        ("745-750/1210", 745, 1210),
        ("?-12/393", 12, 393),
        ("100-?/393", 100, 393),
        ("74", 74, None),
        (74.0, 74, None),
        ("-", None, None),
        (float("nan"), None, None),
        (pd.NA, None, None),
    ],
)
def test_protein_position_parsing(value, start, length):
    assert maf_io.protein_start(value) == start
    assert maf_io.protein_length(value) == length


@pytest.mark.parametrize(
    ("value", "start"),
    [
        ("p.K960R", 960),
        ("p.R1517H", 1517),
        ("p.*123Lext*?", 123),
        ("p.(G12D)", 12),
        ("", None),
    ],
)
def test_hgvsp_parsing(value, start):
    assert maf_io.hgvsp_start(value) == start


@pytest.mark.parametrize(
    ("path", "expected_length"),
    [
        (CBIO, 393),  # bare positions: length inferred from the domain table
        (GDC, 393),
        (FUNCOTATOR, 393),  # Protein_Change only
    ],
)
def test_lollipop_builds_for_every_producer(path, expected_length):
    maf = MAF.read_maf(path)
    with pytest.warns(UserWarning, match="no protein length for TP53"):
        lollipop = maf.plot.lollipop("TP53")
    assert lollipop.protein_length == expected_length
    assert lollipop.mutations


def test_lollipop_from_legacy_protein_change():
    maf = MAF.read_maf(LAML)
    gene = (
        maf.filter_maf(MAF.nonsynonymous_types)["Hugo_Symbol"].value_counts().index[0]
    )
    _, mutations = maf.get_protein_info(gene)
    assert mutations and all(entry["position"] > 0 for entry in mutations)


def test_lollipop_without_any_position_column_is_actionable():
    maf = MAF.read_maf(BRCA)
    gene = maf.filter_maf(MAF.nonsynonymous_types)["Hugo_Symbol"].iloc[0]
    with pytest.raises(
        ValueError, match="Protein_position, HGVSp_Short or Protein_Change"
    ):
        maf.get_protein_info(gene)


def test_read_leading_comments_counts_gdc_block():
    assert len(maf_io.read_leading_comments(GDC)) == 3
    assert maf_io.read_leading_comments(CBIO) == []
