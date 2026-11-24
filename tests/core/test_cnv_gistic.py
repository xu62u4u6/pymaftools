"""GISTIC readers, CNV value_scale contract, and binarisation semantics."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pymaftools.core.Cohort import Cohort
from pymaftools.core.CopyNumberVariationTable import CopyNumberVariationTable
from pymaftools.core.PivotTable import PivotTable


def test_gene_level_accepts_ids_without_underscore(tmp_path):
    # rsplit("_") on TCGA barcodes used to raise "Columns must be same length".
    path = tmp_path / "all_data_by_genes.txt"
    pd.DataFrame(
        {
            "Gene Symbol": ["A", "B"],
            "Gene ID": [1, 2],
            "Cytoband": ["1p36.1", "Xq21"],
            "TCGA-05-4249-01A": [0.1, 0.2],
            "LUAD_001_T": [0.0, 1.0],
        }
    ).to_csv(path, sep="\t", index=False)

    table = CopyNumberVariationTable.read_gistic_gene_level(path)

    metadata = table.sample_metadata
    assert metadata.loc["TCGA-05-4249-01A", "case_ID"] == "TCGA-05-4249-01A"
    assert pd.isna(metadata.loc["TCGA-05-4249-01A", "sample_type"])
    assert metadata.loc["LUAD_001_T", "case_ID"] == "LUAD_001"
    assert metadata.loc["LUAD_001_T", "sample_type"] == "T"


def test_arm_level_keeps_sex_chromosomes_and_does_not_fabricate_arms(tmp_path):
    # Arms were reindexed to 1p..22q: X/Y dropped, missing arms filled with 0.
    path = tmp_path / "broad_values_by_arm.txt"
    pd.DataFrame(
        {
            "Chromosome Arm": ["2q", "1p", "Yq", "Xp", "Xq"],
            "S1": [0.1, -0.2, -0.4, 0.3, 0.0],
        }
    ).to_csv(path, sep="\t", index=False)

    table = CopyNumberVariationTable.read_gistic_arm_level(path)

    assert list(table.index) == ["1p", "2q", "Xp", "Xq", "Yq"]
    assert table.loc["Yq", "S1"] == -0.4
    assert table.feature_metadata["Chromosome"].tolist() == ["1", "2", "X", "X", "Y"]
    assert table.value_scale == "log2_ratio"


def test_gene_level_all_tcga_barcodes(tmp_path):
    path = tmp_path / "all_thresholded.by_genes.txt"
    pd.DataFrame(
        {
            "Gene Symbol": ["A"],
            "Gene ID": [1],
            "Cytoband": ["1p36.1"],
            "TCGA-05-4249-01A": [1],
            "TCGA-05-4250-01A": [0],
        }
    ).to_csv(path, sep="\t", index=False)

    table = CopyNumberVariationTable.read_gistic_gene_level(path)

    assert list(table.columns) == ["TCGA-05-4249-01A", "TCGA-05-4250-01A"]
    assert table.sample_metadata["sample_type"].isna().all()


def test_cnv_value_scale_validation_propagation_and_h5_round_trip(tmp_path):
    with pytest.raises(ValueError, match="Invalid value_scale"):
        CopyNumberVariationTable({"S1": [2]}, index=["TP53"], value_scale="invalid")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = CopyNumberVariationTable(
            {"S1": [0, 2], "S2": [1, 2]},
            index=["TP53", "EGFR"],
            value_scale="absolute_cn",
        )
        table.value_scale = "absolute_cn"
    assert not [w for w in caught if issubclass(w.category, UserWarning)]

    assert table.copy().value_scale == "absolute_cn"
    assert table.subset(features=["TP53"]).value_scale == "absolute_cn"
    assert table.reindex(columns=["S2", "S1"]).value_scale == "absolute_cn"
    assert (
        table.add_sample_metadata(
            pd.Series(["LUAD", "LUSC"], index=["S1", "S2"], name="subtype")
        ).value_scale
        == "absolute_cn"
    )
    assert CopyNumberVariationTable.from_pivot_table(table).value_scale == "absolute_cn"

    pivot_h5 = tmp_path / "cnv.h5"
    table.to_h5(pivot_h5)
    reloaded = PivotTable.read_h5(pivot_h5)
    assert isinstance(reloaded, CopyNumberVariationTable)
    assert reloaded.value_scale == "absolute_cn"

    cohort = Cohort(name="cnv_scale")
    cohort.add_table(table, "cnv")
    cohort_h5 = tmp_path / "cohort.h5"
    cohort.to_h5(cohort_h5)
    reloaded_cohort = Cohort.read_h5(cohort_h5)
    assert isinstance(reloaded_cohort.cnv, CopyNumberVariationTable)
    assert reloaded_cohort.cnv.value_scale == "absolute_cn"


def test_cnv_to_binary_table_absolute_cn_treats_diploid_two_as_neutral():
    # ISS-003: ASCAT3 absolute CN with only {0, 1, 2} used to invert diploid 2 -> True
    # and homozygous deletion 0 -> False.
    ascat_del_only = CopyNumberVariationTable(
        {"S1": [0, 2], "S2": [1, 2], "S3": [2, 2]},
        index=["CDKN2A", "TP53"],
        value_scale="absolute_cn",
    )
    binary = ascat_del_only.to_binary_table()
    assert binary.value_scale is None
    assert binary.loc["CDKN2A"].tolist() == [True, True, False]
    assert binary.loc["TP53"].tolist() == [False, False, False]

    ascat_with_amp_and_na = CopyNumberVariationTable(
        {"S1": [0.0, 2.0], "S2": [4.0, np.nan]},
        index=["MYC", "TP53"],
        value_scale="absolute_cn",
    )
    binary_amp = ascat_with_amp_and_na.to_binary_table()
    assert binary_amp.loc["MYC"].tolist() == [True, True]
    assert binary_amp.loc["TP53"].tolist() == [False, False]

    continuous_abs = CopyNumberVariationTable(
        {"S1": [1.8, 2.0], "S2": [2.3, 2.0]},
        index=["G1", "G2"],
        value_scale="absolute_cn",
    )
    with pytest.raises(ValueError, match="explicit threshold"):
        continuous_abs.to_binary_table()


def test_cnv_to_binary_table_gistic_discrete_log2_ratio_and_ambiguous_none():
    gistic_pos = CopyNumberVariationTable(
        {"S1": [0, 1], "S2": [2, 0]},
        index=["MYC", "EGFR"],
        value_scale="gistic_discrete",
    )
    binary_gistic = gistic_pos.to_binary_table()
    assert binary_gistic.loc["MYC"].tolist() == [False, True]
    assert binary_gistic.loc["EGFR"].tolist() == [True, False]
    assert binary_gistic.value_scale is None

    invalid_gistic = CopyNumberVariationTable(
        {"S1": [0, 4]},
        index=["MYC", "EGFR"],
        value_scale="gistic_discrete",
    )
    with pytest.raises(ValueError, match="outside"):
        invalid_gistic.to_binary_table()

    log2_int = CopyNumberVariationTable(
        {"S1": [0, 1], "S2": [2, 0]},
        index=["MYC", "EGFR"],
        value_scale="log2_ratio",
    )
    with pytest.raises(ValueError, match="explicit threshold"):
        log2_int.to_binary_table()
    assert log2_int.to_binary_table(threshold=0.5).loc["MYC"].tolist() == [False, True]

    # Untagged non-negative {0, 1, 2} is ambiguous between GISTIC and absolute CN.
    ambiguous = CopyNumberVariationTable(
        {"S1": [0, 2], "S2": [1, 2]},
        index=["CDKN2A", "TP53"],
    )
    with pytest.raises(ValueError, match="gistic_discrete.*absolute_cn"):
        ambiguous.to_binary_table()

    # Untagged table with negative calls is unambiguously GISTIC discrete.
    signed_gistic = CopyNumberVariationTable(
        {"S1": [-2, 0], "S2": [1, 2]},
        index=["CDKN2A", "MYC"],
    )
    binary_signed = signed_gistic.to_binary_table()
    assert binary_signed.loc["CDKN2A"].tolist() == [True, True]
    assert binary_signed.loc["MYC"].tolist() == [False, True]


def test_cnv_absolute_cn_ploidy_wgd_and_sex_chromosomes():
    # #61: WGD tetraploid tumors and male/female chrX/chrY baselines.
    cnv = CopyNumberVariationTable(
        {
            "male_diploid": [2, 1, 1, 0],
            "female_diploid": [2, 2, 0, 1],
            "male_wgd": [4, 2, 2, 6],
        },
        index=["TP53", "DMD", "SRY", "MYC"],
        value_scale="absolute_cn",
    )
    cnv.feature_metadata["Chromosome"] = ["chr17", "chrX", "chrY", "chr8"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        binary = cnv.to_binary_table(
            ploidy={"male_diploid": 2.0, "female_diploid": 2.0, "male_wgd": 4.0},
            sex={
                "male_diploid": "male",
                "female_diploid": "female",
                "male_wgd": "M",
            },
        )
    assert not [w for w in caught if issubclass(w.category, UserWarning)]

    # Autosomal TP53: 2 in diploid and 4 in WGD are both copy-neutral (False).
    assert binary.loc["TP53"].tolist() == [False, False, False]
    # chrX DMD: neutral is 1 (male diploid), 2 (female diploid), 2 (male WGD).
    assert binary.loc["DMD"].tolist() == [False, False, False]
    # chrY SRY: neutral is 1 (male diploid), 0 (female diploid), 2 (male WGD).
    assert binary.loc["SRY"].tolist() == [False, False, False]
    # Autosomal MYC: 0 (loss), 1 (loss), 6 (gain in WGD) are all altered (True).
    assert binary.loc["MYC"].tolist() == [True, True, True]

    # Mutations on chrX/chrY/WGD are still flagged when deviating from baseline.
    altered = cnv.copy()
    altered.loc["DMD", "male_diploid"] = 0  # chrX hemizygous deletion
    altered.loc["DMD", "female_diploid"] = 1  # chrX heterozygous loss
    altered.loc["SRY", "female_diploid"] = 1  # unexpected chrY copy in female
    altered.loc["TP53", "male_wgd"] = 2  # relative loss in tetraploid tumor
    binary_altered = altered.to_binary_table(
        ploidy={"male_diploid": 2.0, "female_diploid": 2.0, "male_wgd": 4.0},
        sex={
            "male_diploid": "male",
            "female_diploid": "female",
            "male_wgd": "male",
        },
    )
    assert binary_altered.loc["DMD", "male_diploid"]
    assert binary_altered.loc["DMD", "female_diploid"]
    assert binary_altered.loc["SRY", "female_diploid"]
    assert binary_altered.loc["TP53", "male_wgd"]


def test_cnv_absolute_cn_warns_without_ploidy_or_sex_and_validates_inputs():
    cnv = CopyNumberVariationTable(
        {"S1": [4, 4, 4, 2], "S2": [2, 2, 1, 1]},
        index=["TP53", "EGFR", "DMD", "SRY"],
        value_scale="absolute_cn",
    )
    cnv.feature_metadata["Chromosome"] = ["17", "7", "X", "Y"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cnv.to_binary_table()
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("ploidy" in msg for msg in messages)
    assert any("sex-chromosome" in msg for msg in messages)

    with pytest.raises(ValueError, match="ploidy must be positive"):
        cnv.to_binary_table(ploidy={"S1": 0, "S2": 2})

    with pytest.raises(ValueError, match="missing for sample"):
        cnv.to_binary_table(ploidy={"S1": 2})

    with pytest.raises(ValueError, match="Unrecognized sex value"):
        cnv.to_binary_table(sex={"S1": "unknown", "S2": "male"})

    no_chrom = CopyNumberVariationTable(
        {"S1": [2], "S2": [2]}, index=["TP53"], value_scale="absolute_cn"
    )
    with pytest.raises(ValueError, match="chrom_col"):
        no_chrom.to_binary_table(sex={"S1": "male", "S2": "female"})
