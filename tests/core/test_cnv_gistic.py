"""GISTIC readers: sample-name parsing and arm coverage."""

import pandas as pd

from pymaftools.core.CopyNumberVariationTable import CopyNumberVariationTable


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
        {"Chromosome Arm": ["2q", "1p", "Yq", "Xp", "Xq"], "S1": [0.1, -0.2, -0.4, 0.3, 0.0]}
    ).to_csv(path, sep="\t", index=False)

    table = CopyNumberVariationTable.read_gistic_arm_level(path)

    assert list(table.index) == ["1p", "2q", "Xp", "Xq", "Yq"]
    assert table.loc["Yq", "S1"] == -0.4
    assert table.feature_metadata["Chromosome"].tolist() == ["1", "2", "X", "X", "Y"]


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
