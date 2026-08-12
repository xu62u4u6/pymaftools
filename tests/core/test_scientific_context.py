"""Scientific-context tests for explicit samples and observation states."""

import pandas as pd
import pytest

from pymaftools import (
    Cohort,
    MAF,
    ObservationMask,
    PivotTable,
    SampleManifest,
    TMBAudit,
)


def _manifest() -> SampleManifest:
    return SampleManifest(
        pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "eligible": [True, True, False],
                "exclusion_reason": [pd.NA, pd.NA, "failed_qc"],
                "assay": ["WES", "WES", "WES"],
            },
            index=["S1", "S2", "S3"],
        )
    )


def _maf(sample_id: str = "S1") -> MAF:
    frame = pd.DataFrame(
        {
            "Hugo_Symbol": ["TP53"],
            "Start_Position": [100],
            "End_Position": [100],
            "Reference_Allele": ["C"],
            "Tumor_Seq_Allele1": ["C"],
            "Tumor_Seq_Allele2": ["T"],
            "Variant_Classification": ["Missense_Mutation"],
            "sample_ID": [sample_id],
        }
    )
    frame.index = frame.loc[:, MAF.index_col].apply(
        lambda row: "|".join(row.astype(str)), axis=1
    )
    return MAF(frame)


def test_sample_manifest_adds_eligible_zero_event_samples():
    table = _maf().to_gene_table(sample_manifest=_manifest())

    assert list(table.columns) == ["S1", "S2"]
    assert table.loc["TP53", "S1"] == "Missense_Mutation"
    assert table.loc["TP53", "S2"] == False  # noqa: E712
    assert table.sample_metadata["mutations_count"].to_dict() == {"S1": 1, "S2": 0}
    assert table.sample_metadata["patient_id"].to_dict() == {"S1": "P1", "S2": "P2"}


def test_sample_manifest_rejects_unknown_and_ineligible_event_samples():
    with pytest.raises(ValueError, match="absent from SampleManifest"):
        _maf("S4").to_gene_table(sample_manifest=_manifest())

    with pytest.raises(ValueError, match="marked ineligible"):
        _maf("S3").to_gene_table(sample_manifest=_manifest())


def test_sample_manifest_exposes_repeated_patient_counts():
    manifest = SampleManifest(
        pd.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2"],
                "eligible": [True, True, True],
            },
            index=["S1", "S1-repeat", "S2"],
        )
    )

    assert manifest.patient_counts.to_dict() == {"P1": 2, "P2": 1}
    with pytest.raises(ValueError, match="independent observations"):
        manifest.assert_independent(analysis_unit="patient")

    manifest.assert_independent(analysis_unit="sample")


def test_observation_mask_excludes_unobserved_cells_from_denominator():
    table = PivotTable(
        [["Missense_Mutation", False, False]],
        index=["TP53"],
        columns=["S1", "S2", "S3"],
    )
    mask = ObservationMask(
        pd.DataFrame(
            [[True, True, False]],
            index=table.index,
            columns=table.columns,
        )
    )

    observed_frequency = table.calculate_feature_frequency(observation_mask=mask)
    legacy_frequency = table.calculate_feature_frequency()

    assert observed_frequency.loc["TP53"] == pytest.approx(0.5)
    assert legacy_frequency.loc["TP53"] == pytest.approx(1 / 3)


def test_observation_mask_rejects_present_event_marked_unobserved():
    table = PivotTable([[True]], index=["TP53"], columns=["S1"])
    mask = ObservationMask(
        pd.DataFrame([[False]], index=table.index, columns=table.columns)
    )

    with pytest.raises(ValueError, match="present event"):
        table.calculate_feature_frequency(observation_mask=mask)


def test_add_freq_applies_group_specific_observation_denominators():
    table = PivotTable(
        [[True, False, False, False]],
        index=["TP53"],
        columns=["A1", "A2", "B1", "B2"],
    )
    table.sample_metadata["group"] = ["A", "A", "B", "B"]
    mask = ObservationMask(
        pd.DataFrame(
            [[True, True, True, False]],
            index=table.index,
            columns=table.columns,
        )
    )

    result = table.add_freq(group_col="group", observation_mask=mask)

    assert result.feature_metadata.loc["TP53", "A_freq"] == pytest.approx(0.5)
    assert result.feature_metadata.loc["TP53", "B_freq"] == pytest.approx(0.0)
    assert result.feature_metadata.loc["TP53", "freq"] == pytest.approx(1 / 3)


def _context_table() -> PivotTable:
    table = PivotTable(
        [[True, False], [False, False]],
        index=["TP53", "KRAS"],
        columns=["S1", "S2"],
    )
    mask = ObservationMask(
        pd.DataFrame(
            [[True, True], [True, False]],
            index=table.index,
            columns=table.columns,
        )
    )
    return table.with_scientific_context(
        sample_manifest=_manifest(),
        observation_mask=mask,
    )


def test_attached_context_drives_frequency_and_survives_table_operations():
    table = _context_table()

    assert table.calculate_feature_frequency().to_dict() == {
        "TP53": 0.5,
        "KRAS": 0.0,
    }

    subset = table.subset(features=["KRAS"], samples=["S2"])
    assert not subset.observation_mask.to_frame().loc["KRAS", "S2"]
    assert list(subset.sample_manifest.samples) == ["S1", "S2", "S3"]

    reindexed = table.reindex(index=["TP53", "NEW"], fill_value=False)
    assert not reindexed.observation_mask.to_frame().loc["NEW"].any()
    assert pd.isna(reindexed.calculate_feature_frequency().loc["NEW"])

    binary = table.to_binary_table()
    assert binary.observation_mask.to_frame().equals(
        table.observation_mask.to_frame()
    )
    assert list(binary.sample_manifest.samples) == ["S1", "S2", "S3"]


def test_context_rejects_unknown_reindexed_sample():
    table = _context_table()

    with pytest.raises(ValueError, match="eligible SampleManifest universe"):
        table.reindex(columns=["S1", "UNKNOWN"])


def test_context_round_trips_through_pivot_hdf5(tmp_path):
    path = tmp_path / "context.h5"
    _context_table().to_h5(path)

    restored = PivotTable.read_h5(path)

    assert restored.observation_mask.to_frame().equals(
        _context_table().observation_mask.to_frame()
    )
    assert restored.sample_manifest.to_frame().equals(_manifest().to_frame())
    assert restored.calculate_feature_frequency().loc["TP53"] == pytest.approx(0.5)


def test_context_round_trips_through_cohort_hdf5(tmp_path):
    path = tmp_path / "cohort-context.h5"
    cohort = Cohort("context")
    cohort.add_table(_context_table(), "mutation")
    cohort.to_hdf5(path)

    restored = Cohort.read_hdf5(path).mutation

    assert restored.observation_mask.to_frame().equals(
        _context_table().observation_mask.to_frame()
    )
    assert restored.sample_manifest.to_frame().equals(_manifest().to_frame())


def test_merge_rejects_partial_scientific_context():
    contextual = _context_table().subset(samples=["S1"])
    legacy = PivotTable([[False], [False]], index=contextual.index, columns=["S4"])

    with pytest.raises(ValueError, match="mixed ObservationMask"):
        PivotTable.merge([contextual, legacy])


def test_enrichment_reports_observed_denominators_effect_and_test_family():
    table = PivotTable(
        [[True, True, False, True, False, False], [True, False, False, False, False, False]],
        index=["G1", "G2"],
        columns=["A1", "A2", "A3", "B1", "B2", "B3"],
    )
    table.sample_metadata["group"] = ["A", "A", "A", "B", "B", "B"]
    manifest = SampleManifest(
        pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
                "eligible": True,
            },
            index=table.columns,
        )
    )
    mask = ObservationMask(
        pd.DataFrame(
            [[True, True, True, True, True, False], [True] * 6],
            index=table.index,
            columns=table.columns,
        )
    )
    table = table.with_scientific_context(
        sample_manifest=manifest,
        observation_mask=mask,
    )

    result = table.mutation_enrichment_test(
        "group",
        "A",
        "B",
        minimum_mutations=2,
        analysis_unit="patient",
    )

    assert result.loc["G1", ["A_True", "A_False"]].tolist() == [2, 1]
    assert result.loc["G1", ["B_True", "B_False"]].tolist() == [1, 1]
    assert result.loc["G1", ["A_denominator", "B_denominator"]].tolist() == [3, 2]
    assert result.loc["G1", "odds_ratio"] == pytest.approx(2.0)
    assert result.loc["G1", "ci_low"] < 2 < result.loc["G1", "ci_high"]
    assert bool(result.loc["G1", "tested"])
    assert not bool(result.loc["G2", "tested"])
    assert pd.isna(result.loc["G2", "p_value"])
    assert result.loc["G1", "test_method"] == "fisher"
    assert result.loc["G1", "analysis_unit"] == "patient"


def test_patient_enrichment_rejects_repeated_eligible_patient():
    table = PivotTable(
        [[True, False, True, False]],
        index=["TP53"],
        columns=["A1", "A2", "B1", "B2"],
    )
    table.sample_metadata["group"] = ["A", "A", "B", "B"]
    manifest = SampleManifest(
        pd.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P3"],
                "eligible": True,
            },
            index=table.columns,
        )
    )
    table = table.with_scientific_context(sample_manifest=manifest)

    with pytest.raises(ValueError, match="independent observations"):
        table.mutation_enrichment_test(
            "group", "A", "B", analysis_unit="patient"
        )


def test_enrichment_rejects_omitted_eligible_zero_event_samples():
    table = PivotTable([[True, False]], index=["TP53"], columns=["A1", "B1"])
    table.sample_metadata["group"] = ["A", "B"]
    manifest = SampleManifest(
        pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3", "P4"],
                "eligible": True,
            },
            index=["A1", "A2-zero", "B1", "B2-zero"],
        )
    )
    table = table.with_scientific_context(sample_manifest=manifest)

    with pytest.raises(ValueError, match="every eligible SampleManifest sample"):
        table.mutation_enrichment_test("group", "A", "B", minimum_mutations=0)


def test_enrichment_rejects_identical_groups():
    table = PivotTable([[True, False]], index=["TP53"], columns=["S1", "S2"])
    table.sample_metadata["group"] = ["A", "B"]

    with pytest.raises(ValueError, match="different groups"):
        table.mutation_enrichment_test("group", "A", "A")


def test_tmb_audit_keeps_zero_event_samples_and_deduplicates():
    manifest = SampleManifest(
        pd.DataFrame(
            {
                "patient_id": ["P1", "P2"],
                "eligible": [True, True],
                "callable_mb": [40.0, 30.0],
            },
            index=["S1", "S2"],
        )
    )
    maf = _maf()
    maf = MAF(pd.concat([maf, maf], ignore_index=True))

    audit = maf.calculate_tmb_audit(
        manifest,
        variant_classifications=MAF.nonsynonymous_types,
    )

    assert isinstance(audit, TMBAudit)
    assert audit.summary["mutation_count"].to_dict() == {"S1": 1, "S2": 0}
    assert audit.summary["TMB"].to_dict() == {"S1": 0.025, "S2": 0.0}
    assert audit.exclusion_counts().to_dict() == {"duplicate_event": 1}


def test_tmb_audit_records_filter_reasons_and_build():
    manifest = SampleManifest(
        pd.DataFrame(
            {
                "patient_id": ["P1"],
                "eligible": [True],
                "callable_mb": [40.0],
            },
            index=["S1"],
        )
    )
    maf = MAF(
        pd.concat(
            [
                _maf(),
                _maf().assign(
                    Hugo_Symbol="KRAS",
                    Start_Position=200,
                    End_Position=200,
                    FILTER="FAIL",
                    Somatic_Status="Somatic",
                    NCBI_Build="GRCh38",
                ),
                _maf().assign(
                    Hugo_Symbol="EGFR",
                    Start_Position=300,
                    End_Position=300,
                    FILTER="PASS",
                    Somatic_Status="Germline",
                    NCBI_Build="GRCh38",
                ),
            ],
            ignore_index=True,
        )
    )
    maf.loc[0, ["FILTER", "Somatic_Status", "NCBI_Build"]] = [
        "PASS",
        "Somatic",
        "GRCh38",
    ]

    audit = maf.calculate_tmb_audit(
        manifest,
        pass_col="FILTER",
        somatic_col="Somatic_Status",
        genome_build_col="NCBI_Build",
        expected_genome_build="GRCh38",
        variant_classifications=MAF.nonsynonymous_types,
    )

    assert audit.summary.loc["S1", "mutation_count"] == 1
    assert audit.exclusion_counts().to_dict() == {
        "non_somatic": 1,
        "quality_filter": 1,
    }


def test_tmb_audit_requires_positive_callable_territory():
    manifest = SampleManifest(
        pd.DataFrame(
            {
                "patient_id": ["P1"],
                "eligible": [True],
                "callable_mb": [0.0],
            },
            index=["S1"],
        )
    )

    with pytest.raises(ValueError, match="positive callable territory"):
        _maf().calculate_tmb_audit(manifest)
