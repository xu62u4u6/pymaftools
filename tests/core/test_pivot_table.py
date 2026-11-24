"""
Tests for PivotTable core functionality
"""

import sys
import types
import warnings
from typing import get_type_hints

import pytest
import pandas as pd
import numpy as np
import sqlite3
import pymaftools
from typing_extensions import Self
from pymaftools.core.PivotTable import PivotTable
from pymaftools.core.CopyNumberVariationTable import CopyNumberVariationTable
from pymaftools.core.SmallVariationTable import SmallVariationTable
from pymaftools.core.ObservationMask import ObservationMask
from pymaftools.core.SampleManifest import SampleManifest


class TestPivotTableBasics:
    """Test basic PivotTable functionality"""

    def test_subset_preserves_self_return_annotation(self):
        assert get_type_hints(PivotTable.subset)["return"] is Self
    
    def test_pivot_table_creation(self, sample_mutation_data):
        """Test PivotTable creation from DataFrame"""
        table = PivotTable(sample_mutation_data)
        
        assert isinstance(table, PivotTable)
        assert table.shape == sample_mutation_data.shape
        assert list(table.index) == list(sample_mutation_data.index)
        assert list(table.columns) == list(sample_mutation_data.columns)
    
    def test_metadata_initialization(self, sample_pivot_table):
        """Test that metadata is properly initialized"""
        table = sample_pivot_table
        
        # Check sample_metadata
        assert table.sample_metadata.shape[0] == table.shape[1]
        assert list(table.sample_metadata.index) == list(table.columns)
        assert 'subtype' in table.sample_metadata.columns
        assert 'age' in table.sample_metadata.columns
        
        # Check feature_metadata
        assert table.feature_metadata.shape[0] == table.shape[0]
        assert list(table.feature_metadata.index) == list(table.index)
        assert 'chromosome' in table.feature_metadata.columns
        assert 'gene_type' in table.feature_metadata.columns
    
    def test_pivot_table_copy_preserves_metadata(self, sample_pivot_table):
        """Test that copying preserves metadata"""
        original = sample_pivot_table
        copied = PivotTable(original)
        
        assert copied.sample_metadata.equals(original.sample_metadata)
        assert copied.feature_metadata.equals(original.feature_metadata)
        assert copied.equals(original)

    def test_to_h5_read_h5_roundtrip(self, sample_pivot_table, tmp_path):
        """Single-table HDF5 IO preserves data and both metadata tables."""
        h5_path = tmp_path / "pivot_table.h5"

        sample_pivot_table.to_h5(h5_path)
        loaded = PivotTable.read_h5(h5_path)
        loaded_via_package = pymaftools.read_h5(h5_path)

        assert isinstance(loaded, PivotTable)
        assert loaded.equals(sample_pivot_table)
        assert loaded.sample_metadata.equals(sample_pivot_table.sample_metadata)
        assert loaded.feature_metadata.equals(sample_pivot_table.feature_metadata)
        assert loaded_via_package.equals(sample_pivot_table)

    def test_read_h5_infers_registered_subclass(self, sample_pivot_table, tmp_path):
        """Single-table HDF5 IO restores registered PivotTable subclasses."""
        h5_path = tmp_path / "small_variation_table.h5"
        table = SmallVariationTable(sample_pivot_table)

        table.to_h5(h5_path)
        loaded = pymaftools.read_h5(h5_path)

        assert isinstance(loaded, SmallVariationTable)
        assert loaded.equals(table)
        assert loaded.sample_metadata.equals(table.sample_metadata)
        assert loaded.feature_metadata.equals(table.feature_metadata)

    def test_sqlite_write_failure_preserves_existing_file(
        self, sample_pivot_table, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "pivot.db"
        db_path.write_bytes(b"existing database")

        def fail_to_sql(*args, **kwargs):
            raise RuntimeError("injected write failure")

        monkeypatch.setattr(pd.DataFrame, "to_sql", fail_to_sql)

        with pytest.warns(DeprecationWarning, match="to_h5"):
            with pytest.raises(RuntimeError, match="injected write failure"):
                sample_pivot_table.to_sqlite(db_path)

        assert db_path.read_bytes() == b"existing database"
        assert list(tmp_path.iterdir()) == [db_path]

    def test_sqlite_connections_are_closed(
        self, sample_pivot_table, tmp_path, monkeypatch
    ):
        from pymaftools.core import pivot_io

        connections = []
        original_connect = sqlite3.connect

        class TrackingConnection(sqlite3.Connection):
            was_closed = False

            def close(self):
                self.was_closed = True
                super().close()

        def tracked_connect(*args, **kwargs):
            kwargs["factory"] = TrackingConnection
            connection = original_connect(*args, **kwargs)
            connections.append(connection)
            return connection

        monkeypatch.setattr(pivot_io.sqlite3, "connect", tracked_connect)
        db_path = tmp_path / "pivot.db"

        with pytest.warns(DeprecationWarning, match="to_h5"):
            sample_pivot_table.to_sqlite(db_path)
        with pytest.warns(DeprecationWarning, match="read_h5"):
            PivotTable.read_sqlite(db_path)

        assert len(connections) == 2
        assert all(connection.was_closed for connection in connections)

    def test_sqlite_preserves_numeric_zeros_booleans_and_legacy_wt(self, tmp_path):
        from pymaftools.core.CopyNumberVariationTable import CopyNumberVariationTable

        # Numeric zeros (int and float) must never be coerced to False.
        numeric = PivotTable(
            pd.DataFrame({"s1": [0, 1], "s2": [2.0, 0.0]}, index=["g1", "g2"])
        )
        numeric_db = tmp_path / "numeric.db"
        with pytest.warns(DeprecationWarning, match="to_h5"):
            numeric.to_sqlite(str(numeric_db))
        with pytest.warns(DeprecationWarning, match="read_h5"):
            restored_num = PivotTable.read_sqlite(str(numeric_db))
        assert restored_num.loc["g1", "s1"] == 0
        assert not isinstance(restored_num.loc["g1", "s1"], (bool, np.bool_))
        assert restored_num.loc["g2", "s2"] == pytest.approx(0.0)
        assert not isinstance(restored_num.loc["g2", "s2"], (bool, np.bool_))

        # Object columns containing numeric values stay numeric/non-False;
        # true booleans in mixed or bool columns round-trip cleanly.
        mixed = PivotTable(
            pd.DataFrame(
                {
                    "snv": pd.Series(
                        [False, "Missense_Mutation", True],
                        index=["g1", "g2", "g3"],
                        dtype=object,
                    ),
                    "pure_bool": [False, True, False],
                    "obj_num": pd.Series(
                        [2.0, 0, 1], index=["g1", "g2", "g3"], dtype=object
                    ),
                },
                index=["g1", "g2", "g3"],
            )
        )
        mixed_db = tmp_path / "mixed.db"
        with pytest.warns(DeprecationWarning, match="to_h5"):
            mixed.to_sqlite(str(mixed_db))
        with pytest.warns(DeprecationWarning, match="read_h5"):
            restored_mixed = PivotTable.read_sqlite(str(mixed_db))

        assert restored_mixed.loc["g1", "snv"] is False
        assert restored_mixed.loc["g2", "snv"] == "Missense_Mutation"
        assert restored_mixed.loc["g3", "snv"] is True
        assert restored_mixed.dtypes["pure_bool"] == bool
        assert pd.DataFrame(restored_mixed)["pure_bool"].tolist() == [False, True, False]
        assert pd.DataFrame(restored_mixed)["obj_num"].tolist() == [2.0, 0.0, 1.0]

        # Object columns mixing numbers with strings or booleans cannot round-trip
        # through SQLite without silently coercing 0.0 -> "0.0"; fail loud and
        # point to to_h5(), which preserves exact Python types.
        for bad_col_name, bad_values in [
            ("obj_mixed", [0.0, "x", 2.0]),
            ("num_bool", [0.0, False, 1.0]),
        ]:
            bad_table = PivotTable(
                pd.DataFrame(
                    {bad_col_name: pd.Series(bad_values, index=["g1", "g2", "g3"], dtype=object)},
                    index=["g1", "g2", "g3"],
                )
            )
            bad_db = tmp_path / f"{bad_col_name}.db"
            with pytest.warns(DeprecationWarning, match="to_h5"):
                with pytest.raises(ValueError, match=f"{bad_col_name}.*to_h5"):
                    bad_table.to_sqlite(str(bad_db))
            assert not bad_db.exists()

        # HDF5 preserves mixed numeric + string object columns without coercion.
        h5_mixed = PivotTable(
            pd.DataFrame(
                {"s1": pd.Series([0.0, "x", 2.0], index=["g1", "g2", "g3"], dtype=object)},
                index=["g1", "g2", "g3"],
            )
        )
        h5_path = tmp_path / "mixed.h5"
        h5_mixed.to_h5(h5_path)
        restored_h5 = PivotTable.read_h5(h5_path)
        assert restored_h5.loc["g1", "s1"] == 0.0
        assert isinstance(restored_h5.loc["g1", "s1"], float)
        assert restored_h5.loc["g2", "s1"] == "x"
        assert restored_h5.loc["g3", "s1"] == 2.0

        # Legacy SQLite files encoding False as "WT" still decode to False.
        from contextlib import closing

        legacy_db = tmp_path / "legacy.db"
        with closing(sqlite3.connect(str(legacy_db))) as conn:
            with conn:
                pd.DataFrame(
                    {"s1": ["WT", "Missense_Mutation"]},
                    index=pd.Index(["g1", "g2"], name="feature"),
                ).to_sql("data", conn, index=True)
                pd.DataFrame(index=pd.Index(["s1"], name="sample")).to_sql(
                    "sample_metadata", conn, index=True
                )
                pd.DataFrame(index=pd.Index(["g1", "g2"], name="feature")).to_sql(
                    "feature_metadata", conn, index=True
                )
        with pytest.warns(DeprecationWarning, match="read_h5"):
            legacy_table = PivotTable.read_sqlite(str(legacy_db))
        assert legacy_table.loc["g1", "s1"] is False
        assert legacy_table.loc["g2", "s1"] == "Missense_Mutation"

        # Warn loudly if unpersisted attributes (like value_scale) are attached.
        cnv = CopyNumberVariationTable(
            {"s1": [0, 2]}, index=["g1", "g2"], value_scale="absolute_cn"
        )
        import warnings

        cnv_db = tmp_path / "cnv.db"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cnv.to_sqlite(str(cnv_db))
        assert any(
            issubclass(w.category, DeprecationWarning) and "to_h5" in str(w.message)
            for w in caught
        )
        assert any(
            issubclass(w.category, UserWarning)
            and "value_scale" in str(w.message)
            and "to_h5" in str(w.message)
            for w in caught
        )
    
    def test_subset_functionality(self, sample_pivot_table):
        """Test subset method"""
        table = sample_pivot_table
        
        # Test feature subset
        subset_features = ['TP53', 'KRAS']
        feature_subset = table.subset(features=subset_features)
        
        assert list(feature_subset.index) == subset_features
        assert feature_subset.feature_metadata.shape[0] == len(subset_features)
        assert list(feature_subset.feature_metadata.index) == subset_features
        
        # Test sample subset
        subset_samples = ['sample1', 'sample2']
        sample_subset = table.subset(samples=subset_samples)
        
        assert list(sample_subset.columns) == subset_samples
        assert sample_subset.sample_metadata.shape[0] == len(subset_samples)
        assert list(sample_subset.sample_metadata.index) == subset_samples
        
        # Test combined subset
        combined_subset = table.subset(features=subset_features, samples=subset_samples)
        
        assert combined_subset.shape == (len(subset_features), len(subset_samples))
        assert list(combined_subset.index) == subset_features
        assert list(combined_subset.columns) == subset_samples


class TestPivotTableFrequencyCalculations:
    """Test frequency calculation methods"""
    
    def test_calculate_feature_frequency(self, sample_pivot_table):
        """Test frequency calculation"""
        table = sample_pivot_table
        frequencies = table.calculate_feature_frequency()
        
        assert isinstance(frequencies, pd.Series)
        assert len(frequencies) == table.shape[0]
        assert all(0 <= freq <= 1 for freq in frequencies)
        
        # Check specific frequencies
        # TP53: [True, False, True, False] -> 2/4 = 0.5
        assert frequencies['TP53'] == 0.5
        # KRAS: [False, True, True, False] -> 2/4 = 0.5  
        assert frequencies['KRAS'] == 0.5
    
    def test_add_freq_method(self, sample_pivot_table):
        """Test add_freq writes exact feature frequencies into feature_metadata."""
        table = sample_pivot_table
        table_with_freq = table.add_freq()

        assert "freq" in table_with_freq.feature_metadata.columns
        pd.testing.assert_series_equal(
            table_with_freq.feature_metadata["freq"],
            table.calculate_feature_frequency(),
            check_names=False,
        )

    def test_add_freq_raises_on_drifted_feature_metadata_index(self, sample_pivot_table):
        """add_freq must fail loud, not produce a silently all-NaN freq column.

        Setting ``table.index`` directly relabels the data but leaves
        feature_metadata.index stale; the old label-aligned assignment then
        wrote freq as all-NaN with no error (the Fig5A freq-disappears bug).
        """
        table = sample_pivot_table
        table.index = [f"renamed_{g}" for g in table.index]  # drift only data index

        with pytest.raises(ValueError, match="feature_metadata index"):
            table.add_freq()

    def test_filter_by_freq(self, sample_pivot_table):
        """Test frequency-based filtering actually drops below-threshold features."""
        table = sample_pivot_table.add_freq()

        # TP53=0.5, KRAS=0.5, EGFR=0.75, PIK3CA=0.5 -> only EGFR passes >= 0.6.
        filtered = table.filter_by_freq(threshold=0.6)

        assert list(filtered.index) == ["EGFR"]
        assert list(filtered.feature_metadata.index) == ["EGFR"]
        assert filtered.feature_metadata.loc["EGFR", "freq"] == pytest.approx(0.75)


class TestPivotTableSorting:
    """Test sorting functionality"""

    def test_sort_features(self, sample_pivot_table):
        """Test feature sorting by metadata"""
        table = sample_pivot_table.add_freq()

        # Sort by frequency (descending)
        sorted_table = table.sort_features(by="freq", ascending=False)

        assert sorted_table.index[0] == "EGFR"
        freq_values = sorted_table.feature_metadata["freq"].values
        assert all(
            freq_values[i] >= freq_values[i + 1] for i in range(len(freq_values) - 1)
        )

    def test_sort_samples_by_group(self, sample_pivot_table):
        """Test sample sorting by group preserves exact group order."""
        table = sample_pivot_table

        group_order = ["LUAD", "ASC", "LUSC"]
        sorted_table = table.sort_samples_by_group(
            group_col="subtype",
            group_order=group_order,
            top=2,
        )

        subtypes = sorted_table.sample_metadata["subtype"].tolist()
        assert subtypes == ["LUAD", "LUAD", "ASC", "LUSC"]
        assert set(sorted_table.columns) == set(table.columns)

    def test_sort_samples_by_group_preserves_unlisted_and_missing_groups(
        self, sample_pivot_table
    ):
        table = sample_pivot_table.copy()
        table.sample_metadata.loc[table.columns[-2], "subtype"] = "OTHER"
        table.sample_metadata.loc[table.columns[-1], "subtype"] = np.nan

        sorted_table = table.sort_samples_by_group(
            group_col="subtype",
            group_order=["LUAD"],
        )

        assert set(sorted_table.columns) == set(table.columns)
        assert sorted_table.shape == table.shape
        assert list(sorted_table.columns[-2:]) == list(table.columns[-2:])

    def test_order_preserves_unlisted_groups_and_validates_inputs(
        self, sample_pivot_table
    ):
        table = sample_pivot_table.copy()
        table.sample_metadata.loc[table.columns[-1], "subtype"] = "OTHER"

        ordered = table.order("subtype", ["LUSC", "LUAD"])

        assert ordered.columns[-1] == table.columns[-1]
        assert type(ordered) is type(table)
        with pytest.raises(ValueError, match="not found"):
            table.order("missing", ["LUAD"])
        with pytest.raises(ValueError, match="duplicate"):
            table.order("subtype", ["LUAD", "LUAD"])


class TestPivotTableStatistics:
    """Test statistical methods"""

    def test_to_binary_table(self):
        """Test conversion of mixed string/False/None mutation calls to boolean."""
        table = PivotTable(
            pd.DataFrame(
                {
                    "S1": ["Missense_Mutation", False, None],
                    "S2": ["Multi_Hit", "Nonsense_Mutation", False],
                },
                index=["TP53", "KRAS", "EGFR"],
            )
        )
        binary_table = table.to_binary_table()

        assert all(dtype == bool for dtype in binary_table.dtypes)
        expected = pd.DataFrame(
            {"S1": [True, False, False], "S2": [True, True, False]},
            index=["TP53", "KRAS", "EGFR"],
        )
        pd.testing.assert_frame_equal(pd.DataFrame(binary_table), expected)

    def test_to_binary_table_treats_missing_values_as_absent(self, sample_pivot_table):
        """Missing observations must not be counted as mutations."""
        table = sample_pivot_table.astype(object)
        table.iloc[0, 0] = np.nan

        binary_table = table.to_binary_table()

        assert binary_table.iloc[0, 0] == False  # noqa: E712
        assert binary_table.to_numpy().dtype == bool
        assert binary_table.sample_metadata.equals(table.sample_metadata)
        assert binary_table.feature_metadata.equals(table.feature_metadata)

    def test_to_binary_table_requires_threshold_for_continuous_data(self):
        table = PivotTable([[0.2, -1.5], [2.0, np.nan]])

        with pytest.raises(ValueError, match="explicit threshold"):
            table.to_binary_table()

        binary = table.to_binary_table(threshold=1.0, comparison="abs_gt")

        expected = pd.DataFrame([[False, True], [True, False]])
        assert binary.equals(expected)

    def test_to_binary_table_rejects_threshold_for_categorical_data(self):
        table = PivotTable([["AMP", False]])

        with pytest.raises(ValueError, match="fully numeric"):
            table.to_binary_table(threshold=1.0)

    def test_cnv_binary_conversion_handles_discrete_and_continuous_values(self):
        discrete = CopyNumberVariationTable([[-2, 0, 1]])
        assert discrete.to_binary_table().equals(
            pd.DataFrame([[True, False, True]])
        )

        continuous = CopyNumberVariationTable([[-0.4, 0.1, 0.8]])
        with pytest.raises(ValueError, match="explicit threshold"):
            continuous.to_binary_table()
        assert continuous.to_binary_table(threshold=0.3).equals(
            pd.DataFrame([[True, False, True]])
        )

    def test_PCA_calculation(self, sample_pivot_table):
        """Test PCA coordinates, sample alignment, and variance ordering."""
        table = sample_pivot_table

        pca_df, explained_var, pca_obj = table.pca(to_binary=True)

        assert isinstance(pca_df, pd.DataFrame)
        assert pca_df.shape == (table.shape[1], 2)
        assert pca_df.index.equals(table.columns)
        assert list(pca_df.columns[:2]) == ["PC1", "PC2"]
        assert np.all(np.isfinite(pca_df.values))
        assert len(explained_var) == 2
        assert explained_var[0] >= explained_var[1] >= 0.0
        assert float(np.sum(explained_var)) <= 1.0 + 1e-12
        np.testing.assert_allclose(explained_var, pca_obj.explained_variance_ratio_)

    def test_compute_similarity(self, sample_pivot_table):
        """Test exact cosine similarity values, symmetry, and unit diagonal."""
        table = sample_pivot_table

        similarity_matrix = table.compute_similarity(method="cosine")

        assert similarity_matrix.shape == (table.shape[1], table.shape[1])
        assert similarity_matrix.index.equals(table.columns)
        assert similarity_matrix.columns.equals(table.columns)

        np.testing.assert_allclose(
            np.diag(similarity_matrix.values), np.ones(table.shape[1])
        )
        np.testing.assert_allclose(
            similarity_matrix.values, similarity_matrix.values.T, atol=1e-12
        )
        assert similarity_matrix.loc["sample1", "sample2"] == pytest.approx(0.5)
        assert similarity_matrix.loc["sample1", "sample4"] == pytest.approx(0.5)
        assert similarity_matrix.loc["sample1", "sample3"] == pytest.approx(
            1.0 / np.sqrt(6.0)
        )
        assert similarity_matrix.loc["sample3", "sample4"] == pytest.approx(
            1.0 / np.sqrt(6.0)
        )


class TestPivotTableValidation:
    """Test validation and error handling"""
    
    def test_metadata_validation(self, sample_mutation_data):
        """Test metadata validation"""
        table = PivotTable(sample_mutation_data)
        
        # Add mismatched metadata (should raise error)
        wrong_metadata = pd.DataFrame(
            {'wrong_col': [1, 2]}, 
            index=['wrong1', 'wrong2']
        )
        
        with pytest.raises(ValueError, match="sample_metadata index does not match"):
            table.sample_metadata = wrong_metadata
            table._validate_metadata()
    
    def test_invalid_subset_parameters(self, sample_pivot_table):
        """Test handling of invalid subset parameters"""
        table = sample_pivot_table
        
        # Test with non-existent features - should raise KeyError
        with pytest.raises(KeyError):
            table.subset(features=['nonexistent_gene'])
        
        # Test with non-existent samples - should raise KeyError
        with pytest.raises(KeyError):
            table.subset(samples=['nonexistent_sample'])
    
    def test_empty_table_handling(self):
        """Test handling of empty tables"""
        empty_data = pd.DataFrame()
        
        # Empty DataFrame should create empty PivotTable without error
        table = PivotTable(empty_data)
        assert table.shape == (0, 0)
        assert len(table.sample_metadata) == 0


class TestAnnDataInterop:
    """Test AnnData interoperability methods"""

    def test_to_anndata_basic(self, sample_pivot_table):
        """Test basic PivotTable to AnnData conversion"""
        anndata = pytest.importorskip("anndata")
        adata = sample_pivot_table.to_anndata()

        assert isinstance(adata, anndata.AnnData)
        # AnnData is (samples, features), PivotTable is (features, samples)
        assert adata.shape == (sample_pivot_table.shape[1], sample_pivot_table.shape[0])
        assert list(adata.obs_names) == list(sample_pivot_table.columns)
        assert list(adata.var_names) == list(sample_pivot_table.index)

    def test_to_anndata_preserves_metadata(self, sample_pivot_table):
        """Test that metadata is preserved in AnnData conversion"""
        pytest.importorskip("anndata")
        adata = sample_pivot_table.to_anndata()

        # obs should contain sample_metadata
        assert "subtype" in adata.obs.columns
        assert "age" in adata.obs.columns
        assert list(adata.obs["subtype"]) == list(sample_pivot_table.sample_metadata["subtype"])

        # var should contain feature_metadata
        assert "chromosome" in adata.var.columns
        assert "gene_type" in adata.var.columns

    def test_roundtrip_numeric(self):
        """Test numeric data roundtrip preserves values"""
        pytest.importorskip("anndata")
        data = pd.DataFrame(
            {"s1": [1.5, 2.3, 0.0], "s2": [3.1, 0.0, 1.7]},
            index=["geneA", "geneB", "geneC"],
        )
        table = PivotTable(data)
        table.sample_metadata["group"] = ["A", "B"]
        table.feature_metadata["chr"] = ["1", "2", "3"]

        table2 = PivotTable.from_anndata(table.to_anndata())

        assert np.allclose(table.values, table2.values)
        assert table.sample_metadata.equals(table2.sample_metadata)
        assert table.feature_metadata.equals(table2.feature_metadata)

    def test_roundtrip_boolean(self, sample_pivot_table):
        """Test boolean mutation data roundtrip"""
        pytest.importorskip("anndata")
        adata = sample_pivot_table.to_anndata()
        table2 = PivotTable.from_anndata(adata)

        # Values should match (bool may become float in AnnData)
        original = sample_pivot_table.values.astype(float)
        recovered = table2.values.astype(float)
        assert np.allclose(original, recovered)

    def test_to_anndata_rejects_out_of_order_metadata(self, sample_pivot_table):
        sample_pivot_table.sample_metadata = sample_pivot_table.sample_metadata.iloc[
            ::-1
        ]

        with pytest.raises(
            ValueError, match="sample_metadata.*different order.*reindex"
        ):
            sample_pivot_table.to_anndata()

    def test_scientific_context_roundtrips_with_anndata_contract(self, monkeypatch):
        class StubAnnData:
            def __init__(self, X, obs=None, var=None, layers=None, uns=None, **kwargs):
                self.X = X
                self.obs = obs
                self.var = var
                self.obs_names = obs.index
                self.var_names = var.index
                self.layers = {} if layers is None else layers
                self.uns = {} if uns is None else uns

        module = types.ModuleType("anndata")
        module.AnnData = StubAnnData
        monkeypatch.setitem(sys.modules, "anndata", module)
        table = PivotTable([[True, False]], index=["G1"], columns=["S1", "S2"])
        table.observation_mask = ObservationMask(
            pd.DataFrame([[True, False]], index=table.index, columns=table.columns)
        )
        table.sample_manifest = SampleManifest(
            pd.DataFrame(
                {
                    "patient_id": ["P1", "P2"],
                    "eligible": [True, True],
                },
                index=table.columns,
            )
        )

        adata = table.to_anndata(uns={"caller_key": "kept"})
        restored = PivotTable.from_anndata(adata)

        assert adata.uns["caller_key"] == "kept"
        assert restored.observation_mask.to_frame().equals(
            table.observation_mask.to_frame()
        )
        pd.testing.assert_frame_equal(
            restored.sample_manifest.to_frame().astype("string"),
            table.sample_manifest.to_frame().astype("string"),
        )

    def test_scientific_context_roundtrips_through_h5ad(self, tmp_path):
        anndata = pytest.importorskip("anndata")
        table = PivotTable(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["G1", "G2"],
            columns=["S1", "S2"],
        )
        table.sample_metadata["group"] = ["A", "B"]
        table.feature_metadata["kind"] = ["driver", "passenger"]
        table.observation_mask = ObservationMask(
            pd.DataFrame(
                [[True, False], [True, True]],
                index=table.index,
                columns=table.columns,
            )
        )
        table.sample_manifest = SampleManifest(
            pd.DataFrame(
                {
                    "patient_id": ["P1", "P2", "P3"],
                    "eligible": [True, True, False],
                    "exclusion_reason": [pd.NA, pd.NA, "failed_qc"],
                },
                index=["S1", "S2", "S3"],
            )
        )
        adata = table.to_anndata(
            layers={"caller_layer": np.ones((2, 2))},
            uns={"caller_key": "kept"},
        )
        path = tmp_path / "context.h5ad"
        adata.write_h5ad(path)

        restored = PivotTable.from_anndata(anndata.read_h5ad(path))

        assert "caller_layer" in adata.layers
        assert adata.uns["caller_key"] == "kept"
        assert restored.observation_mask.to_frame().equals(
            table.observation_mask.to_frame()
        )
        pd.testing.assert_frame_equal(
            restored.sample_manifest.to_frame().astype("string"),
            table.sample_manifest.to_frame().astype("string"),
        )

    def test_from_anndata_sparse(self):
        """Test conversion from sparse AnnData"""
        anndata = pytest.importorskip("anndata")
        import scipy.sparse as sp

        X_sparse = sp.csr_matrix(np.array([[1, 0, 3], [0, 2, 0]]))
        adata = anndata.AnnData(
            X=X_sparse,
            obs=pd.DataFrame({"group": ["A", "B"]}, index=["s1", "s2"]),
            var=pd.DataFrame({"chr": ["1", "2", "3"]}, index=["g1", "g2", "g3"]),
        )
        table = PivotTable.from_anndata(adata)

        assert table.shape == (3, 2)  # (features, samples)
        assert table.loc["g1", "s1"] == 1
        assert table.loc["g2", "s1"] == 0
        assert "group" in table.sample_metadata.columns

    def test_to_anndata_missing_anndata(self, sample_pivot_table, monkeypatch):
        """Test ImportError when anndata is not installed"""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "anndata":
                raise ImportError("No module named 'anndata'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="anndata is required"):
            sample_pivot_table.to_anndata()


def test_filter_by_statistical_test_rejects_categorical_mutations():
    table = SmallVariationTable(
        pd.DataFrame(
            {
                "S1": ["Missense_Mutation"],
                "S2": ["Multi_Hit"],
                "S3": [False],
                "S4": [False],
            },
            index=["TP53"],
        )
    )
    table.sample_metadata["group"] = ["A", "A", "B", "B"]

    with pytest.raises(
        ValueError,
        match=r"mutation_enrichment_test\(\).*to_binary_table\(\)",
    ):
        table.filter_by_statistical_test("group")


def test_filter_by_statistical_test_handles_constant_kruskal_rows():
    samples = [f"S{i}" for i in range(8)]
    groups = ["A"] * 4 + ["B"] * 4
    signal_rows = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        index=["G_signal", "G_null1", "G_null2"],
        columns=samples,
    )
    base_table = PivotTable(signal_rows)
    base_table.sample_metadata["group"] = groups
    base_result = base_table.filter_by_statistical_test(
        "group", method="kruskal", alpha=1.01
    )

    zero_rows = pd.DataFrame(
        0.0,
        index=[f"const_{i}" for i in range(20)],
        columns=samples,
    )
    padded_table = PivotTable(pd.concat([signal_rows, zero_rows]))
    padded_table.sample_metadata["group"] = groups

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        padded_result = padded_table.filter_by_statistical_test(
            "group", method="kruskal", alpha=1.01
        )

    assert list(padded_result.index) == list(base_result.index)
    pd.testing.assert_series_equal(
        padded_result.feature_metadata["adjusted_p_value"],
        base_result.feature_metadata["adjusted_p_value"],
    )


def test_add_exon_size_writes_feature_metadata(monkeypatch):
    """add_exon_size looks up each feature (gene) and writes sizes into
    feature_metadata, leaving NaN for genes Ensembl doesn't know."""
    from pymaftools.utils import geneinfo

    def fake_get_exon_size(genes, metric="transcript_length", force_download=False):
        sizes = {"TP53": 2591, "KRAS": 5300}  # TTN intentionally absent
        genes = list(genes)
        return pd.Series([sizes.get(g) for g in genes], index=genes, name=metric)

    monkeypatch.setattr(geneinfo, "get_exon_size", fake_get_exon_size)

    pt = PivotTable(
        pd.DataFrame(
            {"s1": [True, False, True]},
            index=["TP53", "KRAS", "TTN"],
        )
    )
    out = pt.add_exon_size()

    assert "exon_size" in out.feature_metadata.columns
    assert out.feature_metadata.loc["TP53", "exon_size"] == 2591
    assert out.feature_metadata.loc["KRAS", "exon_size"] == 5300
    assert pd.isna(out.feature_metadata.loc["TTN", "exon_size"])
    # original table untouched (add_exon_size returns a copy)
    assert "exon_size" not in pt.feature_metadata.columns


def test_add_freq_group_col_matches_manual_groups():
    """add_freq(group_col=) auto-splits samples and matches the manual groups dict."""
    df = pd.DataFrame(
        {"s1": [True, False], "s2": [True, True], "s3": [False, True], "s4": [False, False]},
        index=["TP53", "KRAS"],
    )
    pt = PivotTable(df)
    pt.sample_metadata["subtype"] = ["A", "A", "B", "B"]

    auto = pt.add_freq(group_col="subtype")
    manual = pt.add_freq(
        groups={g: pt.subset(samples=pt.sample_metadata.subtype == g) for g in ["A", "B"]}
    )

    for col in ["A_freq", "B_freq", "freq"]:
        assert col in auto.feature_metadata.columns
        pd.testing.assert_series_equal(
            auto.feature_metadata[col], manual.feature_metadata[col]
        )


def test_add_freq_rejects_groups_and_group_col_together():
    pt = PivotTable(pd.DataFrame({"s1": [True]}, index=["TP53"]))
    pt.sample_metadata["subtype"] = ["A"]
    with pytest.raises(ValueError, match="not both"):
        pt.add_freq(groups={"A": pt}, group_col="subtype")


def test_sort_features_multikey_keeps_groups_contiguous():
    """A list `by` sorts hierarchically: groups contiguous, then by freq within."""
    pt = PivotTable(
        pd.DataFrame({"s1": [True, True, True, True]}, index=["g1", "g2", "g3", "g4"])
    )
    pt.feature_metadata["band"] = ["Large", "Small", "Large", "Small"]
    pt.feature_metadata["freq"] = [0.1, 0.9, 0.8, 0.2]

    out = pt.sort_features(by=["band", "freq"], ascending=[True, False])
    # bands contiguous (Large, Large, Small, Small); within band freq descending
    assert list(out.index) == ["g3", "g1", "g2", "g4"]


def test_sort_features_unknown_column_raises():
    pt = PivotTable(pd.DataFrame({"s1": [True]}, index=["g1"]))
    with pytest.raises(ValueError, match="bogus"):
        pt.sort_features(by=["bogus"])
