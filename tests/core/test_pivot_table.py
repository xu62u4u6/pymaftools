"""
Tests for PivotTable core functionality
"""

import pytest
import pandas as pd
import numpy as np
import sqlite3
import pymaftools
from pymaftools.core.PivotTable import PivotTable
from pymaftools.core.SmallVariationTable import SmallVariationTable


class TestPivotTableBasics:
    """Test basic PivotTable functionality"""
    
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
        """Test add_freq method"""
        table = sample_pivot_table
        table_with_freq = table.add_freq()
        
        assert 'freq' in table_with_freq.feature_metadata.columns
        assert len(table_with_freq.feature_metadata['freq']) == table.shape[0]
        assert all(0 <= freq <= 1 for freq in table_with_freq.feature_metadata['freq'])
    
    def test_add_freq_with_groups(self, sample_pivot_table):
        """Test add_freq with group-specific frequencies"""
        table = sample_pivot_table
        
        # Create group subsets
        luad_samples = table.sample_metadata['subtype'] == 'LUAD'
        lusc_samples = table.sample_metadata['subtype'] == 'LUSC'
        
        luad_table = table.subset(samples=luad_samples)
        lusc_table = table.subset(samples=lusc_samples)
        
        groups = {'LUAD': luad_table, 'LUSC': lusc_table}
        table_with_group_freq = table.add_freq(groups=groups)
        
        assert 'LUAD_freq' in table_with_group_freq.feature_metadata.columns
        assert 'LUSC_freq' in table_with_group_freq.feature_metadata.columns
        assert 'freq' in table_with_group_freq.feature_metadata.columns

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
        """Test frequency-based filtering"""
        table = sample_pivot_table.add_freq()
        
        # Filter features with frequency >= 0.5
        filtered = table.filter_by_freq(threshold=0.5)
        
        assert all(table.feature_metadata.loc[filtered.index, 'freq'] >= 0.5)
        assert filtered.shape[0] <= table.shape[0]


class TestPivotTableSorting:
    """Test sorting functionality"""
    
    def test_sort_features(self, sample_pivot_table):
        """Test feature sorting by metadata"""
        table = sample_pivot_table.add_freq()
        
        # Sort by frequency (descending)
        sorted_table = table.sort_features(by='freq', ascending=False)
        
        freq_values = sorted_table.feature_metadata['freq'].values
        assert all(freq_values[i] >= freq_values[i+1] for i in range(len(freq_values)-1))
    
    def test_sort_samples_by_group(self, sample_pivot_table):
        """Test sample sorting by group"""
        table = sample_pivot_table
        
        group_order = ['LUAD', 'ASC', 'LUSC']
        sorted_table = table.sort_samples_by_group(
            group_col='subtype', 
            group_order=group_order, 
            top=2
        )
        
        # Check that samples are grouped by subtype
        subtypes = sorted_table.sample_metadata['subtype'].tolist()
        
        # Should have groups in order (though within groups may vary)
        luad_indices = [i for i, s in enumerate(subtypes) if s == 'LUAD']
        asc_indices = [i for i, s in enumerate(subtypes) if s == 'ASC']
        lusc_indices = [i for i, s in enumerate(subtypes) if s == 'LUSC']
        
        # Check that groups appear in order
        if luad_indices and asc_indices:
            assert max(luad_indices) < min(asc_indices)
        if asc_indices and lusc_indices:
            assert max(asc_indices) < min(lusc_indices)

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


class TestPivotTableStatistics:
    """Test statistical methods"""
    
    def test_to_binary_table(self, sample_pivot_table):
        """Test conversion to binary format"""
        table = sample_pivot_table
        binary_table = table.to_binary_table()
        
        # Check that all columns are boolean type
        assert all(dtype == bool for dtype in binary_table.dtypes)
        assert binary_table.shape == table.shape
        expected = table.notna() & table.ne(False)
        assert binary_table.equals(expected)

    def test_to_binary_table_treats_missing_values_as_absent(self, sample_pivot_table):
        """Missing observations must not be counted as mutations."""
        table = sample_pivot_table.astype(object)
        table.iloc[0, 0] = np.nan

        binary_table = table.to_binary_table()

        assert binary_table.iloc[0, 0] == False  # noqa: E712
        assert binary_table.to_numpy().dtype == bool
        assert binary_table.sample_metadata.equals(table.sample_metadata)
        assert binary_table.feature_metadata.equals(table.feature_metadata)
    
    def test_PCA_calculation(self, sample_pivot_table):
        """Test PCA calculation"""
        table = sample_pivot_table
        
        # Test with binary conversion
        pca_df, explained_var, pca_obj = table.PCA(to_binary=True)
        
        assert isinstance(pca_df, pd.DataFrame)
        assert pca_df.shape == (table.shape[1], 2)  # n_samples x 2 components
        assert 'PC1' in pca_df.columns
        assert 'PC2' in pca_df.columns
        assert len(explained_var) == 2
        assert all(0 <= var <= 1 for var in explained_var)
    
    def test_compute_similarity(self, sample_pivot_table):
        """Test similarity matrix computation"""
        table = sample_pivot_table
        
        # Test cosine similarity
        similarity_matrix = table.compute_similarity(method='cosine')
        
        assert similarity_matrix.shape == (table.shape[1], table.shape[1])
        assert similarity_matrix.index.equals(table.columns)
        assert similarity_matrix.columns.equals(table.columns)
        
        # Diagonal should be 1 (self-similarity)
        np.testing.assert_array_almost_equal(
            np.diag(similarity_matrix.values), 
            np.ones(table.shape[1])
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


@pytest.mark.slow
class TestPivotTablePerformance:
    """Performance tests for large datasets"""
    
    def test_large_dataset_creation(self):
        """Test creation with large dataset"""
        from tests.conftest import TestDataBuilder
        
        large_data = TestDataBuilder.create_large_mutation_matrix(
            n_samples=1000, n_genes=5000
        )
        
        table = PivotTable(large_data)
        assert table.shape == (5000, 1000)
    
    def test_large_dataset_frequency_calculation(self):
        """Test frequency calculation on large dataset"""
        from tests.conftest import TestDataBuilder
        
        large_data = TestDataBuilder.create_large_mutation_matrix(
            n_samples=500, n_genes=2000
        )
        
        table = PivotTable(large_data)
        frequencies = table.calculate_feature_frequency()
        
        assert len(frequencies) == 2000
        assert all(0 <= freq <= 1 for freq in frequencies)


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

    def test_from_anndata_basic(self, sample_pivot_table):
        """Test AnnData to PivotTable conversion"""
        pytest.importorskip("anndata")
        adata = sample_pivot_table.to_anndata()
        table = PivotTable.from_anndata(adata)

        assert isinstance(table, PivotTable)
        assert table.shape == sample_pivot_table.shape

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
