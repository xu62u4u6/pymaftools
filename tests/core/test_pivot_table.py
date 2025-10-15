"""
Tests for PivotTable core functionality
"""

import pytest
import pandas as pd
import numpy as np
from pymaftools.core.PivotTable import PivotTable


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


class TestPivotTableStatistics:
    """Test statistical methods"""
    
    def test_to_binary_table(self, sample_pivot_table):
        """Test conversion to binary format"""
        table = sample_pivot_table
        binary_table = table.to_binary_table()
        
        # Check that all columns are boolean type
        assert all(dtype == bool for dtype in binary_table.dtypes)
        assert binary_table.shape == table.shape
        assert (binary_table == (table != False)).all().all()
    
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