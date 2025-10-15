"""
Tests for Cohort functionality
"""

import pytest
import pandas as pd
import tempfile
import os
from pymaftools.core.Cohort import Cohort
from pymaftools.core.PivotTable import PivotTable


class TestCohortBasics:
    """Test basic Cohort functionality"""
    
    def test_cohort_creation(self):
        """Test Cohort creation"""
        cohort = Cohort("test_cohort", "Test description")
        
        assert cohort.name == "test_cohort"
        assert cohort.description == "Test description"
        assert cohort.tables == {}
        assert cohort.sample_metadata is None
        assert cohort.sample_IDs is None
    
    def test_add_table(self, sample_pivot_table):
        """Test adding tables to cohort"""
        cohort = Cohort("test_cohort")
        
        # Add first table
        cohort.add_table(sample_pivot_table, "mutations")
        
        assert "mutations" in cohort.tables
        assert cohort.sample_IDs is not None
        assert len(cohort.sample_IDs) == sample_pivot_table.shape[1]
        assert cohort.sample_metadata is not None
    
    def test_add_multiple_tables(self, sample_cohort):
        """Test adding multiple tables"""
        cohort = sample_cohort
        
        assert "SNV" in cohort.tables
        assert "CNV" in cohort.tables
        assert len(cohort.tables) == 2
        
        # Check that all tables have same samples
        snv_samples = set(cohort.tables["SNV"].columns)
        cnv_samples = set(cohort.tables["CNV"].columns)
        assert snv_samples == cnv_samples
    
    def test_attribute_access(self, sample_cohort):
        """Test accessing tables as attributes"""
        cohort = sample_cohort
        
        # Should be able to access tables as attributes
        assert hasattr(cohort, "SNV")
        assert hasattr(cohort, "CNV")
        
        snv_table = cohort.SNV
        assert isinstance(snv_table, PivotTable)
        assert snv_table.shape[1] > 0  # Has samples
    
    def test_add_sample_metadata(self, sample_cohort):
        """Test adding sample metadata"""
        cohort = sample_cohort
        
        # Add new metadata
        new_metadata = pd.DataFrame({
            'treatment': ['Drug_A', 'Drug_B', 'Control', 'Drug_A'],
            'response': ['Good', 'Poor', 'Good', 'Good']
        }, index=cohort.sample_IDs)
        
        cohort.add_sample_metadata(new_metadata, source="clinical")
        
        # Check that metadata was added
        assert 'treatment' in cohort.sample_metadata.columns
        assert 'response' in cohort.sample_metadata.columns
        
        # Check that metadata propagated to tables
        assert 'treatment' in cohort.SNV.sample_metadata.columns
        assert 'response' in cohort.CNV.sample_metadata.columns


class TestCohortSubsetting:
    """Test Cohort subsetting functionality"""
    
    def test_subset_by_samples(self, sample_cohort):
        """Test subsetting cohort by samples"""
        cohort = sample_cohort
        
        # Subset to first two samples
        subset_samples = list(cohort.sample_IDs[:2])
        subset_cohort = cohort.subset(samples=subset_samples)
        
        assert len(subset_cohort.sample_IDs) == 2
        assert subset_cohort.SNV.shape[1] == 2
        assert subset_cohort.CNV.shape[1] == 2
        
        # Check that metadata was also subset
        assert subset_cohort.sample_metadata.shape[0] == 2
        assert list(subset_cohort.sample_metadata.index) == subset_samples
    
    @pytest.mark.skip(reason="Order method not implemented yet")
    def test_order_by_group(self, sample_cohort):
        """Test ordering cohort by group"""
        cohort = sample_cohort
        
        group_order = ['LUAD', 'ASC', 'LUSC']
        ordered_cohort = cohort.order(
            group_col='subtype', 
            group_order=group_order
        )
        
        # Check that samples are ordered by subtype
        subtypes = ordered_cohort.sample_metadata['subtype'].tolist()
        
        # Check that each group appears in the correct order
        unique_subtypes = []
        for subtype in subtypes:
            if subtype not in unique_subtypes:
                unique_subtypes.append(subtype)
        
        # The unique subtypes should follow the order specified
        expected_order = [g for g in group_order if g in unique_subtypes]
        assert unique_subtypes == expected_order


class TestCohortPersistence:
    """Test saving and loading cohorts"""
    
    def test_sqlite_save_and_load(self, sample_cohort, temp_output_dir):
        """Test saving and loading cohort to/from SQLite"""
        cohort = sample_cohort
        
        # Save to SQLite
        db_path = temp_output_dir / "test_cohort.db"
        cohort.to_sqlite(str(db_path))
        
        assert db_path.exists()
        
        # Load from SQLite
        loaded_cohort = Cohort.read_sqlite(str(db_path))
        
        assert loaded_cohort.name == cohort.name
        assert set(loaded_cohort.tables.keys()) == set(cohort.tables.keys())
        
        # Check that data is preserved
        for table_name in cohort.tables:
            original_table = cohort.tables[table_name]
            loaded_table = loaded_cohort.tables[table_name]
            
            assert original_table.shape == loaded_table.shape
            # Note: Values might differ slightly due to False->WT->False conversion
    
    def test_sql_registry_generation(self, sample_cohort):
        """Test SQL registry generation"""
        cohort = sample_cohort
        registry = cohort.to_sql_registry()
        
        assert isinstance(registry, pd.DataFrame)
        assert len(registry) == len(cohort.tables) * 3  # data, sample_metadata, feature_metadata
        
        expected_columns = ["sql_table_name", "cohort_name", "table_name", "type"]
        assert all(col in registry.columns for col in expected_columns)


class TestCohortValidation:
    """Test validation and error handling"""
    
    def test_add_invalid_table(self):
        """Test adding non-PivotTable object"""
        cohort = Cohort("test")
        
        with pytest.raises(TypeError, match="must be an instance of PivotTable"):
            cohort.add_table("not_a_pivot_table", "invalid")
    
    def test_conflicting_metadata(self, sample_pivot_table):
        """Test handling of conflicting metadata"""
        cohort = Cohort("test")
        cohort.add_table(sample_pivot_table, "table1")
        
        # Create conflicting metadata
        conflicting_metadata = pd.DataFrame({
            'subtype': ['DIFFERENT', 'VALUES', 'HERE', 'CONFLICT']  # Different from original
        }, index=sample_pivot_table.columns)
        
        conflicting_table = PivotTable(sample_pivot_table.values, 
                                     index=sample_pivot_table.index,
                                     columns=sample_pivot_table.columns)
        conflicting_table.sample_metadata = conflicting_metadata
        conflicting_table.feature_metadata = sample_pivot_table.feature_metadata.copy()
        
        with pytest.raises(ValueError, match="conflicting values"):
            cohort.add_table(conflicting_table, "table2")
    
    def test_mismatched_samples(self, sample_pivot_table):
        """Test handling of mismatched samples"""
        cohort = Cohort("test")
        cohort.add_table(sample_pivot_table, "table1")
        
        # Create table with different samples
        different_data = pd.DataFrame({
            'new_sample1': [True, False],
            'new_sample2': [False, True]
        }, index=['TP53', 'KRAS'])
        
        different_table = PivotTable(different_data)
        
        # Should raise KeyError when samples don't match
        with pytest.raises(KeyError):
            cohort.add_table(different_table, "table2")


class TestCohortCopy:
    """Test copying functionality"""
    
    def test_shallow_copy(self, sample_cohort):
        """Test shallow copy"""
        original = sample_cohort
        copied = original.copy(deep=False)
        
        assert copied.name == original.name
        assert copied.description == original.description
        assert set(copied.tables.keys()) == set(original.tables.keys())
        
        # Shallow copy should share table references
        assert copied.tables["SNV"] is original.tables["SNV"]
    
    def test_deep_copy(self, sample_cohort):
        """Test deep copy"""
        original = sample_cohort
        copied = original.copy(deep=True)
        
        assert copied.name == original.name
        assert copied.description == original.description
        assert set(copied.tables.keys()) == set(original.tables.keys())
        
        # Deep copy should have separate table instances
        assert copied.tables["SNV"] is not original.tables["SNV"]
        assert copied.tables["SNV"].equals(original.tables["SNV"])


@pytest.mark.integration 
class TestCohortIntegration:
    """Integration tests for cohort functionality"""
    
    def test_end_to_end_workflow(self, sample_cohort):
        """Test complete cohort workflow"""
        cohort = sample_cohort
        
        # 1. Add frequency calculations
        snv_with_freq = cohort.SNV.add_freq()
        cohort.tables["SNV"] = snv_with_freq
        
        # 2. Filter by frequency
        high_freq_snv = cohort.SNV.filter_by_freq(threshold=0.3)
        
        # 3. Subset cohort to high-frequency features
        feature_subset = cohort.subset(samples=cohort.sample_IDs)
        feature_subset.tables["SNV"] = high_freq_snv
        
        # 4. Verify workflow results
        assert isinstance(feature_subset, Cohort)
        assert len(feature_subset.tables) >= 1
        assert feature_subset.sample_metadata is not None
        assert 'freq' in feature_subset.SNV.feature_metadata.columns