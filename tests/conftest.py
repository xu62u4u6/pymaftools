"""
Test fixtures and utilities for pymaftools tests
"""

import pytest
import pandas as pd
import numpy as np
from pymaftools.core.PivotTable import PivotTable
from pymaftools.core.MAF import MAF
from pymaftools.core.Cohort import Cohort


@pytest.fixture
def sample_mutation_data():
    """Create sample mutation data for testing"""
    data = {
        'sample1': [True, False, True, False],
        'sample2': [False, True, True, False], 
        'sample3': [True, True, False, True],
        'sample4': [False, False, True, True]
    }
    return pd.DataFrame(data, index=['TP53', 'KRAS', 'EGFR', 'PIK3CA'])


@pytest.fixture
def sample_pivot_table(sample_mutation_data):
    """Create a sample PivotTable for testing"""
    table = PivotTable(sample_mutation_data)
    
    # Add sample metadata
    table.sample_metadata['subtype'] = ['LUAD', 'LUSC', 'ASC', 'LUAD']
    table.sample_metadata['age'] = [65, 70, 55, 60]
    table.sample_metadata['sex'] = ['M', 'F', 'M', 'F']
    
    # Add feature metadata
    table.feature_metadata['chromosome'] = ['17p13.1', '12p12.1', '7p11.2', '3q26.32']
    table.feature_metadata['gene_type'] = ['tumor_suppressor', 'oncogene', 'oncogene', 'oncogene']
    
    return table


@pytest.fixture 
def sample_maf_data():
    """Create sample MAF data for testing"""
    maf_data = {
        'Hugo_Symbol': ['TP53', 'KRAS', 'EGFR', 'PIK3CA', 'TP53'],
        'Tumor_Sample_Barcode': ['sample1', 'sample2', 'sample3', 'sample4', 'sample2'],
        'Variant_Classification': ['Missense_Mutation', 'Missense_Mutation', 'Frame_Shift_Del', 
                                  'Missense_Mutation', 'Nonsense_Mutation'],
        'HGVSp': ['p.R175H', 'p.G12D', 'p.E746_A750del', 'p.E545K', 'p.R273*'],
        'Chromosome': ['17', '12', '7', '3', '17'],
        'Start_Position': [7577121, 25245347, 55242464, 178952085, 7577538],
        'End_Position': [7577121, 25245347, 55242479, 178952085, 7577538],
        'Reference_Allele': ['C', 'G', 'AGGAATTAAGAGAAGC', 'G', 'C'],
        'Tumor_Seq_Allele2': ['T', 'A', '-', 'A', 'T']
    }
    return pd.DataFrame(maf_data)


@pytest.fixture
def sample_cnv_data():
    """Create sample CNV data for testing (log2 copy number ratios)"""
    cnv_data = pd.DataFrame({
        'sample1': [0.5, -0.8, 0.2, 1.2],    # log2(copy_number/2): amplifications and deletions
        'sample2': [-0.3, 0.9, -0.5, 0.7], 
        'sample3': [1.1, -0.2, 0.8, -0.9],
        'sample4': [-0.6, 0.4, -0.1, 0.3]
    }, index=['TP53', 'KRAS', 'EGFR', 'PIK3CA'])
    return cnv_data


@pytest.fixture
def sample_cnv_table(sample_cnv_data):
    """Create a sample CNV PivotTable for testing"""
    cnv_table = PivotTable(sample_cnv_data)
    
    # Add sample metadata (same as mutation data for consistency)
    cnv_table.sample_metadata['subtype'] = ['LUAD', 'LUSC', 'ASC', 'LUAD']
    cnv_table.sample_metadata['age'] = [65, 70, 55, 60]
    cnv_table.sample_metadata['sex'] = ['M', 'F', 'M', 'F']
    
    # Add CNV-specific feature metadata
    cnv_table.feature_metadata['chromosome'] = ['17p13.1', '12p12.1', '7p11.2', '3q26.32']
    cnv_table.feature_metadata['gene_type'] = ['tumor_suppressor', 'oncogene', 'oncogene', 'oncogene']
    cnv_table.feature_metadata['cytoband'] = ['17p13.1', '12p12.1', '7p11.2', '3q26.32']
    
    return cnv_table


@pytest.fixture
def sample_cohort(sample_pivot_table, sample_cnv_table):
    """Create a sample Cohort for testing with multiple data types"""
    cohort = Cohort("test_cohort", "Test cohort for unit tests")
    
    # Add tables to cohort
    cohort.add_table(sample_pivot_table, "SNV")    # 突變數據
    cohort.add_table(sample_cnv_table, "CNV")      # 拷貝數數據
    
    return cohort


@pytest.fixture
def sample_model_metrics():
    """Create sample model metrics data for testing"""
    np.random.seed(42)
    data = []
    models = ['SNV', 'CNV-gene', 'CNV-cluster', 'STACK']
    metrics = ['acc', 'f1', 'auc']
    
    for model in models:
        for i in range(10):  # 10 CV folds
            row = {'model': model, 'fold': i}
            for metric in metrics:
                # Generate realistic metric values
                if metric == 'acc':
                    row[metric] = np.random.normal(0.75, 0.05)
                elif metric == 'f1':
                    row[metric] = np.random.normal(0.70, 0.06)
                else:  # auc
                    row[metric] = np.random.normal(0.80, 0.04)
            data.append(row)
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for test outputs"""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir


class TestDataBuilder:
    """Helper class for building test data"""
    
    @staticmethod
    def create_large_mutation_matrix(n_samples=100, n_genes=500, mutation_rate=0.1):
        """Create a large mutation matrix for performance testing"""
        np.random.seed(42)
        data = np.random.choice([True, False], 
                               size=(n_genes, n_samples), 
                               p=[mutation_rate, 1-mutation_rate])
        
        samples = [f"sample_{i}" for i in range(n_samples)]
        genes = [f"gene_{i}" for i in range(n_genes)]
        
        return pd.DataFrame(data, index=genes, columns=samples)
    
    @staticmethod
    def create_maf_with_subtypes(n_samples=50, subtypes=['LUAD', 'LUSC', 'ASC']):
        """Create MAF data with specified subtypes"""
        np.random.seed(42)
        samples = []
        for i, subtype in enumerate(subtypes):
            subtype_samples = [f"{subtype}_{j}" for j in range(n_samples // len(subtypes))]
            samples.extend(subtype_samples)
        
        # Add remaining samples to first subtype if division is not exact
        remaining = n_samples - len(samples)
        if remaining > 0:
            samples.extend([f"{subtypes[0]}_{j}" for j in range(len(samples), n_samples)])
        
        genes = ['TP53', 'KRAS', 'EGFR', 'PIK3CA', 'APC', 'PTEN', 'RB1', 'MYC']
        
        maf_data = []
        for sample in samples:
            # Each sample gets 1-5 random mutations
            n_mutations = np.random.randint(1, 6)
            sample_genes = np.random.choice(genes, n_mutations, replace=False)
            
            for gene in sample_genes:
                maf_data.append({
                    'Hugo_Symbol': gene,
                    'Tumor_Sample_Barcode': sample,
                    'Variant_Classification': np.random.choice([
                        'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 
                        'Frame_Shift_Ins', 'Splice_Site'
                    ]),
                    'HGVSp': f"p.{gene}_{np.random.randint(1, 1000)}X",
                    'Chromosome': str(np.random.randint(1, 23)),
                    'Start_Position': np.random.randint(1000000, 100000000),
                    'End_Position': np.random.randint(1000000, 100000000),
                    'Reference_Allele': np.random.choice(['A', 'T', 'G', 'C']),
                    'Tumor_Seq_Allele2': np.random.choice(['A', 'T', 'G', 'C'])
                })
        
        return pd.DataFrame(maf_data)


# Export commonly used test utilities
__all__ = [
    'sample_mutation_data',
    'sample_pivot_table',
    'sample_cnv_data', 
    'sample_cnv_table',
    'sample_maf_data',
    'sample_cohort',
    'sample_model_metrics',
    'temp_output_dir',
    'TestDataBuilder'
]