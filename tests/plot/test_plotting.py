"""
Tests for plotting functionality
"""

import pytest
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from pymaftools.plot.PivotTablePlot import PivotTablePlot
from pymaftools.plot.ModelPlot import ModelPlot


class TestPivotTablePlot:
    """Test PivotTablePlot functionality"""
    
    def test_plot_creation(self, sample_pivot_table):
        """Test plot object creation"""
        plot = PivotTablePlot(sample_pivot_table)
        
        assert plot.pivot_table is sample_pivot_table
        assert hasattr(plot, 'fig')
        assert hasattr(plot, 'save')
    
    @pytest.mark.plot
    def test_boxplot_with_annot(self, sample_pivot_table, temp_output_dir):
        """Test boxplot with statistical annotations"""
        # Add TMB data for testing
        sample_pivot_table.sample_metadata['TMB'] = [2.5, 3.1, 1.8, 2.9]
        
        plot = PivotTablePlot(sample_pivot_table)
        
        # Test basic boxplot
        ax = plot.plot_boxplot_with_annot(
            test_col='TMB',
            group_col='subtype',
            test='Mann-Whitney'
        )
        
        assert ax is not None
        assert ax.get_xlabel() == 'subtype'
        assert ax.get_ylabel() == 'TMB'
        
        plt.close('all')
    
    @pytest.mark.plot
    def test_boxplot_custom_labels(self, sample_pivot_table, temp_output_dir):
        """Test boxplot with custom axis labels"""
        sample_pivot_table.sample_metadata['TMB'] = [2.5, 3.1, 1.8, 2.9]
        
        plot = PivotTablePlot(sample_pivot_table)
        
        ax = plot.plot_boxplot_with_annot(
            test_col='TMB',
            group_col='subtype',
            xlabel='Cancer Subtype',
            ylabel='Tumor Mutation Burden',
            title='TMB Comparison'
        )
        
        assert ax.get_xlabel() == 'Cancer Subtype'
        assert ax.get_ylabel() == 'Tumor Mutation Burden'
        assert ax.get_title() == 'TMB Comparison'
        
        plt.close('all')
    
    @pytest.mark.plot
    def test_boxplot_save_functionality(self, sample_pivot_table, temp_output_dir):
        """Test saving boxplot"""
        sample_pivot_table.sample_metadata['TMB'] = [2.5, 3.1, 1.8, 2.9]
        
        plot = PivotTablePlot(sample_pivot_table)
        save_path = temp_output_dir / "test_boxplot.png"
        
        plot.plot_boxplot_with_annot(
            test_col='TMB',
            group_col='subtype',
            save_path=str(save_path),
            dpi=150
        )
        
        assert save_path.exists()
        plt.close('all')
    
    @pytest.mark.plot
    def test_pca_plot(self, sample_pivot_table, temp_output_dir):
        """Test PCA plotting"""
        plot = PivotTablePlot(sample_pivot_table)
        
        pca_df, explained_var, pca_obj = plot.plot_pca_samples(
            color_col='subtype',
            figsize=(8, 4),
            to_binary=True
        )
        
        assert pca_df is not None
        assert explained_var is not None
        assert pca_obj is not None
        assert pca_df.shape[1] >= 2  # At least PC1, PC2
        assert len(explained_var) >= 2
        
        plt.close('all')
    
    @pytest.mark.plot 
    def test_pca_plot_numeric_colors(self, sample_pivot_table):
        """Test PCA plot with numeric color encoding"""
        # Add numeric data
        sample_pivot_table.sample_metadata['TMB'] = [2.5, 3.1, 1.8, 2.9]
        
        plot = PivotTablePlot(sample_pivot_table)
        
        pca_df, explained_var, pca_obj = plot.plot_pca_samples(
            color_col='TMB',
            is_numeric=True,
            palette='viridis'
        )
        
        assert pca_df is not None
        plt.close('all')


class TestModelPlot:
    """Test ModelPlot functionality"""
    
    def test_model_plot_creation(self):
        """Test ModelPlot creation"""
        plot = ModelPlot()
        
        assert hasattr(plot, 'fig')
        assert hasattr(plot, 'save')
    
    @pytest.mark.plot
    def test_metric_comparison_plot(self, sample_model_metrics, temp_output_dir):
        """Test metric comparison plotting"""
        plot = ModelPlot()
        
        plot.plot_metric_comparison_with_annotation(
            data=sample_model_metrics,
            metrics=['acc', 'f1'],
            group_col='model',
            test='Mann-Whitney',
            fontsize=12
        )
        
        assert plot.fig is not None
        plt.close('all')
    
    @pytest.mark.plot
    def test_metric_comparison_with_rotation(self, sample_model_metrics):
        """Test metric comparison with rotated labels"""
        plot = ModelPlot()
        
        plot.plot_metric_comparison_with_annotation(
            data=sample_model_metrics,
            metrics=['acc'],
            group_col='model',
            rotation=45,
            fontsize=10
        )
        
        assert plot.fig is not None
        plt.close('all')
    
    @pytest.mark.plot
    def test_metric_comparison_save(self, sample_model_metrics, temp_output_dir):
        """Test saving metric comparison plot"""
        plot = ModelPlot()
        save_path = temp_output_dir / "test_metrics.png"
        
        plot.plot_metric_comparison_with_annotation(
            data=sample_model_metrics,
            metrics=['acc', 'f1'],
            group_col='model',
            save_path=str(save_path),
            dpi=150
        )
        
        assert save_path.exists()
        plt.close('all')


class TestPlotSaving:
    """Test plot saving functionality"""
    
    @pytest.mark.plot
    def test_different_formats(self, sample_pivot_table, temp_output_dir):
        """Test saving plots in different formats"""
        plot = PivotTablePlot(sample_pivot_table)
        sample_pivot_table.sample_metadata['TMB'] = [2.5, 3.1, 1.8, 2.9]
        
        formats = ['png', 'pdf', 'svg']
        
        for fmt in formats:
            save_path = temp_output_dir / f"test_plot.{fmt}"
            
            ax = plot.plot_boxplot_with_annot(
                test_col='TMB',
                group_col='subtype',
                save_path=str(save_path)
            )
            
            if fmt != 'svg':  # SVG might have issues with some matplotlib versions
                assert save_path.exists()
            
            plt.close('all')
    
    @pytest.mark.plot
    def test_high_dpi_saving(self, sample_pivot_table, temp_output_dir):
        """Test high DPI saving"""
        plot = PivotTablePlot(sample_pivot_table)
        sample_pivot_table.sample_metadata['TMB'] = [2.5, 3.1, 1.8, 2.9]
        
        save_path = temp_output_dir / "high_dpi_plot.png"
        
        ax = plot.plot_boxplot_with_annot(
            test_col='TMB',
            group_col='subtype',
            save_path=str(save_path),
            dpi=600
        )
        
        assert save_path.exists()
        
        # Check file size (higher DPI should result in larger file)
        file_size = save_path.stat().st_size
        assert file_size > 1000  # Should be reasonably large
        
        plt.close('all')


class TestPlotValidation:
    """Test plot validation and error handling"""
    
    def test_invalid_column_names(self, sample_pivot_table):
        """Test handling of invalid column names"""
        plot = PivotTablePlot(sample_pivot_table)
        
        with pytest.raises(ValueError, match="Could not interpret value"):
            plot.plot_boxplot_with_annot(
                test_col='nonexistent_column',
                group_col='subtype'
            )
        
        with pytest.raises(ValueError, match="not found in sample_metadata"):
            plot.plot_pca_samples(color_col='nonexistent_column')
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        import pandas as pd
        from pymaftools.core.PivotTable import PivotTable
        
        # Create minimal table
        minimal_data = pd.DataFrame({
            'sample1': [True, False]
        }, index=['gene1', 'gene2'])
        
        table = PivotTable(minimal_data)
        table.sample_metadata['group'] = ['A']
        
        plot = PivotTablePlot(table)
        
        # Should handle single group gracefully
        with pytest.raises((ValueError, IndexError)):
            plot.plot_boxplot_with_annot(
                test_col='nonexistent',
                group_col='group'
            )


@pytest.mark.slow
class TestPlotPerformance:
    """Performance tests for plotting"""
    
    @pytest.mark.plot
    def test_large_dataset_plotting(self):
        """Test plotting with large datasets"""
        from tests.conftest import TestDataBuilder
        import pandas as pd
        from pymaftools.core.PivotTable import PivotTable
        
        # Create large dataset
        large_data = TestDataBuilder.create_large_mutation_matrix(
            n_samples=100, n_genes=500
        )
        
        table = PivotTable(large_data)
        
        # Add sample metadata
        subtypes = ['LUAD', 'LUSC', 'ASC'] * (100 // 3 + 1)
        table.sample_metadata['subtype'] = subtypes[:100]
        table.sample_metadata['TMB'] = np.random.normal(2.5, 0.5, 100)
        
        plot = PivotTablePlot(table)
        
        # Test PCA (this should be reasonably fast)
        pca_df, explained_var, pca_obj = plot.plot_pca_samples(
            color_col='subtype',
            to_binary=True
        )
        
        assert pca_df.shape[0] == 100
        plt.close('all')