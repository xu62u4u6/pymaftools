"""
Base plotting class that provides legend management functionality
"""
import matplotlib.pyplot as plt
from .LegendManager import LegendManager
from .ColorManager import ColorManager


class BasePlot:
    """
    Base class for all plotting classes
    Provides common functionality for legend management and color management
    """
    
    def __init__(self):
        # Initialize legend manager
        self.legend_manager = LegendManager()
        
        # Initialize color manager
        self.color_manager = ColorManager()
        
        # Figure-related attributes
        self.fig = None
        self.ax_legend = None
    
    def add_legend(self, legend_name: str, color_dict: dict):
        """
        Add legend information to LegendManager
        
        Args:
            legend_name: Legend name, such as 'mutation', 'sex', 'subtype', etc.
            color_dict: Color mapping dictionary, such as {'M': 'blue', 'F': 'red'}
            
        Returns:
            self: Support method chaining
        """
        self.legend_manager.add_legend(legend_name, color_dict)
        return self
    
    def plot_all_legends(self, ax=None, fontsize=8, title_fontsize=10, 
                        legend_spacing=0.08, item_spacing=0.02):
        """
        Plot all legends on the specified axis
        
        Args:
            ax: Axis to plot legends on, use self.ax_legend if None
            fontsize: Legend text font size
            title_fontsize: Legend title font size
            legend_spacing: Spacing between different legends
            item_spacing: Spacing between items within the same legend
            
        Returns:
            self: Support method chaining
        """
        target_ax = ax if ax is not None else self.ax_legend
        if target_ax is None:
            raise ValueError("No axis available for legend plotting. Please provide an axis or set self.ax_legend.")
        
        # Set axis for LegendManager and plot legends
        self.legend_manager.set_axis(target_ax).plot_legends(
            fontsize=fontsize,
            title_fontsize=title_fontsize,
            legend_spacing=legend_spacing,
            item_spacing=item_spacing
        )
        return self
    
    def get_legend_manager(self):
        """
        Get the legend manager instance
        
        Returns:
            LegendManager: Legend manager instance
        """
        return self.legend_manager
    
    def remove_legend(self, legend_name: str):
        """
        Remove a specified legend
        
        Args:
            legend_name: Name of the legend to remove
            
        Returns:
            self: Support method chaining
        """
        self.legend_manager.remove_legend(legend_name)
        return self
    
    def clear_legends(self):
        """
        Clear all legends
        
        Returns:
            self: Support method chaining
        """
        self.legend_manager.clear_legends()
        return self
    
    def has_legend(self, legend_name: str) -> bool:
        """
        Check if a legend with the specified name exists
        
        Args:
            legend_name: Legend name
        
        Returns:
            bool: Whether the legend exists
        """
        return self.legend_manager.has_legend(legend_name)
    
    def get_legend_names(self) -> list:
        """
        Get all legend names
        
        Returns:
            list: List of legend names
        """
        return self.legend_manager.get_legend_names()
    
    def get_color_manager(self):
        """
        Get the color manager instance
        
        Returns:
            ColorManager: Color manager instance
        """
        return self.color_manager
    
    def save(self, filename: str, dpi: int = 300, bbox_inches: str = 'tight', 
             transparent: bool = False, **kwargs):
        """
        Save figure to file
        
        Args:
            filename: File name
            dpi: Resolution
            bbox_inches: Bounding box setting
            transparent: Whether to use transparent background
            **kwargs: Other save parameters
        """
        if self.fig is None:
            print("Figure is not exist.")
            return

        try:
            format = filename.split('.')[-1].lower()
            pil_kwargs = {"compression": "tiff_lzw"} if format == "tiff" else {}

            self.fig.savefig(
                filename,
                dpi=dpi,
                bbox_inches=bbox_inches,
                transparent=transparent,
                format=format,
                pil_kwargs=pil_kwargs,
                **kwargs
            )
            print(f"Figure saved to: {filename}")
        except Exception as e:
            print(f"Error while saving figure: {e}")
    
    def save_figure(self, fig, save_path: str, dpi: int = 300, **kwargs):
        """
        Save a specific figure to file (utility method for subclasses)
        
        Args:
            fig: matplotlib Figure object to save
            save_path: Path to save the figure
            dpi: Resolution
            **kwargs: Other save parameters
        """
        if save_path is None:
            return
            
        try:
            format_ext = save_path.split('.')[-1].lower()
            pil_kwargs = {"compression": "tiff_lzw"} if format_ext == "tiff" else {}
            
            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches='tight',
                format=format_ext,
                pil_kwargs=pil_kwargs,
                **kwargs
            )
            print(f"[INFO] Figure saved to: {save_path}")
        except Exception as e:
            print(f"Error while saving figure: {e}")
    
    def close(self):
        """
        Close the figure
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
