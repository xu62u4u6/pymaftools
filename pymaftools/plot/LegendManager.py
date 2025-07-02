import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize
import numpy as np
from typing import Dict, Optional, Tuple, List, Union, Any
import matplotlib.axes as Axes
import matplotlib.gridspec as gridspec


class LegendManager:
    """
    Specialized class for legend management
    
    Responsible for managing and plotting various types of legends, including mutation types, categorical variables, numeric variables, etc.
    Supports advanced features for PCA plots including shape markers, numeric colorbars, and mixed legends.
    """
    
    def __init__(self, ax: Optional[Axes.Axes] = None):
        """
        Initialize legend manager
        
        Args:
            ax: matplotlib axis object, if None will be required when plotting
        """
        self.ax = ax
        self.legend_dict: Dict[str, Dict[str, str]] = {}
        self.shape_dict: Dict[str, Dict[str, str]] = {}  # For shape legends
        self.numeric_legends: Dict[str, Dict[str, Any]] = {}  # For numeric colorbars

    def set_axis(self, ax: Axes.Axes) -> 'LegendManager':
        """
        Set the axis for legend plotting
        
        Args:
            ax: matplotlib axis object
        
        Returns:
            LegendManager: Support method chaining
        """
        self.ax = ax
        return self
    
    def add_legend(self, legend_name: str, color_dict: Dict[str, str]) -> 'LegendManager':
        """
        Add legend information
        
        Args:
            legend_name: Legend name, such as 'Mutation', 'Subtype', 'Sex', etc.
            color_dict: Color mapping dictionary, such as {'M': 'blue', 'F': 'red'}
        
        Returns:
            LegendManager: Support method chaining
        """
        self.legend_dict[legend_name] = color_dict
        return self
    
    def remove_legend(self, legend_name: str) -> 'LegendManager':
        """
        Remove a specified legend
        
        Args:
            legend_name: Name of the legend to remove
        
        Returns:
            LegendManager: Support method chaining
        """
        if legend_name in self.legend_dict:
            del self.legend_dict[legend_name]
        return self
    
    def clear_legends(self) -> 'LegendManager':
        """
        Clear all legends
        
        Returns:
            LegendManager: Support method chaining
        """
        self.legend_dict.clear()
        return self
    
    def get_legend_dict(self) -> Dict[str, Dict[str, str]]:
        """
        Get the current legend dictionary
        
        Returns:
            Dict: Complete legend dictionary
        """
        return self.legend_dict.copy()
    
    def plot_legends(self, 
                    ax: Optional[Axes.Axes] = None,
                    fontsize: int = 8, 
                    title_fontsize: int = 10, 
                    legend_spacing: float = 0.08, 
                    item_spacing: float = 0.02,
                    start_y: float = 0.95,
                    rect_width: float = 0.04,
                    rect_height: float = 0.03,
                    text_offset_x: float = 0.12,
                    title_offset_y: float = 0.05,
                    item_offset_y: float = 0.035,
                    legend_gap: float = 0.03) -> 'LegendManager':
        """
        Plot all legends
        
        Args:
            ax: matplotlib axis object, use initialization axis if None
            fontsize: Legend item font size
            title_fontsize: Legend title font size
            legend_spacing: Spacing between different legends
            item_spacing: Spacing between items within the same legend
            start_y: Starting Y position
            rect_width: Color rectangle width
            rect_height: Color rectangle height
            text_offset_x: Text X offset relative to color rectangle
            title_offset_y: Y offset after title
            item_offset_y: Y offset between items
            legend_gap: Gap between legends
        
        Returns:
            LegendManager: Support method chaining
        """
        # Determine which axis to use
        target_ax = ax if ax is not None else self.ax
        if target_ax is None:
            raise ValueError("No axis provided. Please set axis using set_axis() or provide ax parameter.")
        
        # Return if no legends
        if not self.legend_dict:
            return self
        
        # Clear axis and set basic properties
        target_ax.clear()
        target_ax.axis('off')
        target_ax.set_xlim(0, 1)
        target_ax.set_ylim(0, 1)
        
        # Plot legends from top to bottom
        y_position = start_y
        
        for legend_name, color_dict in self.legend_dict.items():
            # Plot legend title
            target_ax.text(0.05, y_position, legend_name, 
                          fontsize=title_fontsize, fontweight='bold', 
                          va='top', ha='left')
            y_position -= title_offset_y
            
            # Plot legend items
            for label, color in color_dict.items():
                # Plot color rectangle
                rect = Rectangle((0.05, y_position - rect_height/2), 
                               rect_width, rect_height, 
                               facecolor=color, edgecolor='none', linewidth=0)
                target_ax.add_patch(rect)
                
                # Plot label text
                target_ax.text(text_offset_x, y_position, label, 
                              fontsize=fontsize, va='center', ha='left')
                
                y_position -= item_offset_y
            
            # Add spacing between legends
            y_position -= legend_gap
        
        return self
    
    def plot_horizontal_legend(self,
                              ax: Optional[Axes.Axes] = None,
                              fontsize: int = 8,
                              title_fontsize: int = 10,
                              columns: int = 3,
                              rect_size: Tuple[float, float] = (0.03, 0.03),
                              spacing: Tuple[float, float] = (0.15, 0.1)) -> 'LegendManager':
        """
        Plot legends in horizontal layout
        
        Args:
            ax: matplotlib axis object
            fontsize: Item font size
            title_fontsize: Title font size
            columns: Number of items per row
            rect_size: Color rectangle size (width, height)
            spacing: Item spacing (horizontal spacing, vertical spacing)
        
        Returns:
            LegendManager: Support method chaining
        """
        target_ax = ax if ax is not None else self.ax
        if target_ax is None:
            raise ValueError("No axis provided.")
        
        if not self.legend_dict:
            return self
        
        target_ax.clear()
        target_ax.axis('off')
        target_ax.set_xlim(0, 1)
        target_ax.set_ylim(0, 1)
        
        y_position = 0.9
        
        for legend_name, color_dict in self.legend_dict.items():
            # Plot title
            target_ax.text(0.05, y_position, legend_name,
                          fontsize=title_fontsize, fontweight='bold',
                          va='top', ha='left')
            y_position -= 0.08
            
            # Arrange items horizontally
            items = list(color_dict.items())
            rows = len(items) // columns + (1 if len(items) % columns else 0)
            
            for row in range(rows):
                x_position = 0.05
                for col in range(columns):
                    idx = row * columns + col
                    if idx >= len(items):
                        break
                    
                    label, color = items[idx]
                    
                    # Plot color rectangle
                    rect = Rectangle((x_position, y_position - rect_size[1]/2),
                                   rect_size[0], rect_size[1],
                                   facecolor=color, edgecolor='none')
                    target_ax.add_patch(rect)
                    
                    # Plot label
                    target_ax.text(x_position + rect_size[0] + 0.01, y_position,
                                  label, fontsize=fontsize, va='center', ha='left')
                    
                    x_position += spacing[0]
                
                y_position -= spacing[1]
            
            y_position -= 0.05  # Space between legends
        
        return self
    
    def get_legend_info(self, legend_name: str) -> Optional[Dict[str, str]]:
        """
        Get color mapping for the specified legend
        
        Args:
            legend_name: Legend name
        
        Returns:
            Dict: Color mapping dictionary, return None if not exists
        """
        return self.legend_dict.get(legend_name)
    
    def has_legend(self, legend_name: str) -> bool:
        """
        Check if a legend with the specified name exists
        
        Args:
            legend_name: Legend name
        
        Returns:
            bool: Whether the legend exists
        """
        return legend_name in self.legend_dict
    
    def get_legend_names(self) -> List[str]:
        """
        Get all legend names
        
        Returns:
            List: List of legend names
        """
        return list(self.legend_dict.keys())
    
    def count_legends(self) -> int:
        """
        Get the number of legends
        
        Returns:
            int: Number of legends
        """
        return len(self.legend_dict)
    
    def update_legend_colors(self, legend_name: str, color_updates: Dict[str, str]) -> 'LegendManager':
        """
        Update colors for the specified legend
        
        Args:
            legend_name: Legend name
            color_updates: Color mapping to update
        
        Returns:
            LegendManager: Support method chaining
        """
        if legend_name in self.legend_dict:
            self.legend_dict[legend_name].update(color_updates)
        return self
    
    @staticmethod
    def create_from_dict(legend_dict: Dict[str, Dict[str, str]], ax: Optional[Axes.Axes] = None) -> 'LegendManager':
        """
        Create legend manager from dictionary
        
        Args:
            legend_dict: Complete legend dictionary
            ax: matplotlib axis object
        
        Returns:
            LegendManager: New legend manager instance
        """
        manager = LegendManager(ax)
        manager.legend_dict = legend_dict.copy()
        return manager

    def add_shape_legend(self, legend_name: str, shape_dict: Dict[str, str]) -> 'LegendManager':
        """
        Add shape legend information
        
        Args:
            legend_name: Legend name, such as 'Sex', 'Stage', etc.
            shape_dict: Shape mapping dictionary, such as {'M': 'o', 'F': 's'}
        
        Returns:
            LegendManager: Support method chaining
        """
        self.shape_dict[legend_name] = shape_dict
        return self

    def add_numeric_legend(self, 
                          legend_name: str, 
                          colormap: str, 
                          vmin: float, 
                          vmax: float, 
                          label: Optional[str] = None) -> 'LegendManager':
        """
        Add numeric colorbar legend
        
        Args:
            legend_name: Legend name
            colormap: Colormap name (e.g., 'viridis', 'plasma')
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            label: Colorbar label (default to legend_name)
        
        Returns:
            LegendManager: Support method chaining
        """
        self.numeric_legends[legend_name] = {
            'colormap': colormap,
            'vmin': vmin,
            'vmax': vmax,
            'label': label or legend_name
        }
        return self

    def plot_pca_legends(self,
                        ax: Optional[Axes.Axes] = None,
                        color_legend: Optional[str] = None,
                        shape_legend: Optional[str] = None,
                        numeric_legend: Optional[str] = None,
                        fontsize: int = 12,
                        title_fontsize: int = 12,
                        legend_item_spacing: float = 0.04,
                        legend_group_spacing: float = 0.06) -> 'LegendManager':
        """
        Plot PCA-style legends with support for color, shape, and numeric legends
        
        Args:
            ax: matplotlib axis object
            color_legend: Name of categorical color legend to display
            shape_legend: Name of shape legend to display
            numeric_legend: Name of numeric colorbar legend to display
            fontsize: Font size for legend items
            title_fontsize: Font size for legend titles
            legend_item_spacing: Vertical spacing between items within the same legend group
            legend_group_spacing: Vertical spacing between different legend groups
        
        Returns:
            LegendManager: Support method chaining
        """
        target_ax = ax if ax is not None else self.ax
        if target_ax is None:
            raise ValueError("No axis provided. Please set axis using set_axis() or provide ax parameter.")
        
        target_ax.clear()
        target_ax.axis('off')
        target_ax.set_xlim(0, 1)
        target_ax.set_ylim(0, 1)
        
        y_position = 0.95
        
        # Plot numeric colorbar if specified
        if numeric_legend and numeric_legend in self.numeric_legends:
            info = self.numeric_legends[numeric_legend]
            
            # Create vertical colorbar with proper mappable
            cmap = get_cmap(info['colormap'])
            norm = Normalize(vmin=info['vmin'], vmax=info['vmax'])
            sm = ScalarMappable(norm=norm, cmap=cmap)
            data_range = np.linspace(info['vmin'], info['vmax'], 100)
            sm.set_array(data_range)
            
            # 清空 legend 區域
            target_ax.clear()
            target_ax.set_xlim(0, 1)
            target_ax.set_ylim(0, 1)
            target_ax.axis('off')
            
            # 用 inset_axes 產生一個窄且與主圖等高的 colorbar
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cax = inset_axes(target_ax, 
                             width="30%", 
                             height="100%", 
                             loc='center left', 
                             borderpad=0, 
                             )
            cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
            cbar.set_label(info['label'], fontsize=fontsize, labelpad=10)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(
                labelsize=fontsize-2, length=4, width=0.5, color="gray", labelcolor="black", direction='out')
            vmin, vmax = info['vmin'], info['vmax']
            if max(abs(vmin), abs(vmax)) < 0.01 or max(abs(vmin), abs(vmax)) > 1000:
                cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            else:
                cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            cbar.set_ticks(np.linspace(vmin, vmax, 5))
            return self
        
        # Plot categorical color legend if specified
        if color_legend and color_legend in self.legend_dict:
            target_ax.text(0.05, y_position, color_legend, fontsize=title_fontsize, fontweight='bold', 
                          transform=target_ax.transAxes)
            y_position -= 0.05
            
            color_dict = self.legend_dict[color_legend]
            for label, color in color_dict.items():
                # Color rectangle
                rect = Rectangle((0.05, y_position-0.015), 0.06, 0.025, 
                               facecolor=color, transform=target_ax.transAxes)
                target_ax.add_patch(rect)
                # Color label
                target_ax.text(0.15, y_position, str(label), fontsize=fontsize-1, 
                              va='center', transform=target_ax.transAxes)
                y_position -= legend_item_spacing
            
            y_position -= legend_group_spacing
        
        # Plot shape legend if specified
        if shape_legend and shape_legend in self.shape_dict:
            target_ax.text(0.05, y_position, shape_legend, fontsize=title_fontsize, fontweight='bold',
                          transform=target_ax.transAxes)
            y_position -= 0.05
            
            shape_dict = self.shape_dict[shape_legend]
            for label, marker in shape_dict.items():
                # Shape marker
                target_ax.scatter([0.08], [y_position], marker=marker,  # type: ignore
                                c='black', s=40, transform=target_ax.transAxes)
                # Shape label
                target_ax.text(0.15, y_position, str(label), fontsize=fontsize-1, 
                              va='center', transform=target_ax.transAxes)
                y_position -= legend_item_spacing
        
        return self
