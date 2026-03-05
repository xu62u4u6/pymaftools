from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize
import numpy as np
from typing import Any
import matplotlib.axes as Axes
import matplotlib.gridspec as gridspec


class LegendManager:
    """
    Specialized class for legend management.

    Responsible for managing and plotting various types of legends, including
    mutation types, categorical variables, numeric variables, etc.
    Supports advanced features for PCA plots including shape markers, numeric
    colorbars, and mixed legends.
    """

    def __init__(self, ax: Axes.Axes | None = None):
        """
        Initialize legend manager.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis object; if None, axis must be provided when plotting.
        """
        self.ax = ax
        self.legend_dict: dict[str, dict[str, str]] = {}
        self.shape_dict: dict[str, dict[str, str]] = {}  # For shape legends
        self.numeric_legends: dict[str, dict[str, Any]] = {}  # For numeric colorbars

    def set_axis(self, ax: Axes.Axes) -> LegendManager:
        """
        Set the axis for legend plotting.

        Parameters
        ----------
        ax : Axes.Axes
            matplotlib axis object.

        Returns
        -------
        LegendManager
            Support method chaining.
        """
        self.ax = ax
        return self

    def add_legend(self, legend_name: str, color_dict: dict[str, str]) -> LegendManager:
        """
        Add legend information.

        Parameters
        ----------
        legend_name : str
            Legend name, such as 'Mutation', 'Subtype', 'Sex', etc.
        color_dict : dict[str, str]
            Color mapping dictionary, such as ``{'M': 'blue', 'F': 'red'}``.

        Returns
        -------
        LegendManager
            Support method chaining.
        """
        self.legend_dict[legend_name] = color_dict
        return self

    def remove_legend(self, legend_name: str) -> LegendManager:
        """
        Remove a specified legend.

        Parameters
        ----------
        legend_name : str
            Name of the legend to remove.

        Returns
        -------
        LegendManager
            Support method chaining.
        """
        if legend_name in self.legend_dict:
            del self.legend_dict[legend_name]
        return self

    def clear_legends(self) -> LegendManager:
        """
        Clear all legends.

        Returns
        -------
        LegendManager
            Support method chaining.
        """
        self.legend_dict.clear()
        return self

    def get_legend_dict(self) -> dict[str, dict[str, str]]:
        """
        Get the current legend dictionary.

        Returns
        -------
        dict[str, dict[str, str]]
            Complete legend dictionary.
        """
        return self.legend_dict.copy()

    def plot_legends(self,
                    ax: Axes.Axes | None = None,
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
                    legend_gap: float = 0.03) -> LegendManager:
        """
        Plot all legends.

        Parameters
        ----------
        ax : Axes.Axes, optional
            matplotlib axis object, use initialization axis if None.
        fontsize : int
            Legend item font size.
        title_fontsize : int
            Legend title font size.
        legend_spacing : float
            Spacing between different legends.
        item_spacing : float
            Spacing between items within the same legend.
        start_y : float
            Starting Y position.
        rect_width : float
            Color rectangle width.
        rect_height : float
            Color rectangle height.
        text_offset_x : float
            Text X offset relative to color rectangle.
        title_offset_y : float
            Y offset after title.
        item_offset_y : float
            Y offset between items.
        legend_gap : float
            Gap between legends.

        Returns
        -------
        LegendManager
            Support method chaining.
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
                              ax: Axes.Axes | None = None,
                              fontsize: int = 8,
                              title_fontsize: int = 10,
                              columns: int = 3,
                              rect_size: tuple[float, float] = (0.03, 0.03),
                              spacing: tuple[float, float] = (0.15, 0.1)) -> LegendManager:
        """
        Plot legends in horizontal layout.

        Parameters
        ----------
        ax : Axes.Axes, optional
            matplotlib axis object.
        fontsize : int
            Item font size.
        title_fontsize : int
            Title font size.
        columns : int
            Number of items per row.
        rect_size : Tuple[float, float]
            Color rectangle size (width, height).
        spacing : Tuple[float, float]
            Item spacing (horizontal spacing, vertical spacing).

        Returns
        -------
        LegendManager
            Support method chaining.
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

    def get_legend_info(self, legend_name: str) -> dict[str, str] | None:
        """
        Get color mapping for the specified legend.

        Parameters
        ----------
        legend_name : str
            Legend name.

        Returns
        -------
        dict[str, str] or None
            Color mapping dictionary, None if not found.
        """
        return self.legend_dict.get(legend_name)

    def has_legend(self, legend_name: str) -> bool:
        """
        Check if a legend with the specified name exists.

        Parameters
        ----------
        legend_name : str
            Legend name.

        Returns
        -------
        bool
            Whether the legend exists.
        """
        return legend_name in self.legend_dict

    def get_legend_names(self) -> list[str]:
        """
        Get all legend names.

        Returns
        -------
        list[str]
            List of legend names.
        """
        return list(self.legend_dict.keys())

    def count_legends(self) -> int:
        """
        Get the number of legends.

        Returns
        -------
        int
            Number of legends.
        """
        return len(self.legend_dict)

    def update_legend_colors(self, legend_name: str, color_updates: dict[str, str]) -> LegendManager:
        """
        Update colors for the specified legend.

        Parameters
        ----------
        legend_name : str
            Legend name.
        color_updates : dict[str, str]
            Color mapping to update.

        Returns
        -------
        LegendManager
            Support method chaining.
        """
        if legend_name in self.legend_dict:
            self.legend_dict[legend_name].update(color_updates)
        return self

    @staticmethod
    def create_from_dict(legend_dict: dict[str, dict[str, str]], ax: Axes.Axes | None = None) -> LegendManager:
        """
        Create legend manager from dictionary.

        Parameters
        ----------
        legend_dict : dict[str, dict[str, str]]
            Complete legend dictionary.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis object.

        Returns
        -------
        LegendManager
            New legend manager instance.
        """
        manager = LegendManager(ax)
        manager.legend_dict = legend_dict.copy()
        return manager

    def add_shape_legend(self, legend_name: str, shape_dict: dict[str, str]) -> LegendManager:
        """
        Add shape legend information.

        Parameters
        ----------
        legend_name : str
            Legend name, such as 'Sex', 'Stage', etc.
        shape_dict : dict[str, str]
            Shape mapping dictionary, such as ``{'M': 'o', 'F': 's'}``.

        Returns
        -------
        LegendManager
            Support method chaining.
        """
        self.shape_dict[legend_name] = shape_dict
        return self

    def add_numeric_legend(self,
                          legend_name: str,
                          colormap: str,
                          vmin: float,
                          vmax: float,
                          label: str | None = None) -> LegendManager:
        """
        Add numeric colorbar legend.

        Parameters
        ----------
        legend_name : str
            Legend name.
        colormap : str
            Colormap name (e.g., 'viridis', 'plasma').
        vmin : float
            Minimum value for colormap.
        vmax : float
            Maximum value for colormap.
        label : str, optional
            Colorbar label (default to legend_name).

        Returns
        -------
        LegendManager
            Support method chaining.
        """
        self.numeric_legends[legend_name] = {
            'colormap': colormap,
            'vmin': vmin,
            'vmax': vmax,
            'label': label or legend_name
        }
        return self

    def add_numeric_colorbar(self,
                            legend_name: str,
                            scatter_obj: Any,
                            target_ax: Axes.Axes,
                            label: str | None = None,
                            orientation: str = 'vertical',
                            fraction: float = 0.08,
                            pad: float = 0.02) -> LegendManager:
        """
        Add numeric colorbar directly to the target axis with custom format.

        Parameters
        ----------
        legend_name : str
            Legend name.
        scatter_obj : Any
            The scatter plot object for colorbar.
        target_ax : Axes.Axes
            Target axis to add colorbar to.
        label : str, optional
            Colorbar label (default to legend_name).
        orientation : str
            Colorbar orientation ('vertical' or 'horizontal').
        fraction : float
            Size of colorbar relative to parent axis.
        pad : float
            Padding between colorbar and parent axis.

        Returns
        -------
        LegendManager
            Support method chaining.
        """
        # Get the figure from the target axis
        fig = target_ax.figure

        # Create colorbar with custom format
        cbar = fig.colorbar(scatter_obj, ax=target_ax,
                           orientation=orientation,
                           fraction=fraction,
                           pad=pad)
        cbar.set_label(label or legend_name)
        cbar.outline.set_linewidth(0)

        return self

    def plot_pca_legends(self,
                        ax: Axes.Axes | None = None,
                        color_legend: str | None = None,
                        shape_legend: str | None = None,
                        numeric_legend: str | None = None,
                        fontsize: int = 12,
                        title_fontsize: int = 12,
                        legend_item_spacing: float = 0.04,
                        legend_group_spacing: float = 0.06) -> LegendManager:
        """
        Plot PCA-style legends with support for color, shape, and numeric legends.

        Parameters
        ----------
        ax : Axes.Axes, optional
            matplotlib axis object.
        color_legend : str, optional
            Name of categorical color legend to display.
        shape_legend : str, optional
            Name of shape legend to display.
        numeric_legend : str, optional
            Name of numeric colorbar legend to display.
        fontsize : int
            Font size for legend items.
        title_fontsize : int
            Font size for legend titles.
        legend_item_spacing : float
            Vertical spacing between items within the same legend group.
        legend_group_spacing : float
            Vertical spacing between different legend groups.

        Returns
        -------
        LegendManager
            Support method chaining.
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

            # Create colorbar
            cmap = get_cmap(info['colormap'])
            norm = Normalize(vmin=info['vmin'], vmax=info['vmax'])
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

            # Add colorbar to the axis
            cbar_ax = target_ax.inset_axes([0.1, y_position-0.15, 0.8, 0.03])
            cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(info['label'], fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize-2)

            y_position -= 0.25

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
