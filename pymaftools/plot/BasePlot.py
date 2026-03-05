"""Base plotting class that provides legend management functionality."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .ColorManager import ColorManager
from .LegendManager import LegendManager


class BasePlot:
    """
    Base class for all plotting classes.

    Provides common functionality for legend management and color management.
    """

    def __init__(self) -> None:
        """
        Initialize BasePlot.

        Sets up legend manager, color manager, and figure-related attributes.
        """
        # Initialize legend manager
        self.legend_manager: LegendManager = LegendManager()

        # Initialize color manager
        self.color_manager: ColorManager = ColorManager()

        # Figure-related attributes
        self.fig: Figure | None = None
        self.ax_legend: Axes | None = None

    def add_legend(self, legend_name: str, color_dict: dict[str, str]) -> BasePlot:
        """
        Add legend information to LegendManager.

        Parameters
        ----------
        legend_name : str
            Legend name, such as 'mutation', 'sex', 'subtype', etc.
        color_dict : dict[str, str]
            Color mapping dictionary, such as ``{'M': 'blue', 'F': 'red'}``.

        Returns
        -------
        BasePlot
            Self, to support method chaining.
        """
        self.legend_manager.add_legend(legend_name, color_dict)
        return self

    def plot_all_legends(
        self,
        ax: Axes | None = None,
        fontsize: int = 8,
        title_fontsize: int = 10,
        legend_spacing: float = 0.08,
        item_spacing: float = 0.02,
    ) -> BasePlot:
        """
        Plot all legends on the specified axis.

        Parameters
        ----------
        ax : Axes or None, optional
            Axis to plot legends on. Uses ``self.ax_legend`` if *None*.
        fontsize : int, optional
            Legend text font size.
        title_fontsize : int, optional
            Legend title font size.
        legend_spacing : float, optional
            Spacing between different legends.
        item_spacing : float, optional
            Spacing between items within the same legend.

        Returns
        -------
        BasePlot
            Self, to support method chaining.
        """
        target_ax = ax if ax is not None else self.ax_legend
        if target_ax is None:
            raise ValueError(
                "No axis available for legend plotting. "
                "Please provide an axis or set self.ax_legend."
            )

        # Set axis for LegendManager and plot legends
        self.legend_manager.set_axis(target_ax).plot_legends(
            fontsize=fontsize,
            title_fontsize=title_fontsize,
            legend_spacing=legend_spacing,
            item_spacing=item_spacing,
        )
        return self

    def get_legend_manager(self) -> LegendManager:
        """
        Get the legend manager instance.

        Returns
        -------
        LegendManager
            The legend manager instance.
        """
        return self.legend_manager

    def remove_legend(self, legend_name: str) -> BasePlot:
        """
        Remove a specified legend.

        Parameters
        ----------
        legend_name : str
            Name of the legend to remove.

        Returns
        -------
        BasePlot
            Self, to support method chaining.
        """
        self.legend_manager.remove_legend(legend_name)
        return self

    def clear_legends(self) -> BasePlot:
        """
        Clear all legends.

        Returns
        -------
        BasePlot
            Self, to support method chaining.
        """
        self.legend_manager.clear_legends()
        return self

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
        return self.legend_manager.has_legend(legend_name)

    def get_legend_names(self) -> list[str]:
        """
        Get all legend names.

        Returns
        -------
        list[str]
            List of legend names.
        """
        return self.legend_manager.get_legend_names()

    def get_color_manager(self) -> ColorManager:
        """
        Get the color manager instance.

        Returns
        -------
        ColorManager
            The color manager instance.
        """
        return self.color_manager

    def save(
        self,
        filename: str,
        dpi: int = 300,
        bbox_inches: str = "tight",
        transparent: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Save figure to file.

        Parameters
        ----------
        filename : str
            File name.
        dpi : int, optional
            Resolution.
        bbox_inches : str, optional
            Bounding box setting.
        transparent : bool, optional
            Whether to use transparent background.
        **kwargs : Any
            Other save parameters passed to ``fig.savefig``.
        """
        if self.fig is None:
            print("Figure is not exist.")
            return

        try:
            format = filename.split(".")[-1].lower()

            # Prepare save arguments
            save_kwargs: dict[str, Any] = {
                "dpi": dpi,
                "bbox_inches": bbox_inches,
                "transparent": transparent,
                "format": format,
            }

            # Only add pil_kwargs for formats that support it (not SVG/PDF)
            if format in ["tiff", "tif", "png", "jpg", "jpeg"]:
                if format in ["tiff", "tif"]:
                    save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}

            # Add any additional kwargs
            save_kwargs.update(kwargs)

            self.fig.savefig(filename, **save_kwargs)
            print(f"[INFO] Figure saved to: {filename}")
        except Exception as e:
            print(f"Error while saving figure: {e}")

    def close(self) -> None:
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
