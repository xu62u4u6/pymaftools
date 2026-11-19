"""Base plotting class that provides legend management functionality."""

from __future__ import annotations

import os
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
        **kwargs: Any,
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
            **kwargs,
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
        filename: str | os.PathLike[str],
        dpi: int = 300,
        bbox_inches: str = "tight",
        pad_inches: float = 0.08,
        transparent: bool = False,
        summary: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Save figure to file.

        Errors from the backend are raised (nothing is written silently).

        Parameters
        ----------
        filename : str or os.PathLike
            File name.
        dpi : int, optional
            Resolution.
        bbox_inches : str, optional
            Bounding box setting.
        pad_inches : float, optional
            Padding around the tight bounding box.
        transparent : bool, optional
            Whether to use transparent background.
        summary : bool, default False
            Also write :meth:`summary` as JSON next to the figure
            (``plot.png`` -> ``plot.json``) so a program can check what was
            drawn without looking at the image.
        **kwargs : Any
            Other save parameters passed to ``fig.savefig``.
        """
        if self.fig is None:
            raise RuntimeError(
                f"{type(self).__name__} has no figure to save yet; nothing was "
                f"written to {os.fspath(filename)!r}. Draw it first, e.g. "
                "`table.plot.oncoplot().main().render().save(path)` "
                "(OncoPlot requires .render() before .save())."
            )

        path = os.fspath(filename)
        format = path.split(".")[-1].lower()

        # Prepare save arguments
        save_kwargs: dict[str, Any] = {
            "dpi": dpi,
            "bbox_inches": bbox_inches,
            "pad_inches": pad_inches,
            "transparent": transparent,
            "format": format,
        }

        # Only add pil_kwargs for formats that support it (not SVG/PDF)
        if format in ["tiff", "tif"]:
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}

        # Add any additional kwargs
        save_kwargs.update(kwargs)

        self.fig.savefig(path, **save_kwargs)
        print(f"[INFO] Figure saved to: {path}")
        if summary:
            from .summary import write_figure_summary

            record = self.summary(dpi=dpi)
            sidecar = write_figure_summary(record, path, figure_path=path)
            print(f"[INFO] Figure summary saved to: {sidecar}")

    def summary(self, dpi: float | None = None) -> dict[str, Any]:
        """Machine-readable description of the drawn figure.

        Generic part from :func:`pymaftools.plot.summary.figure_summary`
        (per-axes artist counts, ``empty`` flags, ``warnings``); subclasses add
        domain facts through :meth:`_plot_summary`.

        Parameters
        ----------
        dpi : float, optional
            Output resolution used for pixel-size estimates (defaults to the
            figure's dpi).

        Returns
        -------
        dict
            JSON-serialisable (see :func:`pymaftools.to_json`).
        """
        if self.fig is None:
            raise RuntimeError(
                f"{type(self).__name__} has no figure yet; draw it first "
                "(OncoPlot requires .render())."
            )
        from .summary import figure_summary

        record: dict[str, Any] = {"plot_type": type(self).__name__}
        record.update(figure_summary(self.fig))
        extra = self._plot_summary(dpi=dpi if dpi is not None else self.fig.dpi)
        if extra:
            warnings_list = record["warnings"] + list(extra.pop("warnings", []))
            record.update(extra)
            record["warnings"] = warnings_list
        return record

    def _plot_summary(self, dpi: float) -> dict[str, Any]:
        """Subclass hook: domain-specific summary fields (may include
        ``"warnings"``, which are appended to the generic ones)."""
        return {}

    def close(self) -> None:
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
