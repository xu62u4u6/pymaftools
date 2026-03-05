from __future__ import annotations

import os


import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


class FontManager:
    """
    Manager for registering and configuring fonts in matplotlib.

    Provides utilities for registering local font files, scanning
    directories for fonts, and applying font settings globally to
    matplotlib's rcParams.

    Attributes
    ----------
    WEIGHT_MAP : dict
        Mapping from weight name strings to numeric weight values.
    """

    WEIGHT_MAP = {
        "ultralight": 100,
        "light": 200,
        "normal": 400,
        "regular": 400,
        "book": 400,
        "medium": 500,
        "semibold": 600,
        "demibold": 600,
        "bold": 700,
        "heavy": 800,
        "extra bold": 800,
        "black": 900,
    }

    def __init__(self) -> None:
        """Initialize FontManager."""
        pass

    def get_available_fonts(self) -> list[str]:
        """
        Return a sorted list of available font family names.

        Returns
        -------
        list[str]
            Sorted unique font family names registered in matplotlib.
        """
        return sorted(set(f.name for f in fm.fontManager.ttflist))

    @staticmethod
    def normalize_weight(weight: str | int) -> int:
        """
        Normalize a font weight to its numeric value.

        Parameters
        ----------
        weight : str or int
            Font weight as a string name (e.g. "bold") or numeric value.

        Returns
        -------
        int
            Numeric font weight value.

        Raises
        ------
        ValueError
            If the weight string or value is not recognized.
        """
        if isinstance(weight, int):
            if weight not in FontManager.WEIGHT_MAP.values():
                raise ValueError(f"Invalid weight value: {weight}. Must be one of {list(FontManager.WEIGHT_MAP.values())}.")
            return weight
        if isinstance(weight, str):
            if weight.lower() not in FontManager.WEIGHT_MAP:
                raise ValueError(f"Invalid weight string: {weight}. Must be one of {list(FontManager.WEIGHT_MAP.keys())}.")
            return FontManager.WEIGHT_MAP[weight.lower()]

    def entry_equal_properties(self, entry: fm.FontEntry, font_prop: fm.FontProperties) -> bool:
        """
        Check whether a font entry matches the given font properties.

        Parameters
        ----------
        entry : matplotlib.font_manager.FontEntry
            A registered font entry to compare.
        font_prop : matplotlib.font_manager.FontProperties
            Target font properties.

        Returns
        -------
        bool
            True if name, weight, and style all match.
        """
        target_name = font_prop.get_name()
        target_weight = font_prop.get_weight()
        target_style = font_prop.get_style()

        return (entry.name == target_name and
                FontManager.normalize_weight(entry.weight) == FontManager.normalize_weight(target_weight) and
                entry.style == target_style)

    def register_local_font(self, font_path: str) -> str:
        """
        Register a local ``.ttf`` font file.

        Skips registration if a font with the same name, weight, and
        style is already registered.

        Parameters
        ----------
        font_path : str
            Path to a ``.ttf`` font file.

        Returns
        -------
        str
            The registered font family name.
        """
        font_prop = fm.FontProperties(fname=font_path)
        target_name = font_prop.get_name()
        target_weight = font_prop.get_weight()
        target_style = font_prop.get_style()

        # Check for existing font with same name, weight, and style
        for entry in fm.fontManager.ttflist:
            if self.entry_equal_properties(entry, font_prop):
                print(f"Font already registered: {target_name} (weight: {target_weight}, style: {target_style})")
                return target_name

        # If not registered, add it
        fm.fontManager.addfont(font_path)
        print(f"Registered font: {target_name}")
        print(f"Weight: {target_weight}")
        print(f"Style: {target_style}")
        return target_name

    def register_fonts_from_directory(self, directory: str) -> None:
        """
        Recursively register all ``.ttf`` fonts from a directory.

        Parameters
        ----------
        directory : str
            Path to a directory containing font files.

        Raises
        ------
        ValueError
            If the directory does not exist.
        """
        if not os.path.isdir(directory):
            raise ValueError(f"Directory '{directory}' does not exist.")

        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith('.ttf'):
                    font_path = os.path.join(root, filename)
                    self.register_local_font(font_path)

    def setup_matplotlib_fonts(
        self,
        font_family: str = "Source Sans Pro",
        font_weight: str = "normal",
        base_size: int = 12,
        fallback_fonts: list[str] | None = None,
    ) -> str:
        """
        Configure matplotlib font settings globally.

        Parameters
        ----------
        font_family : str, default "Source Sans Pro"
            Desired font family name.
        font_weight : str, default "normal"
            Font weight string (e.g. "normal", "bold").
        base_size : int, default 12
            Base font size in points.
        fallback_fonts : list of str, optional
            Ordered list of fallback font names if *font_family* is
            unavailable.

        Returns
        -------
        str
            The font family name that was actually applied.

        Raises
        ------
        ValueError
            If *font_family* is not found and no fallback fonts are
            provided or available.
        """
        available = self.get_available_fonts()

        # Fallback handling
        if font_family not in available:
            print(f"Font '{font_family}' is not available. Trying fallbacks...")
            if fallback_fonts:
                for fb in fallback_fonts:
                    if fb in available:
                        print(f"Using fallback font: {fb}")
                        font_family = fb
                        break
                else:
                    print("No fallback fonts available. Using default.")
                    font_family = 'DejaVu Sans'
            else:
                raise ValueError(f"Font '{font_family}' not found. Please register it or provide fallback_fonts.")

        # Apply font settings to matplotlib
        plt.rcParams.update({
            'font.family': [font_family],
            'font.weight': font_weight,
            'font.size': base_size,
            'axes.titlesize': base_size + 2,
            'axes.labelsize': base_size,
            'xtick.labelsize': base_size - 1,
            'ytick.labelsize': base_size - 1,
            'legend.fontsize': base_size - 1,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'text.antialiased': True,
            'axes.unicode_minus': False,
        })

        print(f"Matplotlib font configuration set to: {font_family} (weight: {font_weight})")
        return font_family
