import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
from typing import Dict, Union, Optional, List, Any
from matplotlib.colors import to_rgb, to_hex
import colorsys


class ColorManager:
    """
    Unified color mapping and palette management for OncoPlot visualizations.

    This class provides centralized management of color schemes, including
    predefined color maps for different mutation types, CNV data, and utilities
    for generating categorical color mappings.
    """

    # Predefined colormaps
    NONSYNONYMOUS_CMAP = {
        'False': '#FFFFFF',
        'Missense_Mutation': 'gray',
        'Frame_Shift_Ins': '#FF4500',     # Dark red
        'Frame_Shift_Del': "#3E85C0",    # Dark blue
        'In_Frame_Ins': "#E8656E",       # Light red
        'In_Frame_Del': "#ADDBEA",        # Light blue
        'Nonsense_Mutation': "#98E28F",  # Low-saturation green
        'Splice_Site': "#D0875D",        # Low-saturation brown
        'Multi_Hit': "#222222",          # Black (multiple mutations)
    }

    ALL_MUTATION_CMAP = NONSYNONYMOUS_CMAP | {
        "Silent": "#eeeeee",
        "3'UTR": "#bbbbcc",
        "5'UTR": "#bbbbcc",
        "IGR": "#bbbbcc",
        "Intron": "#bbbbcc",
        "RNA": "#bbbbcc",
    }

    CNV_CMAP = {
        "AMP": "salmon",
        "DEL": "steelblue",
        "AMP&DEL": "gray"
    }

    predefined_cmaps = {
        'all_mutation': ALL_MUTATION_CMAP,
        'nonsynonymous': NONSYNONYMOUS_CMAP,
        'cnv': CNV_CMAP,
    }

    def __init__(self):
        """Initialize ColorManager with empty custom colormap registry."""
        self.custom_cmaps = {}

    def add_cmap(self, name: str, cmap: Dict[str, str]) -> None:
        """
        Register a custom colormap for later use.

        Parameters
        ----------
        name : str
            Name identifier for the colormap
        cmap : dict
            Dictionary mapping categories to color values
        """
        self.custom_cmaps[name] = cmap

    def get_cmap(self, name: str, factor: Optional[float] = None, alpha: Optional[float] = None) -> Dict[str, str]:
        """
        Retrieve a colormap by name, optionally adjusting brightness and simulated alpha.

        Parameters
        ----------
        name : str
            Name of the colormap to retrieve
        factor : float, optional
            Brightness factor (>1 = brighter, <1 = darker)
        alpha : float, optional
            Simulated alpha blending with white background (0–1)

        Returns
        -------
        dict
            Adjusted color mapping
        """
        if name in self.custom_cmaps:
            cmap = self.custom_cmaps[name]
        elif name in self.predefined_cmaps:
            cmap = self.predefined_cmaps[name]
        else:
            raise ValueError(f"Unknown colormap: {name}")

        adjusted_cmap = {}

        for k, v in cmap.items():
            color = v
            if factor is not None:
                color = self.adjust_color_brightness(color, factor)
            if alpha is not None:
                color = self.simulate_alpha_blend(color, alpha)
            adjusted_cmap[k] = color

        return adjusted_cmap

    def simulate_alpha_blend(self, color: str, alpha: float, background: str = "#FFFFFF") -> str:
        """
        模擬 alpha 效果，將 color 與背景色混合，實現 alpha 混色後的結果（返回 hex）。
        """
        fg_rgb = to_rgb(color)  # 會自動處理 hex or name
        bg_rgb = to_rgb(background)
        blended_rgb = [(alpha * f + (1 - alpha) * b)
                       for f, b in zip(fg_rgb, bg_rgb)]
        return to_hex(tuple(blended_rgb))

    def generate_categorical_cmap(self,
                                  data: Union[pd.DataFrame, pd.Series],
                                  custom_cmap: Optional[Dict[str, str]] = None,
                                  default_palette: str = "Set1") -> Dict[str, str]:
        """
        Generate color mapping for categorical data.

        Automatically creates a color mapping for unique categories in the data,
        with optional custom color overrides.

        Parameters
        ----------
        data : DataFrame or Series
            Data containing categorical values
        custom_cmap : dict, optional
            Custom color mapping to override defaults
        default_palette : str, default "Set1"
            Name of seaborn palette for default colors

        Returns
        -------
        dict
            Mapping from categories to color values
        """
        # 獲取唯一類別
        if isinstance(data, pd.DataFrame):
            unique_categories = pd.unique(data.values.ravel())
        else:
            unique_categories = data.unique()

        # 移除 NaN 值
        unique_categories = [cat for cat in unique_categories if pd.notna(cat)]

        # 生成默認顏色
        palette = sns.color_palette(default_palette, len(unique_categories))
        default_color_dict = {}
        for i, cat in enumerate(unique_categories):
            # Convert RGB tuple to hex string
            rgb_color = palette[i]
            default_color_dict[cat] = to_hex(rgb_color)

        # 如果有自定義映射，則覆蓋默認顏色
        if custom_cmap:
            for category, color in custom_cmap.items():
                if category in default_color_dict:
                    default_color_dict[category] = color

        return default_color_dict

    def apply_cmap_to_data(self,
                           data: Union[pd.DataFrame, pd.Series],
                           cmap: Dict[str, str],
                           missing_color: str = "#FFFFFF") -> Union[pd.DataFrame, pd.Series]:
        """
        Apply color mapping to data values.

        Parameters
        ----------
        data : DataFrame or Series
            Data to apply color mapping to
        cmap : dict
            Color mapping dictionary
        missing_color : str, default "#FFFFFF"
            Color for missing or unmapped values

        Returns
        -------
        DataFrame or Series
            Data with values replaced by corresponding colors
        """
        return data.map(lambda x: cmap.get(x, missing_color))

    def create_matplotlib_cmap(self,
                               categories: List[str],
                               colors: List[str],
                               unknown_color: str = "white") -> ListedColormap:
        """
        Create matplotlib ListedColormap from categories and colors.

        Parameters
        ----------
        categories : list
            List of category names
        colors : list
            Corresponding list of color values
        unknown_color : str, default "white"
            Color for unknown/unmapped categories

        Returns
        -------
        ListedColormap
            Matplotlib colormap object
        """
        color_list = list(colors)
        color_list.append(unknown_color)  # 為未知類別添加顏色
        return ListedColormap(color_list)

    def adjust_color_brightness(self, color: str, factor: float) -> str:
        """
        Adjust the brightness of a color using HLS color space.

        Parameters
        ----------
        color : str
            Any valid matplotlib color (hex, name, etc.)
        factor : float
            Brightness adjustment factor:
            - 1.0 = no change
            - <1.0 = darker
            - >1.0 = brighter

        Returns
        -------
        str
            Adjusted color in hex format
        """
        # Convert color to RGB (0–1 float)
        r, g, b = to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(r, g, b)

        # 調整亮度
        l = max(0, min(1, l * factor))

        # 轉回 RGB
        r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)

        # 轉成 hex 並回傳
        return to_hex((r_new, g_new, b_new))

    def generate_cmap_from_list(self, 
                               categories: List[Any], 
                               cmap_name: str = "tab20",
                               as_hex: bool = True) -> Dict[str, str]:
        """
        Generate a color mapping dictionary from a list of categories using a matplotlib colormap.

        This method automatically creates evenly spaced colors from the specified colormap
        and maps them to the provided categories. Very useful for creating consistent 
        color mappings for categorical data like case_IDs, sample_types, etc.

        Parameters
        ----------
        categories : list
            List of unique categories to create color mapping for
        cmap_name : str, default "tab20"
            Name of matplotlib colormap to use. Popular choices:
            - "tab20": 20 distinct colors, good for many categories
            - "Set1": 9 bright colors, good for fewer categories  
            - "Set3": 12 pastel colors
            - "viridis", "plasma": continuous colormaps
        as_hex : bool, default True
            If True, return colors as hex strings. If False, return as RGBA tuples.

        Returns
        -------
        dict
            Dictionary mapping categories to color values

        Examples
        --------
        >>> cm = ColorManager()
        >>> case_ids = ["LUAD_001", "LUAD_002", "ASC_001", "ASC_002"]
        >>> colors = cm.generate_cmap_from_list(case_ids, "tab20")
        >>> print(colors)
        {"LUAD_001": "#1f77b4", "LUAD_002": "#ff7f0e", ...}

        >>> # For many categories, tab20 works well
        >>> many_categories = [f"sample_{i}" for i in range(15)]
        >>> colors = cm.generate_cmap_from_list(many_categories, "tab20")

        >>> # For fewer categories, Set1 gives more distinct colors
        >>> few_categories = ["Control", "Treatment", "Placebo"]
        >>> colors = cm.generate_cmap_from_list(few_categories, "Set1")
        """

        
        # Get the colormap
        cmap = get_cmap(cmap_name)
        
        # Generate evenly spaced values for the colormap
        num_categories = len(categories)
        if num_categories == 0:
            return {}
        
        # For discrete colormaps like tab20, use integer indices
        # For continuous colormaps, use linspace
        if cmap_name.startswith(('tab', 'Set', 'Pastel', 'Dark2', 'Paired')):
            # Discrete colormap - use indices with modulo for cycling
            # Use getattr to safely get N attribute, fallback to reasonable number
            cmap_size = getattr(cmap, 'N', 20)
            color_values = [cmap(i % cmap_size) for i in range(num_categories)]
        else:
            # Continuous colormap - use evenly spaced values
            color_values = [cmap(i / max(1, num_categories - 1)) for i in range(num_categories)]
        
        # Create the mapping dictionary
        color_dict = {}
        for category, color_rgba in zip(categories, color_values):
            if as_hex:
                # Convert RGBA to hex string
                color_dict[str(category)] = mcolors.to_hex(color_rgba)
            else:
                # Keep as RGBA tuple
                color_dict[str(category)] = color_rgba
        
        return color_dict
