import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
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
        'Frame_Shift_Ins':'#FF4500',     # Dark red
        'Frame_Shift_Del': "#3E85C0",    # Dark blue
        'In_Frame_Ins': "#E8656E",       # Light red
        'In_Frame_Del':"#ADDBEA",        # Light blue
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
        
    def add_cmap(self, name: str, cmap: Dict[str, str], factor: Optional[float] = None) -> None:
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
        blended_rgb = [(alpha * f + (1 - alpha) * b) for f, b in zip(fg_rgb, bg_rgb)]
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
        default_color_dict = {cat: palette[i] for i, cat in enumerate(unique_categories)}
        
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