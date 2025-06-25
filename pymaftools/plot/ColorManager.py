import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from typing import Dict, Union, Optional, List, Any

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
        'Frame_Shift_Del': '#4682B4',    # Dark blue
        'In_Frame_Ins': '#FF707A',       # Light red
        'In_Frame_Del':'#ADD8E6',        # Light blue
        'Nonsense_Mutation': '#90EE90',  # Low-saturation green
        'Splice_Site': '#CB704D',        # Low-saturation brown
        'Multi_Hit': '#000000',          # Black (multiple mutations)
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
        
    def register_cmap(self, name: str, cmap: Dict[str, str]) -> None:
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
        
    def get_cmap(self, name: str) -> Dict[str, str]:
        """
        Retrieve a colormap by name.
        
        Parameters
        ----------
        name : str
            Name of the colormap to retrieve
            
        Returns
        -------
        dict
            Dictionary mapping categories to colors
            
        Raises
        ------
        ValueError
            If the specified colormap name is not found
        """
        # 優先返回自定義的 cmap
        if name in self.custom_cmaps:
            return self.custom_cmaps[name]
            
        # 返回預定義的 cmap
        if name in self.predefined_cmaps:
            return self.predefined_cmaps[name]
        else:
            raise ValueError(f"Unknown colormap: {name}")
    
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
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple:
        """Convert hexadecimal color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def rgb_to_hex(rgb_color: tuple) -> str:
        """Convert RGB tuple to hexadecimal color string."""
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))
    
    def adjust_color_brightness(self, color: str, factor: float) -> str:
        """
        Adjust the brightness of a color.
        
        Parameters
        ----------
        color : str
            Hexadecimal color string
        factor : float
            Brightness adjustment factor (0-1 for darker, >1 for brighter)
            
        Returns
        -------
        str
            Adjusted hexadecimal color string
        """
        rgb = self.hex_to_rgb(color)
        adjusted_rgb = tuple(min(255, max(0, int(c * factor))) for c in rgb)
        return self.rgb_to_hex(adjusted_rgb)
    
