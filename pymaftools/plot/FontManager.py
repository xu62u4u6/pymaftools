import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

class FontManager:
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

    def __init__(self):
        pass  # You may initialize font paths or other configs here if needed

    def get_available_fonts(self):
        """Return a sorted list of available font family names"""
        return sorted(set(f.name for f in fm.fontManager.ttflist))

    @staticmethod
    def normalize_weight(weight: str | int):
        if isinstance(weight, int):
            if weight not in FontManager.WEIGHT_MAP.values():
                raise ValueError(f"Invalid weight value: {weight}. Must be one of {list(FontManager.WEIGHT_MAP.values())}.")
            return weight
        if isinstance(weight, str):
            if weight.lower() not in FontManager.WEIGHT_MAP:
                raise ValueError(f"Invalid weight string: {weight}. Must be one of {list(FontManager.WEIGHT_MAP.keys())}.")
            return FontManager.WEIGHT_MAP[weight.lower()]

    def entry_equal_properties(self, entry, font_prop):
        target_name = font_prop.get_name()
        target_weight = font_prop.get_weight()
        target_style = font_prop.get_style()

        return (entry.name == target_name and 
                FontManager.normalize_weight(entry.weight) == FontManager.normalize_weight(target_weight) and 
                entry.style == target_style)

    def register_local_font(self, font_path):
        """Register a local .ttf font file only if its name, weight, and style are not already registered"""
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

    def register_fonts_from_directory(self, directory):
        """Recursively register all .ttf fonts from a specified directory and its subdirectories"""
        if not os.path.isdir(directory):
            raise ValueError(f"Directory '{directory}' does not exist.")
        
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith('.ttf'):
                    font_path = os.path.join(root, filename)
                    self.register_local_font(font_path)

    def setup_matplotlib_fonts(self, font_family="Source Sans Pro", font_weight="normal", base_size=12, fallback_fonts=None):
        """
        Configure matplotlib font settings
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