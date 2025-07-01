import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.patches as patches
import matplotlib.lines as mlines
from collections import defaultdict
from .BasePlot import BasePlot

class LollipopPlot(BasePlot):
    """
    LollipopPlot class for creating protein domain and mutation visualizations.
    
    This class provides functionality for visualizing protein domains and mutations
    as lollipop plots, inheriting unified legend management and save functionality
    from BasePlot.
    """
    
    def __init__(self, protein_name, protein_length, domains, mutations, config=None):
        """
        Initialize a new LollipopPlot object.

        Parameters:
        -----------
        protein_name : str
            Protein name
        protein_length : int
            Protein amino acid length
        domains : list
            Domains data list
        mutations : list
            Mutation data list
        config : dict, optional
            Configuration for plot
        """
        # Initialize BasePlot
        super().__init__()
        
        self.protein_name = protein_name
        self.protein_length = protein_length
        self.domains = list(domains)
        self.mutations = list(mutations)

        # --- Default Configuration ---
        self._default_config = {
            'figsize': (15, 5),
            'domain_cmap_name': 'tab20c', # Default colormap for domains
            'mutation_colors': {
                'Missense_Mutation': 'gray', 
                'Frame_Shift_Ins':'#FF4500',     # Dark red
                'Frame_Shift_Del': '#4682B4',    # Dark blue
                'In_Frame_Ins': '#FF707A',       # Light red
                'In_Frame_Del':'#ADD8E6',        # Light blue
                'Nonsense_Mutation': '#90EE90',  # Low-saturation green
                'Splice_Site': '#CB704D',        # Low-saturation brown
            },
            'backbone_y': 0,
            'domain_bottom_y': -0.05,
            'domain_height': 0.1,
            'min_lollipop_candy_y_offset': 0.05, # Offset above domain top
            'max_additional_scaled_height': 0.8,
            'legend_domain_bbox': (1.02, 1.0),
            'legend_mutation_bbox': (1.02, 0.7),
            'tight_layout_rect': [0, 0, 0.88, 1]
        }
        # Merge user config with defaults
        self.config = self._default_config.copy()
        if config:
            self.config.update(config)

        # Calculated Attributes
        self.domain_top_y = self.config['domain_bottom_y'] + self.config['domain_height']
        self.min_lollipop_candy_y = self.domain_top_y + self.config['min_lollipop_candy_y_offset']
        self.max_mutation_count = self._calculate_max_mutation_count()
        self.domain_colors = self._generate_domain_colors()
        
        # Store mutation colors in ColorManager for consistency
        self.color_manager.add_cmap('lollipop_mutations', self.config['mutation_colors'])

        # Placeholders for plot objects
        self.ax_main = None

    def _calculate_max_mutation_count(self):
        """Calculate the maximum mutation count from the mutations list."""
        if not self.mutations:
            return 1
        # Ensure max_mutation_count is at least 1 if there are mutations
        max_count = max(max(m.get('count', 0) for m in self.mutations), 1)
        return max_count

    def _generate_domain_colors(self):
        """Generate a color mapping for unique domain labels."""
        if not self.domains:
            return {}
        unique_domain_labels = sorted(list(set(d.get('Label', 'Unknown') for d in self.domains)))
        cmap_name = self.config['domain_cmap_name']
        num_labels = len(unique_domain_labels)

        try:
            # Adjust cmap name based on number of labels if needed (e.g., Pastel1 has limited colors)
            if 'Set2' in cmap_name and num_labels > 9:
                 print(f"Warning: '{cmap_name}' might not have enough distinct colors for {num_labels} domains. Consider 'tab10' or 'tab20'.")
            if num_labels == 0:
                 return {}
            domain_cmap = cm.get_cmap(cmap_name, num_labels if num_labels > 0 else 1)
        except ValueError:
            print(f"Warning: Colormap '{cmap_name}' failed. Using 'viridis'.")
            domain_cmap = cm.get_cmap('viridis', num_labels if num_labels > 0 else 1)

        return {label: domain_cmap(i) for i, label in enumerate(unique_domain_labels)}

    def _setup_plot(self, ax=None, fig=None):
        """Initialize plot with proper figure handling."""
        if ax is None or fig is None:
            self.fig, self.ax_main = plt.subplots(figsize=self.config['figsize'])
        else:
            self.fig = fig
            self.ax_main = ax

        self.ax_main.set_xlim(-10, self.protein_length + 10)
        self.ax_main.set_ylim(self.config['domain_bottom_y'] - 0.1,
                         self.min_lollipop_candy_y + self.config['max_additional_scaled_height'] + 0.1)
        self.ax_main.set_xlabel("Amino Acid Position", fontsize=12)
        self.ax_main.set_title(f'Protein Domains and Mutations for {self.protein_name}', fontsize=16, pad=20)

        # Hide Y-axis and unnecessary spines
        y_ticks = np.linspace(
            self.config['backbone_y'], 
            self.min_lollipop_candy_y + self.config['max_additional_scaled_height'],
            num=5
        )
        mutation_counts = np.round(
            np.linspace(0, self.max_mutation_count, num=5)
        ).astype(int)

        self.ax_main.set_yticks(y_ticks)
        self.ax_main.set_yticklabels(mutation_counts)
        self.ax_main.set_ylabel("Mutation Count", fontsize=12)
        for spine in ['top', 'right', 'left', 'bottom']:
            if spine != 'bottom':
                self.ax_main.spines[spine].set_visible(False)

    def _plot_backbone(self):
        """Plot protein backbone line."""
        if self.ax_main is None: return
        self.ax_main.plot([0, self.protein_length], [self.config['backbone_y'], self.config['backbone_y']],
                     color='black', linewidth=6, solid_capstyle='round', zorder=1)

    def _plot_domains(self):
        """Plot domains and add to legend manager."""
        if self.ax_main is None: return
        domain_colors = {}
        
        for domain in self.domains:
            start = domain.get("Start")
            end = domain.get("End")
            label = domain.get("Label", "Unknown")
            if start is None or end is None: continue # Skip if data missing

            domain_color = self.domain_colors.get(label, 'lightgrey')
            rect = patches.Rectangle((start, self.config['domain_bottom_y']), end - start, self.config['domain_height'],
                                     linewidth=1, edgecolor='black', facecolor=domain_color, zorder=2)
            self.ax_main.add_patch(rect)
            
            # Collect domain colors for legend
            domain_colors[label] = domain_color
        
        # Add domain legend to LegendManager
        if domain_colors:
            self.add_legend("Domains", domain_colors)

    def _plot_mutations(self):
        """Plot lollipop sticks and candies and add to legend manager."""
        if self.ax_main is None: return
        mutation_colors = {}
        
        for mutation in self.mutations:
            pos = mutation.get('position')
            m_type = mutation.get('type', 'Other')
            count = mutation.get('count', 0)
            if pos is None: continue # Skip if data missing

            mutation_color = self.color_manager.get_cmap('lollipop_mutations').get(m_type, '#CCCCCC')

            # Calculate lollipop top Y position
            scaled_height_addition = (count / self.max_mutation_count) * self.config['max_additional_scaled_height']
            lollipop_top_y = self.min_lollipop_candy_y + scaled_height_addition

            # Draw lollipop stick (zorder=0, bottom layer)
            self.ax_main.plot([pos, pos], [self.config['backbone_y'], lollipop_top_y], color='grey', linewidth=1, zorder=0)

            # Draw lollipop candy (zorder=3, top layer)
            marker_size = 50 + (count / self.max_mutation_count) * 100 # Optional scaling
            self.ax_main.scatter(pos, lollipop_top_y, color=mutation_color, s=marker_size, zorder=3, edgecolors='black', linewidth=0.5)

            # Collect mutation colors for legend
            mutation_colors[m_type] = mutation_color
        
        # Add mutation legend to LegendManager
        if mutation_colors:
            self.add_legend("Mutation Types", mutation_colors)

    def plot(self, ax=None, fig=None):
        """
        Plot lollipop plot with protein domains and mutations.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        fig : matplotlib.figure.Figure, optional
            Figure to use
            
        Returns:
        --------
        self : LollipopPlot
            Returns self for method chaining
        """
        self._setup_plot(ax=ax, fig=fig)
        if self.ax_main is None:
             print("Error: Plot setup failed.")
             return self

        self._plot_mutations()
        self._plot_backbone()
        self._plot_domains()

        # Use inherited legend plotting functionality
        if hasattr(self, 'ax_legend') and self.ax_legend is not None:
            # If we have a dedicated legend axis, use it
            self.plot_all_legends(ax=self.ax_legend)
        else:
            # Otherwise, create a simple layout adjustment
            try:
                if self.fig is not None:
                    self.fig.tight_layout(rect=self.config['tight_layout_rect'])
            except (ValueError, AttributeError) as e:
                 print(f"Warning: tight_layout failed. Legends might overlap or be cut off. Error: {e}")
                 # Fallback to default tight_layout if rect causes issues
                 try:
                     if self.fig is not None:
                         self.fig.tight_layout()
                 except Exception as e2:
                     print(f"Warning: Default tight_layout also failed. Error: {e2}")
        
        return self


    def show(self):
        """
        Show the plot.
        
        Returns:
        --------
        self : LollipopPlot
            Returns self for method chaining
        """
        if self.fig is not None:
            plt.show()
        else:
            print("Error: Plot has not been generated yet. Call plot() first.")
        return self

    @classmethod
    def plot_multi_cohort(cls, 
                          gene, 
                          cohorts_data, 
                          figsize=(20, 15), 
                        width_ratios=[9, 1], 
                         config=None, 
                         save_path=None, 
                         dpi=300,
                         title=None):
        """
        Create lollipop plots for multiple cohorts with unified legend.
        
        Parameters:
        -----------
        gene : str
            Gene name
        cohorts_data : dict
            Dictionary with cohort names as keys and values as tuples:
            (AA_length, mutations_data, domains_data, refseq_ID)
        figsize : tuple, default (20, 15)
            Figure size
        width_ratios : list, default [9, 1]
            Width ratios for main plots and legend
        config : dict, optional
            Configuration options
        save_path : str, optional
            Path to save the figure
        dpi : int, default 300
            DPI for saving
            
        Returns:
        --------
        LollipopPlot : The main plot instance with unified legends
        
        Example:
        --------
        cohorts_data = {
            'LUAD': (AA_length, mutations_data, domains_data, refseq_ID),
            'ASC': (AA_length, mutations_data, domains_data, refseq_ID),
            'LUSC': (AA_length, mutations_data, domains_data, refseq_ID)
        }
        plot = LollipopPlot.plot_multi_cohort('TP53', cohorts_data)
        """
        from matplotlib.gridspec import GridSpec
        
        n_cohorts = len(cohorts_data)
        if n_cohorts == 0:
            raise ValueError("No cohort data provided")
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_cohorts, 2, width_ratios=width_ratios, 
                     height_ratios=[1] * n_cohorts, 
                     wspace=0.02, hspace=0.5)
        
        # Create main plot instance for legend management
        first_cohort_data = list(cohorts_data.values())[0]
        AA_length, mutations_data, domains_data, refseq_ID = first_cohort_data
        
        main_plot = cls(
            protein_name=gene,
            protein_length=AA_length,
            domains=domains_data,
            mutations=mutations_data,
            config=config
        )
        
        # Set up main plot with figure
        main_plot.fig = fig
        main_plot.ax_legend = fig.add_subplot(gs[:, 1])  # Legend spans all rows, right column
        
        # Create individual plots for each cohort
        cohort_plots = []
        all_mutation_colors = {}
        all_domain_colors = {}
        
        for i, (cohort_name, cohort_data) in enumerate(cohorts_data.items()):
            AA_length, mutations_data, domains_data, refseq_ID = cohort_data
            
            # Create subplot for this cohort
            ax = fig.add_subplot(gs[i, 0])  # Left column, specific row
            
            # Create plot instance for this cohort
            cohort_plot = cls(
                protein_name=f"{gene} - {cohort_name}",
                protein_length=AA_length,
                domains=domains_data,
                mutations=mutations_data,
                config=config
            )
            
            # Plot without legend (will be unified later)
            cohort_plot._setup_plot(ax=ax, fig=fig)
            cohort_plot._plot_mutations()
            cohort_plot._plot_backbone()
            cohort_plot._plot_domains()
            
            # Set cohort name as title
            ax.set_title(f"{cohort_name}", fontsize=14, pad=10)
            
            # Collect colors for unified legend
            # Merge domain colors
            for domain in domains_data:
                label = domain.get("Label", "Unknown")
                if label in cohort_plot.domain_colors:
                    all_domain_colors[label] = cohort_plot.domain_colors[label]
            
            # Merge mutation colors
            mutation_types = set(m.get('type', 'Other') for m in mutations_data)
            for m_type in mutation_types:
                if m_type in cohort_plot.color_manager.get_cmap('lollipop_mutations'):
                    all_mutation_colors[m_type] = cohort_plot.color_manager.get_cmap('lollipop_mutations')[m_type]
            
            cohort_plots.append(cohort_plot)
        
        # Add unified legends to main plot
        if all_domain_colors:
            main_plot.add_legend("Domains", all_domain_colors)
        if all_mutation_colors:
            main_plot.add_legend("Mutation Types", all_mutation_colors)
        
        # Plot unified legends on the right axis
        if main_plot.ax_legend is not None:
            main_plot.plot_all_legends(ax=main_plot.ax_legend, 
                                     fontsize=10, 
                                     title_fontsize=12,
                                     legend_spacing=0.15,
                                     item_spacing=0.03)
        
        # Set main title
        if title:
            fig.suptitle(f"Lollipop Plots for Gene: {gene}", fontsize=16, y=0.98)
        
        # Adjust layout
        try:
            # Use bbox_inches='tight' instead of tight_layout for better compatibility
            # with complex layouts and suptitle
            pass  # Skip tight_layout to avoid warnings with complex layouts
        except Exception as e:
            print(f"Warning: tight_layout failed: {e}")
        
        # Save if requested
        if save_path:
            try:
                format_ext = save_path.split('.')[-1].lower()
                pil_kwargs = {"compression": "tiff_lzw"} if format_ext == "tiff" else {}
                
                fig.savefig(
                    save_path,
                    dpi=dpi,
                    bbox_inches='tight',
                    format=format_ext,
                    pil_kwargs=pil_kwargs
                )
                print(f"[INFO] Figure saved to: {save_path}")
            except Exception as e:
                print(f"Error while saving figure: {e}")
        
        return main_plot
