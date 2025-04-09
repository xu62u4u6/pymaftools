import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.lines as mlines
from collections import defaultdict

class LollipopPlot:
    def __init__(self, protein_name, protein_length, domains, mutations, config=None):
        """
        Initialize a new LollipopPlot object.

        Args:
            protein_name (str): Protein name.
            protein_length (int): Protein amino acid length.
            domains (list): Domains data list.
            mutations (list): Mutation data list.
            config (dict, optional): Configuration for plot. Defaults to None.
        """
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
        self.mutation_type_colors = self.config['mutation_colors']

        # Placeholders for plot objects and legend handles
        self.fig = None
        self.ax = None
        self.domain_legend_handles = {}
        self.mutation_legend_handles = {}

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
            domain_cmap = plt.cm.get_cmap(cmap_name, num_labels if num_labels > 0 else 1)
        except ValueError:
            print(f"Warning: Colormap '{cmap_name}' failed. Using 'viridis'.")
            domain_cmap = plt.cm.get_cmap('viridis', num_labels if num_labels > 0 else 1)

        return {label: domain_cmap(i) for i, label in enumerate(unique_domain_labels)}

    def _setup_plot(self, ax=None, fig=None):
        """Init plot"""
        if ax is None or fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.config['figsize'])
        else:
            self.fig = fig
            self.ax = ax

        self.ax.set_xlim(-10, self.protein_length + 10)
        self.ax.set_ylim(self.config['domain_bottom_y'] - 0.1,
                         self.min_lollipop_candy_y + self.config['max_additional_scaled_height'] + 0.1)
        self.ax.set_xlabel("Amino Acid Position", fontsize=12)
        self.ax.set_title(f'Protein Domains and Mutations for {self.protein_name}', fontsize=16, pad=20)

        # Hide Y-axis and unnecessary spines
        y_ticks = np.linspace(
            self.config['backbone_y'], 
            self.min_lollipop_candy_y + self.config['max_additional_scaled_height'],
            num=5
        )
        mutation_counts = np.round(
            np.linspace(0, self.max_mutation_count, num=5)
        ).astype(int)

        self.ax.set_yticks(y_ticks)
        self.ax.set_yticklabels(mutation_counts)
        self.ax.set_ylabel("Mutation Count", fontsize=12)
        for spine in ['top', 'right', 'left', 'bottom']:
            if spine != 'bottom':
                self.ax.spines[spine].set_visible(False)

    def _plot_backbone(self):
        """Plot protein backbone line."""
        if self.ax is None: return
        self.ax.plot([0, self.protein_length], [self.config['backbone_y'], self.config['backbone_y']],
                     color='black', linewidth=6, solid_capstyle='round', zorder=1)

    def _plot_domains(self):
        """Plot domains and prepare legend handles."""
        if self.ax is None: return
        self.domain_legend_handles = {} # Reset handles
        for domain in self.domains:
            start = domain.get("Start")
            end = domain.get("End")
            label = domain.get("Label", "Unknown")
            if start is None or end is None: continue # Skip if data missing

            domain_color = self.domain_colors.get(label, 'lightgrey')
            rect = patches.Rectangle((start, self.config['domain_bottom_y']), end - start, self.config['domain_height'],
                                     linewidth=1, edgecolor='black', facecolor=domain_color, zorder=2)
            self.ax.add_patch(rect)

            if label not in self.domain_legend_handles:
                self.domain_legend_handles[label] = patches.Patch(color=domain_color, label=label)

    def _plot_mutations(self):
        """Plot lollipop sticks and candies and prepare legend handles."""
        if self.ax is None: return
        self.mutation_legend_handles = {} # Reset handles
        for mutation in self.mutations:
            pos = mutation.get('position')
            m_type = mutation.get('type', 'Other')
            count = mutation.get('count', 0)
            if pos is None: continue # Skip if data missing

            mutation_color = self.mutation_type_colors.get(m_type, '#CCCCCC')

            # Calculate lollipop top Y position
            scaled_height_addition = (count / self.max_mutation_count) * self.config['max_additional_scaled_height']
            lollipop_top_y = self.min_lollipop_candy_y + scaled_height_addition

            # Draw lollipop stick (zorder=0, bottom layer)
            self.ax.plot([pos, pos], [self.config['backbone_y'], lollipop_top_y], color='grey', linewidth=1, zorder=0)

            # Draw lollipop candy (zorder=3, top layer)
            marker_size = 50 + (count / self.max_mutation_count) * 100 # Optional scaling
            self.ax.scatter(pos, lollipop_top_y, color=mutation_color, s=marker_size, zorder=3, edgecolors='black', linewidth=0.5)

            # Prepare legend handles
            if m_type not in self.mutation_legend_handles:
                self.mutation_legend_handles[m_type] = mlines.Line2D([], [], color=mutation_color, marker='o', linestyle='None',
                                                                     markersize=8, label=m_type)

    def _add_legends(self):
        """Add legeneds."""
        if self.ax is None: return
        artists = [] # Keep track of legends added

        # Domain Legend
        if self.domain_legend_handles:
            domain_legend = self.ax.legend(handles=self.domain_legend_handles.values(), title='Domains',
                                        loc='upper left', bbox_to_anchor=(1.02, 1.0))  # 頂部位置
            self.ax.add_artist(domain_legend)
            artists.append(domain_legend)

        # Mutation Legend
        if self.mutation_legend_handles:
            mutation_legend = self.ax.legend(handles=self.mutation_legend_handles.values(), title='Mutation Types',
                                            loc='upper left', bbox_to_anchor=(1.02, 0.3))  # 向下移動位置
            artists.append(mutation_legend)


    def plot(self, ax=None, fig=None):
        """Plot lollipop plot."""
        self._setup_plot(ax=ax, fig=fig)  # 傳遞傳入的 ax
        if self.ax is None:
             print("Error: Plot setup failed.")
             return

        self._plot_mutations()
        self._plot_backbone()
        self._plot_domains()
        self._add_legends()

        # Adjust layout
        try:
            self.fig.tight_layout(rect=self.config['tight_layout_rect'])
        except ValueError as e:
             print(f"Warning: tight_layout failed. Legends might overlap or be cut off. Error: {e}")
             # Fallback to default tight_layout if rect causes issues
             try:
                 self.fig.tight_layout()
             except Exception as e2:
                 print(f"Warning: Default tight_layout also failed. Error: {e2}")


    def show(self):
        """Show plot."""
        if self.fig is not None:
            plt.show()
        else:
            print("Error: Plot has not been generated yet. Call plot() first.")

    def save(self, filename, dpi=300, **kwargs):
        """Save plot to file."""
        if self.fig is not None:
            # Ensure bbox_inches='tight' if legends are outside, unless overridden
            if 'bbox_inches' not in kwargs and self.config['tight_layout_rect'] != [0, 0, 1, 1]:
                 kwargs['bbox_inches'] = 'tight'
            self.fig.savefig(filename, dpi=dpi, **kwargs)
            print(f"Plot saved to {filename}")
        else:
            print("Error: Plot has not been generated yet. Call plot() first.")