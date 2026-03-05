from __future__ import annotations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .BasePlot import BasePlot


class MethodsPlot(BasePlot):
    """
    Plotting class for methodology visualizations.

    Inherits from BasePlot to use save method and common plotting functionality.
    Provides 3D cohort demonstration plots and helper drawing utilities.
    """

    def __init__(self) -> None:
        """Initialize MethodsPlot by calling the parent BasePlot constructor."""
        super().__init__()  # Initialize BasePlot

    @staticmethod
    def draw_cube(
        ax: plt.Axes,
        origin: tuple[float, float, float],
        width: float,
        depth: float,
        height: float,
        color: str = 'skyblue',
        alpha: float = 0.8,
    ) -> None:
        """
        Draw a 3D cube on the given axes.

        Parameters
        ----------
        ax : plt.Axes
            The 3D matplotlib axes to draw on.
        origin : tuple[float, float, float]
            ``(x, y, z)`` coordinates for the cube origin corner.
        width : float
            Size along the x-axis.
        depth : float
            Size along the y-axis.
        height : float
            Size along the z-axis.
        color : str, optional
            Face color of the cube. Default is ``'skyblue'``.
        alpha : float, optional
            Transparency level. Default is ``0.8``.
        """
        x, y, z = origin
        vertices = np.array([
            [x, y, z], [x+width, y, z], [x+width, y+depth, z], [x, y+depth, z],
            [x, y, z+height], [x+width, y, z+height], [x+width, y+depth, z+height], [x, y+depth, z+height]
        ])
        faces = [
            [vertices[i] for i in [0,1,2,3]],  # bottom
            [vertices[i] for i in [4,5,6,7]],  # top
            [vertices[i] for i in [0,1,5,4]],  # front
            [vertices[i] for i in [2,3,7,6]],  # back
            [vertices[i] for i in [1,2,6,5]],  # right
            [vertices[i] for i in [0,3,7,4]],  # left
        ]
        cube = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black')
        ax.add_collection3d(cube)

    def plot_cohort_demo(
        self,
        save_path: str | None = None,
        dpi: int = 300,
    ) -> None:
        """
        Plot a 3D cohort demonstration with stacked omics cubes.

        Creates a 3D visualization with two groups of cubes representing
        omics data dimensions (features x samples) for different data types.

        Parameters
        ----------
        save_path : str or None, optional
            Path to save the figure. If ``None``, the figure is only displayed.
        dpi : int, optional
            Resolution for saved figure. Default is ``300``.
        """
        # Set up the 3D plot
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Parameters
        width = 1
        depth = 10  # same for all
        heights = [8, 9, 6, 9, 4]  # variable features
        colors = plt.cm.viridis(np.linspace(0, 1, len(heights)))

        # Draw 5 cubes
        offset_y = depth + -15  # offset along y-axis (depth axis)
        for i, (h, c) in enumerate(zip(heights, colors)):
            x0 = i * (width + 0.5)
            # primary cube
            MethodsPlot.draw_cube(self.ax, origin=(x0, 0, 0), width=width, depth=depth, height=h, color=c, alpha=0.7)
            # side cube: offset along y-axis
            MethodsPlot.draw_cube(self.ax, origin=(x0, offset_y, 0), width=width, depth=depth/3, height=h, color=c, alpha=0.7)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])

        # Set plot limits
        self.ax.set_xlim(0, 5 * (width + 0.5))
        self.ax.set_ylim(-5, depth + 1)
        self.ax.set_zlim(0, max(heights) + 2)

        self.ax.set_xlabel("Omics")
        self.ax.set_ylabel("Samples")
        self.ax.set_zlabel("Features")
        self.ax.view_init(elev=20, azim=45)

        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
        if save_path:
            self.save(save_path, dpi=dpi, bbox_inches='tight')

        plt.show()
