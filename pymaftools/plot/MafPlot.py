"""Plot accessor for MAF objects."""

from __future__ import annotations


class MafPlot:
    """MAF-level plotting and comparison namespace."""

    def __init__(self, maf):
        self.maf = maf

    def summary(self, **kwargs):
        """Draw a compact MAF summary dashboard."""
        from .wes import plot_maf_summary

        return plot_maf_summary(self.maf, **kwargs)

    def titv(self, **kwargs):
        """Draw per-sample Ti/Tv substitution composition."""
        from .wes import plot_titv

        return plot_titv(self.maf, **kwargs)

    def rainfall(self, **kwargs):
        """Draw an inter-mutation distance rainfall plot."""
        from .wes import plot_rainfall

        return plot_rainfall(self.maf, **kwargs)

    def vaf(self, **kwargs):
        """Draw VAF distribution."""
        from .wes import plot_vaf

        return plot_vaf(self.maf, **kwargs)

    def compare_cohorts(self, other, **kwargs):
        """Compare gene mutation frequencies against another MAF cohort."""
        from .wes import compare_cohorts

        return compare_cohorts(self.maf, other, **kwargs)

    def forest(self, compare_result, **kwargs):
        """Draw a forest-style plot from ``compare_cohorts`` output."""
        from .wes import plot_cohort_comparison_forest

        return plot_cohort_comparison_forest(compare_result, **kwargs)

    def lollipop(self, gene, *, protein_domains_path=None, **kwargs):
        """Build a :class:`~pymaftools.plot.LollipopPlot.LollipopPlot` for one gene.

        One-call entry point that wires the MAF's ``get_protein_info`` (mutation
        positions / counts + protein length) and ``get_domain_info`` (domain
        annotations) into a ready-to-draw lollipop. Draw it with ``.plot()``.

        Parameters
        ----------
        gene : str
            Hugo gene symbol to plot.
        protein_domains_path : str, os.PathLike, or None, default None
            Path to a protein domains CSV; ``None`` uses the bundled dataset.
        **kwargs
            Forwarded to :class:`LollipopPlot` (e.g. ``config``,
            ``domain_label_map``, ``mutation_label_map``).

        Raises
        ------
        ValueError
            If the MAF has no usable ``Protein_position`` data for ``gene``.

        Examples
        --------
        >>> maf.plot.lollipop("TP53").plot()
        """
        from .LollipopPlot import LollipopPlot

        AA_length, mutations_data = self.maf.get_protein_info(gene)
        if AA_length is None:
            raise ValueError(
                f"No protein-position data for {gene!r} in this MAF "
                "(a populated 'Protein_position' column is required)."
            )
        domains, _ = self.maf.get_domain_info(gene, AA_length, protein_domains_path)
        return LollipopPlot(
            protein_name=gene,
            protein_length=AA_length,
            domains=domains,
            mutations=mutations_data,
            **kwargs,
        )
