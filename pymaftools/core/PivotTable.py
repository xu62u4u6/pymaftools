"""
PivotTable Module

Extended pandas DataFrame for bioinformatics analysis with integrated metadata support.
Specifically designed for mutation analysis and genomic data visualization.
"""

from __future__ import annotations

# Standard library imports
import sqlite3
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

# Third-party imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests

# Local imports
from .PairwiseMatrix import CooccurrenceMatrix, SimilarityMatrix


class PivotTable(pd.DataFrame):
    """
    Enhanced pandas DataFrame for bioinformatics analysis.
    
    A specialized DataFrame that maintains synchronized metadata for both
    features (rows, typically genes/mutations) and samples (columns).
    Designed for genomic data analysis with built-in support for mutation
    frequency calculations, statistical testing, and visualization.
    
    Attributes
    ----------
    feature_metadata : pd.DataFrame
        Metadata for features (genes/mutations/signatures), indexed by feature names.
    sample_metadata : pd.DataFrame 
        Metadata for samples, indexed by sample names.
        
    Examples
    --------
    >>> # Create a PivotTable from mutation data
    >>> data = pd.DataFrame({'sample1': [True, False], 'sample2': [False, True]},
    ...                     index=['TP53', 'KRAS'])
    >>> table = PivotTable(data)
    >>> table.feature_metadata['freq'] = table.add_freq().feature_metadata['freq']
    """
    # Store metadata attribute names for pandas inheritance
    _metadata: List[str] = ["feature_metadata", "sample_metadata"]

    def __init__(
        self, 
        data: Any = None, 
        *args: Any, 
        **kwargs: Any
    ) -> None:
        """
        Initialize PivotTable with data and empty metadata.
        
        Parameters
        ----------
        data : array-like, dict, or DataFrame
            Data to initialize the PivotTable.
        *args, **kwargs
            Additional arguments passed to pandas DataFrame constructor.
        """
        super().__init__(data, *args, **kwargs)
        # Initialize empty metadata DataFrames with matching indices
        self.feature_metadata: pd.DataFrame = pd.DataFrame(index=self.index)
        self.sample_metadata: pd.DataFrame = pd.DataFrame(index=self.columns)

    @property
    def _constructor(self) -> Callable[..., "PivotTable"]:
        """Return constructor for pandas operations that preserves PivotTable type."""
        def _new_constructor(*args: Any, **kwargs: Any) -> "PivotTable":
            obj = PivotTable(*args, **kwargs)
            obj._validate_metadata()
            return obj
        return _new_constructor

    def _validate_metadata(self) -> None:
        """Validate that metadata indices match DataFrame structure."""
        if not self.feature_metadata.index.equals(self.index):
            raise ValueError(
                "feature_metadata index does not match PivotTable index.")

        if not self.sample_metadata.index.equals(self.columns):
            raise ValueError(
                "sample_metadata index does not match PivotTable columns.")

    def rename_index_and_columns(
        self, 
        index_name: str = "feature", 
        columns_name: str = "sample"
    ) -> "PivotTable":
        """
        Rename the index and columns of the PivotTable.

        Parameters
        ----------
        index_name : str, default "feature"
            New name for the index (features).
        columns_name : str, default "sample"
            New name for the columns (samples).
            
        Returns
        -------
        PivotTable
            PivotTable with renamed index and columns.
        """
        table = self.copy()
        table.index.name = index_name
        table.columns.name = columns_name
        table.feature_metadata.index.name = index_name
        table.sample_metadata.index.name = columns_name
        return table

    def to_sqlite(self, db_path: str) -> None:
        """
        Save PivotTable to SQLite database format.
        
        Parameters
        ----------
        db_path : str
            Path to the SQLite database file.
        """
        db_path = Path(db_path)
        if db_path.exists():
            db_path.unlink()  # Remove existing file

        conn = sqlite3.connect(str(db_path))
        table = self.copy().rename_index_and_columns()
        # TODO: replace False with "WT" in all files
        table = table.replace(False, "WT")
        table.to_sql("data", conn, index=True)
        table.sample_metadata.to_sql("sample_metadata", conn, index=True)
        table.feature_metadata.to_sql("feature_metadata", conn, index=True)
        conn.close()
        print(f"[PivotTable] saved to {db_path}")

    @classmethod
    def read_sqlite(cls, db_path: str) -> "PivotTable":
        """
        Load PivotTable from SQLite database format.
        
        Parameters
        ----------
        db_path : str
            Path to the SQLite database file.
            
        Returns
        -------
        PivotTable
            Loaded PivotTable with metadata.
        """
        conn = sqlite3.connect(db_path)

        # Load main data table
        data = pd.read_sql("SELECT * FROM 'data'", conn, index_col="feature")
        data.columns.name = "sample"

        # Load metadata tables
        sample_metadata = pd.read_sql(
            "SELECT * FROM 'sample_metadata'", conn, index_col="sample")
        feature_metadata = pd.read_sql(
            "SELECT * FROM 'feature_metadata'", conn, index_col="feature")

        # Create PivotTable instance
        table = cls(data)
        # TODO: replace False with "WT" in all files
        table = table.replace("WT", False)
        table.sample_metadata = sample_metadata
        table.feature_metadata = feature_metadata
        table._validate_metadata()
        conn.close()
        print(f"[PivotTable] loaded from {db_path}")
        return table

    def to_hierarchical_clustering(
        self,
        method: str = 'ward',
        metric: str = 'euclidean'
    ) -> Dict[str, np.ndarray]:
        """
        Perform hierarchical clustering on both features and samples.

        Computes hierarchical clustering linkage matrices for both the feature
        dimension (genes/mutations) and sample dimension using scipy's linkage
        function. This enables creation of dendrograms and clustermaps for
        data visualization and pattern discovery.

        Parameters
        ----------
        method : str, default 'ward'
            Linkage algorithm to use. Options include:
            
            - 'ward': Minimizes within-cluster variance (requires euclidean metric)
            - 'single': Nearest point algorithm  
            - 'complete': Farthest point algorithm
            - 'average': UPGMA algorithm
            - 'weighted': WPGMA algorithm
            - 'centroid': UPGMC algorithm
            - 'median': WPGMC algorithm
            
        metric : str, default 'euclidean'
            Distance metric for clustering. Common options:
            
            - 'euclidean': Standard Euclidean distance
            - 'manhattan': L1 distance
            - 'cosine': Cosine distance
            - 'correlation': Correlation distance
            - 'hamming': Hamming distance (for binary data)
            - 'jaccard': Jaccard distance (for binary data)

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing linkage matrices:
            
            - 'gene_linkage' : np.ndarray of shape (n_features-1, 4)
                Linkage matrix for features (genes), where each row represents
                a merge operation in the clustering tree
            - 'sample_linkage' : np.ndarray of shape (n_samples-1, 4)  
                Linkage matrix for samples, where each row represents
                a merge operation in the clustering tree

        Notes
        -----
        The linkage matrices returned follow scipy's format where each row
        contains [cluster1_id, cluster2_id, distance, cluster_size].

        For binary mutation data, consider using 'hamming' or 'jaccard' metrics.
        The 'ward' method works only with 'euclidean' metric.

        Examples
        --------
        >>> # Basic hierarchical clustering
        >>> clustering = pivot_table.to_hierarchical_clustering()
        >>> gene_linkage = clustering['gene_linkage']
        >>> sample_linkage = clustering['sample_linkage']

        >>> # Using Jaccard distance for binary mutation data
        >>> clustering = pivot_table.to_hierarchical_clustering(
        ...     method='average', 
        ...     metric='jaccard'
        ... )

        >>> # Create dendrogram from results
        >>> from scipy.cluster.hierarchy import dendrogram
        >>> import matplotlib.pyplot as plt
        >>> 
        >>> plt.figure(figsize=(10, 6))
        >>> dendrogram(clustering['gene_linkage'])
        >>> plt.title('Gene Clustering Dendrogram')
        >>> plt.show()

        See Also
        --------
        scipy.cluster.hierarchy.linkage : The underlying clustering function
        scipy.cluster.hierarchy.dendrogram : For visualizing clustering results
        seaborn.clustermap : For creating clustered heatmaps
        """
        from scipy.cluster.hierarchy import linkage

        # Feature clustering (genes/mutations along rows)
        gene_linkage = linkage(self.values,
                               method=method,
                               metric=metric)

        # Sample clustering (samples along columns, so transpose data)
        sample_linkage = linkage(self.values.T,
                                 method=method,
                                 metric=metric)

        return {
            'gene_linkage': gene_linkage,
            'sample_linkage': sample_linkage
        }

    def copy(self, deep: bool = True) -> "PivotTable":
        """
        Make a copy of this object's indices and data.

        Creates a deep or shallow copy of the PivotTable and its associated
        feature_metadata and sample_metadata.

        Parameters
        ----------
        deep : bool, default True
            Whether to make a deep copy or shallow copy.

        Returns
        -------
        PivotTable
            Copy of the PivotTable with preserved metadata.

        Examples
        --------
        >>> pivot_copy = pivot_table.copy()
        >>> pivot_shallow = pivot_table.copy(deep=False)

        See Also
        --------
        pandas.DataFrame.copy : The underlying pandas copy method.
        """
        pivot_table = super().copy(deep=deep)
        pivot_table.feature_metadata = self.feature_metadata.copy(deep=deep)
        pivot_table.sample_metadata = self.sample_metadata.copy(deep=deep)
        return pivot_table

    @classmethod
    def _from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_meta_src: pd.DataFrame,
        sample_meta_src: pd.DataFrame,
        *,
        feature_fill: Any = np.nan,
        sample_fill: Any = np.nan,
    ) -> "PivotTable":
        """
        Create a PivotTable from a DataFrame with synchronized metadata.

        This is an internal method used to construct PivotTable objects
        with properly aligned feature and sample metadata.

        Parameters
        ----------
        df : pd.DataFrame
            The main data DataFrame to wrap as PivotTable.
        feature_meta_src : pd.DataFrame
            Source DataFrame for feature metadata.
        sample_meta_src : pd.DataFrame
            Source DataFrame for sample metadata.
        feature_fill : scalar, default np.nan
            Value to use for missing values when reindexing feature metadata.
        sample_fill : scalar, default np.nan
            Value to use for missing values when reindexing sample metadata.

        Returns
        -------
        PivotTable
            New PivotTable with synchronized metadata.

        Notes
        -----
        This method automatically reindexes the metadata DataFrames to match
        the indices of the input DataFrame, filling missing values as specified.
        """
        new = cls(df)
        new.feature_metadata = feature_meta_src.reindex(
            df.index, fill_value=feature_fill)
        new.sample_metadata = sample_meta_src.reindex(
            df.columns, fill_value=sample_fill)
        return new

    def __getitem__(self, key: Union[str, List, Tuple, slice, pd.Series, Callable]) -> "PivotTable":
        """
        Enhanced indexing support for PivotTable.

        Always returns a PivotTable object with preserved metadata, regardless
        of the indexing operation performed. Supports multiple indexing patterns
        including boolean Series and 2D indexing.

        Parameters
        ----------
        key : str, list, tuple, slice, pd.Series, or callable
            Index specification. Can be:

            - str : Single column name
                Example: ``tbl["gene1"]``
            - list : Multiple column names  
                Example: ``tbl[["gene1", "gene2"]]``
            - tuple : (row_selector, column_selector) for 2D indexing
                Example: ``tbl[["sample1", "sample2"], ["gene1", "gene2"]]``
            - slice : Row or column slice
                Example: ``tbl[:10]`` or ``tbl[:, :5]``
            - pd.Series (bool) : Boolean mask for selection
                Example: ``tbl[high_expression_mask]``
            - callable : Function for conditional selection
                Example: ``tbl[lambda x: x > 0.5]``

        Returns
        -------
        PivotTable
            Always returns a PivotTable with corresponding feature_metadata 
            and sample_metadata preserved. Scalar results and Series are 
            automatically converted to single-element DataFrames.

        Notes
        -----
        All results are wrapped as PivotTable objects to maintain consistency:

        - Scalar values become 1x1 PivotTable
        - Series become single-row or single-column PivotTable  
        - DataFrames maintain their structure as PivotTable

        The metadata (feature_metadata and sample_metadata) are automatically
        synchronized with the resulting data selection.

        Examples
        --------
        >>> # Single column selection
        >>> gene_data = pivot_table["TP53"]

        >>> # Multiple column selection
        >>> oncogenes = pivot_table[["TP53", "KRAS", "EGFR"]]

        >>> # 2D indexing 
        >>> subset = pivot_table[["sample1", "sample2"], ["gene1", "gene2"]]

        >>> # Boolean indexing
        >>> high_freq = pivot_table.feature_metadata["freq"] > 0.1
        >>> frequent_genes = pivot_table[high_freq, :]
        """
        if isinstance(key, tuple) and len(key) == 2:
            # Handle 2D indexing: tbl[row_sel, col_sel]
            row_sel, col_sel = key
            result = super().loc[row_sel, col_sel]
        else:
            # Handle 1D indexing: tbl["col"] or tbl[["col1", "col2"]]
            result = super().__getitem__(key)

        # Convert scalar results to DataFrame for consistency
        if np.isscalar(result):
            # Handle scalar selection: tbl["row1", "col1"]
            result = pd.DataFrame({col_sel: [result]}, index=[row_sel])

        # Convert Series to DataFrame for consistency
        if isinstance(result, pd.Series):
            if result.name in self.columns:
                # Column selection: convert to n × 1 DataFrame
                result = result.to_frame()
            else:
                # Row selection: convert to 1 × n DataFrame
                result = result.to_frame().T

        # Always wrap as PivotTable with metadata
        return self._from_dataframe(
            result,
            self.feature_metadata,
            self.sample_metadata,
        )

    def subset(
        self,
        *,
        features: Optional[Union[List, pd.Series, slice]] = None,
        samples: Optional[Union[List, pd.Series, slice]] = None
    ) -> "PivotTable":
        """
        Subset PivotTable by features and/or samples.

        This method provides a convenient interface for selecting specific
        features (rows) and samples (columns) from the PivotTable while
        preserving metadata alignment.

        Parameters
        ----------
        features : list, pd.Series, or slice, optional
            Features (rows) to select. Can be:

            - list of str : Feature names to select
                Example: ``["TP53", "KRAS", "EGFR"]``
            - pd.Series (bool) : Boolean mask for feature selection
                Example: ``pivot_table.feature_metadata["freq"] > 0.1``
            - slice : Slice object for feature selection
                Example: ``slice(None, 10)`` for first 10 features
            - None : Select all features (default)

        samples : list, pd.Series, or slice, optional  
            Samples (columns) to select. Can be:

            - list of str : Sample names to select
                Example: ``["sample1", "sample2", "sample3"]``
            - pd.Series (bool) : Boolean mask for sample selection
                Example: ``pivot_table.sample_metadata["subtype"] == "LUAD"``
            - slice : Slice object for sample selection
                Example: ``slice(None, 20)`` for first 20 samples
            - None : Select all samples (default)

        Returns
        -------
        PivotTable
            Subset PivotTable with synchronized metadata. Only keeps existing
            labels (inner join behavior).

        Notes
        -----
        This method uses inner join behavior, meaning only existing labels
        are kept. Missing labels are silently ignored. For outer join behavior
        that includes missing labels with NaN values, use the ``reindex`` method.

        Examples
        --------
        >>> # Select specific genes and samples
        >>> subset = pivot_table.subset(
        ...     features=["TP53", "KRAS"], 
        ...     samples=["sample1", "sample2"]
        ... )

        >>> # Select high-frequency mutations 
        >>> high_freq = pivot_table.feature_metadata["freq"] > 0.1
        >>> frequent_mutations = pivot_table.subset(features=high_freq)

        >>> # Select samples by subtype
        >>> luad_samples = pivot_table.sample_metadata["subtype"] == "LUAD"  
        >>> luad_data = pivot_table.subset(samples=luad_samples)

        The metadata is subset based on the DataFrame's index (features)
        and columns (samples). Missing indices in metadata will result
        in NaN values in the new PivotTable's metadata.

        See Also
        --------
        PivotTable.reindex : For outer join behavior with missing labels.
        PivotTable.__getitem__ : For direct indexing operations.
        """
        features = slice(None) if features is None else features
        samples = slice(None) if samples is None else samples
        return self[features, samples]

    def reindex(
        self,
        index: Optional[Union[List, pd.Index]] = None,
        columns: Optional[Union[List, pd.Index]] = None,
        *args: Any,
        fill_value: Any = np.nan,
        feature_fill_value: Any = np.nan,
        sample_fill_value: Any = np.nan,
        **kwargs: Any,
    ) -> "PivotTable":
        """
        Conform PivotTable to new index and/or columns with synchronized metadata.

        This method extends pandas DataFrame.reindex to also reindex the
        associated feature_metadata and sample_metadata, maintaining consistency
        across all components of the PivotTable.

        Parameters
        ----------
        index : array-like, optional
            New labels for the rows. If None, use existing index.
        columns : array-like, optional
            New labels for the columns. If None, use existing columns.
        *args
            Additional positional arguments passed to pandas.DataFrame.reindex.
        fill_value : scalar, default np.nan
            Value to use for missing values in the main DataFrame.
        feature_fill_value : scalar, default np.nan
            Value to use for missing values when reindexing feature_metadata.
        sample_fill_value : scalar, default np.nan
            Value to use for missing values when reindexing sample_metadata.
        **kwargs
            Additional keyword arguments passed to pandas.DataFrame.reindex.

        Returns
        -------
        PivotTable
            Reindexed PivotTable with synchronized metadata.

        Notes
        -----
        Unlike the ``subset`` method which uses inner join behavior, this method
        uses outer join behavior and will include missing labels filled with
        the specified fill values.

        The metadata DataFrames are automatically reindexed to match the new
        structure of the main DataFrame.

        Examples
        --------
        >>> # Reindex with new features, filling missing with 0
        >>> new_features = ["TP53", "KRAS", "NEW_GENE"]
        >>> reindexed = pivot_table.reindex(
        ...     index=new_features, 
        ...     fill_value=0,
        ...     feature_fill_value="Unknown"
        ... )

        >>> # Reindex with new samples
        >>> new_samples = ["sample1", "sample2", "new_sample"] 
        >>> reindexed = pivot_table.reindex(
        ...     columns=new_samples,
        ...     sample_fill_value="Missing"
        ... )

        >>> # Reindex both dimensions
        >>> reindexed = pivot_table.reindex(
        ...     index=new_features,
        ...     columns=new_samples,
        ...     fill_value=np.nan
        ... )

        See Also
        --------
        pandas.DataFrame.reindex : The underlying pandas reindex method.
        PivotTable.subset : For inner join behavior.
        """
        reindexed_df = super().reindex(
            index=index,
            columns=columns,
            *args,
            fill_value=fill_value,
            **kwargs,
        )

        return self._from_dataframe(
            reindexed_df,
            self.feature_metadata,
            self.sample_metadata,
            feature_fill=feature_fill_value,
            sample_fill=sample_fill_value,
        )

    @staticmethod
    def merge(
        tables: List["PivotTable"],
        fill_value: Any = np.nan,
        feature_fill_value: Any = np.nan,
        sample_fill_value: Any = np.nan,
        join: Literal["inner", "outer"] = "outer"
    ) -> "PivotTable":
        """
        Merge multiple PivotTables into a single PivotTable.

        Concatenates multiple PivotTable objects along the sample axis (columns)
        and aligns features (rows) according to the selected join strategy.
        Metadata (feature_metadata and sample_metadata) is automatically
        synchronized with the resulting data matrix.

        Parameters
        ----------
        tables : List[PivotTable]
            List of PivotTable objects to merge. All should have compatible structure.
        fill_value : scalar, default np.nan
            Value to use for missing values in the main data matrix after merging.
        feature_fill_value : scalar, default np.nan
            Value to use for missing values when reindexing feature_metadata.
        sample_fill_value : scalar, default np.nan
            Value to use for missing values when reindexing sample_metadata.
        join : {'inner', 'outer'}, default 'outer'
            Strategy for aligning features (rows) across tables:
            - 'inner': Keep only features shared by all tables.
            - 'outer': Keep all features from all tables (union).

            Note: samples (columns) are always unioned.

        Returns
        -------
        PivotTable
            A new PivotTable with:
            - Merged data matrix
            - Reindexed feature and sample metadata
            - Missing values filled with specified defaults

        Raises
        ------
        ValueError
            If `join` is not 'inner' or 'outer'.
        ValueError
            If sample (column) names overlap across tables.

        Examples
        --------
        >>> # Outer merge (default): keeps all features
        >>> merged = PivotTable.merge([table_A, table_B])

        >>> # Inner merge: keeps only shared features
        >>> merged = PivotTable.merge([table_A, table_B], join='inner')

        >>> # Fill missing values with 0
        >>> merged = PivotTable.merge([table_A, table_B], fill_value=0)
        """
        if join not in {"inner", "outer"}:
            raise ValueError("join must be either 'inner' or 'outer'.")

        # Step 1: merge main data (along sample axis)
        merged_data = pd.concat(tables, axis=1)

        if not merged_data.columns.is_unique:
            duplicated = merged_data.columns[merged_data.columns.duplicated(
            )].tolist()
            raise ValueError(
                f"Merged PivotTable has duplicate sample names: {duplicated}. "
                f"Please ensure sample names are unique across tables."
            )

        # Step 2: merge metadata before trimming
        merged_sample_meta = pd.concat(
            [t.sample_metadata for t in tables], axis=0)
        merged_feature_meta = pd.concat(
            [t.feature_metadata for t in tables], axis=0)
        merged_feature_meta = merged_feature_meta[~merged_feature_meta.index.duplicated(
            keep="first")]
        # Step 3: determine features to keep
        if join == "inner":
            features = sorted(set.intersection(
                *(set(t.index) for t in tables)))
        else:
            features = merged_data.index

        samples = merged_data.columns  # always union of all samples

        # Step 4: create merged PivotTable and align metadata
        merged = PivotTable(merged_data.fillna(fill_value))
        merged.feature_metadata = merged_feature_meta.fillna(
            feature_fill_value)
        merged.sample_metadata = merged_sample_meta.fillna(sample_fill_value)
        merged = merged.reindex(
            index=features,
            columns=samples,
            fill_value=fill_value,
            feature_fill_value=feature_fill_value,
            sample_fill_value=sample_fill_value,
        )

        return merged

    def calculate_TMB(
        self, 
        default_capture_size: float = 40, 
        group_col: str = "subtype", 
        capture_size_dict: Optional[Dict[str, float]] = None
    ) -> "PivotTable":
        table = self.copy()
        table.sample_metadata["capture_size"] = default_capture_size

        if capture_size_dict is not None:

            for group, size in capture_size_dict.items():
                print(group, size)
                mask = table.sample_metadata[group_col] == group
                table.sample_metadata.loc[mask, "capture_size"] = size

        table.sample_metadata["TMB"] = table.sample_metadata["mutations_count"] / \
            table.sample_metadata["capture_size"]
        return table

    @staticmethod
    def plot_boxplot_with_annot(
        data: pd.DataFrame,
        group_col: str = "subtype",
        test_col: str = "mutations_count",
        palette: Optional[Union[str, Dict]] = None,
        title: Optional[str] = None,
        ax: Optional[Axes] = None,
        test: str = 'Mann-Whitney',
        alpha: float = 0.8,
        order: Optional[List[str]] = None,
        fontsize: int = 12,
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        if title is None:
            title = f"Boxplot of {test_col} by {group_col}"

        gb = data.groupby(group_col)
        group_pairs = list(combinations(gb.groups.keys(), 2))
        ax.set_title(title, fontsize=fontsize)

        # boxplot
        boxplot = sns.boxplot(data=data, x=group_col, y=test_col,
                              ax=ax, hue=group_col, palette=palette, order=order)

        # alpha
        for patch in boxplot.patches:
            patch.set_facecolor((patch.get_facecolor()[0],  # R
                                patch.get_facecolor()[1],  # G
                                patch.get_facecolor()[2],  # B
                                alpha))

        # stat
        annotator = Annotator(ax=ax, pairs=group_pairs,
                              data=data, x=group_col, y=test_col, order=order)
        annotator.configure(test=test, text_format='star',
                            loc='inside', verbose=False)
        annotator.apply_and_annotate()

        # xticklabel with sample size
        sample_counts = gb.size().to_dict()
        xticks_labels = [
            f"{group} (n={sample_counts[group]})" for group in order]
        ax.set_xticklabels(xticks_labels, fontsize=fontsize)

        ax.set_xlabel(group_col, fontsize=fontsize)
        ax.set_ylabel(test_col, fontsize=fontsize)

        # save if path is given
        if save_path is not None:
            pil_kwargs = {
                "compression": "tiff_lzw"} if format == "tiff" else {}
            fig.savefig(save_path, 
                        dpi=dpi, 
                        bbox_inches='tight',
                        pil_kwargs=pil_kwargs)
            print(f"[INFO] Figure saved to: {save_path}")

        return ax

    def calculate_feature_frequency(self) -> pd.Series[float]:
        """
        Calculate mutation frequency for each feature.

        Computes the frequency of each feature (gene/mutation/signature) across 
        all samples by converting the data to binary format and calculating the 
        proportion of samples with mutations for each feature.

        Treats any non-False value as indicating the presence of a mutation, 
        effectively converting the data to binary (mutated/not mutated) before 
        frequency calculation.

        Returns
        -------
        pd.Series
            Mutation frequency for each feature, indexed by feature names.
            Values range from 0.0 (no mutations in any sample) to 1.0 
            (mutations in all samples).

        Notes
        -----
        This method is equivalent to calling:
        
        1. Convert PivotTable to binary format using `to_binary_table()`
        2. Sum mutations across samples (axis=1) 
        3. Divide by total number of samples

        The frequency represents the proportion of samples that have a mutation
        for each feature, regardless of the specific mutation type or value.

        Examples
        --------
        >>> # Create example mutation data
        >>> data = pd.DataFrame({
        ...     'sample1': [True, False, True], 
        ...     'sample2': [False, True, True],
        ...     'sample3': [True, True, False]
        ... }, index=['TP53', 'KRAS', 'EGFR'])
        >>> table = PivotTable(data)
        >>> frequencies = table.calculate_feature_frequency()
        >>> print(frequencies)
        TP53     0.666667
        KRAS     0.666667  
        EGFR     0.666667
        dtype: float64

        >>> # Frequency shows proportion of samples with each mutation
        >>> print(f"TP53 is mutated in {frequencies['TP53']:.1%} of samples")
        TP53 is mutated in 66.7% of samples

        See Also
        --------
        to_binary_table : Convert PivotTable to binary mutation format
        add_freq : Add frequency columns to feature_metadata
        filter_by_freq : Filter features by frequency threshold
        """
        binary_table = self.to_binary_table()
        # Ensure result is float64 type, not object
        frequency = binary_table.sum(axis=1).astype(float) / binary_table.shape[1]
        return frequency


    def add_freq(self, groups: Dict[str, "PivotTable"] = {}) -> "PivotTable":
        """
        Add mutation frequency columns to feature_metadata.

        Calculates overall mutation frequency and optionally group-specific 
        frequencies for all features, adding these as new columns to the 
        feature_metadata DataFrame. This enables frequency-based filtering 
        and analysis operations.

        Parameters
        ----------
        groups : Dict[str, PivotTable], default {}
            Dictionary mapping group names to PivotTable objects for calculating
            group-specific mutation frequencies. Each PivotTable should represent
            a subset of samples belonging to a specific group (e.g., cancer subtypes,
            treatment groups, etc.).

            Example: {"LUAD": luad_table, "LUSC": lusc_table, "Control": control_table}

        Returns
        -------
        PivotTable
            A new PivotTable (copy) with frequency columns added to feature_metadata:
            
            - "{group_name}_freq" : float
                Frequency for each group specified in the groups dictionary
            - "freq" : float  
                Overall frequency across all samples in the current PivotTable

        Raises
        ------
        TypeError
            If any value in the groups dictionary is not a PivotTable instance.

        Notes
        -----
        The frequency calculation treats any non-False value as indicating mutation
        presence. Frequencies are calculated as:
        
        frequency = (number of mutated samples) / (total number of samples)
        
        Group-specific frequencies are calculated independently for each group's
        PivotTable, while the overall frequency uses all samples in the current
        PivotTable.

        Examples
        --------
        >>> # Add overall frequency only
        >>> table_with_freq = pivot_table.add_freq()
        >>> print(table_with_freq.feature_metadata.columns)
        Index(['freq'], dtype='object')

        >>> # Add group-specific frequencies  
        >>> luad_subset = pivot_table.subset(samples=luad_sample_mask)
        >>> lusc_subset = pivot_table.subset(samples=lusc_sample_mask)
        >>> groups = {"LUAD": luad_subset, "LUSC": lusc_subset}
        >>> table_with_freq = pivot_table.add_freq(groups=groups)
        >>> print(table_with_freq.feature_metadata.columns)
        Index(['LUAD_freq', 'LUSC_freq', 'freq'], dtype='object')

        >>> # Use frequencies for filtering
        >>> high_freq_features = table_with_freq.filter_by_freq(threshold=0.1)
        >>> luad_specific = table_with_freq[
        ...     (table_with_freq.feature_metadata['LUAD_freq'] > 0.2) &
        ...     (table_with_freq.feature_metadata['LUSC_freq'] < 0.05)
        ... ]

        See Also
        --------
        calculate_feature_frequency : Calculate frequency for current PivotTable
        filter_by_freq : Filter features by frequency threshold
        sort_features : Sort features by metadata columns including frequency
        """
        pivot_table = self.copy()
        freq_data = pd.DataFrame(index=pivot_table.index)

        for group, table in groups.items():
            if not isinstance(table, PivotTable):
                raise TypeError(f"Expected PivotTable for group '{group}', got {type(table)}.")
            freq_data[f"{group}_freq"] = table.calculate_feature_frequency()

        freq_data["freq"] = pivot_table.calculate_feature_frequency()
        pivot_table.feature_metadata[freq_data.columns] = freq_data
        return pivot_table


    def sort_features(self, by: str = "freq", ascending: bool = False) -> "PivotTable":
        """
        Sort features (rows) by a column in feature_metadata.
        
        Parameters
        ----------
        by : str, default "freq"
            Column name in feature_metadata to sort by.
        ascending : bool, default False
            Sort order. False for descending (highest values first).
            
        Returns
        -------
        PivotTable
            New PivotTable with features sorted by the specified column.
            
        Raises
        ------
        ValueError
            If the specified column is not found in feature_metadata.
        """
        if by not in self.feature_metadata.columns:
            raise ValueError(f"Column '{by}' not found in feature_metadata.")
        table = self.copy()
        # get sorted index based on the specified column
        sorted_index = table.feature_metadata.sort_values(
            by=by, ascending=ascending).index
        return table.subset(features=sorted_index)

    def sort_samples_by_mutations(self, top: int = 10) -> "PivotTable":
        """
        Sort samples by their mutation patterns.
        
        Uses a binary encoding approach where mutation patterns of the top 
        mutated features are converted to integers for sorting.
        
        Parameters
        ----------
        top : int, default 10
            Number of top features to consider for sorting.
            
        Returns
        -------
        PivotTable
            New PivotTable with samples sorted by mutation patterns.
            The mutation weight is added to sample_metadata.
        """
        def binary_sort_key(column: pd.Series) -> int:
            """Convert binary mutation pattern to integer for sorting."""
            # Convert mutation status (True/False) to binary string (1/0)
            binary_str = "".join(column.astype(int).astype(str))
            # Convert binary string to integer (e.g., "101" -> 5)
            return int(binary_str, 2)

        # tmp_pivot_table = pivot_table.drop(columns=freq_columns)
        pivot_table = self.copy()
        binary_pivot_table = pivot_table.to_binary_table()
        mutations_weight = (binary_pivot_table
                            .head(top)
                            .apply(binary_sort_key, axis=0))
        pivot_table.sample_metadata["mutations_weight"] = mutations_weight
        sorted_samples = (mutations_weight
                          .sort_values(ascending=False)
                          .index
                          .tolist())
        return pivot_table.subset(samples=sorted_samples)

    def sort_samples_by_group(self, group_col: str, group_order: List[str], top: int = 10) -> "PivotTable":
        """
        Sort samples first by the given subtype order, then within each subtype, 
        apply sort_samples_by_mutations.

        Parameters:
        - group_col (str): The column in sample_metadata containing group information.
        - group_order (list): The order to sort the groups.
        - top (int): The number of top features used for sorting within each subtype.

        Returns:
        - PivotTable: A new PivotTable sorted by subtype and mutations.
        """
        pivot_table = self.copy()

        # 確保 group_col 存在於 sample_metadata
        if group_col not in pivot_table.sample_metadata.columns:
            raise ValueError(
                f"Column '{group_col}' not found in sample_metadata.")

        sorted_samples = []

        # 依照 subtype_order 進行分組排序
        for subtype in group_order:
            subtype_samples = pivot_table.sample_metadata[
                pivot_table.sample_metadata[group_col] == subtype].index

            if len(subtype_samples) > 0:
                # 篩選出該 subtype 的樣本，並應用 sort_samples_by_mutations
                subtype_pivot = pivot_table.subset(samples=subtype_samples)
                sorted_subtype_pivot = subtype_pivot.sort_samples_by_mutations(
                    top=top)

                sorted_samples.extend(
                    sorted_subtype_pivot.columns)  # 儲存排序後的樣本順序

        # 重新排列 PivotTable
        return pivot_table.subset(samples=sorted_samples)

    def PCA(self, to_binary: bool) -> Tuple[pd.DataFrame, np.ndarray, PCA]:
        """
        Perform Principal Component Analysis on the PivotTable.

        Parameters
        ----------
        to_binary : bool
            Whether to convert the data to binary format before PCA.
            
        Returns
        -------
        tuple
            - pca_result_df : pd.DataFrame with PC1 and PC2 for each sample
            - explained_variance : np.ndarray of variance ratios for PC1 and PC2  
            - pca : sklearn.decomposition.PCA fitted object
        """
        pivot_table = self.to_binary_table() if to_binary else self
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pivot_table.T)  # 樣本在 row，所以轉置
        explained_variance = pca.explained_variance_ratio_
        pca_result_df = pd.DataFrame(
            pca_result, index=pivot_table.columns, columns=["PC1", "PC2"])
        return pca_result_df, explained_variance, pca

    def plot_pca_samples(
        self,
        group_col: str = "subtype",
        figsize: Tuple[int, int] = (8, 6),
        to_binary: bool = False,
        palette_dict: Optional[Dict[str, str]] = None,
        alpha: float = 0.8,
        title: str = "PCA of samples",
        cmap: str = "summer",
        is_numeric: bool = False,
        save_path: Optional[str] = None,
        fontsize: int = 12,
        titlesize: int = 14
    ) -> Tuple[pd.DataFrame, np.ndarray, PCA]:
        """
        Plot PCA scatter plot of samples colored by group_col.
        """
        # caculte PCA result
        pca_result_df, explained_variance, pca = self.PCA(to_binary=to_binary)

        # ensure group_col exists in sample_metadata
        if group_col not in self.sample_metadata.columns:
            raise ValueError(
                f"Column '{group_col}' not found in sample_metadata.")
        pca_result_df[group_col] = self.sample_metadata[group_col]

        # plot PCA scatter plot
        plt.figure(figsize=figsize)

        if is_numeric:
            # numeric → cmap
            scatter = plt.scatter(pca_result_df["PC1"],
                                  pca_result_df["PC2"],
                                  c=pca_result_df[group_col],
                                  cmap=cmap,
                                  alpha=alpha)
            plt.colorbar(scatter, label=group_col)  # 手動加上 colorbar
        else:
            # categorical → palette
            scatter = sns.scatterplot(data=pca_result_df,
                                      x="PC1",
                                      y="PC2",
                                      hue=group_col,
                                      palette=palette_dict or "Set1",
                                      alpha=alpha)

        plt.title(title, fontsize=titlesize)
        plt.xlabel(
            f"Principal Component 1 ({explained_variance[0] * 100:.2f}%)", fontsize=fontsize)
        plt.ylabel(
            f"Principal Component 2 ({explained_variance[1] * 100:.2f}%)", fontsize=fontsize)

        if not is_numeric:
            plt.legend(title=group_col, title_fontsize=fontsize)
        if save_path:
            format = save_path.split('.')[-1].lower()
            pil_kwargs = {
                "compression": "tiff_lzw"} if format == "tiff" else {}

            plt.savefig(save_path,
                        dpi=300,
                        format=format,
                        bbox_inches='tight',
                        pil_kwargs=pil_kwargs)
        plt.show()
        return pca_result_df, explained_variance, pca

    def head(self, n: int = 50) -> "PivotTable":
        """
        Return the first n features (rows) subset of the PivotTable.
        
        Parameters
        ----------
        n : int, default 50
            Number of features to return.
            
        Returns
        -------
        PivotTable
            PivotTable subset containing only the first n features.
        """
        pivot_table = self.copy()
        head_indices = pivot_table.index[:n].tolist()
        return pivot_table.subset(features=head_indices)

    def filter_by_freq(self, threshold: float = 0.05) -> "PivotTable":
        """
        Filter features by their mutation frequency.
        
        Parameters
        ----------
        threshold : float, default 0.05
            Minimum frequency threshold (0 to 1).
            
        Returns
        -------
        PivotTable
            PivotTable containing only features with freq >= threshold.
            
        Raises
        ------
        ValueError
            If 'freq' column is not found in feature_metadata.
        """
        if "freq" not in self.feature_metadata.columns:
            raise ValueError("freq column not found in feature_metadata. Please perform table.add_freq() first")
        pivot_table = self.copy()
        return pivot_table.subset(features=pivot_table.feature_metadata.freq >= threshold)

    def to_cooccur_matrix(self, freq: bool = True) -> 'CooccurrenceMatrix':
        """
        Convert to co-occurrence matrix format.
        
        Parameters
        ----------
        freq : bool, default True
            If True, normalize by sample count to get frequencies.
            If False, return raw co-occurrence counts.
            
        Returns
        -------
        CooccurrenceMatrix
            Matrix showing feature co-occurrence patterns.
        """
        table = self.to_binary_table()
        matrix = table.astype(int)
        cooccur_matrix = matrix.dot(matrix.T)
        if freq:
            cooccur_matrix = cooccur_matrix / matrix.shape[1]

        return CooccurrenceMatrix(cooccur_matrix)

    def to_binary_table(self) -> "PivotTable":
        """
        Convert PivotTable to binary format.
        
        Converts all non-False values to True, creating a binary representation
        of the mutation data.
        
        Returns
        -------
        PivotTable
            Bool PivotTable where True indicates mutation presence.
        """
        binary_pivot_table = self.copy()
        # Convert to boolean and ensure proper dtype
        binary_data = (binary_pivot_table != False).astype(bool)
        binary_pivot_table[:] = binary_data
        return binary_pivot_table

    def mutation_enrichment_test(
        self,
        group_col: str,
        group1: str,
        group2: str,
        alpha: float = 0.05,
        minimum_mutations: int = 2,
        method: Literal["chi2", "fisher"] = "chi2"
    ) -> pd.DataFrame:
        """
        Perform statistical enrichment test for mutations between two groups.
        
        Tests whether specific mutations are significantly enriched in one group
        compared to another using either Chi-squared test or Fisher's exact test.
        Multiple testing correction is applied using the Benjamini-Hochberg method.
        
        Parameters
        ----------
        group_col : str
            Column name in sample_metadata that contains group assignments.
        group1 : str
            Name of the first group to compare.
        group2 : str  
            Name of the second group to compare.
        alpha : float, default 0.05
            Significance level for multiple testing correction.
        minimum_mutations : int, default 2
            Minimum number of mutations required in either group to include
            a feature in the analysis.
        method : {"chi2", "fisher"}, default "chi2"
            Statistical test method to use:
            - "chi2": Chi-squared test of independence
            - "fisher": Fisher's exact test
            
        Returns
        -------
        pd.DataFrame
            Results DataFrame with the following columns:
            - "{group1}_True": Count of mutated samples in group1
            - "{group1}_False": Count of non-mutated samples in group1  
            - "{group2}_True": Count of mutated samples in group2
            - "{group2}_False": Count of non-mutated samples in group2
            - "p_value": Raw p-values from statistical test
            - "adjusted_p_value": FDR-corrected p-values
            - "is_significant": Boolean indicating significance after correction
            - "test_method": Method used for testing
            
        Raises
        ------
        ValueError
            If unsupported statistical method is specified.
            
        Notes
        -----
        The method creates 2x2 contingency tables for each feature:
        
                    Group1   Group2
        Mutated       a        b
        Not mutated   c        d
        
        Features with fewer than `minimum_mutations` in both groups are excluded
        to avoid testing rare mutations that may not be statistically meaningful.
        
        Examples
        --------
        >>> # Test for mutations enriched in LUAD vs LUSC
        >>> results = pivot_table.mutation_enrichment_test(
        ...     group_col="subtype",
        ...     group1="LUAD", 
        ...     group2="LUSC",
        ...     method="fisher"
        ... )
        >>> significant = results[results["is_significant"]]
        >>> print(f"Found {len(significant)} significantly enriched mutations")
        
        See Also
        --------
        scipy.stats.chi2_contingency : Chi-squared test implementation
        scipy.stats.fisher_exact : Fisher's exact test implementation
        statsmodels.stats.multitest.multipletests : Multiple testing correction
        """

        binary_pivot_table = self.to_binary_table()
        sample_metadata = binary_pivot_table.sample_metadata

        subset1 = binary_pivot_table.subset(
            samples=sample_metadata[group_col] == group1)
        subset2 = binary_pivot_table.subset(
            samples=sample_metadata[group_col] == group2)

        df = pd.DataFrame(index=binary_pivot_table.index,
                          columns=[f"{group1}_True", f"{group1}_False", f"{group2}_True", f"{group2}_False"])

        df[f"{group1}_True"] = subset1.sum(axis=1)
        df[f"{group1}_False"] = len(subset1.columns) - df[f"{group1}_True"]

        df[f"{group2}_True"] = subset2.sum(axis=1)
        df[f"{group2}_False"] = len(subset2.columns) - df[f"{group2}_True"]

        # Filter out mutations not observed enough
        df = df[(df[f"{group1}_True"] >= minimum_mutations) |
                (df[f"{group2}_True"] >= minimum_mutations)]

        def get_p_value(row: pd.Series) -> float:
            contingency_table = row.values.astype(int).reshape(2, 2)
            if method == "chi2":
                _, p, _, _ = chi2_contingency(contingency_table)
            elif method == "fisher":
                _, p = fisher_exact(contingency_table)
            else:
                raise ValueError(f"Unsupported method: {method}")
            return p

        df["p_value"] = df.apply(get_p_value, axis=1)

        p_values = df["p_value"].values
        reject, adjusted_p_value, _, _ = multipletests(
            p_values, method="fdr_bh", alpha=alpha)

        df["adjusted_p_value"] = adjusted_p_value
        df["is_significant"] = reject
        df["test_method"] = method

        return df

    def compute_similarity(self, method: Literal["cosine", "hamming", "jaccard"] = "cosine") -> 'SimilarityMatrix':
        """
        Compute sample similarity matrix using specified metric.
        
        Parameters
        ----------
        method : {"cosine", "hamming", "jaccard"}, default "cosine"
            Similarity metric to use.
            
        Returns
        -------
        SimilarityMatrix
            Pairwise similarity matrix between samples.
            
        Raises
        ------
        ValueError
            If unsupported similarity method is specified.
        """
        X = self.T.values if hasattr(self.T, "values") else np.array(self.T)
        if method == "cosine":
            similarity = cosine_similarity(X)
        elif method in {"hamming", "jaccard"}:
            similarity = 1 - pairwise_distances(X, metric=method)
        else:
            raise ValueError(f"Unsupported similarity method: {method}")

        similarity_df = pd.DataFrame(similarity,
                                     index=self.columns,
                                     columns=self.columns)
        return SimilarityMatrix(similarity_df)

    def order(self, group_col: str, group_order: List[str]) -> "PivotTable":
        """
        Reorder samples by group membership.
        
        Parameters
        ----------
        group_col : str
            Column name in sample_metadata containing group information.
        group_order : List[str]
            Order of groups for sample arrangement.
            
        Returns
        -------
        PivotTable
            PivotTable with samples ordered by group membership.
        """
        subset_list = []
        assert group_col in self.sample_metadata.columns
        for group in group_order:
            subset_list.append(self.subset(
                samples=self.sample_metadata[group_col] == group))
        table = PivotTable.merge(subset_list)
        return table

    @staticmethod
    def prepare_data(maf: 'MAF') -> "PivotTable":
        """
        Prepare and process MAF data into a sorted PivotTable.
        
        Filters MAF for nonsynonymous mutations, converts to PivotTable,
        adds frequency calculations, and sorts by feature frequency 
        and sample mutation patterns.
        
        Parameters
        ----------
        maf : MAF
            Input MAF object containing mutation data.
            
        Returns
        -------
        PivotTable
            Processed and sorted PivotTable ready for analysis.
        """
        from .MAF import MAF
        filtered_all_case_maf = maf.filter_maf(MAF.nonsynonymous_types)
        pivot_table = filtered_all_case_maf.to_pivot_table()
        sorted_pivot_table = (pivot_table
                              .add_freq()
                              .sort_features()
                              .sort_samples_by_mutations()
                              )
        return sorted_pivot_table


def capture_size(bed_path: str) -> float:
    """
    Calculate the total capture size (in megabases) from a BED file.

    The BED file must have at least three columns: chrom, start, end.

    Parameters:
        bed_path (str): Path to the BED file.

    Returns:
        float: Total capture region size in megabases (Mb).
    """
    bed = pd.read_csv(bed_path, sep='\t', header=None)
    bed = bed.iloc[:, 0:3]
    bed.columns = ['chrom', 'start', 'end']
    capture_length = bed.end - bed.start
    return capture_length.sum()/1e6  # in MB
