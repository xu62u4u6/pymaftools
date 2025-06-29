import numpy as np
import pandas as pd
import re
from typing import Optional, List
from .PivotTable import PivotTable


class CopyNumberVariationTable(PivotTable):
    @property
    def _constructor(self):
        """Return constructor for pandas operations that preserves CNV type."""
        def _new_constructor(*args, **kwargs):
            obj = CopyNumberVariationTable(*args, **kwargs)
            # attempt to preserve metadata if available
            if hasattr(self, 'sample_metadata') and not self.sample_metadata.empty:
                try:
                    obj.sample_metadata = self.sample_metadata.copy()
                except:
                    pass
            if hasattr(self, 'feature_metadata') and not self.feature_metadata.empty:
                try:
                    obj.feature_metadata = self.feature_metadata.copy()
                except:
                    pass
            return obj
        return _new_constructor
    
    @classmethod
    def from_pivot_table(cls, table: PivotTable) -> 'CopyNumberVariationTable':
        """
        Create a CopyNumberVariationTable object from a PivotTable object, preserving all metadata.

        Parameters
        ----------
        table : PivotTable
            A PivotTable object containing sample_metadata and feature_metadata attributes.

        Returns
        -------
        CopyNumberVariationTable
            A CopyNumberVariationTable object with original sample_metadata and feature_metadata preserved.
        """
        if not hasattr(table, 'sample_metadata') or not hasattr(table, 'feature_metadata'):
            raise ValueError(
                "PivotTable must have sample_metadata and feature_metadata attributes.")

        cnv_table = cls(table.values, index=table.index, columns=table.columns)
        cnv_table.sample_metadata = table.sample_metadata.copy()
        cnv_table.feature_metadata = table.feature_metadata.copy()
        cnv_table._validate_metadata()
        return cnv_table

    @classmethod
    def read_gistic_arm_level(
        cls,
        file_path: str,
    ):
        """
        Read GISTIC broad data by arm level file.

        Parameters
        ----------
        file_path : str
            Path to the GISTIC arm-level results file.

        Returns
        -------
        CopyNumberVariationTable
            A CopyNumberVariationTable object with arm-level copy number data.
        """
        df = pd.read_csv(file_path,
                         sep="\t", index_col=0)
        df.columns = df.columns.str.replace(".call", "", regex=False)
        df = df.fillna(0)
        order = [arm for i in range(1, 23) for arm in (f"{i}p", f"{i}q")]
        table = CopyNumberVariationTable(df).reindex(index=order, fill_value=0)
        table.feature_metadata["Chromosome"] = table.index.str.replace(
            "p", "").str.replace("q", "")
        table.feature_metadata["Arm"] = table.feature_metadata.index.str[-1]
        return table.rename_index_and_columns()

    @classmethod
    def read_gistic_gene_level(
        cls,
        file_path: str,
        feature_columns: List[str] = ["Gene Symbol", "Gene ID", "Cytoband"],
        samples: Optional[List[str]] = None
    ):
        """
        Read GISTIC results file and create a CopyNumberVariationTable object.

        This method reads GISTIC output files (typically all_data_by_genes.txt or 
        all_thresholded.by_genes.txt) and converts them into a CopyNumberVariationTable object with 
        properly formatted feature and sample metadata.

        Parameters
        ----------
        file_path : str
            Path to the GISTIC results file (tab-separated format).
        feature_columns : list of str, default ["Gene Symbol", "Gene ID", "Cytoband"]
            List of column names to be treated as feature metadata. These columns
            will be separated from the main data matrix.
        samples : None or list of str, optional
            List of sample names to subset. If None, all samples are kept.
            Only samples present in both the data and this list will be retained.

        Returns
        -------
        CopyNumberVariationTable
            A CopyNumberVariationTable object containing:
            - Main data matrix with gene symbols as index
            - feature_metadata with gene information and parsed chromosome data
            - sample_metadata with case_ID and sample_type extracted from column names

        Raises
        ------
        ValueError
            If 'Gene Symbol' column is not found in the input file.

        Notes
        -----
        The method performs several data processing steps:
        1. Removes '.call' suffix from column names
        2. Separates feature metadata from data columns
        3. Parses sample names to extract case_ID and sample_type (split by last '_')
        4. Parses Cytoband information into Chromosome, Arm, and Band columns
        5. Subsets data to specified samples if provided

        The Cytoband parsing supports both numeric chromosomes (1-22) and 
        sex chromosomes (X, Y) using the pattern: chromosome + arm (p/q) + band.

        Examples
        --------
        >>> cnv = CopyNumberVariationTable.read_gistic_gene_level('data/all_data_by_genes.txt')
        >>> cnv = CopyNumberVariationTable.read_gistic_gene_level('data/all_thresholded.by_genes.txt', 
        ...                       feature_columns=['Gene Symbol', 'Gene ID', 'Cytoband', 'Locus ID'])
        >>> cnv = CopyNumberVariationTable.read_gistic_gene_level('data/all_data_by_genes.txt', 
        ...                       samples=['LUAD_001_T', 'LUAD_002_T'])
        """

        df = pd.read_csv(file_path, sep="\t")
        if "Gene Symbol" not in df.columns:
            raise ValueError("Gene Symbol column is required.")

        df = df.copy()  # Avoid modifying the original DataFrame
        df = df.set_index("Gene Symbol")
        df.columns = df.columns.str.replace(".call", "", regex=False)

        # Separate feature columns from data columns
        feature_mask = df.columns.isin(feature_columns)
        data_columns = df.loc[:, ~feature_mask]
        metadata_columns = df.loc[:, feature_mask]

        # Create CopyNumberVariationTable object (inherits from PivotTable)
        table = cls(data_columns)
        table.feature_metadata = metadata_columns.copy()

        # Process sample metadata - extract case_ID and sample_type from column names
        table.sample_metadata[["case_ID", "sample_type"]] = (
            table.columns.to_series()
            .str.rsplit("_", n=1)
            .apply(pd.Series)
        )

        # Process chromosome information - parse cytoband into components
        if "Cytoband" in table.feature_metadata.columns:
            cytoband_extract = table.feature_metadata["Cytoband"].str.extract(
                r"(\w+)([pq])([\d.]+)", expand=True
            )
            table.feature_metadata[["Chromosome",
                                    "Arm", "Band"]] = cytoband_extract

            # Prepare chromosomal data for sorting
            table._prepare_chromosomal_sorting()

        # Subset samples if specified
        if samples is not None:
            available_samples = table.columns[table.columns.isin(
                samples)].tolist()
            if available_samples:
                table = table.subset(samples=available_samples)
            else:
                print(
                    f"Warning: No matching samples found from {len(samples)} specified samples")

        return table.rename_index_and_columns()

    def _prepare_chromosomal_sorting(self) -> None:
        """
        Prepare chromosome, arm, and band columns for proper sorting.

        Converts chromosome, arm, and band columns to categorical data types
        with proper ordering for genomic position sorting.

        Notes
        -----
        This method modifies the feature_metadata in place by:
        1. Converting Chromosome to categorical with order: 1, 2, ..., 22, X, Y
        2. Converting Arm to categorical with order: p, q
        3. Adding a Band_numeric column for numerical sorting of bands
        """
        if not all(col in self.feature_metadata.columns for col in ['Chromosome', 'Arm', 'Band']):
            return

        # Convert chromosome to categorical for proper sorting
        # Define the order: 1, 2, ..., 22, X, Y
        chrom_order = [str(i) for i in range(1, 23)] + ['X', 'Y']
        self.feature_metadata['Chromosome'] = pd.Categorical(
            self.feature_metadata['Chromosome'],
            categories=chrom_order,
            ordered=True
        )

        # Convert arm to categorical (p comes before q)
        arm_order = ['p', 'q']
        self.feature_metadata['Arm'] = pd.Categorical(
            self.feature_metadata['Arm'],
            categories=arm_order,
            ordered=True
        )

        # Convert band to numeric for proper sorting
        self.feature_metadata['Band_numeric'] = pd.to_numeric(
            self.feature_metadata['Band'],
            errors='coerce'
        )

    def sort_by_chromosome(self, ascending: bool = True) -> 'CopyNumberVariationTable':
        """
        Sort CopyNumberVariationTable data by chromosomal position.

        Sorts the CopyNumberVariationTable data by chromosome number, arm (p/q), and band position.
        Handles both numeric chromosomes (1-22) and sex chromosomes (X, Y).

        Parameters
        ----------
        ascending : bool, default True
            Whether to sort in ascending order. If False, sorts in descending order.

        Returns
        -------
        CopyNumberVariationTable
            A new CopyNumberVariationTable object with features sorted by chromosomal position.

        Notes
        -----
        The sorting order is:
        1. Chromosome: 1, 2, ..., 22, X, Y
        2. Arm: p (short arm) before q (long arm)
        3. Band: numerical order (e.g., 11.1, 11.2, 12.1)

        Requires the feature_metadata to have 'Chromosome', 'Arm', and 'Band' columns,
        which are typically created by the read_gistic method when parsing Cytoband information.

        Examples
        --------
        >>> cnv_sorted = cnv_table.sort_by_chromosome()
        >>> cnv_desc = cnv_table.sort_by_chromosome(ascending=False)
        """
        if not all(col in self.feature_metadata.columns for col in ['Chromosome', 'Arm', 'Band']):
            raise ValueError("Chromosome, Arm, and Band columns are required for sorting. "
                             "These are typically created by parsing Cytoband information.")

        # Create a copy to avoid modifying the original object
        sorted_cnv_table = self.copy()

        # Prepare chromosomal data for sorting if not already done
        if 'Band_numeric' not in sorted_cnv_table.feature_metadata.columns:
            sorted_cnv_table._prepare_chromosomal_sorting()

        # Sort by chromosome, arm, and band
        sort_columns = ['Chromosome', 'Arm', 'Band_numeric']
        sort_order = [ascending] * len(sort_columns)

        sorted_indices = sorted_cnv_table.feature_metadata.sort_values(
            sort_columns,
            ascending=sort_order
        ).index

        # Reorder the CopyNumberVariationTable data and metadata
        sorted_cnv_table = sorted_cnv_table.subset(features=sorted_indices)

        # Clean up temporary column
        sorted_cnv_table.feature_metadata.drop(
            'Band_numeric', axis=1, inplace=True)

        return sorted_cnv_table

    @staticmethod
    def read_all_gistic(all_data_by_genes_file,
                        sample_cutoffs_file,
                        all_thresholded_by_genes_file,
                        broad_values_by_arm_file):
        """
        Read all GISTIC output files and create CopyNumberVariationTable objects.

        Parameters
        ----------
        all_data_by_genes_file : str
            Path to the GISTIC all_data_by_genes.txt file.
        sample_cutoffs_file : str
            Path to the GISTIC sample_cutoffs.txt file.
        all_thresholded_by_genes_file : str
            Path to the GISTIC all_thresholded.by_genes.txt file.
        broad_values_by_arm_file : str
            Path to the GISTIC broad_values_by_arm.txt file.

        Returns
        -------
        tuple
            A tuple containing:
            - all_data_by_genes_table : CopyNumberVariationTable
            - sample_cutoff_df : pd.DataFrame
            - thresholded_cnv_table : CopyNumberVariationTable
            - broad_values_by_arm_table : CopyNumberVariationTable
        """

        all_data_by_genes_table = CopyNumberVariationTable.read_gistic_gene_level(all_data_by_genes_file)
        sample_cutoff_df = read_sample_cutoff_file(sample_cutoffs_file)
        thresholded_cnv_table = CopyNumberVariationTable.read_gistic_gene_level(
            all_thresholded_by_genes_file,
            feature_columns=["Gene Symbol", "Gene ID", "Cytoband", "Locus ID"])
        broad_values_by_arm_table = CopyNumberVariationTable.read_gistic_arm_level(broad_values_by_arm_file)

        # Add sample metadata to the tables
        all_data_by_genes_table = all_data_by_genes_table.add_sample_metadata(sample_cutoff_df, fill_value=np.nan)
        thresholded_cnv_table = thresholded_cnv_table.add_sample_metadata(sample_cutoff_df, fill_value=np.nan)
        return (all_data_by_genes_table,
                sample_cutoff_df,
                thresholded_cnv_table,
                broad_values_by_arm_table)


def read_sample_cutoff_file(sample_cutoff_file):
    """
    Read the sample cutoff file and extract the amp_thresh and del_thresh values.

    Parameters
    ----------
    sample_cutoff_file : str
        Path to the sample cutoff file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing sample cutoff data with amp_threshold and del_threshold columns.
    """
    with open(sample_cutoff_file, "r") as f:
        first_line = f.readline().strip()

    match = re.search(r'amp_thresh=([\d.]+), del_thresh=([\d.]+)', first_line)
    if not match:
        raise ValueError(
            "Could not find amp_thresh and del_thresh in the first line of the file.")

    amp_threshold = float(match.group(1))
    del_threshold = float(match.group(2))
    sample_cutoffs_df = pd.read_csv(sample_cutoff_file,
                                    sep="\t",
                                    skiprows=1,
                                    index_col=0)  # pass header
    sample_cutoffs_df.index = sample_cutoffs_df.index.str.replace(".call", "")
    sample_cutoffs_df = sample_cutoffs_df.rename(columns={
        "High": "amp_high_threshold",
        "Low": "del_high_threshold"
    })
    sample_cutoffs_df["amp_low_threshold"] = amp_threshold
    sample_cutoffs_df["del_low_threshold"] = -del_threshold
    column_order = [
        "del_high_threshold",
        "del_low_threshold",
        "amp_low_threshold",
        "amp_high_threshold"
    ]
    sample_cutoffs_df = sample_cutoffs_df.reindex(columns=column_order)
    return sample_cutoffs_df
