import numpy as np
import pandas as pd
import os
import re
from typing import Optional, List, TYPE_CHECKING
from .PivotTable import PivotTable

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class CopyNumberVariationTable(PivotTable):
    """
    CopyNumberVariationTable class for handling copy number data.
    """

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


    def to_thresholded_cnv(self):
        """
        Convert cnv table to a thresholded version.
        
        del_high_threshold, del_low_threshold, amp_low_threshold, amp_high_threshold must be defined in sample_metadata.
        """
        cutoffs = self.sample_metadata.loc[:, ["del_high_threshold", 
                                                "del_low_threshold", 
                                                "amp_low_threshold", 
                                                "amp_high_threshold", 
                                                ]]
        def classify_cnv_column(column, cutoffs):
            sample = column.name
            thresholds = cutoffs.loc[sample]
            return column.apply(lambda x: 
                -2 if x < thresholds['del_high_threshold'] else
                -1 if x < thresholds['del_low_threshold'] else
                +2 if x > thresholds['amp_high_threshold'] else
                +1 if x > thresholds['amp_low_threshold'] else
                0
            )
        thresholded_cnv = self.apply(lambda col: classify_cnv_column(col, cutoffs), axis=0)
        return thresholded_cnv

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
    
    def to_cluster_table(self, cluster_col="cluster") -> pd.DataFrame:
        """
        cluster must in feature_metadata.
        """
        if cluster_col not in self.feature_metadata.columns:
            raise ValueError(f"Column '{cluster_col}' not found in feature_metadata.")
        # save clustering results
        table = self.copy()
        table.feature_metadata["chr_arm"] = table.feature_metadata["Chromosome"].astype(str) + table.feature_metadata["Arm"]
        table[cluster_col] = table.feature_metadata[cluster_col]
        
        # to cluster table
        cluster_table = CopyNumberVariationTable(pd.DataFrame(table).groupby(cluster_col).mean())
        cluster_table.sample_metadata = table.sample_metadata

        gb = table.feature_metadata.groupby(cluster_col)
        cluster_table.feature_metadata["unique_chr_arm"]  = gb["chr_arm"].unique()
        cluster_table.feature_metadata["features"] = gb.apply(lambda df: list(df.index))
        cluster_table.feature_metadata["features_count"] = cluster_table.feature_metadata["features"].apply(len)
        return cluster_table.rename_index_and_columns()

    def plot_cnv_band_ratio(
        self,
        cluster_id: str,
        mode: str = "gain",
        threshold: float = 0.1,
        sample_type: str = "T",
        subtype_order: Optional[List[str]] = None,
        ax: Optional["plt.Axes"] = None,
        cmap: Optional[str] = None,
        show: bool = True,
        title: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Plot gain or loss frequency across cytobands for a specific CNV cluster and sample type.

        Parameters
        ----------
        cluster_id : str
            Cluster ID to extract features (e.g., "C47" or "C6").
        mode : {"gain", "loss"}
            Type of alteration to compute.
        threshold : float
            Threshold for gain or loss (default: 0.1).
        sample_type : str
            Sample type to subset (default: "T").
        subtype_order : list of str, optional
            Order of subtypes to show in columns. If None, uses ["LUAD", "ASC", "LUSC"].
        ax : matplotlib Axes, optional
            If provided, plot on this Axes object.
        cmap : str, optional
            Colormap (default: "Reds" for gain, "Blues" for loss).
        show : bool
            Whether to show the plot.
        title : str, optional
            Title to display on plot.
        
        Returns
        -------
        pd.DataFrame
            Cytoband × Subtype frequency table.
        """
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(f"Required plotting libraries not available: {e}. "
                            "Please install matplotlib and seaborn.")
        
        # Set default values
        if subtype_order is None:
            subtype_order = ["LUAD", "ASC", "LUSC"]
        
        # Select default colormap
        if cmap is None:
            cmap = "Reds" if mode == "gain" else "Blues"
        
        # Validate mode
        if mode not in ["gain", "loss"]:
            raise ValueError("mode must be either 'gain' or 'loss'")
        
        # Check if required columns exist
        if "cluster" not in self.feature_metadata.columns:
            raise ValueError("Column 'cluster' not found in feature_metadata.")
        if "Band" not in self.feature_metadata.columns:
            raise ValueError("Column 'Band' not found in feature_metadata. This is typically created by parsing Cytoband information.")
        if "sample_type" not in self.sample_metadata.columns:
            raise ValueError("Column 'sample_type' not found in sample_metadata.")
        if "subtype" not in self.sample_metadata.columns:
            raise ValueError("Column 'subtype' not found in sample_metadata.")
        
        # 1. Subset cluster & sample type
        cluster_features = self.feature_metadata[self.feature_metadata["cluster"] == cluster_id].index
        sample_type_samples = self.sample_metadata[self.sample_metadata["sample_type"] == sample_type].index
        
        if len(cluster_features) == 0:
            raise ValueError(f"No features found for cluster '{cluster_id}'")
        if len(sample_type_samples) == 0:
            raise ValueError(f"No samples found for sample_type '{sample_type}'")
        
        subset = self.subset(features=cluster_features, samples=sample_type_samples)
        
        # Order samples by subtype if specified
        if subtype_order:
            available_subtypes = [s for s in subtype_order if s in subset.sample_metadata["subtype"].values]
            if available_subtypes:
                ordered_samples = []
                for subtype in available_subtypes:
                    subtype_samples = subset.sample_metadata[subset.sample_metadata["subtype"] == subtype].index
                    ordered_samples.extend(subtype_samples.tolist())
                subset = subset.subset(samples=ordered_samples)

        # 2. Compute cytoband average table
        tmp = pd.DataFrame(subset.copy())
        tmp["Band"] = subset.feature_metadata["Band"]
        band_df = tmp.groupby("Band").mean()

        # 3. Compute binary gain/loss
        if mode == "gain":
            binary_df = band_df > threshold
        elif mode == "loss":
            binary_df = band_df < -threshold

        # 4. Compute frequency table by subtype
        freq_df = (
            binary_df.T
            .assign(subtype=subset.sample_metadata["subtype"].values)
            .groupby("subtype")
            .mean()
            .T
        )

        if subtype_order:
            freq_df = freq_df[[s for s in subtype_order if s in freq_df.columns]]

        # 5. Plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 10))

        sns.heatmap(
            freq_df, ax=ax, cmap=cmap,
            vmin=0, vmax=1,
            annot=True, fmt=".0%",
            cbar_kws={"label": f"{mode.capitalize()} Frequency"}
        )
        ax.set_title(title or f"{mode.capitalize()} Ratio - Cluster {cluster_id}")
        ax.set_xlabel("Subtype")
        ax.set_ylabel("Cytoband")
        
        if show and ax is None:
            plt.show()
        
        return freq_df
    

    def to_cnv_table(all_sample_df):
        cnv_matrix = all_sample_df.pivot_table(
            index="gene_id", # ENSG ID to avoid duplication
            columns="sample_ID",  
            values="copy_number" 
        )

        feature_metadata = (
            all_sample_df
            .drop_duplicates(subset=["gene_id"])
            .set_index("gene_id")[["gene_name", "chromosome", "start", "end"]] 
        )

        # create CopyNumberVariationTable object
        table = CopyNumberVariationTable(cnv_matrix)
        table = table.add_feature_metadata(feature_metadata, fill_value=np.nan)
        table._validate_metadata()

        # make gene_name unique
        dup_genes = table.feature_metadata.gene_name[table.feature_metadata.gene_name.duplicated()].unique()
        dup_mask = table.feature_metadata.gene_name.isin(dup_genes) # if gene_name is duplicated
        table.feature_metadata.loc[dup_mask, "gene_name"] = table.feature_metadata.gene_name[dup_mask] + "|" + table.feature_metadata.index[dup_mask] # add ENSG ID to duplicated gene_name

        # set index to gene_name
        table.index = table.feature_metadata.gene_name
        table.feature_metadata.index = table.feature_metadata.gene_name
        return table

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

def TCGA_sample_type(TCGA_barcode):
    pattern = r"-([0-2][0-9])[A-Z]$"
    match = re.search(pattern, TCGA_barcode)
    if not match:
        raise ValueError(f"Barcode {TCGA_barcode} does not match expected format.")
    sample_type = int(match.group(1))
    if sample_type < 10:
        return "T"  # Tumor
    elif sample_type < 20:
        return "N"  # Normal
    elif sample_type < 30:
        return "C"  # Control
    else:
        raise ValueError(f"Sample type code {sample_type} is not recognized.")

def get_target_sample_ID(paired_sample_IDs, target_sample_type):
    paired_sample_IDs_list = paired_sample_IDs.split(",")
    for sample_ID in paired_sample_IDs_list:
        sample_ID = sample_ID.strip()
        sample_type = TCGA_sample_type(sample_ID)
        if sample_type == target_sample_type:
            return sample_ID
    raise ValueError("No matching sample type found.")

def read_TCGA_ASCAT3_CNV_file_sheet(file_path, file_suffix="ascat3.gene_level_copy_number.v36.tsv"):
    file_sheet = pd.read_csv(file_path, sep="\t", header=0)
    file_sheet = file_sheet[file_sheet["File Name"].str.contains(file_suffix, regex=True)].copy()
    file_sheet["tumor_sample_ID"] = file_sheet["Sample ID"].apply(lambda x: get_target_sample_ID(x, "T"))
    file_sheet["normal_sample_ID"] = file_sheet["Sample ID"].apply(lambda x: get_target_sample_ID(x, "N"))
    return file_sheet

def read_cnv_files(base_dir, file_sheet):
	all_samples = []
	for _, row in file_sheet.iterrows():
		tumor_sample_ID = row["tumor_sample_ID"]
		path = os.path.join(base_dir, row["File Name"])
		cnv_df = pd.read_csv(path, sep="\t")
		cnv_df["sample_ID"] = tumor_sample_ID
		cnv_df = cnv_df.dropna()
		all_samples.append(cnv_df)
	all_sample_df = pd.concat(all_samples, axis=0)
	return all_sample_df
