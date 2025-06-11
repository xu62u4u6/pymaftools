import pandas as pd
from typing import Optional, List
from .PivotTable import PivotTable

class CNV(PivotTable):
    
    @classmethod
    def read_gistic(cls, 
                    file_path: str,
                    feature_columns: List[str] = ["Gene Symbol", "Gene ID", "Cytoband"], 
                    samples: Optional[List[str]] = None):
        """
        Read GISTIC results file and create a CNV object.
        
        This method reads GISTIC output files (typically all_data_by_genes.txt or 
        all_thresholded.by_genes.txt) and converts them into a CNV object with 
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
        CNV
            A CNV object containing:
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
        >>> cnv = CNV.read_gistic('data/all_data_by_genes.txt')
        >>> cnv = CNV.read_gistic('data/all_thresholded.by_genes.txt', 
        ...                       feature_columns=['Gene Symbol', 'Gene ID', 'Cytoband', 'Locus ID'])
        >>> cnv = CNV.read_gistic('data/all_data_by_genes.txt', 
        ...                       samples=['LUAD_001_T', 'LUAD_002_T'])
        """
        
        df = pd.read_csv(file_path, sep="\t")
        if "Gene Symbol" not in df.columns:
            raise ValueError("Gene Symbol column is required.")
        
        
        df = df.copy() # Avoid modifying the original DataFrame
        df = df.set_index("Gene Symbol")
        df.columns = df.columns.str.replace(".call", "", regex=False)
        
        # Separate feature columns from data columns
        feature_mask = df.columns.isin(feature_columns)
        data_columns = df.loc[:, ~feature_mask]
        metadata_columns = df.loc[:, feature_mask]
        
        # Create CNV object (inherits from PivotTable)
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
            table.feature_metadata[["Chromosome", "Arm", "Band"]] = cytoband_extract
            
            # Prepare chromosomal data for sorting
            table._prepare_chromosomal_sorting()
        
        # Subset samples if specified
        if samples is not None:
            available_samples = table.columns[table.columns.isin(samples)].tolist()
            if available_samples:
                table = table.subset(samples=available_samples)
            else:
                print(f"Warning: No matching samples found from {len(samples)} specified samples")
            
        return table
        
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
        
    def sort_by_chromosome(self, ascending: bool = True) -> 'CNV':
        """
        Sort CNV data by chromosomal position.
        
        Sorts the CNV data by chromosome number, arm (p/q), and band position.
        Handles both numeric chromosomes (1-22) and sex chromosomes (X, Y).
        
        Parameters
        ----------
        ascending : bool, default True
            Whether to sort in ascending order. If False, sorts in descending order.
            
        Returns
        -------
        CNV
            A new CNV object with features sorted by chromosomal position.
            
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
        >>> cnv_sorted = cnv.sort_by_chromosome()
        >>> cnv_desc = cnv.sort_by_chromosome(ascending=False)
        """
        if not all(col in self.feature_metadata.columns for col in ['Chromosome', 'Arm', 'Band']):
            raise ValueError("Chromosome, Arm, and Band columns are required for sorting. "
                           "These are typically created by parsing Cytoband information.")
        
        # Create a copy to avoid modifying the original object
        sorted_cnv = self.copy()
        
        # Prepare chromosomal data for sorting if not already done
        if 'Band_numeric' not in sorted_cnv.feature_metadata.columns:
            sorted_cnv._prepare_chromosomal_sorting()
        
        # Sort by chromosome, arm, and band
        sort_columns = ['Chromosome', 'Arm', 'Band_numeric']
        sort_order = [ascending] * len(sort_columns)
        
        sorted_indices = sorted_cnv.feature_metadata.sort_values(
            sort_columns, 
            ascending=sort_order
        ).index
        
        # Reorder the CNV data and metadata
        sorted_cnv = sorted_cnv.subset(features=sorted_indices)
        
        # Clean up temporary column
        sorted_cnv.feature_metadata.drop('Band_numeric', axis=1, inplace=True)
        
        return sorted_cnv