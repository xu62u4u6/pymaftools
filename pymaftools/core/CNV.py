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
        
        # Subset samples if specified
        if samples is not None:
            available_samples = table.columns[table.columns.isin(samples)].tolist()
            if available_samples:
                table = table.subset(samples=available_samples)
            else:
                print(f"Warning: No matching samples found from {len(samples)} specified samples")
            
        return table
        

        
        
        