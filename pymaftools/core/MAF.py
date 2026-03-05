from __future__ import annotations

import os
import warnings
from typing import Any

import pandas as pd

from .PivotTable import PivotTable


class MAF(pd.DataFrame):
    """
    A pandas DataFrame subclass for Mutation Annotation Format (MAF) files.

    Provides methods to read, filter, merge, and convert MAF data commonly
    used in cancer genomics pipelines.

    Attributes
    ----------
    index_col : list[str]
        Default columns used to build the row index.
    vaild_variant_classfication : list[str]
        All recognised variant classification labels.
    nonsynonymous_types : list[str]
        Variant classifications considered nonsynonymous.
    """

    index_col = [
        "Hugo_Symbol",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele1",
        "Tumor_Seq_Allele2",
    ]

    # GDC MAF file fields:
    # https://docs.gdc.cancer.gov/Encyclopedia/pages/Mutation_Annotation_Format_TCGAv2/

    vaild_variant_classfication = [
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "In_Frame_Del",
        "In_Frame_Ins",
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Silent",
        "Splice_Site",
        "Translation_Start_Site",
        "Nonstop_Mutation",
        "3'UTR",
        "3'Flank",
        "5'UTR",
        "5'Flank",
        "IGR",
        "Intron",
        "RNA",
        "Targeted_Region",
    ]

    nonsynonymous_types = [
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "In_Frame_Del",
        "In_Frame_Ins",
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Splice_Site",
        "Translation_Start_Site",
        "Nonstop_Mutation",
    ]

    @classmethod
    def read_maf(
        cls,
        maf_path: str | os.PathLike,
        sample_ID: str,
        preffix: str = "",
        suffix: str = "",
    ) -> MAF:
        """
        Read a MAF file and return a MAF object.

        Parameters
        ----------
        maf_path : str or os.PathLike
            Path to the MAF file (first row is skipped as a comment line).
        sample_ID : str
            Sample identifier to assign to all rows.
        preffix : str, default ""
            Prefix prepended to the sample ID.
        suffix : str, default ""
            Suffix appended to the sample ID.

        Returns
        -------
        MAF
            A MAF DataFrame with a composite index built from ``index_col``.
        """
        maf = cls(pd.read_csv(maf_path, skiprows=1, sep="\t"))
        maf["sample_ID"] = f"{preffix}{sample_ID}{suffix}"
        maf.index = maf.loc[:, cls.index_col].apply(
            lambda row: "|".join(row.astype(str)), axis=1
        )  # concat column
        # maf = maf.filter_maf(cls.vaild_variant_classfication)
        return cls(maf)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialise a MAF DataFrame.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to ``pd.DataFrame.__init__``.
        **kwargs : Any
            Keyword arguments forwarded to ``pd.DataFrame.__init__``.
        """
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self) -> type[MAF]:
        """
        Return the constructor for this subclass.

        Returns
        -------
        type[MAF]
            The MAF class, ensuring pandas operations return MAF instances.
        """
        # make sure returned object is MAF type
        return MAF

    def filter_maf(self, mutation_types: list[str]) -> MAF:
        """
        Filter rows by variant classification.

        Parameters
        ----------
        mutation_types : list[str]
            Variant classification values to keep.

        Returns
        -------
        MAF
            Filtered MAF containing only the specified mutation types.
        """
        return self[self.Variant_Classification.isin(mutation_types)]

    @staticmethod
    def merge_mutations(column: pd.Series) -> str | bool:
        """
        Merge multiple mutations for a single gene–sample pair.

        If all values are ``False`` the result is ``False``.  When more than
        one non-false mutation exists the result is ``"Multi_Hit"``, following
        the maftools convention (see maftools issue #347).

        Parameters
        ----------
        column : pd.Series
            Series of variant classifications (or ``False``) for one
            gene–sample combination.

        Returns
        -------
        str or bool
            ``False`` if no mutation, a single classification string, or
            ``"Multi_Hit"`` when multiple mutations are present.
        """
        if (column == False).all():  # noqa: E712
            return False
        # If a gene has ≥2 mutations in a sample, mark as 'Multi_Hit' (even if types are the same).
        # Behavior aligned with maftools fix for issue #347: https://github.com/PoisonAlien/maftools/issues/347
        non_false_mutations = column[column != False]  # noqa: E712
        if len(non_false_mutations) > 1:
            return "Multi_Hit"
        elif len(non_false_mutations) == 1:
            return non_false_mutations[0]

    def to_pivot_table(self) -> PivotTable:
        """
        Create a gene-by-sample pivot table of variant classifications.

        Returns
        -------
        PivotTable
            Pivot table with genes as rows, samples as columns, and variant
            classifications (or ``"Multi_Hit"`` / ``False``) as values.
        """
        pivot_table = self.pivot_table(
            values="Variant_Classification",
            index="Hugo_Symbol",
            columns="sample_ID",
            aggfunc=MAF.merge_mutations,
        ).fillna(False)
        pivot_table = PivotTable(pivot_table)
        pivot_table.sample_metadata["mutations_count"] = self.mutations_count
        # pivot_table.sample_metadata["TMB"] = self.mutations_count / 40
        return pivot_table

    def to_mutation_table(self) -> PivotTable:
        """
        Create a mutation-level pivot table.

        Each row corresponds to a unique mutation (composite index) rather
        than a gene, providing finer resolution than :meth:`to_pivot_table`.

        Returns
        -------
        PivotTable
            Pivot table indexed by individual mutations.
        """
        mutation_table = self.pivot_table(
            index=self.index,
            columns="sample_ID",
            values="Variant_Classification",
            aggfunc="first",
        ).fillna(False)
        mutation_table = PivotTable(mutation_table)
        mutation_table.sample_metadata["mutations_count"] = self.mutations_count
        return mutation_table

    def change_index_level(self, index_col: list[str] | None = None) -> MAF:
        """
        Rebuild the row index from the specified columns.

        Parameters
        ----------
        index_col : list[str] or None, default None
            Columns to concatenate into the index.  When ``None`` the class
            default :attr:`index_col` is used.

        Returns
        -------
        MAF
            A copy of this MAF with the new composite index.
        """
        maf = self.copy()
        if index_col is None:
            index_col = self.index_col
        new_index_col = maf.loc[:, index_col].apply(
            lambda row: "|".join(row.astype(str)), axis=1
        )
        maf.index = new_index_col
        return maf

    @property
    def mutations_count(self) -> pd.Series:
        """
        Count the number of mutations per sample.

        Returns
        -------
        pd.Series
            Series indexed by sample ID with mutation counts as values.
        """
        return self.groupby(self.sample_ID).size()

    def sort_by_chrom(self) -> MAF:
        """
        Sort rows by genomic coordinates.

        Returns
        -------
        MAF
            MAF sorted by Chromosome, Start_Position, and End_Position.
        """
        return self.sort_values(by=["Chromosome", "Start_Position", "End_Position"])

    @staticmethod
    def merge_mafs(mafs: list[MAF]) -> MAF:
        """
        Concatenate multiple MAF objects into one.

        Parameters
        ----------
        mafs : list[MAF]
            MAF objects to concatenate.

        Returns
        -------
        MAF
            A single MAF containing all rows from the input MAFs.
        """
        return MAF(pd.concat(mafs))

    @classmethod
    def read_csv(
        cls,
        csv_path: str | os.PathLike,
        sep: str = "\t",
        reindex: bool = False,
    ) -> MAF:
        """
        Read a CSV/TSV file into a MAF object.

        Parameters
        ----------
        csv_path : str or os.PathLike
            Path to the CSV or TSV file.
        sep : str, default "\\t"
            Column delimiter.
        reindex : bool, default False
            If ``True``, rebuild the composite index from :attr:`index_col`
            after reading.  Otherwise the first column is used as the index.

        Returns
        -------
        MAF
            MAF constructed from the file contents.
        """
        if reindex:
            maf = cls(pd.read_csv(csv_path, sep=sep))
            maf = maf.change_index_level()
        else:
            maf = cls(pd.read_csv(csv_path, sep=sep, index_col=0))
        return maf

    def to_csv(self, csv_path: str | os.PathLike, **kwargs: Any) -> None:
        """
        Write the MAF to a CSV/TSV file.

        Default behaviour writes a tab-separated file with the index included.

        Parameters
        ----------
        csv_path : str or os.PathLike
            Destination file path.
        **kwargs : Any
            Additional keyword arguments forwarded to
            ``pd.DataFrame.to_csv``.
        """
        # Set default arguments
        kwargs.setdefault("index", True)  # Ensure index is saved by default
        kwargs.setdefault("sep", "\t")  # Default to tab-separated values

        # Call the parent class's to_csv method
        super().to_csv(csv_path, **kwargs)

    def to_MAF(self, maf_path: str | os.PathLike, **kwargs: Any) -> None:
        """
        Write the data as a standard MAF file (no index column).

        Parameters
        ----------
        maf_path : str or os.PathLike
            Destination file path.
        **kwargs : Any
            Additional keyword arguments forwarded to
            ``pd.DataFrame.to_csv``.
        """
        # Set default arguments
        kwargs.setdefault("index", False)  # Ensure index is saved by default
        kwargs.setdefault("sep", "\t")  # Default to tab-separated values

        # Call the parent class's to_csv method
        super().to_csv(maf_path, **kwargs)

    def to_base_change_pivot_table(self) -> PivotTable:
        """
        Build a base-change pivot table with transition/transversion stats.

        Only SNPs are considered.  The returned PivotTable has base-change
        categories as rows and samples as columns, with ``ti``, ``tv``, and
        ``ti/tv`` ratio stored in ``sample_metadata``.

        Returns
        -------
        PivotTable
            Pivot table of base-change counts with ti/tv metadata.
        """
        maf = self.copy()
        base_change = maf.loc[
            maf.Variant_Type == "SNP",
            ["Reference_Allele", "Tumor_Seq_Allele2", "sample_ID"],
        ]
        base_change["Base_Change"] = (
            base_change["Reference_Allele"] + "→" + base_change["Tumor_Seq_Allele2"]
        )
        pivot_table = base_change.pivot_table(
            values="Reference_Allele",
            index="sample_ID",
            columns="Base_Change",
            aggfunc="count",
            fill_value=0,
        )
        pivot_table = PivotTable(pivot_table.T)
        pivot_table.sample_metadata["ti"] = pivot_table.loc[
            ["A→G", "C→T", "G→A", "T→C"]
        ].sum()
        pivot_table.sample_metadata["tv"] = pivot_table.loc[
            ["A→C", "A→T", "C→A", "C→G", "G→C", "G→T", "T→A", "T→G"]
        ].sum()
        pivot_table.sample_metadata["ti/tv"] = (
            pivot_table.sample_metadata.ti / pivot_table.sample_metadata.tv
        )
        return pivot_table

    def get_protein_info(self, gene: str) -> tuple[int | None, list[dict]]:
        """
        Extract protein mutation information for a given gene.

        Parameters
        ----------
        gene : str
            Hugo gene symbol to query.

        Returns
        -------
        AA_length : int or None
            Total amino-acid length of the protein, or ``None`` if
            unavailable.
        mutations_data : list[dict]
            List of dicts with keys ``"position"``, ``"type"``, and
            ``"count"`` describing nonsynonymous mutations.
        """

        def extract_protein_start(pos):
            if pd.isna(pos):
                return None
            pos = str(pos).split("/")[0]
            if "-" in pos:
                return int(pos.split("-")[0])
            try:
                return int(pos)
            except (ValueError, TypeError):
                return None

        maf = self.filter_maf(self.nonsynonymous_types)
        sub_df = maf.loc[
            maf["Hugo_Symbol"] == gene,
            ["Protein_position", "Variant_Classification", "Variant_Type"],
        ].copy()

        # add amino acid position
        sub_df["AA_Position"] = sub_df["Protein_position"].apply(extract_protein_start)

        # get total AA length（858/1210 → 1210）
        try:
            AA_length = int(
                sub_df["Protein_position"].dropna().values[0].split("/")[-1]
            )
        except (ValueError, TypeError, IndexError):
            AA_length = None

        # count mutations and to dict
        mutations_data = (
            sub_df.dropna(subset=["AA_Position", "Variant_Classification"])
            .groupby(["AA_Position", "Variant_Classification"])
            .size()
            .reset_index(name="count")
            .rename(
                columns={"AA_Position": "position", "Variant_Classification": "type"}
            )
            .to_dict(orient="records")
        )

        return AA_length, mutations_data

    @staticmethod
    def get_domain_info(
        gene_name: str,
        AA_length: int,
        protein_domains_path: str | os.PathLike | None = None,
    ) -> tuple[list[dict], str]:
        """
        Look up protein domain annotations for a gene.

        Parameters
        ----------
        gene_name : str
            HGNC gene symbol.
        AA_length : int
            Amino-acid length used to match the correct transcript.
        protein_domains_path : str, os.PathLike, or None, default None
            Path to a protein domains CSV.  When ``None`` the bundled
            dataset (derived from maftools) is used.

        Returns
        -------
        domains : list[dict]
            List of dicts with ``"Start"``, ``"End"``, and ``"Label"`` keys.
        refseq_id : str
            The RefSeq transcript ID used.

        Raises
        ------
        ValueError
            If no domain information is found for the given gene and length.
        """
        if protein_domains_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # get domain info from https://github.com/PoisonAlien/maftools/blob/master/inst/extdata/protein_domains.RDs
            protein_domains_path = os.path.join(
                script_dir, "../data/protein_domains.csv"
            )

        protein_domains = pd.read_csv(
            protein_domains_path, index_col=0, low_memory=False
        )
        subset = protein_domains.loc[
            (protein_domains.HGNC == gene_name)
            & (protein_domains["aa.length"] == AA_length)
        ]
        if subset.empty:
            raise ValueError(
                f"No domain info found for {gene_name} with length {AA_length}"
            )

        refseq_ids = subset["refseq.ID"].unique()
        if len(refseq_ids) != 1:
            warnings.warn(
                f"Multiple refseq.IDs found for {gene_name} with length {AA_length}: {refseq_ids}. "
                f"Selecting the first one: {refseq_ids[0]}"
            )
            subset = subset[subset["refseq.ID"] == refseq_ids[0]]
        return subset[["Start", "End", "Label"]].to_dict(orient="records"), refseq_ids[
            0
        ]

    def write_maf(self, file_path: str | os.PathLike) -> None:
        """
        Write the MAF to a tab-separated file without the index.

        Parameters
        ----------
        file_path : str or os.PathLike
            Destination file path.
        """
        self.to_csv(file_path, sep="\t", index=False)

    def write_SigProfilerMatrixGenerator_format(
        self, output_path: str | os.PathLike
    ) -> None:
        """
        Convert and write the MAF in SigProfilerMatrixGenerator format.

        Renames columns to match the SigProfilerMatrixGenerator standard and
        filters to rows whose ``Variant_Type`` is SNP, INS, or DEL.

        Parameters
        ----------
        output_path : str or os.PathLike
            Destination TSV file path.
        """
        rename_dict = {
            "Sample": "sample_ID",
            "chrom": "Chromosome",
            "pos_start": "Start_Position",
            "pos_end": "End_Position",
            "ref": "Reference_Allele",
            "alt": "Tumor_Seq_Allele2",
            "mut_type": "Variant_Type",
        }
        maf = self.copy().rename(columns=rename_dict)

        maf = maf[maf["Variant_Type"].isin(["SNP", "INS", "DEL"])]
        maf.to_csv(output_path, sep="\t", index=False)

    def select_samples(self, sample_IDs: list[str]) -> MAF:
        """
        Select rows belonging to specific samples.

        Parameters
        ----------
        sample_IDs : list[str]
            Sample identifiers to keep.

        Returns
        -------
        MAF
            A copy containing only rows for the requested samples.
        """
        return self[self.sample_ID.isin(sample_IDs)].copy()
