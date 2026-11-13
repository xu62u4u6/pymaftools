from __future__ import annotations

import gzip
import os
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from . import maf_io
from .PivotTable import PivotTable
from .SmallVariationTable import SmallVariationTable
from .variant_groups import FUNCTIONAL_GROUP, FUNCTIONAL_ORDER

if TYPE_CHECKING:
    from ..plot.MafPlot import MafPlot
    from .SampleManifest import SampleManifest
    from .TMBAudit import TMBAudit


class MAF(pd.DataFrame):
    """
    A pandas DataFrame subclass for Mutation Annotation Format (MAF) files.

    Provides methods to read, filter, merge, and convert MAF data commonly
    used in cancer genomics pipelines.
    """

    # Preserve builder-attached per-sample provenance as pandas propagates a
    # MAF through filtering and concatenation.  Raw MAFs are not PivotTables,
    # so this remains private until conversion to a gene-level table.
    _metadata = ["_sample_metadata", "_ingest_report"]

    index_col = [
        "Hugo_Symbol",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele1",
        "Tumor_Seq_Allele2",
    ]
    """list[str]: Default columns used to build the row index."""

    # GDC MAF file fields:
    # https://docs.gdc.cancer.gov/Encyclopedia/pages/Mutation_Annotation_Format_TCGAv2/

    valid_variant_classification = [
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
        "Splice_Region",
        "3'UTR",
        "3'Flank",
        "5'UTR",
        "5'Flank",
        "IGR",
        "Intron",
        "RNA",
        "Targeted_Region",
    ]
    """list[str]: All recognised variant classification labels."""

    # Deprecated misspelling retained for callers written against <=0.4.
    vaild_variant_classfication = valid_variant_classification

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
    """list[str]: Variant classifications considered nonsynonymous."""

    @staticmethod
    def _count_leading_comment_lines(
        maf_path: str | os.PathLike,
        comment: str = "#",
    ) -> int:
        """
        Count the consecutive comment lines at the top of a MAF file.

        MAF files may begin with zero, one (e.g. the GDC ``#version 2.4``
        line), or several ``#``-prefixed comment lines before the column
        header.  This helper reports how many such lines to skip so the
        header row is parsed correctly regardless of file flavour.

        Parameters
        ----------
        maf_path : str or os.PathLike
            Path to the MAF file.
        comment : str, default "#"
            Prefix that marks a comment line.

        Returns
        -------
        int
            Number of leading comment lines to skip.
        """
        n = 0
        path = os.fspath(maf_path)
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, mode="rt", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(comment):
                    n += 1
                else:
                    break
        return n

    @staticmethod
    def _detect_delimiter(
        maf_path: str | os.PathLike,
        *,
        skiprows: int = 0,
    ) -> str:
        """Detect a comma- or tab-delimited MAF header.

        Detection is deliberately limited to the two formats supported by
        :meth:`read_maf`.  Inspecting the header instead of the data rows avoids
        treating punctuation inside annotations as a delimiter.
        """
        path = os.fspath(maf_path)
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, mode="rt", encoding="utf-8") as handle:
            for _ in range(skiprows):
                next(handle, None)
            header = next(handle, "").rstrip("\r\n")

        if not header:
            raise ValueError(f"MAF file '{maf_path}' has no column header.")

        tab_count = header.count("\t")
        comma_count = header.count(",")
        if tab_count and not comma_count:
            return "\t"
        if comma_count and not tab_count:
            return ","
        if tab_count > comma_count:
            return "\t"
        if comma_count > tab_count:
            return ","
        raise ValueError(
            f"Could not detect whether MAF file '{maf_path}' is comma- or "
            "tab-delimited; pass sep=',' or sep='\\t' explicitly."
        )

    @classmethod
    def read_maf(
        cls,
        maf_path: str | os.PathLike,
        sample_ID: str | None = None,
        preffix: str = "",
        suffix: str = "",
        sample_col: str = "Tumor_Sample_Barcode",
        sep: str | None = None,
        *,
        validate: Literal["strict", "warn", "off"] = "strict",
        deduplicate: bool = False,
    ) -> MAF:
        """
        Read a MAF file and return a MAF object.

        Inputs from GDC, cBioPortal, vcf2maf, Funcotator/Oncotator and
        maftools example data are accepted.  Everything the reader detects or
        changes is recorded in :attr:`MAF.ingest_report`
        (``print(maf.ingest_report)``), so nothing is altered silently:

        - leading ``#`` comment lines (GDC ``#version gdc-1.0.0``, Funcotator
          ``##`` blocks) are skipped; ``.gz`` is decompressed; comma- or
          tab-delimited headers are detected unless ``sep`` is given;
        - case variants of standard columns are renamed
          (``End_position`` -> ``End_Position``);
        - identifier columns (sample barcodes, ``Chromosome``,
          ``Hugo_Symbol``, alleles) are kept as text, so ``"001"`` stays
          ``"001"`` and a chromosome column never mixes ``int`` and ``str``;
          ``Start_Position``/``End_Position`` are nullable integers;
        - a missing ``Tumor_Seq_Allele1`` is filled from
          ``Reference_Allele`` (it is only used to build the row index).

        Problems that would silently change results are audited: blank
        ``Hugo_Symbol``, duplicate events, mixed genome builds, mixed
        ``chr``/no-``chr`` naming and the legacy allele layout (alternate
        allele in ``Tumor_Seq_Allele1``).  ``validate`` decides what happens
        when one is found.

        Sample identity is resolved as follows:

        - If ``sample_ID`` is given, every row is assigned that single value
          (treats the whole file as one sample).
        - Otherwise the per-row value of ``sample_col``
          (``Tumor_Sample_Barcode`` by default) is used, so a standard
          multi-sample MAF keeps its samples distinct.

        Parameters
        ----------
        maf_path : str or os.PathLike
            Path to the MAF file (optionally ``.gz``).
        sample_ID : str or None, default None
            Sample identifier to assign to all rows.  When ``None`` the
            ``sample_col`` column is used instead.
        preffix : str, default ""
            Prefix prepended to each sample ID.
        suffix : str, default ""
            Suffix appended to each sample ID.
        sample_col : str, default "Tumor_Sample_Barcode"
            Column holding the per-row sample identifier, used when
            ``sample_ID`` is ``None``.
        sep : {``,`` , ``\\t``} or None, default None
            Input delimiter. When ``None``, inspect the first non-comment
            header line and detect comma- or tab-delimited input.
        validate : {"strict", "warn", "off"}, default "strict"
            ``"strict"`` raises :class:`MAFValidationError` listing every
            problem found; ``"warn"`` loads the data unchanged and emits one
            warning; ``"off"`` only records problems in the report.
        deduplicate : bool, default False
            Drop exact duplicate events (same sample, chromosome, gene,
            positions and alleles) instead of reporting them as a problem.

        Returns
        -------
        MAF
            A MAF DataFrame with a composite index built from ``index_col``
            and the :class:`IngestReport` in :attr:`ingest_report`.

        Raises
        ------
        ValueError
            If required columns are missing, or ``sample_ID`` is ``None`` and
            ``sample_col`` is not present.
        MAFValidationError
            Under ``validate="strict"`` when an audited problem is found.
        """
        if validate not in maf_io.VALIDATION_MODES:
            raise ValueError(
                f"validate must be one of {maf_io.VALIDATION_MODES}, got {validate!r}."
            )
        comment_lines = maf_io.read_leading_comments(maf_path)
        skiprows = len(comment_lines)
        detected_sep = cls._detect_delimiter(maf_path, skiprows=skiprows)
        if sep is None:
            sep = detected_sep
        elif sep not in {",", "\t"}:
            raise ValueError("sep must be ',' or '\\t'.")

        header_columns = maf_io.split_header(
            maf_io.read_header_line(maf_path, skiprows), sep
        )
        report = maf_io.IngestReport(
            path=os.fspath(maf_path),
            source=maf_io.detect_source(comment_lines, header_columns),
            delimiter=sep,
            comment_lines=skiprows,
            renamed_columns=maf_io.resolve_column_aliases(header_columns),
        )
        maf = cls(
            maf_io.read_typed_table(
                maf_path, skiprows=skiprows, sep=sep, header_columns=header_columns
            )
        )

        if "Tumor_Seq_Allele1" not in maf.columns and "Reference_Allele" in maf.columns:
            maf["Tumor_Seq_Allele1"] = maf["Reference_Allele"]
            report.filled_columns["Tumor_Seq_Allele1"] = (
                "copied from Reference_Allele (index only)"
            )

        missing_index_columns = [
            column for column in cls.index_col if column not in maf.columns
        ]
        if missing_index_columns:
            if sep != detected_sep:
                expected = "comma" if detected_sep == "," else "tab"
                received = "comma" if sep == "," else "tab"
                raise ValueError(
                    f"Delimiter mismatch for MAF file '{maf_path}': the header "
                    f"appears {expected}-delimited, but sep selected {received}-"
                    "delimited parsing. Remove sep to auto-detect it or pass the "
                    "matching delimiter explicitly."
                )
            raise ValueError(
                f"MAF file '{maf_path}' is missing required column(s): "
                f"{missing_index_columns}. Required: {cls.index_col} "
                "(Tumor_Seq_Allele1 may be absent if Reference_Allele is present)."
            )
        if sample_ID is not None:
            maf["sample_ID"] = f"{preffix}{sample_ID}{suffix}"
        elif sample_col in maf.columns:
            barcodes = maf[sample_col].astype(str)
            placeholders = sorted(
                set(barcodes.unique()) & maf_io.PLACEHOLDER_SAMPLE_IDS
            )
            report.placeholder_sample_ids = placeholders
            maf["sample_ID"] = preffix + barcodes + suffix
        else:
            raise ValueError(
                f"sample_ID not provided and column '{sample_col}' not found "
                f"in {maf_path}; pass sample_ID explicitly or set sample_col."
            )

        blank_genes = maf["Hugo_Symbol"].isin(maf_io.BLANK_GENE_VALUES)
        maf.loc[blank_genes, "Hugo_Symbol"] = pd.NA

        event_keys = ["sample_ID", "Chromosome", *cls.index_col]
        if deduplicate:
            present_keys = [key for key in event_keys if key in maf.columns]
            duplicated = maf.duplicated(subset=present_keys)
            report.dropped_duplicate_rows = int(duplicated.sum())
            maf = maf.loc[~duplicated]

        maf_io.audit_events(
            maf,
            report,
            key_columns=event_keys,
            known_classifications=cls.valid_variant_classification,
        )
        maf_io.summarize_issues(report, deduplicate=deduplicate)
        report.n_rows = len(maf)
        report.n_samples = int(maf["sample_ID"].nunique())

        if report.issues and validate == "strict":
            raise maf_io.MAFValidationError(
                f"MAF file '{maf_path}' has input problem(s) that would change "
                "results:\n  - "
                + "\n  - ".join(report.issues)
                + "\nFix the file, or pass validate='warn' to load it anyway "
                "(see the error's .report for the full ingest report).",
                report,
            )
        if report.issues and validate == "warn":
            warnings.warn(
                f"MAF file '{maf_path}' has input problem(s): "
                + " | ".join(report.issues),
                UserWarning,
                stacklevel=2,
            )

        maf.index = maf.loc[:, cls.index_col].apply(
            lambda row: "|".join(row.astype(str)), axis=1
        )  # concat column
        maf = cls(maf)
        maf._ingest_report = report
        return maf

    @property
    def ingest_report(self) -> maf_io.IngestReport | None:
        """What :meth:`read_maf` detected, changed and found (``None`` for a
        MAF built in memory)."""
        return getattr(self, "_ingest_report", None)

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

    @property
    def plot(self) -> "MafPlot":
        """
        Access MAF-level plotting functionality.

        Returns
        -------
        MafPlot
            Plotting interface for raw MAF summaries, Ti/Tv, rainfall, VAF,
            and cohort comparison plots.
        """
        from ..plot.MafPlot import MafPlot

        return MafPlot(self)

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

    def calculate_tmb_audit(
        self,
        sample_manifest: "SampleManifest",
        *,
        callable_mb_col: str = "callable_mb",
        pass_col: str | None = None,
        pass_values: tuple[object, ...] = ("PASS",),
        somatic_col: str | None = None,
        somatic_values: tuple[object, ...] = ("Somatic", "SOMATIC", True),
        variant_classifications: list[str] | None = None,
        genome_build_col: str | None = None,
        expected_genome_build: str | None = None,
        deduplicate: bool = True,
    ) -> "TMBAudit":
        """Calculate TMB with an event-level inclusion and exclusion ledger.

        Unlike :meth:`PivotTable.calculate_tmb`, this method requires an
        explicit sample universe and per-sample callable territory. Optional
        event filters are applied before the numerator is counted, and every
        excluded row retains a machine-readable reason.

        Parameters
        ----------
        sample_manifest : SampleManifest
            Complete sample universe. Eligible samples with zero events remain
            in the output summary.
        callable_mb_col : str, default "callable_mb"
            Positive callable territory in megabases stored in the manifest.
        pass_col, somatic_col : str, optional
            Event columns used for quality and somatic-status filtering. A
            named column is required to exist; no undocumented guessing occurs.
        variant_classifications : list of str, optional
            Allowed ``Variant_Classification`` values. For nonsynonymous TMB,
            pass :attr:`MAF.nonsynonymous_types` explicitly.
        genome_build_col, expected_genome_build : str, optional
            Both must be provided together to enforce a single build.
        deduplicate : bool, default True
            After all event filters, exclude repeated
            sample/coordinate/allele events from the numerator.

        Returns
        -------
        TMBAudit
            Event ledger and per-sample TMB summary.
        """
        from .SampleManifest import SampleManifest
        from .TMBAudit import TMBAudit

        if not isinstance(sample_manifest, SampleManifest):
            raise TypeError("sample_manifest must be a SampleManifest.")
        if "sample_ID" not in self.columns:
            raise ValueError("MAF must contain sample_ID for auditable TMB.")

        sample_manifest.validate_event_samples(self["sample_ID"])
        manifest = sample_manifest.eligible_frame()
        if callable_mb_col not in manifest.columns:
            raise ValueError(
                f"SampleManifest must contain '{callable_mb_col}' for TMB."
            )
        callable_mb = pd.to_numeric(manifest[callable_mb_col], errors="coerce")
        invalid_territory = callable_mb.isna() | (callable_mb <= 0)
        if invalid_territory.any():
            raise ValueError(
                "Eligible samples must have positive callable territory; invalid "
                f"sample(s): {callable_mb.index[invalid_territory].tolist()}."
            )

        if (genome_build_col is None) != (expected_genome_build is None):
            raise ValueError(
                "genome_build_col and expected_genome_build must be provided together."
            )

        events = pd.DataFrame(self).copy().reset_index(drop=True)
        events["input_event_id"] = events.index
        events["included"] = True
        events["exclusion_reason"] = pd.Series(
            pd.NA, index=events.index, dtype="string"
        )

        def exclude(mask: pd.Series, reason: str) -> None:
            newly_excluded = events["included"] & mask.fillna(True)
            events.loc[newly_excluded, "included"] = False
            events.loc[newly_excluded, "exclusion_reason"] = reason

        for column in [pass_col, somatic_col, genome_build_col]:
            if column is not None and column not in events.columns:
                raise ValueError(
                    f"MAF is missing requested TMB filter column '{column}'."
                )

        if pass_col is not None:
            exclude(~events[pass_col].isin(pass_values), "quality_filter")
        if somatic_col is not None:
            exclude(~events[somatic_col].isin(somatic_values), "non_somatic")
        if variant_classifications is not None:
            if "Variant_Classification" not in events.columns:
                raise ValueError(
                    "MAF is missing requested TMB filter column "
                    "'Variant_Classification'."
                )
            exclude(
                ~events["Variant_Classification"].isin(variant_classifications),
                "variant_classification",
            )
        if genome_build_col is not None:
            exclude(
                events[genome_build_col].astype(str) != str(expected_genome_build),
                "genome_build",
            )

        if deduplicate:
            dedup_columns = ["sample_ID", *self.index_col]
            missing_dedup = [c for c in dedup_columns if c not in events.columns]
            if missing_dedup:
                raise ValueError(
                    f"MAF is missing TMB deduplication column(s): {missing_dedup}."
                )
            duplicate = pd.Series(False, index=events.index)
            included_index = events.index[events["included"]]
            duplicate.loc[included_index] = events.loc[included_index].duplicated(
                subset=dedup_columns, keep="first"
            )
            exclude(duplicate, "duplicate_event")

        counts = (
            events.loc[events["included"], "sample_ID"]
            .value_counts()
            .reindex(manifest.index, fill_value=0)
            .astype(int)
        )
        summary = manifest.copy()
        summary["mutation_count"] = counts
        summary["callable_mb"] = callable_mb
        summary["TMB"] = summary["mutation_count"] / summary["callable_mb"]
        summary["tmb_pass_col"] = pass_col
        summary["tmb_somatic_col"] = somatic_col
        summary["tmb_variant_policy"] = (
            "all"
            if variant_classifications is None
            else ",".join(sorted(variant_classifications))
        )
        summary["tmb_genome_build"] = expected_genome_build
        summary["tmb_deduplicated"] = deduplicate
        return TMBAudit(events=events, summary=summary)

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
            return non_false_mutations.iloc[0]

    def to_gene_table(
        self,
        sample_manifest: "SampleManifest | None" = None,
        *,
        samples: Iterable[object] | None = None,
    ) -> "SmallVariationTable":
        """
        Create a gene-level (gene × sample) pivot table of variant classifications.

        This is the gene-level counterpart to :meth:`to_mutation_table` (one row
        per mutation); the name makes the granularity explicit. ``to_pivot_table``
        is a backward-compatible alias.

        The gene-by-sample matrix is built directly from the event table. This
        avoids materializing the much larger mutation-by-sample dense matrix;
        the direct path is essential for cohort-scale MAFs. Multiple events in
        one gene and sample are represented as ``"Multi_Hit"``.

        **Denominators.** An event table cannot list samples that were
        sequenced but have no called events, so mutation frequencies computed
        from it are only correct when the full sample universe is supplied,
        either as ``samples`` (a list of IDs) or as a ``SampleManifest``.
        Samples that have events but none placeable in a gene row (blank
        ``Hugo_Symbol``) are kept as all-``False`` columns with a warning,
        rather than silently disappearing.

        Parameters
        ----------
        sample_manifest : SampleManifest, optional
            Full, auditable sample universe (eligible samples become columns).
        samples : iterable of sample IDs, optional
            Full sample universe as plain IDs, including samples with zero
            events. Mutually exclusive with ``sample_manifest``.

        Returns
        -------
        SmallVariationTable
            Gene × sample matrix with gene-level feature_metadata.

        Raises
        ------
        ValueError
            If events belong to samples outside ``samples`` /
            ``sample_manifest``, or both are given.
        """
        from .SmallVariationTable import SmallVariationTable

        if sample_manifest is not None and samples is not None:
            raise ValueError("Pass either sample_manifest or samples, not both.")
        sample_universe = None
        if samples is not None:
            sample_universe = pd.Index(pd.unique(pd.Index(list(samples))))
            unknown = pd.Index(self["sample_ID"].unique()).difference(sample_universe)
            if not unknown.empty:
                raise ValueError(
                    f"{len(unknown)} sample(s) with mutation events are missing "
                    f"from samples: {unknown[:5].tolist()}"
                    f"{' ...' if len(unknown) > 5 else ''}. Check that the IDs "
                    "use the same barcode level (aliquot / sample / patient)."
                )

        if sample_manifest is not None:
            from .SampleManifest import SampleManifest

            if not isinstance(sample_manifest, SampleManifest):
                raise TypeError("sample_manifest must be a SampleManifest.")
            sample_manifest.validate_event_samples(self["sample_ID"])

        events = self
        if "Hugo_Symbol" not in events.columns:
            if events.index.name != "Hugo_Symbol":
                raise ValueError(
                    "MAF must contain 'Hugo_Symbol' as a column or named index."
                )
            gene_symbols = events.index.to_numpy()
            events = self.copy().reset_index(drop=True)
            events["Hugo_Symbol"] = gene_symbols

        blank_gene = events["Hugo_Symbol"].isna()
        if blank_gene.any():
            warnings.warn(
                f"{int(blank_gene.sum())} event(s) with a blank Hugo_Symbol "
                "cannot be placed in a gene row and are left out of the gene "
                "table (their samples are kept).",
                UserWarning,
                stacklevel=2,
            )
        grouped = events.groupby(["Hugo_Symbol", "sample_ID"], sort=True)[
            "Variant_Classification"
        ].agg(event_count="size", first_classification="first")
        grouped["classification"] = grouped["first_classification"].where(
            grouped["event_count"].eq(1), "Multi_Hit"
        )
        gene_matrix = grouped["classification"].unstack("sample_ID", fill_value=False)
        gene_matrix = gene_matrix.fillna(False)
        result = SmallVariationTable(gene_matrix)

        if sample_universe is not None:
            result = result.reindex(
                columns=sample_universe,
                fill_value=False,
                sample_fill_value=pd.NA,
            )
        elif sample_manifest is None:
            event_samples = pd.Index(self["sample_ID"].dropna().unique())
            lost = event_samples.difference(result.columns)
            if not lost.empty:
                warnings.warn(
                    f"{len(lost)} sample(s) have events but none in a gene "
                    f"row (e.g. {lost[:3].tolist()}); kept as all-False "
                    "columns so frequency denominators stay correct.",
                    UserWarning,
                    stacklevel=2,
                )
                result = result.reindex(
                    columns=result.columns.append(lost),
                    fill_value=False,
                    sample_fill_value=pd.NA,
                )

        if sample_manifest is not None:
            result = result.reindex(
                columns=sample_manifest.eligible_samples,
                fill_value=False,
                sample_fill_value=pd.NA,
            )
            result.sample_metadata = sample_manifest.eligible_frame()

        result.sample_metadata["mutations_count"] = self.mutations_count.reindex(
            result.columns, fill_value=0
        )
        for column, values in self.tmb_by_functional_group.items():
            result.sample_metadata[column] = values.reindex(
                result.columns, fill_value=0
            )

        feature_metadata = events.copy()
        feature_metadata.index = feature_metadata["Hugo_Symbol"]
        present_gene_columns = [
            column
            for column in SmallVariationTable._GENE_LEVEL_COLS
            if column in feature_metadata.columns
        ]
        gene_metadata = feature_metadata[present_gene_columns].groupby(level=0).first()
        for column, aggregate in SmallVariationTable._GENE_LEVEL_AGG.items():
            if column in feature_metadata.columns:
                gene_metadata[column] = (
                    feature_metadata[column].groupby(level=0).agg(aggregate)
                )
        result.feature_metadata = gene_metadata.reindex(result.index)

        if sample_manifest is not None:
            result = result.with_scientific_context(sample_manifest=sample_manifest)
        result._validate_metadata()
        return result

    def to_pivot_table(
        self,
        sample_manifest: "SampleManifest | None" = None,
        *,
        samples: Iterable[object] | None = None,
    ) -> "SmallVariationTable":
        """Alias for :meth:`to_gene_table` (gene-level pivot table).

        Kept for backward compatibility; ``to_gene_table`` is preferred in new
        code as its name states the granularity.
        """
        return self.to_gene_table(sample_manifest=sample_manifest, samples=samples)

    # Columns to carry into mutation-level feature_metadata
    _FEATURE_META_COLS = [
        # position / structure
        "Hugo_Symbol",
        "Chromosome",
        "Start_Position",
        "End_Position",
        "Strand",
        "EXON",
        "INTRON",
        "cDNA_position",
        "CDS_position",
        "Protein_position",
        # variant identity
        "Variant_Type",
        "Variant_Classification",
        "VARIANT_CLASS",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Consequence",
        "One_Consequence",
        "CONTEXT",
        # functional impact
        "IMPACT",
        "HGVSc",
        "HGVSp_Short",
        "Amino_acids",
        "Codons",
        "SIFT",
        "PolyPhen",
        "DOMAINS",
        # gene-level annotation
        "Entrez_Gene_Id",
        "Gene",
        "BIOTYPE",
        "TRANSCRIPT_STRAND",
        "HGNC_ID",
        "RefSeq",
        "MANE",
        "APPRIS",
        # clinical / databases
        "hotspot",
        "COSMIC",
        "Existing_variation",
        "CLIN_SIG",
        "GENE_PHENO",
        "dbSNP_RS",
        "callers",
    ]

    def to_mutation_table(
        self,
        sample_manifest: "SampleManifest | None" = None,
    ) -> "SmallVariationTable":
        """
        Create a mutation-level pivot table.

        Each row corresponds to a unique mutation (composite index) rather
        than a gene, providing finer resolution than :meth:`to_pivot_table`.
        feature_metadata is populated with per-mutation annotation columns.

        Returns
        -------
        SmallVariationTable
            Pivot table indexed by individual mutations.
        """
        if sample_manifest is not None:
            from .SampleManifest import SampleManifest

            if not isinstance(sample_manifest, SampleManifest):
                raise TypeError("sample_manifest must be a SampleManifest.")
            sample_manifest.validate_event_samples(self["sample_ID"])

        mutation_table = self.pivot_table(
            index=self.index,
            columns="sample_ID",
            values="Variant_Classification",
            aggfunc="first",
        ).fillna(False)
        mutation_table = SmallVariationTable(mutation_table)
        if sample_manifest is not None:
            mutation_table = mutation_table.reindex(
                columns=sample_manifest.eligible_samples,
                fill_value=False,
                sample_fill_value=pd.NA,
            )
            mutation_table.sample_metadata = sample_manifest.eligible_frame()

        mutation_table.sample_metadata["mutations_count"] = (
            self.mutations_count.reindex(mutation_table.columns, fill_value=0)
        )
        # Cache the per-functional-group TMB breakdown (the stacked-TMB source a
        # single pivot cell can't reconstruct). Aligns by sample label.
        for col, values in self.tmb_by_functional_group.items():
            mutation_table.sample_metadata[col] = values.reindex(
                mutation_table.columns, fill_value=0
            )

        # Build feature_metadata from MAF rows (one row per unique mutation index)
        metadata = pd.DataFrame(self).copy()
        if (
            self.index.name in self._FEATURE_META_COLS
            and self.index.name not in metadata.columns
        ):
            metadata[self.index.name] = self.index.to_numpy()
        present_cols = [c for c in self._FEATURE_META_COLS if c in metadata.columns]
        deduped = metadata.loc[~metadata.index.duplicated(keep="first"), present_cols]
        mutation_table.feature_metadata = deduped.reindex(mutation_table.index)

        if sample_manifest is not None:
            mutation_table = mutation_table.with_scientific_context(
                sample_manifest=sample_manifest
            )

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

    # Prefix for the per-functional-group TMB columns stored in sample_metadata.
    TMB_PREFIX = "tmb_"

    @property
    def tmb_by_functional_group(self) -> pd.DataFrame:
        """Per-sample mutation counts split by functional group.

        Buckets ``Variant_Classification`` into the six coarse functional groups
        (``core.variant_groups.FUNCTIONAL_GROUP``) and counts mutations per
        sample per group. This is the stacked-TMB source that a single ``pivot``
        cell cannot reconstruct (``to_mutation_table`` keeps only the *first*
        class per gene x sample), so it is computed here while the raw events
        are still in hand and cached into ``sample_metadata``.

        Note: counts reflect *this MAF's* events. If the MAF was filtered (e.g.
        nonsynonymous-only) before conversion, the breakdown is of that subset.

        Returns
        -------
        pd.DataFrame
            Indexed by sample ID; columns ``tmb_<group>`` in FUNCTIONAL_ORDER
            (``tmb_Truncating``, ``tmb_Splice``, ...), integer counts.
        """
        groups = self.Variant_Classification.map(FUNCTIONAL_GROUP).fillna("Other")
        counts = pd.crosstab(self.sample_ID, groups).reindex(
            columns=FUNCTIONAL_ORDER, fill_value=0
        )
        counts.columns = [f"{self.TMB_PREFIX}{g}" for g in counts.columns]
        return counts

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
    def merge_mafs(mafs: list[MAF], *, allow_mixed_builds: bool = False) -> MAF:
        """
        Concatenate multiple MAF objects into one.

        Merging files from different sources (e.g. GDC GRCh38 with cBioPortal
        GRCh37) silently puts two coordinate systems in one table, so mixed
        ``NCBI_Build`` values or mixed ``chr1``/``1`` naming raise unless
        explicitly allowed.

        Parameters
        ----------
        mafs : list[MAF]
            MAF objects to concatenate.
        allow_mixed_builds : bool, default False
            Skip the genome-build and chromosome-naming consistency check.

        Returns
        -------
        MAF
            A single MAF containing all rows from the input MAFs.

        Raises
        ------
        ValueError
            If the inputs mix genome builds or chromosome naming styles and
            ``allow_mixed_builds`` is False.
        """
        merged = MAF(pd.concat(mafs))
        if not allow_mixed_builds:
            problems = []
            if "NCBI_Build" in merged.columns:
                builds = sorted(merged["NCBI_Build"].dropna().astype(str).unique())
                if len(builds) > 1:
                    problems.append(f"genome builds {builds}")
            if "Chromosome" in merged.columns:
                styles = sorted(
                    {
                        maf_io.chromosome_style(value)
                        for value in merged["Chromosome"].dropna()
                    }
                )
                if len(styles) > 1:
                    problems.append("chromosome naming 'chr1' and '1'")
            if problems:
                raise ValueError(
                    "merge_mafs inputs mix " + " and ".join(problems) + ". "
                    "Lift over / harmonize the inputs first, or pass "
                    "allow_mixed_builds=True if this is intended."
                )
        return merged

    @classmethod
    def read_csv(
        cls,
        csv_path: str | os.PathLike,
        sep: str = "\t",
        reindex: bool = False,
        sample_col: str | None = None,
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
            When that column is ``Hugo_Symbol``, it is preserved as an event
            column instead because gene symbols are not unique mutation IDs.
        sample_col : str or None, default None
            Source column to copy into the canonical ``sample_ID`` column.
            Leave as ``None`` when the input already contains ``sample_ID``.

        Returns
        -------
        MAF
            MAF constructed from the file contents.

        Raises
        ------
        ValueError
            If ``sample_col`` is provided but is absent from the input.
        """
        if reindex:
            maf = cls(pd.read_csv(csv_path, sep=sep))
            maf = maf.change_index_level()
        else:
            maf = cls(pd.read_csv(csv_path, sep=sep, index_col=0))
            if maf.index.name == "Hugo_Symbol":
                maf = cls(maf.reset_index())

        if sample_col is not None:
            if sample_col not in maf.columns:
                raise ValueError(
                    f"MAF is missing requested sample column '{sample_col}'."
                )
            maf["sample_ID"] = maf[sample_col]
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

    def to_maf(self, maf_path: str | os.PathLike, **kwargs: Any) -> None:
        """
        Write the data as a standard MAF file (tab-separated, no index column).

        This is the canonical MAF writer; ``to_MAF`` and ``write_maf`` are
        deprecated aliases kept for backward compatibility.

        Parameters
        ----------
        maf_path : str or os.PathLike
            Destination file path.
        **kwargs : Any
            Additional keyword arguments forwarded to
            ``pd.DataFrame.to_csv``.
        """
        # Set default arguments
        kwargs.setdefault("index", False)  # MAF files have no index column
        kwargs.setdefault("sep", "\t")  # Default to tab-separated values

        # Call the parent class's to_csv method
        super().to_csv(maf_path, **kwargs)

    def to_MAF(self, maf_path: str | os.PathLike, **kwargs: Any) -> None:
        """Deprecated alias for :meth:`to_maf`."""
        warnings.warn(
            "MAF.to_MAF is deprecated; use to_maf() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.to_maf(maf_path, **kwargs)

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

        maf = self.filter_maf(self.nonsynonymous_types)
        gene_rows = maf.loc[maf["Hugo_Symbol"] == gene]

        # Position source, best first: VEP/GDC/cBioPortal Protein_position,
        # then HGVSp_Short (vcf2maf/cBioPortal), then Protein_Change
        # (Funcotator/Oncotator/legacy TCGA).
        if "Protein_position" in gene_rows.columns:
            positions = gene_rows["Protein_position"].map(maf_io.protein_start)
            if positions.isna().all():
                positions = None
        else:
            positions = None
        if positions is None:
            for column in ("HGVSp_Short", "Protein_Change"):
                if column in gene_rows.columns:
                    positions = gene_rows[column].map(maf_io.hgvsp_start)
                    break
        if positions is None:
            raise ValueError(
                "Lollipop needs amino-acid positions: the MAF has none of "
                "Protein_position, HGVSp_Short or Protein_Change."
            )

        sub_df = pd.DataFrame(
            {
                "AA_Position": pd.to_numeric(positions, errors="coerce"),
                "Variant_Classification": gene_rows["Variant_Classification"],
            }
        )

        # Total AA length from "858/1210" -> 1210. cBioPortal stores bare
        # positions, so the length may be unknown (None).
        AA_length = None
        if "Protein_position" in gene_rows.columns:
            lengths = gene_rows["Protein_position"].map(maf_io.protein_length)
            lengths = lengths.dropna()
            if not lengths.empty:
                AA_length = int(lengths.iloc[0])

        # count mutations and to dict
        mutations_data = (
            sub_df.dropna(subset=["AA_Position", "Variant_Classification"])
            .astype({"AA_Position": int})
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
    def infer_protein_length(
        gene: str,
        max_position: int,
        protein_domains_path: str | os.PathLike | None = None,
    ) -> int | None:
        """Pick a protein length for ``gene`` from the domain table.

        Used when the MAF stores bare amino-acid positions (cBioPortal) and so
        carries no ``pos/length``. The first listed transcript long enough to
        contain ``max_position`` is chosen, with a warning naming it.
        Returns ``None`` when no transcript fits.
        """
        if protein_domains_path is None:
            protein_domains_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../data/protein_domains.csv",
            )
        domains = pd.read_csv(protein_domains_path, index_col=0, low_memory=False)
        lengths = (
            domains.loc[domains["HGNC"] == gene, ["refseq.ID", "aa.length"]]
            .drop_duplicates()
            .loc[lambda frame: frame["aa.length"] >= max_position]
        )
        if lengths.empty:
            return None
        chosen = lengths.iloc[0]
        warnings.warn(
            f"The MAF has no protein length for {gene}; using "
            f"{int(chosen['aa.length'])} aa from {chosen['refseq.ID']} "
            "(domain table). Positions may refer to another transcript.",
            UserWarning,
            stacklevel=2,
        )
        return int(chosen["aa.length"])

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
        """Deprecated alias for :meth:`to_maf`."""
        warnings.warn(
            "MAF.write_maf is deprecated; use to_maf() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.to_maf(file_path)

    def write_SigProfilerMatrixGenerator_format(
        self, output_path: str | os.PathLike
    ) -> None:
        """Deprecated alias for :meth:`to_sigprofiler`."""
        warnings.warn(
            "write_SigProfilerMatrixGenerator_format is deprecated; "
            "use to_sigprofiler() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.to_sigprofiler(output_path)

    def to_sigprofiler(self, output_path: str | os.PathLike) -> None:
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
