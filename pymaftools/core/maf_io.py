"""MAF ingestion: header parsing, column aliasing, typed reading and input audits.

MAF files reach pymaftools from many producers (GDC, cBioPortal, vcf2maf,
Funcotator/Oncotator, maftools example data, hand-edited spreadsheets).  They
share the column vocabulary but differ in header comments, column-name casing,
missing optional columns, identifier formatting and genome build.  This module
turns that variation into one typed event table **and** an
:class:`IngestReport` that records every change made and every problem found,
so that nothing is altered silently.

Scope (see CLAUDE.md §6): column/type harmonization and input audits only.
Variant normalization (left-alignment, multi-allelic splitting, liftover) is
delegated to ``bcftools`` / ``vcf2maf`` / ``GATK``.
"""

from __future__ import annotations

import csv
import gzip
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import IO

import pandas as pd

# Columns that identify things and must never be type-inferred: pandas would
# turn sample "001" into 1, a chromosome column into a mix of int and str, and
# the gene symbol "NA" into NaN.
TEXT_COLUMNS = (
    "Hugo_Symbol",
    "Entrez_Gene_Id",
    "Chromosome",
    "Strand",
    "Variant_Classification",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele1",
    "Tumor_Seq_Allele2",
    "Match_Norm_Seq_Allele1",
    "Match_Norm_Seq_Allele2",
    "Tumor_Sample_Barcode",
    "Matched_Norm_Sample_Barcode",
    "NCBI_Build",
    "HGVSc",
    "HGVSp",
    "HGVSp_Short",
    "Protein_Change",
    "Protein_position",
    "Transcript_ID",
    "FILTER",
)

POSITION_COLUMNS = ("Start_Position", "End_Position")

# Columns matched case-insensitively when the exact spelling is absent
# (e.g. maftools' ``tcga_laml.maf.gz`` spells ``End_position``).
CANONICAL_COLUMNS = (
    *TEXT_COLUMNS,
    *POSITION_COLUMNS,
    "t_depth",
    "t_ref_count",
    "t_alt_count",
    "n_depth",
    "n_ref_count",
    "n_alt_count",
)

BLANK_GENE_VALUES = frozenset(
    {"", "NA", "N/A", "NaN", "nan", "None", "none", "NULL", "null", "."}
)
"""Hugo_Symbol spellings that mean "no gene" and are treated as missing."""

UNASSIGNED_GENE_VALUES = frozenset({"Unknown"})
"""Placeholder gene names written by vcf2maf/Funcotator for intergenic events."""

PLACEHOLDER_SAMPLE_IDS = frozenset({"TUMOR", "Unknown", "NORMAL"})
"""Tumor_Sample_Barcode values written when a caller was not given sample IDs."""

# pandas' default NA spellings, used for every non-identifier column.
DEFAULT_NA_VALUES = (
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
)

VALIDATION_MODES = ("strict", "warn", "off")


class MAFValidationError(ValueError):
    """Raised by ``read_maf(validate="strict")`` when an input problem would
    silently change downstream results.  ``.report`` holds the full
    :class:`IngestReport`."""

    def __init__(self, message: str, report: "IngestReport") -> None:
        super().__init__(message)
        self.report = report


@dataclass
class IngestReport:
    """What :meth:`MAF.read_maf` detected, changed and found in one file.

    Print it (``print(maf.ingest_report)``) for a human summary.
    """

    path: str
    source: str = "maf"
    """Best-effort producer guess: gdc, funcotator, vcf2maf, legacy-tcga,
    minimal or maf.  Informational only; reading never depends on it."""
    delimiter: str = "\t"
    comment_lines: int = 0
    n_rows: int = 0
    n_samples: int = 0
    renamed_columns: dict[str, str] = field(default_factory=dict)
    filled_columns: dict[str, str] = field(default_factory=dict)
    blank_gene_rows: int = 0
    unassigned_gene_rows: int = 0
    duplicate_rows: int = 0
    dropped_duplicate_rows: int = 0
    unknown_classifications: dict[str, int] = field(default_factory=dict)
    genome_builds: list[str] = field(default_factory=list)
    chromosome_styles: list[str] = field(default_factory=list)
    legacy_allele_rows: int = 0
    placeholder_sample_ids: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    """Problems that change results if ignored (raised under ``strict``)."""
    notes: list[str] = field(default_factory=list)
    """Changes made and observations that do not block reading."""

    def __str__(self) -> str:
        sep_name = "tab" if self.delimiter == "\t" else "comma"
        lines = [
            f"MAF ingest report: {self.path}",
            f"  source guess : {self.source} ({sep_name}-delimited, "
            f"{self.comment_lines} comment line(s) skipped)",
            f"  events       : {self.n_rows} rows, {self.n_samples} samples",
        ]
        if self.genome_builds:
            lines.append(f"  genome build : {', '.join(self.genome_builds)}")
        if self.issues:
            lines.append("  ISSUES:")
            lines.extend(f"    - {issue}" for issue in self.issues)
        if self.notes:
            lines.append("  notes:")
            lines.extend(f"    - {note}" for note in self.notes)
        if not self.issues and not self.notes:
            lines.append("  no changes made, no problems found")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# header handling
# --------------------------------------------------------------------------


def _open_text(path: str) -> IO[str]:
    opener = gzip.open if path.endswith(".gz") else open
    return opener(path, mode="rt", encoding="utf-8-sig", newline="")


def read_leading_comments(maf_path: str | os.PathLike, comment: str = "#") -> list[str]:
    """Return the consecutive ``#`` comment lines at the top of a MAF file."""
    comments = []
    with _open_text(os.fspath(maf_path)) as handle:
        for line in handle:
            if line.startswith(comment):
                comments.append(line.rstrip("\r\n"))
            else:
                break
    return comments


def read_header_line(maf_path: str | os.PathLike, skiprows: int) -> str:
    """Return the column-header line that follows ``skiprows`` comments."""
    with _open_text(os.fspath(maf_path)) as handle:
        for _ in range(skiprows):
            next(handle, None)
        return next(handle, "").rstrip("\r\n")


def split_header(header: str, sep: str) -> list[str]:
    """Split a header line honoring CSV quoting."""
    return [name.strip() for name in next(csv.reader([header], delimiter=sep))]


def resolve_column_aliases(columns: Sequence[str]) -> dict[str, str]:
    """Map case-variant spellings of canonical MAF columns to the canonical name.

    Only applied when the canonical spelling itself is absent, so an input
    that already has ``End_Position`` is never touched.
    """
    present = set(columns)
    lower_to_canonical = {name.lower(): name for name in CANONICAL_COLUMNS}
    renames: dict[str, str] = {}
    for column in columns:
        canonical = lower_to_canonical.get(column.lower())
        if (
            canonical is not None
            and canonical != column
            and canonical not in present
            and canonical not in renames.values()
        ):
            renames[column] = canonical
    return renames


def detect_source(comment_lines: Sequence[str], columns: Iterable[str]) -> str:
    """Guess which tool produced a MAF from its header comments and columns."""
    text = "\n".join(comment_lines).lower()
    cols = set(columns)
    if "funcotator" in text:
        return "funcotator"
    if "gdc" in text or "annotation.spec" in text:
        return "gdc"
    if {"vcf_pos", "vcf_id", "all_effects"} & cols:
        return "vcf2maf"
    if "Protein_Change" in cols and "HGVSp_Short" not in cols:
        return "legacy-tcga"
    if len(cols) <= 12:
        return "minimal"
    return "maf"


def read_typed_table(
    maf_path: str | os.PathLike,
    *,
    skiprows: int,
    sep: str,
    header_columns: Sequence[str],
) -> pd.DataFrame:
    """Read a MAF body with identifier columns kept as text.

    Identifier columns are read as ``str`` with only the empty string treated
    as missing; every other column keeps pandas' default inference and NA
    spellings.  ``Start_Position``/``End_Position`` become nullable ``Int64``
    so one missing coordinate cannot turn every position into a float.
    """
    renames = resolve_column_aliases(header_columns)
    canonical_names = [renames.get(name, name) for name in header_columns]
    text_set = set(TEXT_COLUMNS)
    dtype: dict[str, type] = {}
    na_values: dict[str, list[str]] = {}
    for original, canonical in zip(header_columns, canonical_names):
        if canonical in text_set:
            dtype[original] = str
            na_values[original] = [""]
        else:
            na_values[original] = list(DEFAULT_NA_VALUES)

    frame = pd.read_csv(
        maf_path,
        skiprows=skiprows,
        sep=sep,
        dtype=dtype,
        keep_default_na=False,
        na_values=na_values,
        low_memory=False,
        encoding="utf-8-sig",
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns=renames)
    for column in POSITION_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(
                "Int64"
            )
    return frame


# --------------------------------------------------------------------------
# audits
# --------------------------------------------------------------------------


def chromosome_style(value: object) -> str:
    return "chr-prefixed" if str(value).lower().startswith("chr") else "unprefixed"


def audit_events(
    events: pd.DataFrame,
    report: IngestReport,
    *,
    key_columns: Sequence[str],
    known_classifications: Iterable[str],
) -> None:
    """Fill the audit fields of ``report`` from an event table (in place)."""
    if "Hugo_Symbol" in events.columns:
        genes = events["Hugo_Symbol"]
        report.blank_gene_rows = int(genes.isna().sum())
        report.unassigned_gene_rows = int(genes.isin(UNASSIGNED_GENE_VALUES).sum())

    present_keys = [column for column in key_columns if column in events.columns]
    if present_keys:
        report.duplicate_rows = int(events.duplicated(subset=present_keys).sum())

    if "Variant_Classification" in events.columns:
        known = set(known_classifications)
        counts = events["Variant_Classification"].dropna().value_counts()
        report.unknown_classifications = {
            str(label): int(count)
            for label, count in counts.items()
            if label not in known
        }

    if "NCBI_Build" in events.columns:
        report.genome_builds = sorted(
            events["NCBI_Build"].dropna().astype(str).unique().tolist()
        )

    if "Chromosome" in events.columns:
        report.chromosome_styles = sorted(
            {chromosome_style(value) for value in events["Chromosome"].dropna()}
        )

    allele_columns = {"Variant_Type", "Reference_Allele"}
    allele_columns |= {"Tumor_Seq_Allele1", "Tumor_Seq_Allele2"}
    if allele_columns.issubset(events.columns):
        snp = events["Variant_Type"].eq("SNP")
        ref = events["Reference_Allele"].str.upper()
        allele1 = events["Tumor_Seq_Allele1"].str.upper()
        allele2 = events["Tumor_Seq_Allele2"].str.upper()
        legacy = snp & allele2.eq(ref) & allele1.ne(ref) & allele1.notna()
        report.legacy_allele_rows = int(legacy.sum())


def summarize_issues(report: IngestReport, *, deduplicate: bool) -> None:
    """Translate audit counts into human-readable issues and notes."""
    for original, canonical in report.renamed_columns.items():
        report.notes.append(f"renamed column '{original}' -> '{canonical}'")
    for column, reason in report.filled_columns.items():
        report.notes.append(f"filled missing column '{column}': {reason}")

    if report.blank_gene_rows:
        report.issues.append(
            f"{report.blank_gene_rows} event(s) have a blank Hugo_Symbol; "
            "gene-level tables cannot place them. Fill them (e.g. 'Unknown') "
            "or drop them before analysis."
        )
    if report.unassigned_gene_rows:
        report.notes.append(
            f"{report.unassigned_gene_rows} event(s) have Hugo_Symbol "
            "'Unknown' (intergenic placeholder); they form an 'Unknown' row "
            "in gene-level tables."
        )
    if report.dropped_duplicate_rows:
        report.notes.append(
            f"dropped {report.dropped_duplicate_rows} duplicate event row(s) "
            "(same sample, gene, position and alleles)"
        )
    elif report.duplicate_rows and not deduplicate:
        report.issues.append(
            f"{report.duplicate_rows} duplicate event row(s) (same sample, "
            "gene, position and alleles), typically unmerged multi-caller "
            "output; each extra copy becomes a false Multi_Hit. Pass "
            "deduplicate=True to drop them."
        )
    if len(report.genome_builds) > 1:
        report.issues.append(
            f"mixed genome builds in one file: {report.genome_builds}. "
            "Lift over to one build first (e.g. with Picard/CrossMap)."
        )
    if len(report.chromosome_styles) > 1:
        report.issues.append(
            "mixed chromosome naming ('chr1' and '1'); the same chromosome "
            "would be counted twice. Harmonize the Chromosome column first."
        )
    if report.legacy_allele_rows:
        report.issues.append(
            f"{report.legacy_allele_rows} SNP row(s) carry the alternate "
            "allele in Tumor_Seq_Allele1 while Tumor_Seq_Allele2 equals "
            "Reference_Allele (legacy TCGA layout); Ti/Tv and spectra read "
            "Tumor_Seq_Allele2 and would count these as no change."
        )
    if report.unknown_classifications:
        report.notes.append(
            "Variant_Classification values outside the GDC vocabulary "
            f"(not counted as nonsynonymous): {report.unknown_classifications}"
        )
    if report.placeholder_sample_ids:
        report.notes.append(
            f"Tumor_Sample_Barcode is a caller placeholder "
            f"{report.placeholder_sample_ids}; pass sample_ID=... so files "
            "from different samples do not collapse into one."
        )


_FIRST_INT = re.compile(r"(\d+)")
_LENGTH = re.compile(r"/\s*(\d+)\s*$")
_HGVSP_POS = re.compile(r"p\.\(?[A-Za-z*]*?(\d+)")


def protein_start(value: object) -> int | None:
    """Parse the first amino-acid position from a Protein_position value.

    Handles VEP/GDC ``858/1210``, ranges ``745-750/1210``, unknown bounds
    ``?-12/393`` / ``100-?/393`` and cBioPortal's bare ``858``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if value is pd.NA:
        return None
    text = str(value).split("/")[0]
    match = _FIRST_INT.search(text)
    return int(match.group(1)) if match else None


def protein_length(value: object) -> int | None:
    """Parse the protein length from ``pos/length``; ``None`` when absent."""
    if value is None or value is pd.NA:
        return None
    match = _LENGTH.search(str(value))
    return int(match.group(1)) if match else None


def hgvsp_start(value: object) -> int | None:
    """Parse the first amino-acid position from ``p.K960R``-style notation."""
    if value is None or value is pd.NA:
        return None
    match = _HGVSP_POS.search(str(value))
    return int(match.group(1)) if match else None
