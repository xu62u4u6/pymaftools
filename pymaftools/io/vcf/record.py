from __future__ import annotations

import re
from dataclasses import dataclass

CHROM_PATTERN = re.compile(r"^(?:chr)?([1-9]|1\d|2[0-2]|X|Y)$")
"""Matches standard human nuclear chromosomes with or without a ``chr`` prefix."""

VCF_RECORD_COLUMNS: tuple[str, ...] = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "variant_type",
    "filter",
    "caller",
    "tumor_dp",
    "tumor_ref",
    "tumor_ad",
    "tumor_af",
    "normal_dp",
    "normal_ref",
    "normal_ad",
)
"""Canonical column schema for a harmonized :class:`pymaftools.core.VCF.VCF` table."""

_CHROM_ORDER: dict[str, int] = {
    **{str(index): index for index in range(1, 23)},
    "X": 23,
    "Y": 24,
}


def classify_variant_type(ref: str, alt: str) -> str:
    """Classify a normalized REF/ALT allele pair into a variant class.

    Parameters
    ----------
    ref : str
        Reference allele sequence.
    alt : str
        Alternate allele sequence.

    Returns
    -------
    str
        One of ``"SNV"``, ``"MNV"``, ``"INS"``, ``"DEL"``, or ``"COMPLEX"``.
    """
    ref_len = len(ref)
    alt_len = len(alt)
    if ref_len == 1 and alt_len == 1:
        return "SNV"
    if ref_len == alt_len:
        return "MNV"
    if ref_len == 1 and alt_len > 1:
        return "INS"
    if ref_len > 1 and alt_len == 1:
        return "DEL"
    return "COMPLEX"


def variant_category(variant_type: str) -> str:
    """Collapse a granular ``variant_type`` into ``"SNV"`` or ``"INDEL"``.

    ``"SNV"`` and ``"MNV"`` map to ``"SNV"``; ``"INS"``, ``"DEL"``, and
    ``"COMPLEX"`` map to ``"INDEL"``.
    """
    if variant_type in {"SNV", "MNV"}:
        return "SNV"
    return "INDEL"


def chrom_sort_key(chrom: str) -> tuple[int, str]:
    """Return a genomic sort key for human chromosomes with or without ``chr``."""
    stripped = chrom[3:] if chrom.startswith("chr") else chrom
    return (_CHROM_ORDER.get(stripped, 999), chrom)


def has_chr_prefix(chrom: str) -> bool:
    """Return ``True`` when *chrom* uses the UCSC ``chr`` prefix."""
    return chrom.startswith("chr")


@dataclass
class VCFRecord:
    """Validated representation of a parsed harmonized VCF variant row."""

    chrom: str
    pos: int
    ref: str
    alt: str
    filter: str
    caller: str
    tumor_dp: int
    tumor_ref: int
    tumor_ad: int
    tumor_af: float
    normal_dp: int | None = None
    normal_ref: int | None = None
    normal_ad: int | None = None
    variant_type: str = ""

    def __post_init__(self) -> None:
        """Validate parsed VCF row values and populate ``variant_type``."""
        if not CHROM_PATTERN.match(self.chrom):
            raise ValueError(
                f"Invalid chromosome '{self.chrom}'. Expected chr1-22, chrX, chrY "
                "or 1-22, X, Y."
            )
        if not self.variant_type:
            self.variant_type = classify_variant_type(self.ref, self.alt)
        if not 0.0 <= self.tumor_af <= 1.0:
            raise ValueError(
                f"Invalid tumor_af '{self.tumor_af}'. Expected a value between 0.0 and 1.0."
            )
        if self.tumor_dp < 0:
            raise ValueError(
                f"Invalid tumor_dp '{self.tumor_dp}'. Expected a non-negative integer."
            )
        if self.tumor_ref < 0:
            raise ValueError(
                f"Invalid tumor_ref '{self.tumor_ref}'. Expected a non-negative integer."
            )
        if self.tumor_ad < 0:
            raise ValueError(
                f"Invalid tumor_ad '{self.tumor_ad}'. Expected a non-negative integer."
            )
        for field_name in ("normal_dp", "normal_ref", "normal_ad"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(
                    f"Invalid {field_name} '{value}'. Expected a non-negative integer."
                )
