from __future__ import annotations

import re
from dataclasses import dataclass


CHROM_PATTERN = re.compile(r"^chr([1-9]|1\d|2[0-2]|X|Y)$")


@dataclass
class VCFRecord:
    """
    A validated representation of a parsed VCF variant row.

    Attributes
    ----------
    chrom : str
        Chromosome label in ``chr`` notation.
    pos : int
        1-based genomic position.
    ref : str
        Reference allele.
    alt : str
        Alternate allele.
    filter : str
        VCF FILTER value.
    caller : str
        Variant caller name.
    tumor_dp : int
        Tumour sample total read depth.
    tumor_ad : int
        Tumour sample alternate allele depth.
    tumor_af : float
        Tumour sample alternate allele fraction.
    normal_dp : int or None, default None
        Normal sample total read depth.
    normal_ad : int or None, default None
        Normal sample alternate allele depth.
    """

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

    def __post_init__(self) -> None:
        """
        Validate parsed VCF row values.

        Raises
        ------
        ValueError
            If chromosome format, allele fraction, or depth values are invalid.
        """
        if not CHROM_PATTERN.match(self.chrom):
            raise ValueError(
                f"Invalid chromosome '{self.chrom}'. Expected chr1-22, chrX, or chrY."
            )
        if not 0.0 <= self.tumor_af <= 1.0:
            raise ValueError(
                f"Invalid tumor_af '{self.tumor_af}'. Expected a value between 0.0 and 1.0."
            )
        if self.tumor_dp < 0:
            raise ValueError(
                f"Invalid tumor_dp '{self.tumor_dp}'. Expected a non-negative integer."
            )
        if self.tumor_ad < 0:
            raise ValueError(
                f"Invalid tumor_ad '{self.tumor_ad}'. Expected a non-negative integer."
            )
