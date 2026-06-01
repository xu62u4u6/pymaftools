from __future__ import annotations

import gzip
import re
from typing import Any

import pandas as pd

from pymaftools.io.vcf import parse_vcf_rows
from pymaftools.io.vcf.parsers import _resolve_samples


def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


class VCF(pd.DataFrame):
    """
    A pandas DataFrame subclass for Variant Call Format (VCF) files.

    Provides methods to read, filter, and retain header metadata from VCF
    data commonly used in cancer genomics pipelines.

    Attributes
    ----------
    header : dict[str, Any]
        Parsed VCF header metadata including INFO, FORMAT, and sample names.
    """

    _metadata = ["header"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialise a VCF DataFrame.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to ``pd.DataFrame.__init__``.
        **kwargs : Any
            Keyword arguments forwarded to ``pd.DataFrame.__init__``.
        """
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self) -> type[VCF]:
        """
        Return the constructor for this subclass.

        Returns
        -------
        type[VCF]
            The VCF class, ensuring pandas operations return VCF instances.
        """
        return VCF

    @classmethod
    def from_file(cls, vcf_path: str, caller: str) -> VCF:
        """
        Read a VCF file and return a VCF object.

        Parameters
        ----------
        vcf_path : str
            Path to the VCF file.
        caller : str
            Variant caller name assigned to each parsed row.

        Returns
        -------
        VCF
            A VCF DataFrame with a composite variant index and parsed header.
        """
        header: dict[str, Any] = {
            "INFO": {},
            "FORMAT": {},
            "samples": [],
            "tumor_sample": None,
            "normal_sample": None,
        }

        id_pattern = re.compile(r"ID=([^,>]+)")
        tumor_meta_re = re.compile(r"^##tumor_sample=(.+)$", re.IGNORECASE)
        normal_meta_re = re.compile(r"^##normal_sample=(.+)$", re.IGNORECASE)
        with _open_text(vcf_path) as handle:
            for line in handle:
                line = line.rstrip("\n")
                if line.startswith("##INFO="):
                    match = id_pattern.search(line)
                    if match is not None:
                        header["INFO"][match.group(1)] = line
                elif line.startswith("##FORMAT="):
                    match = id_pattern.search(line)
                    if match is not None:
                        header["FORMAT"][match.group(1)] = line
                elif m := tumor_meta_re.match(line):
                    header["tumor_sample"] = m.group(1).strip()
                elif m := normal_meta_re.match(line):
                    header["normal_sample"] = m.group(1).strip()
                elif line.startswith("#CHROM"):
                    columns = line.lstrip("#").split("\t")
                    header["samples"] = columns[9:]
                    break

        # If ##tumor_sample= was absent, resolve from column names
        if header["tumor_sample"] is None and header["samples"]:
            t, n = _resolve_samples(None, None, header["samples"])
            header["tumor_sample"] = t
            header["normal_sample"] = n

        rows = parse_vcf_rows(vcf_path, caller)
        vcf = cls(pd.DataFrame(rows))
        if not vcf.empty:
            vcf.index = (
                vcf["chrom"]
                + "|"
                + vcf["pos"].astype(str)
                + "|"
                + vcf["ref"]
                + "|"
                + vcf["alt"]
            )
        else:
            vcf.index = pd.Index([], dtype="object")
        vcf.header = header
        return vcf

    def to_file(self, path: str) -> None:
        """
        Write harmonized VCF to a plain-text file compatible with vcf2maf.

        FORMAT per record: GT:DP:AD:AF
          - AD = ref_count,alt_count  (Number=R, required by vcf2maf)
          - normal column written first, tumor column second

        Parameters
        ----------
        path : str
            Output file path (uncompressed VCF).
        """
        tumor_id  = self.header.get("tumor_sample", "TUMOR")
        normal_id = self.header.get("normal_sample", "NORMAL")
        has_normal = normal_id is not None and "normal_dp" in self.columns

        lines: list[str] = [
            "##fileformat=VCFv4.2",
            f"##tumor_sample={tumor_id}",
        ]
        if has_normal:
            lines.append(f"##normal_sample={normal_id}")
        lines += [
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">',
            '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for ref and alt alleles">',
            '##FORMAT=<ID=AF,Number=A,Type=Float,Description="Allele fraction of the alt allele">',
        ]
        sample_cols = [normal_id, tumor_id] if has_normal else [tumor_id]
        lines.append(
            "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + sample_cols)
        )

        for idx, row in self.iterrows():
            t_ref = int(row.get("tumor_ref", row["tumor_dp"] - row["tumor_ad"]))
            t_ad  = int(row["tumor_ad"])
            t_dp  = int(row["tumor_dp"])
            t_af  = round(float(row["tumor_af"]), 4)
            tumor_fmt = f"0/1:{t_dp}:{t_ref},{t_ad}:{t_af}"

            if has_normal and row.get("normal_dp") is not None:
                n_ref = int(row.get("normal_ref", row["normal_dp"] - row["normal_ad"]))
                n_ad  = int(row["normal_ad"])
                n_dp  = int(row["normal_dp"])
                normal_fmt = f"0/0:{n_dp}:{n_ref},{n_ad}:0.0"
            else:
                normal_fmt = None

            chrom, pos, ref, alt = str(row["chrom"]), str(int(row["pos"])), str(row["ref"]), str(row["alt"])
            fixed = [chrom, pos, ".", ref, alt, ".", "PASS", ".", "GT:DP:AD:AF"]
            samples = ([normal_fmt, tumor_fmt] if normal_fmt is not None else [tumor_fmt])
            lines.append("\t".join(fixed + samples))

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def filter_variants(
        self, min_dp: int = 30, min_ad: int = 3, min_af: float = 0.05
    ) -> VCF:
        """
        Filter variants by tumour depth and allele fraction thresholds.

        Parameters
        ----------
        min_dp : int, default 30
            Minimum tumour read depth.
        min_ad : int, default 3
            Minimum tumour alternate allele depth.
        min_af : float, default 0.05
            Minimum tumour alternate allele fraction.

        Returns
        -------
        VCF
            Filtered VCF preserving the parsed header metadata.
        """
        filtered = self[
            (self["tumor_dp"] >= min_dp)
            & (self["tumor_ad"] >= min_ad)
            & (self["tumor_af"] >= min_af)
        ]
        filtered.header = getattr(self, "header", {})
        return filtered

    @classmethod
    def merge_callers(cls, vcfs: list[VCF], min_callers: int = 2) -> VCF:
        """
        Merge VCF objects from multiple callers and retain variants supported
        by at least ``min_callers`` distinct callers.

        Parameters
        ----------
        vcfs : list[VCF]
            One VCF per caller, ordered by priority (first = highest priority).
            When a variant is called by multiple callers, the metrics (DP/AD/AF)
            from the highest-priority caller are retained.
        min_callers : int, default 2
            Minimum number of distinct callers required to keep a variant.

        Returns
        -------
        VCF
            Merged VCF with a ``callers`` column (e.g. ``"mutect2;muse"``)
            and header inherited from the first VCF in ``vcfs``.
        """
        if not vcfs:
            empty = cls()
            empty.header = {}
            return empty

        combined = pd.concat(vcfs)
        callers_per_variant = (
            combined.groupby(level=0)["caller"]
            .apply(lambda x: ";".join(x.unique()))
        )
        caller_count = combined.groupby(level=0)["caller"].nunique()

        deduped = combined[~combined.index.duplicated(keep="first")].copy()
        deduped["callers"] = callers_per_variant
        result = cls(deduped[caller_count >= min_callers])
        result.header = getattr(vcfs[0], "header", {})
        return result
