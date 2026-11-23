from __future__ import annotations

import gzip
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from pymaftools._deprecation import warn_deprecated
from pymaftools.io.vcf.adapters import (
    BaseCallerAdapter,
    list_caller_adapters,
    safe_af,
)
from pymaftools.io.vcf.consensus import (
    build_concordance_summary,
    merge_caller_vcfs,
    sort_vcf_dataframe,
    summarize_cohort_consensus,
)
from pymaftools.io.vcf.parsers import parse_vcf_file
from pymaftools.io.vcf.record import VCF_RECORD_COLUMNS


def _open_write_text(path: str | os.PathLike[str]):
    path_str = os.fspath(path)
    if path_str.endswith(".gz"):
        return gzip.open(path_str, "wt", encoding="utf-8")
    return open(path_str, "w", encoding="utf-8")


def _normalize_vcf_paths(
    vcf_path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> list[str | os.PathLike[str]]:
    if isinstance(vcf_path, (str, os.PathLike)):
        return [vcf_path]
    paths = list(vcf_path)
    if not paths:
        raise ValueError("vcf_path sequence must contain at least one VCF file path.")
    return paths


class VCF(pd.DataFrame):
    """Harmonized pandas DataFrame subclass for somatic VCF tables.

    Provides caller-adapter parsing, variant filtering, natural genomic
    sorting, multi-caller consensus merging, concordance summaries, and
    ``vcf2maf``-compatible export.
    """

    _metadata = ["header"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self) -> type[VCF]:
        return VCF

    @staticmethod
    def available_callers() -> list[str]:
        """Return sorted names of all registered VCF caller adapters."""
        return list_caller_adapters()

    @classmethod
    def from_file(
        cls,
        vcf_path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        caller: str | BaseCallerAdapter = "auto",
        **kwargs: Any,
    ) -> VCF:
        """Deprecated alias for :meth:`read_vcf` (reads from a file path)."""
        warn_deprecated(
            "VCF.from_file()",
            "VCF.read_vcf()",
            "(read_* is the convention for file readers).",
        )
        return cls.read_vcf(vcf_path, caller, **kwargs)

    @classmethod
    def read_vcf(
        cls,
        vcf_path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        caller: str | BaseCallerAdapter = "auto",
        *,
        tumor_sample: str | None = None,
        normal_sample: str | None = None,
        accepted_filters: Iterable[str] | None = None,
    ) -> VCF:
        """Read one or more VCF files from the same caller into a harmonized table.

        Parameters
        ----------
        vcf_path : str, os.PathLike, or Sequence of paths
            Single VCF path (plain text or ``.gz``), or a sequence of paths
            from the same caller (for example VarScan2 SNP + INDEL VCFs or
            Strelka2 SNV + INDEL VCFs) to concatenate into one ``VCF``.
        caller : str or BaseCallerAdapter, default ``"auto"``
            Registered caller name (``"mutect2"``, ``"strelka2"``,
            ``"varscan2"``, ``"muse"``, ``"pindel"``, ``"deepsomatic"``,
            ``"dragen"``, ``"lofreq"``), ``"auto"`` for header auto-detection,
            or a custom :class:`~pymaftools.io.vcf.adapters.BaseCallerAdapter`.
        tumor_sample : str or None, default None
            Explicit tumour sample column name overriding header resolution.
        normal_sample : str or None, default None
            Explicit normal sample column name overriding header resolution.
        accepted_filters : Iterable[str] or None, default None
            ``FILTER`` values to keep (defaults to ``{"PASS", "."}``).

        Returns
        -------
        VCF
            Harmonized VCF DataFrame indexed by ``chrom|pos|ref|alt`` with
            parsed metadata in ``vcf.header``.
        """
        paths = _normalize_vcf_paths(vcf_path)
        combined_rows: list[dict[str, Any]] = []
        merged_header: dict[str, Any] | None = None

        for path in paths:
            header, rows = parse_vcf_file(
                path,
                caller=caller,
                tumor_sample=tumor_sample,
                normal_sample=normal_sample,
                accepted_filters=accepted_filters,
            )
            if merged_header is None:
                merged_header = header
            else:
                if header.get("caller") != merged_header.get("caller"):
                    raise ValueError(
                        "All files passed to a single VCF.read_vcf() call must come "
                        f"from the same caller (got '{merged_header.get('caller')}' "
                        f"and '{header.get('caller')}'). Use VCF.merge_callers() "
                        "to combine different callers."
                    )
                merged_header["INFO"].update(header.get("INFO", {}))
                merged_header["FORMAT"].update(header.get("FORMAT", {}))
                merged_header["multiallelic_sites_expanded"] = merged_header.get(
                    "multiallelic_sites_expanded", 0
                ) + header.get("multiallelic_sites_expanded", 0)
                merged_header["filtered_non_pass_count"] = merged_header.get(
                    "filtered_non_pass_count", 0
                ) + header.get("filtered_non_pass_count", 0)
                merged_header["filtered_chrom_count"] = merged_header.get(
                    "filtered_chrom_count", 0
                ) + header.get("filtered_chrom_count", 0)
            combined_rows.extend(rows)

        if combined_rows:
            frame = pd.DataFrame(combined_rows, columns=list(VCF_RECORD_COLUMNS))
            frame.index = (
                frame["chrom"]
                + "|"
                + frame["pos"].astype(str)
                + "|"
                + frame["ref"]
                + "|"
                + frame["alt"]
            )
            if len(paths) > 1:
                frame = frame[~frame.index.duplicated(keep="first")]
        else:
            frame = pd.DataFrame(
                columns=list(VCF_RECORD_COLUMNS),
                index=pd.Index([], dtype="object"),
            )

        vcf = cls(frame)
        vcf.header = merged_header or {}
        return vcf

    def sort_variants(self) -> VCF:
        """Return a new :class:`VCF` sorted in natural genomic coordinate order."""
        sorted_df = sort_vcf_dataframe(self)
        result = VCF(sorted_df)
        result.header = getattr(self, "header", {})
        return result

    def to_file(self, path: str | os.PathLike[str], *, sort: bool = False) -> None:
        """Write harmonized VCF to a plain-text or ``.gz`` file compatible with ``vcf2maf``.

        ``FORMAT`` per record is ``GT:DP:AD:AF`` where ``AD = ref_count,alt_count``
        (``Number=R``, required by ``vcf2maf``). Normal is written before tumour
        when present; rows missing normal counts emit ``.`` in the normal column
        rather than dropping the column.

        Parameters
        ----------
        path : str or os.PathLike
            Output path (``.vcf`` or ``.vcf.gz``).
        sort : bool, default False
            When ``True``, sort variants in natural genomic order before writing.
        """
        header = getattr(self, "header", None) or {}
        tumor_id = header.get("tumor_sample") or "TUMOR"
        normal_id = header.get("normal_sample")
        has_normal = (
            normal_id is not None
            and "normal_dp" in self.columns
            and bool(self["normal_dp"].notna().any())
        )
        has_callers_info = "callers" in self.columns and not self.empty

        lines: list[str] = [
            "##fileformat=VCFv4.2",
            f"##tumor_sample={tumor_id}",
        ]
        if has_normal:
            lines.append(f"##normal_sample={normal_id}")
        if has_callers_info:
            lines += [
                '##INFO=<ID=CALLERS,Number=1,Type=String,Description="Supporting somatic callers">',
                '##INFO=<ID=N_CALLERS,Number=1,Type=Integer,Description="Number of supporting somatic callers">',
            ]
        lines += [
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">',
            '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for ref and alt alleles">',
            '##FORMAT=<ID=AF,Number=A,Type=Float,Description="Allele fraction of the alt allele">',
        ]
        sample_cols = [normal_id, tumor_id] if has_normal else [tumor_id]
        lines.append(
            "\t".join(
                [
                    "#CHROM",
                    "POS",
                    "ID",
                    "REF",
                    "ALT",
                    "QUAL",
                    "FILTER",
                    "INFO",
                    "FORMAT",
                ]
                + sample_cols
            )
        )

        source_df = sort_vcf_dataframe(self) if sort and not self.empty else self

        for _, row in source_df.iterrows():
            t_ad = int(row["tumor_ad"])
            t_dp = int(row["tumor_dp"])
            t_ref_raw = row.get("tumor_ref")
            t_ref = (
                max(t_dp - t_ad, 0)
                if t_ref_raw is None or pd.isna(t_ref_raw)
                else int(t_ref_raw)
            )
            t_af = round(float(row["tumor_af"]), 4)
            tumor_fmt = f"0/1:{t_dp}:{t_ref},{t_ad}:{t_af}"

            if has_normal:
                n_dp_raw = row.get("normal_dp")
                n_ad_raw = row.get("normal_ad")
                if (
                    n_dp_raw is None
                    or pd.isna(n_dp_raw)
                    or n_ad_raw is None
                    or pd.isna(n_ad_raw)
                ):
                    normal_fmt = "."
                else:
                    n_dp = int(n_dp_raw)
                    n_ad = int(n_ad_raw)
                    n_ref_raw = row.get("normal_ref")
                    n_ref = (
                        max(n_dp - n_ad, 0)
                        if n_ref_raw is None or pd.isna(n_ref_raw)
                        else int(n_ref_raw)
                    )
                    n_af = round(safe_af(n_ad, n_dp), 4)
                    normal_fmt = f"0/0:{n_dp}:{n_ref},{n_ad}:{n_af}"
                samples = [normal_fmt, tumor_fmt]
            else:
                samples = [tumor_fmt]

            chrom = str(row["chrom"])
            pos = str(int(row["pos"]))
            ref = str(row["ref"])
            alt = str(row["alt"])
            filter_raw = row.get("filter")
            filter_val = (
                "PASS"
                if filter_raw is None or pd.isna(filter_raw) or str(filter_raw) == ""
                else str(filter_raw)
            )
            if has_callers_info and not pd.isna(row.get("callers")):
                callers_str = str(row["callers"])
                n_callers_raw = row.get("n_callers")
                n_callers_val = (
                    len(callers_str.split(";"))
                    if n_callers_raw is None or pd.isna(n_callers_raw)
                    else int(n_callers_raw)
                )
                info_val = f"CALLERS={callers_str};N_CALLERS={n_callers_val}"
            else:
                info_val = "."

            fixed = [
                chrom,
                pos,
                ".",
                ref,
                alt,
                ".",
                filter_val,
                info_val,
                "GT:DP:AD:AF",
            ]
            lines.append("\t".join(fixed + samples))

        with _open_write_text(path) as fh:
            fh.write("\n".join(lines) + "\n")

    def filter_variants(
        self, min_dp: int = 30, min_ad: int = 3, min_af: float = 0.05
    ) -> VCF:
        """Filter variants by tumour depth and allele fraction thresholds.

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
            Filtered VCF preserving parsed header metadata.
        """
        if self.empty:
            empty = VCF(self.copy())
            empty.header = getattr(self, "header", {})
            return empty

        filtered = self[
            (self["tumor_dp"] >= min_dp)
            & (self["tumor_ad"] >= min_ad)
            & (self["tumor_af"] >= min_af)
        ]
        result = VCF(filtered)
        result.header = getattr(self, "header", {})
        return result

    @classmethod
    def merge_callers(
        cls,
        vcfs: Sequence[VCF],
        min_callers: int = 2,
        *,
        min_callers_snv: int | None = None,
        min_callers_indel: int | None = None,
        aggregation: str = "priority",
        vaf_spread_warn: float | None = 0.2,
        sort: bool = False,
    ) -> VCF:
        """Merge VCF tables across multiple somatic callers for one sample.

        Parameters
        ----------
        vcfs : Sequence[VCF]
            One :class:`VCF` per caller, ordered by priority (first = highest).
        min_callers : int, default 2
            Default minimum number of distinct callers required to keep a variant.
        min_callers_snv : int or None, default None
            Optional caller threshold specifically for SNVs/MNVs. Defaults to
            ``min_callers``.
        min_callers_indel : int or None, default None
            Optional caller threshold specifically for INDELs/complex variants
            (useful when combining SNV-only callers like MuSE with INDEL callers).
        aggregation : {"priority", "median", "mean"}, default "priority"
            Strategy for combining numeric DP/AD/AF metrics across callers.
        vaf_spread_warn : float or None, default 0.2
            Emit a ``UserWarning`` when retained consensus variants show
            cross-caller ``max(AF) - min(AF)`` above this threshold. Pass
            ``None`` to silence.
        sort : bool, default False
            When ``True``, sort merged consensus variants in natural genomic order.

        Returns
        -------
        VCF
            Merged consensus VCF with ``callers``, ``n_callers``, and
            ``vaf_spread`` columns, inheriting header metadata from ``vcfs[0]``.
        """
        merged_df = merge_caller_vcfs(
            vcfs,
            min_callers=min_callers,
            min_callers_snv=min_callers_snv,
            min_callers_indel=min_callers_indel,
            aggregation=aggregation,
            vaf_spread_warn=vaf_spread_warn,
            sort=sort,
        )
        result = cls(merged_df)
        result.header = getattr(vcfs[0], "header", {}) if vcfs else {}
        return result

    @classmethod
    def concordance_summary(
        cls,
        vcfs: Sequence[VCF],
        *,
        min_callers: int = 2,
        min_callers_snv: int | None = None,
        min_callers_indel: int | None = None,
    ) -> dict[str, Any]:
        """Compute per-caller, pairwise Jaccard, and exact exclusive subset counts."""
        return build_concordance_summary(
            vcfs,
            min_callers=min_callers,
            min_callers_snv=min_callers_snv,
            min_callers_indel=min_callers_indel,
        )

    @classmethod
    def merge_cohort_vcfs(
        cls,
        sample_vcfs: Mapping[str, Sequence[VCF]],
        *,
        min_callers: int = 2,
        min_callers_snv: int | None = None,
        min_callers_indel: int | None = None,
        aggregation: str = "priority",
        filter_kwargs: Mapping[str, Any] | None = None,
        vaf_spread_warn: float | None = 0.2,
        sort: bool = True,
    ) -> tuple[dict[str, VCF], pd.DataFrame]:
        """Merge multi-caller VCFs independently for every sample in a cohort.

        Each sample's consensus is computed strictly within that sample so
        single-caller hits from different patients can never combine into a
        spurious multi-caller call.

        Parameters
        ----------
        sample_vcfs : Mapping[str, Sequence[VCF]]
            Mapping from ``sample_id`` to its caller :class:`VCF` tables.
        min_callers : int, default 2
            Default minimum distinct callers per sample.
        min_callers_snv : int or None, default None
            Optional SNV-specific caller threshold.
        min_callers_indel : int or None, default None
            Optional INDEL-specific caller threshold.
        aggregation : {"priority", "median", "mean"}, default "priority"
            Metric aggregation rule passed to :meth:`merge_callers`.
        filter_kwargs : Mapping[str, Any] or None, default None
            Optional keyword arguments forwarded to :meth:`filter_variants`
            for each caller table before merging.
        vaf_spread_warn : float or None, default 0.2
            Cross-caller VAF spread warning threshold passed to :meth:`merge_callers`.
        sort : bool, default True
            When ``True``, sort each sample's merged variants in genomic order.

        Returns
        -------
        tuple[dict[str, VCF], pandas.DataFrame]
            ``(merged_by_sample, cohort_summary_df)``.
        """
        merged_by_sample: dict[str, VCF] = {}
        per_sample_summaries: dict[str, dict[str, Any]] = {}

        for sample_id, caller_tables in sample_vcfs.items():
            prepared = (
                [table.filter_variants(**filter_kwargs) for table in caller_tables]
                if filter_kwargs
                else list(caller_tables)
            )
            per_sample_summaries[sample_id] = cls.concordance_summary(
                prepared,
                min_callers=min_callers,
                min_callers_snv=min_callers_snv,
                min_callers_indel=min_callers_indel,
            )
            merged_by_sample[sample_id] = cls.merge_callers(
                prepared,
                min_callers=min_callers,
                min_callers_snv=min_callers_snv,
                min_callers_indel=min_callers_indel,
                aggregation=aggregation,
                vaf_spread_warn=vaf_spread_warn,
                sort=sort,
            )

        summary_df = summarize_cohort_consensus(per_sample_summaries)
        return merged_by_sample, summary_df
