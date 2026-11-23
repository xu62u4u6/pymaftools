from __future__ import annotations

import csv
import gzip
import os
import re
from collections.abc import Iterable
from typing import Any

from .adapters import (
    BaseCallerAdapter,
    detect_caller_adapter,
    get_caller_adapter,
    parse_format_mapping,
    safe_af,
)
from .record import CHROM_PATTERN, VCFRecord, classify_variant_type

_ID_PATTERN = re.compile(r"ID=([^,>]+)")
_TUMOR_META_RE = re.compile(r"^##tumor_sample=(.+)$", re.IGNORECASE)
_NORMAL_META_RE = re.compile(r"^##normal_sample=(.+)$", re.IGNORECASE)


def _open_text(path: str | os.PathLike[str]):
    path_str = os.fspath(path)
    if path_str.endswith(".gz"):
        return gzip.open(path_str, "rt", encoding="utf-8", newline="")
    return open(path_str, encoding="utf-8", newline="")


def sample_type(sample_id: str) -> str:
    """Classify a VCF sample column as tumour or normal.

    Parameters
    ----------
    sample_id : str
        Sample identifier from the VCF header.

    Returns
    -------
    str
        ``"TUMOR"`` or ``"NORMAL"``.

    Raises
    ------
    ValueError
        If the sample identifier does not end with ``T`` or ``N`` and is not
        literally ``"TUMOR"`` or ``"NORMAL"``.
    """
    if sample_id.upper() == "TUMOR" or sample_id.endswith("T"):
        return "TUMOR"
    if sample_id.upper() == "NORMAL" or sample_id.endswith("N"):
        return "NORMAL"
    raise ValueError(
        f"Unable to determine sample type for '{sample_id}'. "
        "Expected suffix 'T' or 'N', or literal 'TUMOR'/'NORMAL'."
    )


def _safe_af(ad: int, dp: int) -> float:
    return safe_af(ad, dp)


def parse_info(
    format_str: str,
    value_str: str,
    *,
    alt_index: int = 0,
) -> tuple[int, int, int, float]:
    """Parse caller-specific ``FORMAT`` data into ``(DP, ref, alt, AF)``.

    Kept for backwards compatibility; new code should use a
    :class:`~pymaftools.io.vcf.adapters.BaseCallerAdapter`.
    """
    parsed = parse_format_mapping(format_str, value_str)
    try:
        # VarScan2: GT:GQ:DP:RD:AD:FREQ:DP4 (RD = ref, AD = alt-only)
        if "RD" in parsed and "AD" in parsed and "," not in parsed["AD"]:
            if alt_index != 0:
                raise ValueError("VarScan FORMAT does not encode per-ALT depths.")
            dp = int(parsed["DP"])
            ref = int(parsed["RD"])
            ad = int(parsed["AD"])
            af = (
                float(parsed["FREQ"].rstrip("%")) / 100.0
                if "FREQ" in parsed
                else safe_af(ad, dp)
            )
            return dp, ref, ad, af

        # Standard Number=R AD callers (Mutect2, MuSE, Pindel, DeepSomatic, DRAGEN)
        if "AD" in parsed:
            ad_values = [int(x) for x in parsed["AD"].split(",")]
            ref = ad_values[0]
            ad = ad_values[alt_index + 1]
            dp = int(parsed["DP"]) if "DP" in parsed else sum(ad_values)
            if "AF" in parsed:
                af = float(parsed["AF"].split(",")[alt_index])
            elif "VAF" in parsed:
                af = float(parsed["VAF"].split(",")[alt_index])
            else:
                af = safe_af(ad, dp)
            return dp, ref, ad, af
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Failed to parse FORMAT '{format_str}' with value '{value_str}'."
        ) from exc

    raise ValueError(f"Unrecognized FORMAT pattern: {format_str}")


def _resolve_samples(
    meta_tumor: str | None,
    meta_normal: str | None,
    sample_columns: list[str],
    *,
    tumor_sample: str | None = None,
    normal_sample: str | None = None,
    adapter: BaseCallerAdapter | None = None,
) -> tuple[str | None, str | None]:
    """Resolve ``(tumor_column, normal_column)`` using an adapter or default rules."""
    resolved_adapter = adapter or get_caller_adapter("mutect2")
    return resolved_adapter.resolve_samples(
        meta_tumor,
        meta_normal,
        sample_columns,
        tumor_sample=tumor_sample,
        normal_sample=normal_sample,
    )


def parse_vcf_file(
    vcf_path: str | os.PathLike[str],
    caller: str | BaseCallerAdapter = "auto",
    *,
    tumor_sample: str | None = None,
    normal_sample: str | None = None,
    accepted_filters: Iterable[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse a VCF file in a single pass using a caller adapter.

    Parameters
    ----------
    vcf_path : str or os.PathLike
        Path to a plain-text or gzipped VCF file.
    caller : str or BaseCallerAdapter, default ``"auto"``
        Registered caller name (e.g. ``"mutect2"``, ``"strelka2"``,
        ``"varscan2"``, ``"muse"``, ``"pindel"``, ``"deepsomatic"``,
        ``"dragen"``, ``"lofreq"``), ``"auto"`` for header auto-detection, or
        a custom :class:`BaseCallerAdapter` instance.
    tumor_sample : str or None, default None
        Explicit tumour sample column name overriding automatic resolution.
    normal_sample : str or None, default None
        Explicit normal sample column name overriding automatic resolution.
    accepted_filters : Iterable[str] or None, default None
        ``FILTER`` values to retain. Defaults to the caller adapter's
        ``default_accepted_filters`` (typically ``{"PASS", "."}``).

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]]]
        ``(header, rows)`` where ``header`` stores parsed metadata and ``rows``
        contains validated harmonized variant dictionaries.
    """
    header_lines: list[str] = []
    column_header: list[str] | None = None
    meta_tumor: str | None = None
    meta_normal: str | None = None
    info_defs: dict[str, str] = {}
    format_defs: dict[str, str] = {}

    allowed_filters = (
        frozenset(accepted_filters) if accepted_filters is not None else None
    )

    rows: list[dict[str, Any]] = []
    multiallelic_sites_expanded = 0
    filtered_non_pass_count = 0
    filtered_chrom_count = 0

    with _open_text(vcf_path) as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.startswith("##"):
                header_lines.append(line)
                if line.startswith("##INFO="):
                    match = _ID_PATTERN.search(line)
                    if match is not None:
                        info_defs[match.group(1)] = line
                elif line.startswith("##FORMAT="):
                    match = _ID_PATTERN.search(line)
                    if match is not None:
                        format_defs[match.group(1)] = line
                elif m := _TUMOR_META_RE.match(line):
                    meta_tumor = m.group(1).strip()
                elif m := _NORMAL_META_RE.match(line):
                    meta_normal = m.group(1).strip()
                continue
            if line.startswith("#CHROM"):
                column_header = line.lstrip("#").split("\t")
                break

        if column_header is None:
            raise ValueError(f"No #CHROM header line found in VCF file: {vcf_path}")

        if isinstance(caller, str) and caller.strip().lower() == "auto":
            adapter = detect_caller_adapter(header_lines)
        else:
            adapter = get_caller_adapter(caller)

        sample_columns = column_header[9:] if len(column_header) > 9 else []
        resolved_tumor, resolved_normal = adapter.resolve_samples(
            meta_tumor,
            meta_normal,
            sample_columns,
            tumor_sample=tumor_sample,
            normal_sample=normal_sample,
        )

        reader = csv.DictReader(handle, fieldnames=column_header, delimiter="\t")
        for raw_row in reader:
            if not raw_row or not raw_row.get("CHROM"):
                continue
            chrom = raw_row["CHROM"]
            filter_val = raw_row.get("FILTER", ".")
            if not adapter.accept_filter(filter_val, allowed_filters):
                filtered_non_pass_count += 1
                continue
            if not CHROM_PATTERN.match(chrom):
                filtered_chrom_count += 1
                continue

            pos = int(raw_row["POS"])
            ref = raw_row["REF"]
            raw_alts = raw_row["ALT"].split(",")
            if len(raw_alts) > 1:
                multiallelic_sites_expanded += 1

            info_str = raw_row.get("INFO", ".") or "."
            format_str = raw_row.get("FORMAT")
            tumor_value = (
                raw_row.get(resolved_tumor) if resolved_tumor is not None else None
            )
            normal_value = (
                raw_row.get(resolved_normal) if resolved_normal is not None else None
            )

            for alt_index, alt in enumerate(raw_alts):
                if alt in {".", "*", ""}:
                    continue
                tumor_metrics = adapter.parse_record_metrics(
                    ref=ref,
                    alt=alt,
                    format_str=format_str,
                    sample_value=tumor_value,
                    info_str=info_str,
                    alt_index=alt_index,
                )
                normal_dp: int | None = None
                normal_ref: int | None = None
                normal_ad: int | None = None
                if resolved_normal is not None and normal_value is not None:
                    normal_metrics = adapter.parse_record_metrics(
                        ref=ref,
                        alt=alt,
                        format_str=format_str,
                        sample_value=normal_value,
                        info_str=info_str,
                        alt_index=alt_index,
                    )
                    normal_dp = normal_metrics.dp
                    normal_ref = normal_metrics.ref_count
                    normal_ad = normal_metrics.alt_count

                parsed_row: dict[str, Any] = {
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "variant_type": classify_variant_type(ref, alt),
                    "filter": filter_val,
                    "caller": adapter.name,
                    "tumor_dp": tumor_metrics.dp,
                    "tumor_ref": tumor_metrics.ref_count,
                    "tumor_ad": tumor_metrics.alt_count,
                    "tumor_af": tumor_metrics.af,
                    "normal_dp": normal_dp,
                    "normal_ref": normal_ref,
                    "normal_ad": normal_ad,
                }
                VCFRecord(**parsed_row)
                rows.append(parsed_row)

    header: dict[str, Any] = {
        "INFO": info_defs,
        "FORMAT": format_defs,
        "samples": sample_columns,
        "tumor_sample": resolved_tumor or "TUMOR",
        "normal_sample": resolved_normal,
        "caller": adapter.name,
        "supported_variant_types": sorted(adapter.supported_variant_types),
        "multiallelic_sites_expanded": multiallelic_sites_expanded,
        "filtered_non_pass_count": filtered_non_pass_count,
        "filtered_chrom_count": filtered_chrom_count,
    }
    return header, rows


def parse_vcf_rows(
    vcf_path: str | os.PathLike[str],
    caller: str | BaseCallerAdapter = "auto",
    *,
    tumor_sample: str | None = None,
    normal_sample: str | None = None,
    accepted_filters: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse accepted VCF rows into validated harmonized dictionaries."""
    _, rows = parse_vcf_file(
        vcf_path,
        caller=caller,
        tumor_sample=tumor_sample,
        normal_sample=normal_sample,
        accepted_filters=accepted_filters,
    )
    return rows
