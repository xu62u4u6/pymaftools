from __future__ import annotations

import csv
import gzip

from .record import CHROM_PATTERN, VCFRecord


def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


def sample_type(sample_id: str) -> str:
    """
    Classify a VCF sample column as tumour or normal.

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
        If the sample identifier does not end with ``T`` or ``N``.
    """
    if sample_id.upper() == "TUMOR" or sample_id.endswith("T"):
        return "TUMOR"
    if sample_id.upper() == "NORMAL" or sample_id.endswith("N"):
        return "NORMAL"
    raise ValueError(
        f"Unable to determine sample type for '{sample_id}'. Expected suffix 'T' or 'N', or literal 'TUMOR'/'NORMAL'."
    )


def _safe_af(ad: int, dp: int) -> float:
    """
    Derive allele fraction from depth values.

    Parameters
    ----------
    ad : int
        Alternate allele depth.
    dp : int
        Total read depth.

    Returns
    -------
    float
        Alternate allele fraction.

    Raises
    ------
    ValueError
        If depth is zero while alternate depth is positive.
    """
    if dp == 0:
        if ad == 0:
            return 0.0
        raise ValueError("Unable to derive AF when DP is zero and AD is positive.")
    return ad / dp


def parse_info(format_str: str, value_str: str) -> tuple[int, int, int, float]:
    """
    Parse caller-specific FORMAT data into depth metrics.

    Parameters
    ----------
    format_str : str
        FORMAT column value.
    value_str : str
        Sample-specific value string matching ``format_str``.

    Returns
    -------
    tuple[int, int, int, float]
        Parsed ``(DP, ref_count, alt_count, AF)`` values.
        ref_count is the reference allele depth.

    Raises
    ------
    ValueError
        If the FORMAT pattern is not recognised or cannot be parsed.
    """
    keys = format_str.split(":")
    values = value_str.split(":")
    parsed = dict(zip(keys, values))

    try:
        # varscan: GT:GQ:DP:RD:AD:FREQ:DP4
        # RD = ref depth, AD = alt depth (Number=1, alt only)
        if format_str == "GT:GQ:DP:RD:AD:FREQ:DP4":
            dp  = int(values[2])
            ref = int(values[3])  # RD
            ad  = int(values[4])  # alt only
            af  = float(values[5].rstrip("%")) / 100
            return dp, ref, ad, af

        # mutect2: GT:AD:AF:DP:... (AD = Number=R → ref,alt)
        if "AD" in parsed and (("AF" in parsed) or ("DP" in parsed)):
            ad_values = [int(x) for x in parsed["AD"].split(",")]
            ref = ad_values[0]
            ad  = ad_values[1]
            dp  = int(parsed["DP"]) if "DP" in parsed else sum(ad_values)
            af  = float(parsed["AF"].split(",")[0]) if "AF" in parsed else _safe_af(ad, dp)
            return dp, ref, ad, af

        # muse: GT:DP:AD:BQ:SS (AD = ref,alt)
        if format_str == "GT:DP:AD:BQ:SS":
            dp  = int(parsed["DP"])
            ad_values = [int(x) for x in parsed["AD"].split(",")]
            ref = ad_values[0]
            ad  = ad_values[1]
            af  = _safe_af(ad, dp)
            return dp, ref, ad, af

        # pindel: GT:AD (AD = ref,alt)
        if format_str == "GT:AD":
            ad_values = [int(x) for x in parsed["AD"].split(",")]
            ref = ad_values[0]
            ad  = ad_values[1]
            dp  = sum(ad_values)
            af  = _safe_af(ad, dp)
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
) -> tuple[str, str | None]:
    """
    Determine which sample column is tumor and which is normal.

    Resolution order:
    1. ``##tumor_sample=`` / ``##normal_sample=`` metadata (Mutect2 style).
    2. Column name is literally ``TUMOR`` or ``NORMAL`` (MuSE / VarScan style).
    3. Last character of sample ID is ``T`` (tumor) or ``N`` (normal) (Pindel style).

    Parameters
    ----------
    meta_tumor : str or None
        Value of ``##tumor_sample=`` header line, if present.
    meta_normal : str or None
        Value of ``##normal_sample=`` header line, if present.
    sample_columns : list[str]
        Sample column names from the ``#CHROM`` line (index 9 onwards).

    Returns
    -------
    tuple[str, str | None]
        ``(tumor_column, normal_column)`` where ``normal_column`` may be ``None``.

    Raises
    ------
    ValueError
        If no tumor column can be identified.
    """
    tumor_col: str | None = None
    normal_col: str | None = None

    # Priority 1: explicit metadata headers (e.g. Mutect2 ##tumor_sample=AS_02T)
    if meta_tumor is not None:
        for col in sample_columns:
            if col == meta_tumor:
                tumor_col = col
            elif col == meta_normal:
                normal_col = col

    # Priority 2: literal TUMOR / NORMAL column names (MuSE, VarScan)
    if tumor_col is None:
        for col in sample_columns:
            if col.upper() == "TUMOR":
                tumor_col = col
            elif col.upper() == "NORMAL":
                normal_col = col

    # Priority 3: sample ID suffix T / N (Pindel, custom IDs)
    if tumor_col is None:
        for col in sample_columns:
            if col.endswith("T"):
                tumor_col = col
            elif col.endswith("N"):
                normal_col = col

    if tumor_col is None:
        raise ValueError(
            f"Unable to identify tumor sample column from: {sample_columns}. "
            "Expected ##tumor_sample= header, a 'TUMOR' column, or a sample ID ending in 'T'."
        )

    return tumor_col, normal_col


def parse_vcf_rows(vcf_path: str, caller: str) -> list[dict]:
    """
    Parse PASS VCF rows into validated dictionaries.

    Parameters
    ----------
    vcf_path : str
        Path to the VCF file.
    caller : str
        Variant caller name assigned to each row.

    Returns
    -------
    list[dict]
        Parsed and validated VCF rows.
    """
    rows: list[dict] = []
    column_header: list[str] | None = None
    meta_tumor: str | None = None
    meta_normal: str | None = None
    rename_map = {
        "CHROM": "chrom",
        "POS": "pos",
        "REF": "ref",
        "ALT": "alt",
        "FILTER": "filter",
    }

    import re
    _tumor_meta_re = re.compile(r"^##tumor_sample=(.+)$", re.IGNORECASE)
    _normal_meta_re = re.compile(r"^##normal_sample=(.+)$", re.IGNORECASE)

    with _open_text(vcf_path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("##"):
                m = _tumor_meta_re.match(line)
                if m:
                    meta_tumor = m.group(1).strip()
                    continue
                m = _normal_meta_re.match(line)
                if m:
                    meta_normal = m.group(1).strip()
                continue
            if line.startswith("#CHROM"):
                column_header = line.lstrip("#").split("\t")
                break

        if column_header is None:
            raise ValueError(f"No #CHROM header line found in VCF file: {vcf_path}")

        tumor_sample, normal_sample = _resolve_samples(
            meta_tumor, meta_normal, column_header[9:]
        )

        if tumor_sample is None:
            raise ValueError(f"No tumor sample column found in VCF file: {vcf_path}")

        reader = csv.DictReader(handle, fieldnames=column_header, delimiter="\t")
        for raw_row in reader:
            row = {rename_map.get(key, key): value for key, value in raw_row.items()}
            if row["filter"] != "PASS":
                continue
            if not CHROM_PATTERN.match(row["chrom"]):
                continue

            tumor_dp, tumor_ref, tumor_ad, tumor_af = parse_info(row["FORMAT"], row[tumor_sample])
            normal_dp = normal_ref = normal_ad = None
            if normal_sample is not None:
                normal_dp, normal_ref, normal_ad, _ = parse_info(row["FORMAT"], row[normal_sample])

            parsed_row = {
                "chrom": row["chrom"],
                "pos": int(row["pos"]),
                "ref": row["ref"],
                "alt": row["alt"],
                "filter": row["filter"],
                "caller": caller,
                "tumor_dp": tumor_dp,
                "tumor_ref": tumor_ref,
                "tumor_ad": tumor_ad,
                "tumor_af": tumor_af,
                "normal_dp": normal_dp,
                "normal_ref": normal_ref,
                "normal_ad": normal_ad,
            }
            VCFRecord(**parsed_row)
            rows.append(parsed_row)

    return rows
