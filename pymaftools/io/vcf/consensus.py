from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from pymaftools.provenance import SCHEMA_VERSION

from .adapters import get_caller_adapter
from .record import (
    VCF_RECORD_COLUMNS,
    chrom_sort_key,
    classify_variant_type,
    has_chr_prefix,
    variant_category,
)

CONSENSUS_EXTRA_COLUMNS: tuple[str, ...] = ("callers", "n_callers", "vaf_spread")
VALID_AGGREGATIONS: frozenset[str] = frozenset({"priority", "median", "mean"})


def _ensure_variant_type_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        if "variant_type" not in out.columns:
            out["variant_type"] = pd.Series(dtype="object")
        return out
    if "variant_type" in df.columns:
        return df
    out = df.copy()
    out["variant_type"] = [
        classify_variant_type(str(ref), str(alt))
        for ref, alt in zip(out["ref"], out["alt"])
    ]
    return out


def _caller_name_for_vcf(vcf: pd.DataFrame, index: int) -> str:
    header = getattr(vcf, "header", None) or {}
    if not vcf.empty and "caller" in vcf.columns:
        unique_callers = [str(c) for c in vcf["caller"].dropna().unique()]
        if len(unique_callers) == 1:
            return unique_callers[0]
    if header.get("caller"):
        return str(header["caller"])
    return f"caller_{index}"


def _supported_types_for_vcf(vcf: pd.DataFrame, caller_name: str) -> frozenset[str]:
    header = getattr(vcf, "header", None) or {}
    declared = header.get("supported_variant_types")
    if declared:
        return frozenset(str(x).upper() for x in declared)
    try:
        adapter = get_caller_adapter(caller_name)
        return adapter.supported_variant_types
    except ValueError:
        return frozenset({"SNV", "INDEL"})


def _validate_chrom_prefix_consistency(vcfs: Sequence[pd.DataFrame]) -> None:
    chr_callers: list[str] = []
    bare_callers: list[str] = []
    for idx, vcf in enumerate(vcfs):
        if vcf.empty or "chrom" not in vcf.columns:
            continue
        caller = _caller_name_for_vcf(vcf, idx)
        prefixes = {has_chr_prefix(str(c)) for c in vcf["chrom"].dropna().unique()}
        if True in prefixes:
            chr_callers.append(caller)
        if False in prefixes:
            bare_callers.append(caller)
    if chr_callers and bare_callers:
        raise ValueError(
            "Inconsistent chromosome naming across callers: "
            f"chr-prefixed={sorted(set(chr_callers))} vs "
            f"unprefixed={sorted(set(bare_callers))}. "
            "Harmonize chromosome prefixes (e.g. with 'bcftools annotate --rename-chrs') "
            "before merging callers."
        )


def _warn_normalization_and_indel_support(
    vcfs: Sequence[pd.DataFrame],
    *,
    indel_threshold: int,
    explicit_indel_threshold: bool,
) -> None:
    multiallelic_callers: list[str] = []
    snv_only_callers: list[str] = []
    indel_capable_callers: set[str] = set()
    any_indel_present = False

    for idx, vcf in enumerate(vcfs):
        caller = _caller_name_for_vcf(vcf, idx)
        header = getattr(vcf, "header", None) or {}
        if header.get("multiallelic_sites_expanded", 0) > 0:
            multiallelic_callers.append(caller)
        supported = _supported_types_for_vcf(vcf, caller)
        if "INDEL" in supported:
            indel_capable_callers.add(caller)
        else:
            snv_only_callers.append(caller)
        if not vcf.empty:
            typed = _ensure_variant_type_column(vcf)
            if any(
                variant_category(str(vt)) == "INDEL" for vt in typed["variant_type"]
            ):
                any_indel_present = True

    if multiallelic_callers:
        warnings.warn(
            "Multi-allelic sites were expanded in-memory without reference-based "
            f"left-alignment for caller(s) {sorted(set(multiallelic_callers))}. "
            "For accurate multi-caller INDEL matching, normalize VCFs beforehand "
            "with 'bcftools norm -m -any -f ref.fa'.",
            UserWarning,
            stacklevel=3,
        )

    if (
        not explicit_indel_threshold
        and snv_only_callers
        and any_indel_present
        and indel_threshold > len(indel_capable_callers)
    ):
        warnings.warn(
            f"SNV-only caller(s) {sorted(set(snv_only_callers))} cannot call INDELs, "
            f"leaving only {len(indel_capable_callers)} INDEL-capable caller(s) "
            f"while min_callers_indel defaults to {indel_threshold}. All INDELs "
            "will be dropped unless you pass min_callers_indel="
            f"{max(1, len(indel_capable_callers))} explicitly.",
            UserWarning,
            stacklevel=3,
        )


def _aggregate_metrics(
    combined: pd.DataFrame,
    deduped: pd.DataFrame,
    aggregation: str,
) -> pd.DataFrame:
    if aggregation == "priority" or deduped.empty:
        return deduped

    agg_func = "median" if aggregation == "median" else "mean"
    grouped = combined.groupby(level=0, sort=False)

    result = deduped.copy()
    for int_col in ("tumor_dp", "tumor_ref", "tumor_ad"):
        if int_col in combined.columns:
            agg_vals = grouped[int_col].agg(agg_func).reindex(result.index)
            result[int_col] = np.rint(agg_vals.to_numpy(dtype=float)).astype(int)

    if "tumor_af" in combined.columns:
        result["tumor_af"] = (
            grouped["tumor_af"].agg(agg_func).reindex(result.index).astype(float)
        )

    for opt_int_col in ("normal_dp", "normal_ref", "normal_ad"):
        if opt_int_col in combined.columns:
            numeric = pd.to_numeric(combined[opt_int_col], errors="coerce")
            agg_vals = (
                numeric.groupby(level=0, sort=False).agg(agg_func).reindex(result.index)
            )
            result[opt_int_col] = [
                None if pd.isna(v) else int(round(float(v))) for v in agg_vals
            ]

    return result


def sort_vcf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a harmonized VCF DataFrame in natural genomic order."""
    if df.empty:
        return df.copy()
    order_df = pd.DataFrame(
        {
            "_chrom_key": [chrom_sort_key(str(c)) for c in df["chrom"]],
            "_pos": df["pos"].astype(int).to_numpy(),
            "_ref": df["ref"].astype(str).to_numpy(),
            "_alt": df["alt"].astype(str).to_numpy(),
        },
        index=df.index,
    )
    sorted_positions = np.lexsort(
        (
            order_df["_alt"].to_numpy(),
            order_df["_ref"].to_numpy(),
            order_df["_pos"].to_numpy(),
            [k[1] for k in order_df["_chrom_key"]],
            [k[0] for k in order_df["_chrom_key"]],
        )
    )
    return df.iloc[sorted_positions].copy()


def merge_caller_vcfs(
    vcfs: Sequence[pd.DataFrame],
    *,
    min_callers: int = 2,
    min_callers_snv: int | None = None,
    min_callers_indel: int | None = None,
    aggregation: str = "priority",
    vaf_spread_warn: float | None = 0.2,
    sort: bool = False,
) -> pd.DataFrame:
    """Merge harmonized VCF tables across callers with SNV/INDEL-aware voting.

    Returns a plain DataFrame; :meth:`pymaftools.core.VCF.VCF.merge_callers`
    wraps it back into a :class:`~pymaftools.core.VCF.VCF` with header metadata.
    """
    if aggregation not in VALID_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation '{aggregation}'. "
            f"Expected one of: {sorted(VALID_AGGREGATIONS)}."
        )

    snv_threshold = min_callers if min_callers_snv is None else min_callers_snv
    indel_threshold = min_callers if min_callers_indel is None else min_callers_indel
    if snv_threshold < 1 or indel_threshold < 1:
        raise ValueError(
            "Caller consensus thresholds must be >= 1 "
            f"(got min_callers_snv={snv_threshold}, min_callers_indel={indel_threshold})."
        )

    empty_columns = list(VCF_RECORD_COLUMNS) + list(CONSENSUS_EXTRA_COLUMNS)
    if not vcfs:
        return pd.DataFrame(columns=empty_columns, index=pd.Index([], dtype="object"))

    _validate_chrom_prefix_consistency(vcfs)
    _warn_normalization_and_indel_support(
        vcfs,
        indel_threshold=indel_threshold,
        explicit_indel_threshold=min_callers_indel is not None,
    )

    non_empty = [_ensure_variant_type_column(v) for v in vcfs if not v.empty]
    if not non_empty:
        return pd.DataFrame(columns=empty_columns, index=pd.Index([], dtype="object"))

    # Deduplicate duplicate keys within each caller before cross-caller voting
    deduped_inputs: list[pd.DataFrame] = []
    for frame in non_empty:
        deduped_inputs.append(frame[~frame.index.duplicated(keep="first")])

    combined = pd.concat(deduped_inputs, axis=0)
    grouped = combined.groupby(level=0, sort=False)

    callers_per_variant = grouped["caller"].apply(
        lambda series: ";".join(dict.fromkeys(str(x) for x in series))
    )
    caller_count = grouped["caller"].nunique()
    af_min = grouped["tumor_af"].min()
    af_max = grouped["tumor_af"].max()
    vaf_spread = (af_max - af_min).round(6)

    deduped = combined[~combined.index.duplicated(keep="first")].copy()
    deduped = _aggregate_metrics(combined, deduped, aggregation)
    deduped["callers"] = callers_per_variant.reindex(deduped.index)
    deduped["n_callers"] = caller_count.reindex(deduped.index).astype(int)
    deduped["vaf_spread"] = vaf_spread.reindex(deduped.index).astype(float)

    categories = [variant_category(str(vt)) for vt in deduped["variant_type"]]
    required = np.array(
        [snv_threshold if cat == "SNV" else indel_threshold for cat in categories],
        dtype=int,
    )
    keep_mask = deduped["n_callers"].to_numpy(dtype=int) >= required
    result = deduped.loc[keep_mask].copy()

    if (
        vaf_spread_warn is not None
        and not result.empty
        and (result["vaf_spread"] > vaf_spread_warn).any()
    ):
        n_discordant = int((result["vaf_spread"] > vaf_spread_warn).sum())
        max_spread = float(result["vaf_spread"].max())
        warnings.warn(
            f"{n_discordant} consensus variant(s) show cross-caller VAF spread "
            f"> {vaf_spread_warn:.2f} (max spread={max_spread:.3f}). Inspect the "
            "'vaf_spread' column before downstream VAF/clonality analyses.",
            UserWarning,
            stacklevel=3,
        )

    if sort and not result.empty:
        result = sort_vcf_dataframe(result)

    return result


def build_concordance_summary(
    vcfs: Sequence[pd.DataFrame],
    *,
    min_callers: int = 2,
    min_callers_snv: int | None = None,
    min_callers_indel: int | None = None,
) -> dict[str, Any]:
    """Compute exact per-caller, pairwise Jaccard, and mutually-exclusive subset stats."""
    snv_threshold = min_callers if min_callers_snv is None else min_callers_snv
    indel_threshold = min_callers if min_callers_indel is None else min_callers_indel

    if not vcfs:
        return {
            "schema_version": SCHEMA_VERSION,
            "callers": [],
            "total_unique_variants": 0,
            "min_callers_snv": snv_threshold,
            "min_callers_indel": indel_threshold,
            "consensus_count": 0,
            "snv_consensus_count": 0,
            "indel_consensus_count": 0,
            "per_caller": {},
            "pairwise_jaccard": {},
            "exclusive_subsets": {},
        }

    _validate_chrom_prefix_consistency(vcfs)

    caller_names: list[str] = []
    caller_sets: dict[str, set[str]] = {}
    variant_categories: dict[str, str] = {}

    for idx, vcf in enumerate(vcfs):
        name = _caller_name_for_vcf(vcf, idx)
        if name not in caller_sets:
            caller_names.append(name)
            caller_sets[name] = set()
        if vcf.empty:
            continue
        typed = _ensure_variant_type_column(vcf)
        for vid, vtype in zip(typed.index, typed["variant_type"]):
            key = str(vid)
            caller_sets[name].add(key)
            if key not in variant_categories:
                variant_categories[key] = variant_category(str(vtype))

    all_variants = sorted(variant_categories)
    exclusive_subsets: dict[str, int] = {}
    snv_consensus = 0
    indel_consensus = 0

    for vid in all_variants:
        present = [c for c in caller_names if vid in caller_sets[c]]
        subset_key = ";".join(present)
        exclusive_subsets[subset_key] = exclusive_subsets.get(subset_key, 0) + 1
        cat = variant_categories[vid]
        if cat == "SNV" and len(present) >= snv_threshold:
            snv_consensus += 1
        elif cat == "INDEL" and len(present) >= indel_threshold:
            indel_consensus += 1

    per_caller: dict[str, dict[str, int]] = {}
    for name in caller_names:
        vids = caller_sets[name]
        snv_n = sum(1 for v in vids if variant_categories[v] == "SNV")
        indel_n = sum(1 for v in vids if variant_categories[v] == "INDEL")
        unique_only = exclusive_subsets.get(name, 0)
        per_caller[name] = {
            "total": len(vids),
            "snv": snv_n,
            "indel": indel_n,
            "unique_only": unique_only,
        }

    pairwise_jaccard: dict[str, float] = {}
    for left, right in itertools.combinations(caller_names, 2):
        union_n = len(caller_sets[left] | caller_sets[right])
        inter_n = len(caller_sets[left] & caller_sets[right])
        pairwise_jaccard[f"{left}|{right}"] = (
            round(inter_n / union_n, 6) if union_n > 0 else 0.0
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "callers": caller_names,
        "total_unique_variants": len(all_variants),
        "min_callers_snv": snv_threshold,
        "min_callers_indel": indel_threshold,
        "consensus_count": snv_consensus + indel_consensus,
        "snv_consensus_count": snv_consensus,
        "indel_consensus_count": indel_consensus,
        "per_caller": per_caller,
        "pairwise_jaccard": pairwise_jaccard,
        "exclusive_subsets": exclusive_subsets,
    }


def summarize_cohort_consensus(
    per_sample_summaries: Mapping[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a cohort-level QC table from per-sample concordance summaries."""
    all_callers: list[str] = []
    for summary in per_sample_summaries.values():
        for caller in summary.get("callers", []):
            if caller not in all_callers:
                all_callers.append(caller)

    records: list[dict[str, Any]] = []
    for sample_id, summary in per_sample_summaries.items():
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "total_unique_variants": summary["total_unique_variants"],
            "consensus_count": summary["consensus_count"],
            "snv_consensus_count": summary["snv_consensus_count"],
            "indel_consensus_count": summary["indel_consensus_count"],
        }
        per_caller = summary.get("per_caller", {})
        for caller in all_callers:
            row[f"{caller}_count"] = per_caller.get(caller, {}).get("total", 0)
        records.append(row)

    if not records:
        return pd.DataFrame(
            columns=[
                "total_unique_variants",
                "consensus_count",
                "snv_consensus_count",
                "indel_consensus_count",
            ],
            index=pd.Index([], name="sample_id"),
        )
    return pd.DataFrame(records).set_index("sample_id")
