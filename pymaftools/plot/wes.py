"""Common WES/MAF summary plots and statistics."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from ..provenance import (
    Provenance,
    attach_provenance,
    provenance_for_table,
    sample_universe,
    table_inputs,
)


BASE_CHANGE_ORDER = [
    "C>A",
    "C>G",
    "C>T",
    "T>A",
    "T>C",
    "T>G",
]


def _canonical_substitution(ref, alt) -> str | float:
    """Normalize one SNV to the six pyrimidine-context substitutions."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    ref = str(ref).upper()
    alt = str(alt).upper()
    if ref not in comp or alt not in comp or ref == alt:
        return np.nan
    if ref in {"A", "G"}:
        ref, alt = comp[ref], comp[alt]
    return f"{ref}>{alt}"


def _get_sample_col(maf: pd.DataFrame) -> str:
    if "sample_ID" in maf.columns:
        return "sample_ID"
    if "Tumor_Sample_Barcode" in maf.columns:
        return "Tumor_Sample_Barcode"
    raise ValueError("MAF must contain 'sample_ID' or 'Tumor_Sample_Barcode'.")


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Required column(s) missing: {missing}.")


def mutation_burden_by_class(maf: pd.DataFrame) -> pd.DataFrame:
    """Return sample x variant-class mutation counts."""
    sample_col = _get_sample_col(maf)
    _require_columns(maf, [sample_col, "Variant_Classification"])
    return pd.crosstab(maf[sample_col], maf["Variant_Classification"])


def top_mutated_genes(maf: pd.DataFrame, top: int = 20) -> pd.Series:
    """Return top mutated genes by unique mutated samples."""
    sample_col = _get_sample_col(maf)
    _require_columns(maf, [sample_col, "Hugo_Symbol"])
    return (
        maf.drop_duplicates(["Hugo_Symbol", sample_col])
        .groupby("Hugo_Symbol")
        .size()
        .sort_values(ascending=False)
        .head(top)
    )


def summarize_titv(maf: pd.DataFrame) -> pd.DataFrame:
    """Return per-sample six-class substitution counts in pyrimidine context."""
    sample_col = _get_sample_col(maf)
    _require_columns(
        maf, [sample_col, "Variant_Type", "Reference_Allele", "Tumor_Seq_Allele2"]
    )
    snp = maf.loc[maf["Variant_Type"].eq("SNP")].copy()

    def normalize(row):
        return _canonical_substitution(
            row["Reference_Allele"], row["Tumor_Seq_Allele2"]
        )

    snp["Base_Change"] = snp.apply(normalize, axis=1)
    counts = pd.crosstab(snp[sample_col], snp["Base_Change"])
    return counts.reindex(columns=BASE_CHANGE_ORDER, fill_value=0)


def plot_titv(
    maf: pd.DataFrame,
    fraction: bool = True,
    figsize=(10, 5),
    show_sample_labels: bool | None = None,
):
    """Draw Ti/Tv six-class substitution composition per sample."""
    from .ColorManager import ColorManager
    from . import style

    counts = summarize_titv(maf)
    if counts.empty:
        raise ValueError("No SNP substitutions available for Ti/Tv plot.")
    plot_data = (
        counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        if fraction
        else counts
    )
    cols = list(plot_data.columns)
    colors = [ColorManager.TITV_CMAP[c] for c in cols]
    fig, ax, legend_ax = style.fig_with_legend(figsize, legend_width=0.16)
    plot_data.plot(
        kind="bar", stacked=True, width=0.9, color=colors, ax=ax, legend=False
    )
    ax.set_title("Transition / Transversion Composition")
    ax.set_xlabel(f"sample (n={len(plot_data)})")
    ax.set_ylabel("Fraction" if fraction else "SNP count")
    ax.margins(x=0.01)
    style.style_axes(ax)
    style.apply_sample_xticklabels(ax, plot_data.index, show=show_sample_labels)
    style.draw_legend_cards(
        legend_ax, {"Base change": {c: ColorManager.TITV_CMAP[c] for c in cols}}
    )
    fig.tight_layout()
    return fig


def _chrom_sort_key(chrom) -> tuple[int, str]:
    label = str(chrom).replace("chr", "").replace("CHR", "")
    if label.isdigit():
        return (int(label), "")
    special = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    return (special.get(label.upper(), 99), label)


KATAEGIS_COLUMNS = [
    "cluster_id",
    "sample",
    "chrom",
    "start",
    "end",
    "span_bp",
    "n_mutations",
    "mean_inter_mutation_distance",
    "median_inter_mutation_distance",
    "dominant_substitution",
    "c_to_t_g_fraction",
    "apobec_like",
    "genes",
]


def _find_kataegis_intervals(
    positions: np.ndarray,
    *,
    min_mutations: int,
    max_mean_distance: float,
) -> list[tuple[int, int]]:
    """Return ``(start_idx, end_idx)`` inclusive index ranges of kataegis clusters."""
    n = len(positions)
    if n < min_mutations:
        return []
    window_span_limit = max_mean_distance * (min_mutations - 1)
    qualifying: list[tuple[int, int]] = []
    for start_idx in range(0, n - min_mutations + 1):
        end_idx = start_idx + min_mutations - 1
        if float(positions[end_idx] - positions[start_idx]) <= window_span_limit:
            if qualifying and start_idx <= qualifying[-1][1]:
                qualifying[-1] = (qualifying[-1][0], max(qualifying[-1][1], end_idx))
            else:
                qualifying.append((start_idx, end_idx))

    clusters: list[tuple[int, int]] = []
    for start_idx, end_idx in qualifying:
        # Trim outlying flanking mutations pulled into an edge window by a dense core.
        while (
            end_idx - start_idx + 1 > min_mutations
            and float(positions[start_idx + 1] - positions[start_idx])
            > max_mean_distance
        ):
            start_idx += 1
        while (
            end_idx - start_idx + 1 > min_mutations
            and float(positions[end_idx] - positions[end_idx - 1]) > max_mean_distance
        ):
            end_idx -= 1
        count = end_idx - start_idx + 1
        if count < min_mutations:
            continue
        mean_dist = float(positions[end_idx] - positions[start_idx]) / (count - 1)
        if mean_dist <= max_mean_distance:
            clusters.append((start_idx, end_idx))
    return clusters


def detect_kataegis(
    maf: pd.DataFrame,
    *,
    sample: str | None = None,
    min_mutations: int = 6,
    max_mean_distance: float = 1000.0,
    snv_only: bool = True,
) -> pd.DataFrame:
    """Detect localized hypermutation (kataegis) clusters in a MAF event table.

    A kataegis focus is defined as a contiguous run of at least
    ``min_mutations`` variants on the same chromosome with mean inter-mutation
    distance ``<= max_mean_distance`` bp (standard cancer-genomics default:
    ``>= 6`` mutations, mean distance ``<= 1,000`` bp). Flanking steps larger
    than ``max_mean_distance`` are trimmed so distant unrelated variants are
    not absorbed into a dense core.

    Parameters
    ----------
    maf : pandas.DataFrame
        MAF event table.
    sample : str, optional
        Restrict detection to a single sample. When ``None``, evaluates every
        sample in ``maf``.
    min_mutations : int, default 6
        Minimum consecutive variants required to call a kataegis cluster
        (must be ``>= 2``).
    max_mean_distance : float, default 1000.0
        Maximum mean inter-mutation distance in base pairs (must be ``> 0``).
    snv_only : bool, default True
        Restrict to SNPs (``Variant_Type == "SNP"``) matching canonical
        kataegis detection. Set ``False`` to include all variant types.

    Returns
    -------
    pandas.DataFrame
        Detected kataegis clusters (empty DataFrame with canonical columns when
        none are found), with provenance recorded in ``attrs`` (readable via
        :func:`pymaftools.provenance.get_provenance`).
    """
    if (
        isinstance(min_mutations, bool)
        or not isinstance(min_mutations, (int, np.integer))
        or int(min_mutations) < 2
    ):
        raise ValueError(
            f"min_mutations must be an integer >= 2; got {min_mutations!r}."
        )
    min_mutations = int(min_mutations)
    if (
        isinstance(max_mean_distance, bool)
        or not isinstance(max_mean_distance, (int, float, np.integer, np.floating))
        or not np.isfinite(max_mean_distance)
        or float(max_mean_distance) <= 0
    ):
        raise ValueError(
            "max_mean_distance must be a positive finite number; "
            f"got {max_mean_distance!r}."
        )
    max_mean_distance = float(max_mean_distance)

    sample_col = _get_sample_col(maf)
    required = [sample_col, "Chromosome", "Start_Position"]
    if snv_only:
        required.append("Variant_Type")
    _require_columns(maf, required)

    data = maf.copy()
    all_samples = data[sample_col].dropna().astype(str).unique().tolist()
    if sample is not None:
        sample = str(sample)
        if sample not in set(all_samples):
            raise ValueError(
                f"Sample {sample!r} not found in MAF. Available samples: "
                f"{all_samples[:10]}."
            )
        data = data.loc[data[sample_col].astype(str).eq(sample)].copy()
        evaluated_samples = [sample]
    else:
        evaluated_samples = all_samples

    if snv_only:
        data = data.loc[data["Variant_Type"].eq("SNP")].copy()

    data["Start_Position"] = pd.to_numeric(data["Start_Position"], errors="coerce")
    data = data.dropna(subset=[sample_col, "Chromosome", "Start_Position"])

    has_alleles = {"Reference_Allele", "Tumor_Seq_Allele2"}.issubset(data.columns)
    if has_alleles and not data.empty:
        data["_sub"] = data.apply(
            lambda row: _canonical_substitution(
                row["Reference_Allele"], row["Tumor_Seq_Allele2"]
            ),
            axis=1,
        )
    has_gene = "Hugo_Symbol" in data.columns

    rows: list[dict] = []
    if not data.empty:
        data["_sample_str"] = data[sample_col].astype(str)
        data["_chrom_order"] = data["Chromosome"].map(lambda c: _chrom_sort_key(c)[0])
        data["_chrom_label"] = data["Chromosome"].map(lambda c: _chrom_sort_key(c)[1])
        data = data.sort_values(
            ["_sample_str", "_chrom_order", "_chrom_label", "Start_Position"],
            kind="mergesort",
        )
        cluster_idx = 0
        for (sample_id, chrom), grp in data.groupby(
            ["_sample_str", "Chromosome"], sort=False
        ):
            positions = grp["Start_Position"].to_numpy(dtype=np.int64)
            intervals = _find_kataegis_intervals(
                positions,
                min_mutations=min_mutations,
                max_mean_distance=max_mean_distance,
            )
            for s_i, e_i in intervals:
                cluster_idx += 1
                sub_grp = grp.iloc[s_i : e_i + 1]
                pos_slice = positions[s_i : e_i + 1]
                diffs = np.diff(pos_slice).astype(float)
                span_bp = int(pos_slice[-1] - pos_slice[0])
                n_mut = int(len(pos_slice))
                mean_dist = float(span_bp) / (n_mut - 1)
                median_dist = float(np.median(diffs))

                dominant_sub = None
                c_to_t_g_fraction = np.nan
                apobec_like = False
                if has_alleles:
                    subs = sub_grp["_sub"].dropna()
                    if not subs.empty:
                        counts = subs.value_counts()
                        dominant_sub = str(counts.index[0])
                        c_to_t_g_fraction = float(subs.isin(["C>T", "C>G"]).mean())
                        apobec_like = bool(c_to_t_g_fraction >= 0.75)

                genes_str = ""
                if has_gene:
                    genes = [
                        str(g)
                        for g in pd.unique(sub_grp["Hugo_Symbol"].dropna())
                        if str(g).strip()
                    ]
                    genes_str = ",".join(genes)

                rows.append(
                    {
                        "cluster_id": f"K{cluster_idx}",
                        "sample": str(sample_id),
                        "chrom": str(chrom),
                        "start": int(pos_slice[0]),
                        "end": int(pos_slice[-1]),
                        "span_bp": span_bp,
                        "n_mutations": n_mut,
                        "mean_inter_mutation_distance": mean_dist,
                        "median_inter_mutation_distance": median_dist,
                        "dominant_substitution": dominant_sub,
                        "c_to_t_g_fraction": c_to_t_g_fraction,
                        "apobec_like": apobec_like,
                        "genes": genes_str,
                    }
                )

    result = pd.DataFrame(rows, columns=KATAEGIS_COLUMNS)
    provenance = Provenance(
        analysis="detect_kataegis",
        parameters={
            "sample": sample,
            "min_mutations": min_mutations,
            "max_mean_distance": max_mean_distance,
            "snv_only": snv_only,
        },
        inputs=table_inputs(maf),
        sample_universe=sample_universe(evaluated_samples, source="maf"),
    )
    return attach_provenance(result, provenance)


def plot_rainfall(
    maf: pd.DataFrame,
    sample: str | None = None,
    figsize=(12, 4),
    point_size: float = 10,
    color_by: str = "substitution",
    chrom_lengths: dict | None = None,
    *,
    detect_kataegis: bool = True,
    min_mutations: int = 6,
    max_mean_distance: float = 1000.0,
    return_clusters: bool = False,
):
    """Draw inter-mutation distance across genomic coordinates.

    Designed to surface localized hypermutation (**kataegis**) clusters —
    sudden downward plumes of consecutive mutations with short inter-mutation
    distances (typically ``<= 1 kb``) and characteristic SBS-6 substitution
    enrichment (e.g. APOBEC-driven ``C>T`` / ``C>G``).

    Parameters
    ----------
    maf : pandas.DataFrame
        MAF event table.
    sample : str, optional
        Sample to draw. With multiple samples, defaults to the first sample.
    figsize : tuple, default (12, 4)
        Figure size.
    point_size : float, default 10
        Scatter marker area.
    color_by : {"substitution", "variant_classification"}, \
default "substitution"
        ``substitution`` plots SNPs by the six canonical pyrimidine classes;
        ``variant_classification`` preserves consequence-based colouring.
    chrom_lengths : dict, optional
        Reference chromosome lengths keyed by chromosome label (``1`` or
        ``chr1``). When provided, every plotted chromosome must be present and
        at least as long as its largest observed position.
    detect_kataegis : bool, default True
        Run :func:`detect_kataegis` on the plotted sample, draw the distance
        threshold guide line, highlight any detected kataegis clusters with
        shaded bands and labels, and report detection status in the plot title.
    min_mutations : int, default 6
        Passed to :func:`detect_kataegis` when ``detect_kataegis=True``.
    max_mean_distance : float, default 1000.0
        Passed to :func:`detect_kataegis` when ``detect_kataegis=True``.
    return_clusters : bool, default False
        When ``True``, return ``(fig, clusters_df)`` instead of only ``fig``.
        The detected table is also attached to ``fig.kataegis_clusters``.
    """
    from . import style
    from .ColorManager import ColorManager

    valid_color_modes = {"substitution", "variant_classification"}
    if color_by not in valid_color_modes:
        raise ValueError(
            f"color_by must be one of {sorted(valid_color_modes)}; got {color_by!r}."
        )
    sample_col = _get_sample_col(maf)
    required = [sample_col, "Chromosome", "Start_Position"]
    if color_by == "substitution":
        required += ["Variant_Type", "Reference_Allele", "Tumor_Seq_Allele2"]
    else:
        required += ["Variant_Classification"]
    _require_columns(maf, required)
    data = maf.copy()
    resolved_sample = sample
    if resolved_sample is not None:
        resolved_sample = str(resolved_sample)
        data = data.loc[data[sample_col].astype(str).eq(resolved_sample)].copy()
    elif data[sample_col].nunique() > 1:
        resolved_sample = str(data[sample_col].iloc[0])
        data = data.loc[data[sample_col].astype(str).eq(resolved_sample)].copy()
    elif not data.empty:
        resolved_sample = str(data[sample_col].iloc[0])
    if data.empty:
        raise ValueError("No variants available for rainfall plot.")

    if color_by == "substitution":
        data = data.loc[data["Variant_Type"].eq("SNP")].copy()
        data["_rainfall_class"] = data.apply(
            lambda row: _canonical_substitution(
                row["Reference_Allele"], row["Tumor_Seq_Allele2"]
            ),
            axis=1,
        )
        data = data.dropna(subset=["_rainfall_class"])
        color_map = ColorManager.TITV_CMAP
        legend_title = "Substitution"
        if data.empty:
            raise ValueError(
                "No valid SNP substitutions available for rainfall plot; "
                "check Variant_Type, Reference_Allele, and Tumor_Seq_Allele2."
            )
    else:
        data["_rainfall_class"] = data["Variant_Classification"]
        color_map = ColorManager.ALL_MUTATION_CMAP
        legend_title = "Mutation"

    data["Start_Position"] = pd.to_numeric(data["Start_Position"], errors="coerce")
    data = data.dropna(subset=["Start_Position"])
    chroms = sorted(data["Chromosome"].dropna().unique(), key=_chrom_sort_key)
    reference_lengths = None
    if chrom_lengths is not None:
        reference_lengths = {
            str(chrom).lower().removeprefix("chr"): length
            for chrom, length in chrom_lengths.items()
        }
        missing = [
            str(chrom)
            for chrom in chroms
            if str(chrom).lower().removeprefix("chr") not in reference_lengths
        ]
        if missing:
            raise ValueError(
                f"chrom_lengths is missing plotted chromosome(s): {missing}."
            )
    offsets = {}
    chrom_spans: list[tuple[str, float, float]] = []
    offset = 0
    for chrom in chroms:
        offsets[chrom] = offset
        max_pos = data.loc[data["Chromosome"].eq(chrom), "Start_Position"].max()
        if reference_lengths is None:
            span = int(max_pos) + 1
        else:
            key = str(chrom).lower().removeprefix("chr")
            try:
                span = int(reference_lengths[key])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"chrom_lengths[{key!r}] must be a positive integer."
                ) from error
            if span <= 0 or span < max_pos:
                raise ValueError(
                    f"chrom_lengths[{key!r}]={span} is shorter than the largest "
                    f"observed position {int(max_pos)}."
                )
        chrom_spans.append((str(chrom), float(offset), float(offset + span)))
        offset += span
    total_span = max(float(offset), 1.0)
    n_events_before_diff = int(len(data))

    data["genomic_pos"] = data.apply(
        lambda row: offsets[row["Chromosome"]] + row["Start_Position"],
        axis=1,
    )
    data["_chrom_order"] = data["Chromosome"].map(
        lambda chrom: _chrom_sort_key(chrom)[0]
    )
    data["_chrom_label"] = data["Chromosome"].map(
        lambda chrom: _chrom_sort_key(chrom)[1]
    )
    data = data.sort_values(["_chrom_order", "_chrom_label", "Start_Position"])
    data["distance"] = data.groupby("Chromosome")["Start_Position"].diff()
    data = data.dropna(subset=["distance"])
    if data.empty:
        raise ValueError(
            "Rainfall plot requires at least two variants on one chromosome."
        )

    clusters = pd.DataFrame(columns=KATAEGIS_COLUMNS)
    if detect_kataegis:
        kataegis_finder = globals()["detect_kataegis"]
        clusters = kataegis_finder(
            maf,
            sample=resolved_sample,
            min_mutations=min_mutations,
            max_mean_distance=max_mean_distance,
            snv_only=(color_by == "substitution"),
        )

    fig, ax, legend_ax = style.fig_with_legend(figsize, legend_width=0.16)

    # Subtle alternating chromosome background bands for genomic orientation.
    if len(chrom_spans) > 1:
        for idx, (_chrom_name, c_start, c_end) in enumerate(chrom_spans):
            if idx % 2 == 1:
                ax.axvspan(c_start, c_end, color="#f3f4f6", alpha=0.85, zorder=0)

    present = {}
    class_order = (
        BASE_CHANGE_ORDER
        if color_by == "substitution"
        else list(pd.unique(data["_rainfall_class"]))
    )
    for cls in class_order:
        sub = data.loc[data["_rainfall_class"].eq(cls)]
        if sub.empty:
            continue
        color = color_map.get(cls, style.MUTED)
        present[cls] = color
        ax.scatter(
            sub["genomic_pos"],
            np.log10(sub["distance"].clip(lower=1)),
            s=point_size,
            color=color,
            alpha=0.8,
            edgecolors="none",
            zorder=3,
        )

    if len(chrom_spans) > 1:
        centers = [(c_start + c_end) / 2.0 for _, c_start, c_end in chrom_spans]
        labels = [
            c_name.removeprefix("chr").removeprefix("CHR")
            for c_name, _, _ in chrom_spans
        ]
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, fontsize=8 if len(labels) <= 24 else 7)
        ax.set_xlabel("Chromosome")
    else:
        ax.set_xlabel("Genomic position")

    title_base = f"Rainfall Plot{f' - {resolved_sample}' if resolved_sample else ''}"
    if detect_kataegis:
        thresh_y = float(np.log10(max(max_mean_distance, 1.0)))
        ax.axhline(
            thresh_y,
            color="#6b7280",
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
            zorder=1,
        )
        ymin, ymax = ax.get_ylim()
        y_top = max(ymax, thresh_y + 0.6)
        if not clusters.empty:
            ax.set_ylim(min(ymin, 0.0), y_top + 0.8)
            min_band_w = total_span * 0.008
            chrom_offset_lookup = {str(k): float(v) for k, v in offsets.items()}
            summaries: list[str] = []
            for idx, row in clusters.iterrows():
                c_offset = chrom_offset_lookup.get(str(row["chrom"]), 0.0)
                x0 = c_offset + float(row["start"])
                x1 = c_offset + float(row["end"])
                if x1 - x0 < min_band_w:
                    mid = (x0 + x1) / 2.0
                    x0 = max(0.0, mid - min_band_w / 2.0)
                    x1 = min(total_span, mid + min_band_w / 2.0)
                ax.axvspan(x0, x1, color="#dc2626", alpha=0.16, zorder=1)
                mid_x = (x0 + x1) / 2.0
                tag_parts = [
                    f"{row['cluster_id']}: {row['chrom']} (n={row['n_mutations']}"
                ]
                if pd.notna(row["dominant_substitution"]):
                    tag_parts.append(f", {row['dominant_substitution']}")
                if bool(row["apobec_like"]):
                    tag_parts.append(", APOBEC-like")
                tag_parts.append(")")
                tag = "".join(tag_parts)
                summaries.append(
                    f"{row['cluster_id']} {row['chrom']} n={row['n_mutations']}"
                )
                ax.annotate(
                    tag,
                    xy=(
                        mid_x,
                        max(
                            thresh_y,
                            float(
                                np.log10(max(row["mean_inter_mutation_distance"], 1.0))
                            ),
                        ),
                    ),
                    xytext=(mid_x, y_top + 0.15 + (idx % 2) * 0.35),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#991b1b",
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#dc2626",
                        "lw": 0.9,
                        "shrinkA": 2,
                        "shrinkB": 2,
                    },
                    zorder=4,
                )
            n_k = len(clusters)
            plural = "cluster" if n_k == 1 else "clusters"
            ax.set_title(
                f"{title_base}  |  {n_k} kataegis {plural} detected "
                f"({'; '.join(summaries[:3])}{'...' if n_k > 3 else ''})",
                fontsize=10.5,
            )
        else:
            dist_txt = (
                f"{int(max_mean_distance):,} bp"
                if float(max_mean_distance).is_integer()
                else f"{max_mean_distance:g} bp"
            )
            ax.set_title(
                f"{title_base}\n"
                f"No kataegis clusters detected "
                f"(min_mutations={min_mutations}, max_mean_distance<={dist_txt}; "
                f"{n_events_before_diff} variants)",
                fontsize=10,
            )
    else:
        ax.set_title(title_base)

    ax.set_ylabel("log10 inter-mutation distance")
    style.style_axes(ax)
    style.draw_legend_cards(legend_ax, {legend_title: present})
    fig.tight_layout()
    fig.kataegis_clusters = clusters
    fig.pymaftools_summary = {
        "plot_type": "rainfall",
        "sample": resolved_sample,
        "color_by": color_by,
        "n_events": n_events_before_diff,
        "n_plotted_distances": int(len(data)),
        "n_chromosomes": int(len(chrom_spans)),
        "detect_kataegis": bool(detect_kataegis),
        "n_kataegis_clusters": int(len(clusters)),
        "kataegis_clusters": [
            {
                "cluster_id": str(row["cluster_id"]),
                "chrom": str(row["chrom"]),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "n_mutations": int(row["n_mutations"]),
                "mean_inter_mutation_distance": float(
                    row["mean_inter_mutation_distance"]
                ),
                "dominant_substitution": (
                    str(row["dominant_substitution"])
                    if pd.notna(row["dominant_substitution"])
                    else None
                ),
                "apobec_like": bool(row["apobec_like"]),
            }
            for _, row in clusters.iterrows()
        ],
    }
    if return_clusters:
        return fig, clusters
    return fig


def infer_vaf(maf: pd.DataFrame, vaf_col: str | None = None) -> pd.Series:
    """Infer VAF from a named column or t_alt_count / t_depth."""
    candidates = (
        [vaf_col]
        if vaf_col
        else [
            "t_vaf",
            "VAF",
            "Tumor_VAF",
            "i_TumorVAF_WU",
            "i_TumorVAF",
            "tumor_vaf",
        ]
    )
    for col in candidates:
        if col and col in maf.columns:
            vaf = pd.to_numeric(maf[col], errors="coerce")
            return (
                vaf / 100 if vaf.max(skipna=True) and vaf.max(skipna=True) > 1 else vaf
            )
    if {"t_alt_count", "t_depth"}.issubset(maf.columns):
        alt = pd.to_numeric(maf["t_alt_count"], errors="coerce")
        depth = pd.to_numeric(maf["t_depth"], errors="coerce")
        return alt / depth.replace(0, np.nan)
    raise ValueError("No VAF column found and t_alt_count/t_depth are unavailable.")


def plot_vaf(
    maf: pd.DataFrame,
    vaf_col: str | None = None,
    by: str | None = None,
    figsize=(8, 5),
):
    """Draw VAF distribution, optionally grouped by a MAF column."""
    import matplotlib.pyplot as plt

    from . import style

    data = maf.copy()
    data["VAF"] = infer_vaf(data, vaf_col=vaf_col)
    data = data.dropna(subset=["VAF"])
    if data.empty:
        raise ValueError("No numeric VAF values available.")
    fig, ax = plt.subplots(figsize=figsize)
    if by and by in data.columns:
        for label, sub in data.groupby(by):
            ax.hist(sub["VAF"], bins=30, alpha=0.5, label=str(label), density=True)
        ax.legend(title=by, frameon=False)
        ax.set_ylabel("Density")
    else:
        ax.hist(data["VAF"], bins=30, color=style.ACCENT, alpha=0.9)
        ax.set_ylabel("Variants")
    ax.set_title("Variant Allele Frequency")
    ax.set_xlabel("VAF")
    style.style_axes(ax)
    fig.tight_layout()
    return fig


def _interaction_observation_frame(binary) -> pd.DataFrame:
    """Return the validated observed-cell frame for an interaction analysis.

    Without an attached :class:`ObservationMask` every cell is observed. With
    one, the mask must align exactly with the table and must not mark a present
    event as unobserved (the same contract as observed-denominator frequency).
    """
    observation_mask = getattr(binary, "observation_mask", None)
    if observation_mask is None:
        return pd.DataFrame(True, index=binary.index, columns=binary.columns)
    # Validates alignment and rejects present events marked unobserved.
    observation_mask.calculate_feature_frequency(binary)
    return observation_mask.to_frame().astype(bool)


_INTERACTION_COLUMNS = [
    "gene1",
    "gene2",
    "odds_ratio",
    "p_value",
    "both_mutated",
    "gene2_only",
    "gene1_only",
    "neither_mutated",
    "jointly_observed",
    "tested",
    "adjusted_p_value",
    "tested_family_size",
    "interaction",
    "is_significant",
]


def somatic_interactions(
    table,
    top: int = 25,
    alpha: float = 0.05,
    *,
    _warning_stacklevel: int = 2,
) -> pd.DataFrame:
    """Pairwise Fisher tests for co-occurrence / mutual exclusivity.

    The interaction hypothesis is undirected, so each selected gene pair is
    tested once. The Benjamini-Hochberg correction is applied to the unique
    unordered-pair family of *testable* pairs; self-pairs and mirrored matrix
    entries are not included. This is intentional and differs from the
    matrix-cell correction used by ``maftools::somaticInteractions``.

    When the table carries an :class:`~pymaftools.core.ObservationMask`, each
    pair's 2 x 2 table is built only from samples observed for *both* genes
    (reported as ``jointly_observed``); an unobserved cell is never counted as
    mutation absence. A pair with no jointly observed sample is reported with
    ``tested=False`` and missing statistics, and is excluded from the BH family.
    ``tested_family_size`` records the number of pairs in that family.
    """
    if isinstance(top, bool) or not isinstance(top, (int, np.integer)) or top < 1:
        raise ValueError(
            f"top must be a positive integer >= 1, got {top!r}. "
            "Pass e.g. top=25 to test pairs among the top 25 mutated genes."
        )
    top = int(top)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float, np.number))
        or not (0.0 < float(alpha) < 1.0)
    ):
        raise ValueError(
            f"alpha must be a significance threshold strictly between 0 and 1, "
            f"got {alpha!r}. Pass e.g. alpha=0.05."
        )
    alpha = float(alpha)

    binary = table.to_binary_table().astype(bool)
    observed = _interaction_observation_frame(binary)
    present = pd.DataFrame(binary).astype(bool) & observed
    genes = present.sum(axis=1).sort_values(ascending=False).head(top).index
    present = present.loc[genes]
    observed = observed.loc[genes]
    rows = []
    # ``combinations`` enumerates each biological (unordered) pair exactly once.
    for gene1, gene2 in combinations(genes, 2):
        jointly_observed = observed.loc[gene1] & observed.loc[gene2]
        gene1_mutated = present.loc[gene1] & jointly_observed
        gene2_mutated = present.loc[gene2] & jointly_observed
        gene1_absent = ~present.loc[gene1] & jointly_observed
        gene2_absent = ~present.loc[gene2] & jointly_observed
        both_mutated = int((gene1_mutated & gene2_mutated).sum())
        gene2_only = int((gene1_absent & gene2_mutated).sum())
        gene1_only = int((gene1_mutated & gene2_absent).sum())
        neither_mutated = int((gene1_absent & gene2_absent).sum())
        jointly_observed_count = int(jointly_observed.sum())
        tested = jointly_observed_count > 0
        if tested:
            odds_ratio, p_value = fisher_exact(
                [
                    [both_mutated, gene2_only],
                    [gene1_only, neither_mutated],
                ]
            )
        else:
            # No sample informs this pair; do not fabricate evidence.
            odds_ratio, p_value = np.nan, np.nan
        rows.append(
            {
                "gene1": gene1,
                "gene2": gene2,
                "odds_ratio": odds_ratio,
                "p_value": p_value,
                "both_mutated": both_mutated,
                "gene2_only": gene2_only,
                "gene1_only": gene1_only,
                "neither_mutated": neither_mutated,
                "jointly_observed": jointly_observed_count,
                "tested": tested,
            }
        )
    result = pd.DataFrame(rows)
    provenance = provenance_for_table(
        "somatic_interactions", table, {"top": top, "alpha": alpha}
    )
    if result.empty:
        return attach_provenance(pd.DataFrame(columns=_INTERACTION_COLUMNS), provenance)
    tested_pairs = result["tested"].to_numpy(dtype=bool)
    result["adjusted_p_value"] = np.nan
    if tested_pairs.any():
        # Correct only the distinct, testable interaction hypotheses.
        result.loc[tested_pairs, "adjusted_p_value"] = multipletests(
            result.loc[tested_pairs, "p_value"], method="fdr_bh"
        )[1]
    result["tested_family_size"] = int(tested_pairs.sum())
    result["interaction"] = np.select(
        [~tested_pairs, result["odds_ratio"] >= 1],
        ["Not tested", "Co-occurrence"],
        default="Mutual exclusivity",
    )
    result["is_significant"] = tested_pairs & (result["adjusted_p_value"] < alpha)
    _warn_interaction_pitfalls(
        result, n_samples=binary.shape[1], stacklevel=_warning_stacklevel + 1
    )
    result = result.sort_values("adjusted_p_value", na_position="last")
    return attach_provenance(result, provenance)


def _warn_interaction_pitfalls(
    result: pd.DataFrame, n_samples: int, *, stacklevel: int = 3
) -> None:
    """Warn (never alter results) when the interaction table is likely misleading.

    The numbers stay exactly as computed; these warnings exist because a
    plausible-looking heatmap is easy to over-interpret, by people and by
    automated agents alike.
    """
    import warnings

    if n_samples < 10:
        warnings.warn(
            f"somatic_interactions on only {n_samples} samples: Fisher tests "
            "have almost no power here; treat the table as descriptive.",
            UserWarning,
            stacklevel=stacklevel,
        )
    # Untested pairs (no jointly observed sample) are outside the tested family.
    tested = result[result["tested"]] if "tested" in result else result
    n_pairs = len(tested)
    if n_pairs >= 10:
        cooccur = tested["is_significant"] & tested["interaction"].eq("Co-occurrence")
        share = float(cooccur.mean())
        if share > 0.5:
            warnings.warn(
                f"{share:.0%} of {n_pairs} gene pairs are significant "
                "co-occurrences. Such pervasive co-occurrence usually reflects "
                "differences in mutation burden (hypermutated samples, long genes "
                "such as TTN/MUC16) rather than gene-gene interaction; consider "
                "stratifying by TMB or excluding hypermutators before "
                "interpreting individual pairs.",
                UserWarning,
                stacklevel=stacklevel,
            )


def plot_somatic_interactions(
    table,
    top: int = 25,
    alpha: float = 0.05,
    figsize=(8, 7),
    *,
    _warning_stacklevel: int = 2,
):
    """Draw a signed -log10(FDR) interaction heatmap.

    Each upper-triangle cell is ``sign(log OR) * -log10(BH-FDR)``: red for
    co-occurrence, blue for mutual exclusivity, and colour strength proportional
    to the *evidence*, not the effect size. A pair that was never co-mutated in
    a few samples (odds ratio 0) but is far from significant therefore stays
    pale instead of looking like the strongest exclusivity in the figure.
    ``*`` marks FDR < ``alpha``; grey cells were not tested (no jointly
    observed sample). Odds ratios themselves are in the returned statistics.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, pandas.DataFrame)
        The figure and the statistics from :func:`somatic_interactions`. The
        figure's :func:`~pymaftools.plot.summary.figure_summary` includes
        ``n_tested``, ``n_significant`` and ``n_infinite_or_zero_odds_ratio``.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    from . import style

    stats = somatic_interactions(
        table, top=top, alpha=alpha, _warning_stacklevel=_warning_stacklevel + 1
    )
    binary = table.to_binary_table().astype(bool)
    genes = (
        binary.astype(int)
        .sum(axis=1)
        .sort_values(ascending=False)
        .head(top)
        .index.tolist()
    )
    mat = pd.DataFrame(np.nan, index=genes, columns=genes)
    untested = pd.DataFrame(False, index=genes, columns=genes)
    for _, row in stats.iterrows():
        if not row["tested"]:
            # no jointly observed samples: drawn grey, never as evidence
            untested.loc[row["gene1"], row["gene2"]] = True
            untested.loc[row["gene2"], row["gene1"]] = True
            continue
        direction = 1.0 if row["odds_ratio"] >= 1 else -1.0
        # BH-adjusted p-values are never 0 in practice; floor avoids inf.
        evidence = -np.log10(max(float(row["adjusted_p_value"]), 1e-300))
        value = direction * evidence
        mat.loc[row["gene1"], row["gene2"]] = value
        mat.loc[row["gene2"], row["gene1"]] = value
    annotations = pd.DataFrame("", index=genes, columns=genes)
    for _, row in stats.loc[stats["is_significant"]].iterrows():
        annotations.loc[row["gene1"], row["gene2"]] = "*"
        annotations.loc[row["gene2"], row["gene1"]] = "*"
    threshold = -np.log10(alpha)
    observed_max = (
        float(np.nanmax(np.abs(mat.values))) if mat.notna().any().any() else 0.0
    )
    # Keep the significance threshold well inside the scale so "significant"
    # always reads as a clearly saturated colour.
    vmax = max(observed_max, 2 * threshold)
    triangle_mask = np.tril(np.ones(mat.shape, dtype=bool))
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    sns.heatmap(
        mat,
        cmap=style.DIVERGING_CMAP,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        square=True,
        mask=mat.isna() | triangle_mask,
        annot=annotations,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar=False,
    )
    if (untested.to_numpy() & ~triangle_mask).any():
        sns.heatmap(
            untested.where(untested & ~triangle_mask).astype(float),
            cmap=["#d9d9d9"],
            ax=ax,
            square=True,
            mask=~(untested.to_numpy() & ~triangle_mask),
            linewidths=0.5,
            linecolor="white",
            cbar=False,
        )
    ax.set_title("Somatic interactions  (* FDR < %.2g; grey = not tested)" % alpha)
    ax.tick_params(length=0)
    style.add_vertical_colorbar(
        ax,
        style.DIVERGING_CMAP,
        -vmax,
        vmax,
        label="signed -log10(FDR)\n(+ co-occurrence / - exclusivity)",
    )
    fig.tight_layout()
    fig.pymaftools_summary = _interaction_figure_summary(stats, alpha)
    return fig, stats


def _interaction_figure_summary(stats: pd.DataFrame, alpha: float) -> dict:
    """Domain facts for ``figure_summary`` of the interaction heatmap."""
    if stats.empty:
        return {
            "plot_type": "somatic_interactions",
            "n_pairs": 0,
            "warnings": ["no gene pairs to test: the heatmap is empty"],
        }
    tested = stats[stats["tested"]]
    odds = tested["odds_ratio"].astype(float)
    extreme = int((np.isinf(odds) | odds.eq(0)).sum())
    significant = tested[tested["is_significant"]]
    record = {
        "plot_type": "somatic_interactions",
        "color_scale": "sign(log odds ratio) * -log10(BH FDR)",
        "alpha": alpha,
        "n_pairs": int(len(stats)),
        "n_tested": int(len(tested)),
        "n_significant": int(len(significant)),
        "n_significant_cooccurrence": int(
            significant["interaction"].eq("Co-occurrence").sum()
        ),
        "n_significant_exclusivity": int(
            significant["interaction"].eq("Mutual exclusivity").sum()
        ),
        "n_infinite_or_zero_odds_ratio": extreme,
        "significant_pairs": [
            {
                "gene1": str(row.gene1),
                "gene2": str(row.gene2),
                "interaction": row.interaction,
                "adjusted_p_value": float(row.adjusted_p_value),
            }
            for row in significant.itertuples()
        ],
    }
    warnings_list = []
    if record["n_significant"] == 0:
        warnings_list.append(
            f"no gene pair reaches FDR < {alpha:g}; colours show weak evidence only"
        )
    if extreme:
        warnings_list.append(
            f"{extreme} of {len(tested)} tested pairs have an odds ratio of 0 or "
            "infinity (a zero cell, usually sparse counts); do not read them as "
            "strong effects"
        )
    if len(tested) >= 10 and record["n_significant_cooccurrence"] > 0.5 * len(tested):
        warnings_list.append(
            f"{record['n_significant_cooccurrence']} of {len(tested)} pairs are "
            "significant co-occurrences: likely mutation-burden confounding "
            "(hypermutators, long genes), not gene-gene interaction"
        )
    record["warnings"] = warnings_list
    return record


def compare_cohorts(
    cohort1_maf: pd.DataFrame,
    cohort2_maf: pd.DataFrame,
    cohort1_name: str | None = None,
    cohort2_name: str | None = None,
    min_mutated: int = 1,
    *,
    name1: str | None = None,
    name2: str | None = None,
) -> pd.DataFrame:
    """Compare gene mutation frequencies between two MAF cohorts."""
    if cohort1_name is not None and name1 is not None:
        raise ValueError(
            f"Cannot pass both cohort1_name={cohort1_name!r} and legacy alias "
            f"name1={name1!r}; pass only cohort1_name."
        )
    if cohort2_name is not None and name2 is not None:
        raise ValueError(
            f"Cannot pass both cohort2_name={cohort2_name!r} and legacy alias "
            f"name2={name2!r}; pass only cohort2_name."
        )
    cohort1_name = (
        cohort1_name
        if cohort1_name is not None
        else (name1 if name1 is not None else "Cohort1")
    )
    cohort2_name = (
        cohort2_name
        if cohort2_name is not None
        else (name2 if name2 is not None else "Cohort2")
    )
    if (
        isinstance(min_mutated, bool)
        or not isinstance(min_mutated, (int, np.integer))
        or min_mutated < 1
    ):
        raise ValueError(
            f"min_mutated must be a positive integer >= 1, got {min_mutated!r}. "
            "Pass min_mutated=1 to include every gene mutated in at least one cohort."
        )
    min_mutated = int(min_mutated)

    cohort1_sample_col = _get_sample_col(cohort1_maf)
    cohort2_sample_col = _get_sample_col(cohort2_maf)
    _require_columns(cohort1_maf, [cohort1_sample_col, "Hugo_Symbol"])
    _require_columns(cohort2_maf, [cohort2_sample_col, "Hugo_Symbol"])

    genes = sorted(
        set(cohort1_maf["Hugo_Symbol"].dropna())
        | set(cohort2_maf["Hugo_Symbol"].dropna())
    )
    c1_samples = cohort1_maf[cohort1_sample_col].dropna().unique().tolist()
    c2_samples = cohort2_maf[cohort2_sample_col].dropna().unique().tolist()
    cohort1_sample_count = len(c1_samples)
    cohort2_sample_count = len(c2_samples)
    cohort1_gene_counts = (
        cohort1_maf.drop_duplicates(["Hugo_Symbol", cohort1_sample_col])
        .groupby("Hugo_Symbol")
        .size()
    )
    cohort2_gene_counts = (
        cohort2_maf.drop_duplicates(["Hugo_Symbol", cohort2_sample_col])
        .groupby("Hugo_Symbol")
        .size()
    )
    rows = []
    for gene in genes:
        cohort1_mutated = int(cohort1_gene_counts.get(gene, 0))
        cohort2_mutated = int(cohort2_gene_counts.get(gene, 0))
        if max(cohort1_mutated, cohort2_mutated) < min_mutated:
            continue
        cohort1_wildtype = cohort1_sample_count - cohort1_mutated
        cohort2_wildtype = cohort2_sample_count - cohort2_mutated
        odds_ratio, p_value = fisher_exact(
            [
                [cohort1_mutated, cohort1_wildtype],
                [cohort2_mutated, cohort2_wildtype],
            ]
        )
        rows.append(
            {
                "gene": gene,
                f"{cohort1_name}_mutated": cohort1_mutated,
                f"{cohort2_name}_mutated": cohort2_mutated,
                f"{cohort1_name}_freq": (
                    cohort1_mutated / cohort1_sample_count
                    if cohort1_sample_count
                    else np.nan
                ),
                f"{cohort2_name}_freq": (
                    cohort2_mutated / cohort2_sample_count
                    if cohort2_sample_count
                    else np.nan
                ),
                "odds_ratio": odds_ratio,
                "p_value": p_value,
            }
        )
    provenance = Provenance(
        analysis="compare_cohorts",
        parameters={
            "cohort1_name": cohort1_name,
            "cohort2_name": cohort2_name,
            "min_mutated": min_mutated,
        },
        inputs=table_inputs(cohort1_maf) + table_inputs(cohort2_maf),
        sample_universe=sample_universe(
            [f"{cohort1_name}:{s}" for s in c1_samples]
            + [f"{cohort2_name}:{s}" for s in c2_samples],
            source="maf_cohorts",
        ),
    )
    result = pd.DataFrame(rows)
    if result.empty:
        empty = pd.DataFrame(
            columns=[
                "gene",
                f"{cohort1_name}_mutated",
                f"{cohort2_name}_mutated",
                f"{cohort1_name}_freq",
                f"{cohort2_name}_freq",
                "odds_ratio",
                "p_value",
                "adjusted_p_value",
            ]
        )
        return attach_provenance(empty, provenance)
    result["adjusted_p_value"] = multipletests(result["p_value"], method="fdr_bh")[1]
    result = result.sort_values("adjusted_p_value")
    return attach_provenance(result, provenance)


def plot_cohort_comparison_forest(
    compare_result: pd.DataFrame,
    top: int = 20,
    figsize=(7, 6),
):
    """Draw cohort-comparison odds ratios as a forest-style plot."""
    import matplotlib.pyplot as plt

    from . import style

    required_columns = {"gene", "odds_ratio", "adjusted_p_value"}
    missing_columns = sorted(required_columns.difference(compare_result.columns))
    if missing_columns:
        raise ValueError(
            "compare_result is missing required column(s): "
            f"{missing_columns}. Pass the DataFrame returned by "
            "maf.compare_cohorts(other)."
        )
    if compare_result.empty:
        raise ValueError("compare_result is empty.")
    data = compare_result.dropna(subset=["odds_ratio"]).head(top).copy()
    if data.empty:
        raise ValueError("No odds ratios available for plotting.")
    finite_odds = data.loc[
        np.isfinite(data["odds_ratio"]) & data["odds_ratio"].gt(0),
        "odds_ratio",
    ]
    upper_cap = max(10.0, finite_odds.max() * 10 if not finite_odds.empty else 10.0)
    lower_cap = min(0.1, finite_odds.min() / 10 if not finite_odds.empty else 0.1)
    data["plot_odds_ratio"] = data["odds_ratio"].replace(
        {np.inf: upper_cap, -np.inf: lower_cap}
    )
    data.loc[data["plot_odds_ratio"].eq(0), "plot_odds_ratio"] = lower_cap
    data = data.iloc[::-1]
    capped = data["odds_ratio"].apply(lambda v: not np.isfinite(v) or v == 0)
    fdr = -np.log10(data["adjusted_p_value"].clip(lower=np.finfo(float).tiny))
    fdr_max = float(fdr.max()) if len(fdr) else 1.0
    fdr_max = max(fdr_max, 1e-6)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        data["plot_odds_ratio"],
        data["gene"],
        c=fdr,
        cmap=style.SEQUENTIAL_CMAP,
        vmin=0,
        vmax=fdr_max,
        s=55,
        edgecolors=style.SPINE_COLOR,
        linewidths=0.5,
        zorder=3,
    )
    # Mark genes whose odds ratio was infinite/zero (capped to the axis edge).
    if capped.any():
        ax.scatter(
            data.loc[capped, "plot_odds_ratio"],
            data.loc[capped, "gene"],
            marker=">",
            color=style.ACCENT_2,
            s=55,
            zorder=4,
            label="OR = inf/0 (capped)",
        )
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.axvline(1, color=style.SPINE_COLOR, linestyle="--", linewidth=1, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio")
    ax.set_ylabel("Gene")
    ax.set_title("Differentially Mutated Genes")
    style.style_axes(ax)
    style.add_vertical_colorbar(
        ax, style.SEQUENTIAL_CMAP, 0.0, fdr_max, label="-log10 FDR"
    )
    fig.tight_layout()
    return fig


plot_forest = plot_cohort_comparison_forest


# --------------------------------------------------------------------------- #
#  MAF overview dashboard primitives (each draws on a given ax; the accessor /
#  overview composes them). Colours come from ColorManager's fixed palettes.
# --------------------------------------------------------------------------- #


def _functional_series(maf: pd.DataFrame) -> pd.Series:
    """Map ``Variant_Classification`` to a coarse functional group."""
    from .ColorManager import ColorManager

    return maf["Variant_Classification"].map(
        lambda c: ColorManager.FUNCTIONAL_GROUP.get(c, "Other")
    )


def _functional_order(present) -> list[str]:
    from .ColorManager import ColorManager

    return [g for g in ColorManager.FUNCTIONAL_ORDER if g in set(present)]


def summary_stats(maf: pd.DataFrame) -> dict:
    """Cohort-level summary numbers for the overview header (no plotting)."""
    sample_col = _get_sample_col(maf)
    _require_columns(maf, [sample_col, "Hugo_Symbol", "Variant_Classification"])
    per_sample = maf.groupby(sample_col).size()
    func = _functional_series(maf)
    stats = {
        "samples": int(maf[sample_col].nunique()),
        "genes_mutated": int(maf["Hugo_Symbol"].nunique()),
        "variants": int(len(maf)),
        "median_per_sample": float(per_sample.median()) if len(per_sample) else 0.0,
        "missense_fraction": float((func == "Missense").mean()) if len(func) else 0.0,
    }
    if "Variant_Type" in maf.columns:
        stats["snv_fraction"] = float(maf["Variant_Type"].eq("SNP").mean())
    try:
        titv = summarize_titv(maf).sum()
        ti = titv.get("C>T", 0) + titv.get("T>C", 0)
        tv = titv.sum() - ti
        stats["titv_ratio"] = float(ti / tv) if tv else float("nan")
    except (ValueError, KeyError):
        pass
    return stats


def plot_sample_burden(
    maf: pd.DataFrame, ax=None, figsize=(12, 4), show_sample_labels: bool | None = None
):
    """Per-sample mutation burden, stacked by functional consequence and sorted
    by total burden, with a median line."""
    from .ColorManager import ColorManager
    from . import style

    sample_col = _get_sample_col(maf)
    _require_columns(maf, [sample_col, "Variant_Classification"])
    func = _functional_series(maf)
    burden = pd.crosstab(maf[sample_col], func)
    order = _functional_order(burden.columns)
    burden = burden[order]
    burden = burden.loc[burden.sum(axis=1).sort_values(ascending=False).index]

    standalone = ax is None
    legend_ax = None
    if standalone:
        fig, ax, legend_ax = style.fig_with_legend(figsize, legend_width=0.16)
    burden.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=1.0,
        legend=False,
        color=[ColorManager.FUNCTIONAL_CMAP[g] for g in order],
    )
    median = burden.sum(axis=1).median()
    ax.axhline(median, color=style.SPINE_COLOR, linestyle="--", linewidth=1)
    ax.text(
        len(burden) - 0.5,
        median,
        f"median {median:.0f} ",
        va="bottom",
        ha="right",
        fontsize=11,
        color=style.SPINE_COLOR,
    )
    ax.set_title("Mutation burden per sample")
    ax.set_xlabel(f"sample (n={len(burden)}, sorted)")
    ax.set_ylabel("variants")
    ax.margins(x=0.005)
    style.style_axes(ax)
    style.apply_sample_xticklabels(ax, burden.index, show=show_sample_labels)
    if standalone:
        style.draw_legend_cards(
            legend_ax,
            {"Consequence": {g: ColorManager.FUNCTIONAL_CMAP[g] for g in order}},
        )
        return ax.figure
    return ax


def plot_mutation_composition(maf: pd.DataFrame, ax=None, figsize=(8, 4)):
    """Nested composition: one bar per ``Variant_Type`` (SNP/DEL/INS), stacked by
    functional consequence — variant level + consequence level in one panel."""
    from .ColorManager import ColorManager
    from . import style

    _require_columns(maf, ["Variant_Type", "Variant_Classification"])
    func = _functional_series(maf)
    comp = pd.crosstab(maf["Variant_Type"], func)
    order = _functional_order(comp.columns)
    comp = comp[order]
    comp = comp.loc[comp.sum(axis=1).sort_values(ascending=False).index]

    standalone = ax is None
    legend_ax = None
    if standalone:
        fig, ax, legend_ax = style.fig_with_legend(figsize, legend_width=0.2)
    comp.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=0.7,
        legend=False,
        color=[ColorManager.FUNCTIONAL_CMAP[g] for g in order],
    )
    ax.set_title("Mutation composition")
    ax.set_xlabel("variant type")
    ax.set_ylabel("variants")
    ax.tick_params(axis="x", labelrotation=0)
    style.style_axes(ax)
    if standalone:
        style.draw_legend_cards(
            legend_ax,
            {"Consequence": {g: ColorManager.FUNCTIONAL_CMAP[g] for g in order}},
        )
        return ax.figure
    return ax


def plot_snv_spectrum(maf: pd.DataFrame, ax=None, figsize=(7, 4)):
    """Cohort-aggregate six-class SNV substitution spectrum (proportions), with
    the Ti/Tv ratio annotated."""
    import matplotlib.pyplot as plt

    from .ColorManager import ColorManager
    from . import style

    counts = summarize_titv(maf).sum()
    if counts.sum() == 0:
        raise ValueError("No SNP substitutions available for SNV spectrum.")
    prop = counts / counts.sum()
    ti = counts.get("C>T", 0) + counts.get("T>C", 0)
    tv = counts.sum() - ti
    titv_ratio = ti / tv if tv else float("nan")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        BASE_CHANGE_ORDER,
        [prop.get(c, 0) for c in BASE_CHANGE_ORDER],
        color=[ColorManager.TITV_CMAP[c] for c in BASE_CHANGE_ORDER],
        width=0.8,
    )
    ax.set_title("SNV spectrum  (Ti/Tv = %.2f)" % titv_ratio)
    ax.set_ylabel("fraction")
    ax.set_ylim(0, max(0.05, float(prop.max()) * 1.15))
    style.style_axes(ax)
    if standalone:
        ax.figure.tight_layout()
        return ax.figure
    return ax


def plot_gene_recurrence(maf: pd.DataFrame, ax=None, figsize=(7, 4)):
    """Gene recurrence structure: how many genes are mutated in N samples
    (long-tail private vs recurrent driver structure)."""
    import matplotlib.pyplot as plt

    from . import style

    sample_col = _get_sample_col(maf)
    _require_columns(maf, [sample_col, "Hugo_Symbol"])
    gene_samples = (
        maf.drop_duplicates(["Hugo_Symbol", sample_col]).groupby("Hugo_Symbol").size()
    )
    recurrence = gene_samples.value_counts().sort_index()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    ax.bar(recurrence.index, recurrence.values, color=style.ACCENT, width=0.8)
    ax.set_yscale("log")
    ax.set_title("Gene recurrence")
    ax.set_xlabel("samples mutated")
    ax.set_ylabel("genes (log)")
    style.style_axes(ax)
    if standalone:
        ax.figure.tight_layout()
        return ax.figure
    return ax


def plot_top_genes(maf: pd.DataFrame, ax=None, top: int = 10, figsize=(7, 4)):
    """Top recurrently-mutated genes as horizontal bars, stacked by functional
    consequence (shares the overview's Consequence palette/legend)."""
    from .ColorManager import ColorManager
    from . import style

    sample_col = _get_sample_col(maf)
    _require_columns(maf, [sample_col, "Hugo_Symbol", "Variant_Classification"])
    n_samples = maf[sample_col].nunique()
    ranked = (
        maf.drop_duplicates(["Hugo_Symbol", sample_col])
        .groupby("Hugo_Symbol")
        .size()
        .sort_values(ascending=False)
        .head(top)
    )
    genes = ranked.index
    sub = maf[maf["Hugo_Symbol"].isin(genes)].copy()
    sub["_func"] = _functional_series(sub)
    mat = pd.crosstab(sub["Hugo_Symbol"], sub["_func"]).reindex(genes).fillna(0)
    order = _functional_order(mat.columns)
    mat = mat[order].iloc[::-1]  # highest-ranked gene on top

    standalone = ax is None
    legend_ax = None
    if standalone:
        fig, ax, legend_ax = style.fig_with_legend(figsize, legend_width=0.2)
    mat.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        width=0.75,
        legend=False,
        color=[ColorManager.FUNCTIONAL_CMAP[g] for g in order],
    )
    # Frequency label (% of samples mutated) at the end of each bar.
    for gene, y in zip(mat.index, range(len(mat))):
        freq = ranked[gene] / n_samples if n_samples else 0
        ax.text(
            mat.loc[gene].sum(),
            y,
            f"  {freq * 100:.0f}%",
            va="center",
            ha="left",
            fontsize=10,
            color=style.TEXT_COLOR,
        )
    total_genes = maf["Hugo_Symbol"].nunique()
    ax.set_title(f"Top {len(genes)} of {total_genes:,} mutated genes")
    ax.set_xlabel("variants")
    ax.margins(x=0.12)
    style.style_axes(ax)
    if standalone:
        style.draw_legend_cards(
            legend_ax,
            {"Consequence": {g: ColorManager.FUNCTIONAL_CMAP[g] for g in order}},
        )
        return ax.figure
    return ax


def plot_overview(maf: pd.DataFrame, figsize=(15, 10)):
    """MAF-only overview dashboard: a summary header plus complementary panels
    at the sample / variant / nucleotide / gene levels, composed from the
    primitives above. Colours/legend come from ColorManager's fixed palettes."""
    import matplotlib.pyplot as plt

    from .ColorManager import ColorManager
    from . import style

    _require_columns(maf, ["Variant_Classification", "Hugo_Symbol"])
    stats = summary_stats(maf)

    # Bump the base font sizes — this is a large multi-panel figure, so the
    # matplotlib defaults read too small.
    rc = {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2,
            4,
            height_ratios=[1.0, 1.0],
            # composition is 3 vertical bars -> narrower; top_genes spans the
            # right two cells (widest) so its horizontal bars get room.
            width_ratios=[0.55, 0.95, 0.95, 0.55],
            hspace=0.38,
            wspace=0.4,
            top=0.86,
            bottom=0.09,
            left=0.06,
            right=0.99,
        )
        plot_sample_burden(
            maf, ax=fig.add_subplot(gs[0, 0:3]), show_sample_labels=False
        )
        plot_mutation_composition(maf, ax=fig.add_subplot(gs[1, 0]))
        plot_snv_spectrum(maf, ax=fig.add_subplot(gs[1, 1]))
        plot_top_genes(maf, ax=fig.add_subplot(gs[1, 2:4]), top=10)

        # Top-right corner: shared Consequence legend, filling the white triangle
        # left by the burden plot's sorted taper (each panel annotates its own
        # numbers; cohort-level numbers go in the subtitle).
        legend_ax = fig.add_subplot(gs[0, 3])
        legend_ax.axis("off")
        func_present = _functional_order(_functional_series(maf).unique())
        style.draw_legend_cards(
            legend_ax,
            {"Consequence": {g: ColorManager.FUNCTIONAL_CMAP[g] for g in func_present}},
        )

        # Title + cohort headline subtitle (full width -> never clipped).
        subtitle = (
            f"{stats['samples']} samples  ·  {stats['variants']:,} variants  ·  "
            f"{stats['genes_mutated']:,} genes mutated"
        )
        if "snv_fraction" in stats:
            subtitle += f"  ·  {stats['snv_fraction'] * 100:.0f}% SNV"
        fig.suptitle(
            "MAF Overview",
            fontsize=16,
            fontweight="semibold",
            x=0.06,
            ha="left",
            y=0.97,
        )
        fig.text(0.06, 0.915, subtitle, fontsize=11, ha="left", color=style.TEXT_COLOR)
    return fig
