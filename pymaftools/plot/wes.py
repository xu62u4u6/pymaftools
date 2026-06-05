"""Common WES/MAF summary plots and statistics."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


BASE_CHANGE_ORDER = [
    "C>A", "C>G", "C>T", "T>A", "T>C", "T>G",
]


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


def plot_maf_summary(maf: pd.DataFrame, top: int = 20, figsize=(12, 9)):
    """Draw a compact MAF summary dashboard."""
    import matplotlib.pyplot as plt

    from .ColorManager import ColorManager
    from . import style

    _require_columns(maf, ["Variant_Classification", "Variant_Type", "Hugo_Symbol"])
    burden = mutation_burden_by_class(maf)
    variant_class = maf["Variant_Classification"].value_counts()
    variant_type = maf["Variant_Type"].value_counts()
    top_genes = top_mutated_genes(maf, top=top)

    # One colour per variant classification, shared by the stacked burden and
    # the classification bars (so a class is the same colour in both).
    def class_color(name):
        return ColorManager.ALL_MUTATION_CMAP.get(name, style.MUTED)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.32])
    ax_burden = fig.add_subplot(gs[0, 0])
    ax_class = fig.add_subplot(gs[0, 1])
    ax_type = fig.add_subplot(gs[1, 0])
    ax_genes = fig.add_subplot(gs[1, 1])
    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.axis("off")

    burden.plot(
        kind="bar",
        stacked=True,
        ax=ax_burden,
        width=0.9,
        legend=False,
        color=[class_color(c) for c in burden.columns],
    )
    ax_burden.set_title("Mutation Burden by Sample")
    ax_burden.set_xlabel("Sample")
    ax_burden.set_ylabel("Mutations")
    ax_burden.tick_params(axis="x", labelrotation=90, labelsize=7)

    vc = variant_class.sort_values()
    vc.plot(kind="barh", ax=ax_class, color=[class_color(c) for c in vc.index])
    ax_class.set_title("Variant Classification")
    ax_class.set_xlabel("Mutations")

    # Plain count panels carry no category meaning -> one neutral house accent
    # (not a second arbitrary colour competing with the classification palette).
    variant_type.sort_values().plot(kind="barh", ax=ax_type, color=style.ACCENT)
    ax_type.set_title("Variant Type")
    ax_type.set_xlabel("Mutations")

    top_genes.sort_values().plot(kind="barh", ax=ax_genes, color=style.ACCENT)
    ax_genes.set_title(f"Top {len(top_genes)} Mutated Genes")
    ax_genes.set_xlabel("Mutated samples")

    for ax in (ax_burden, ax_class, ax_type, ax_genes):
        style.style_axes(ax)

    style.draw_legend_cards(
        legend_ax,
        {"Classification": {c: class_color(c) for c in variant_class.index}},
    )
    fig.tight_layout()
    return fig


def summarize_titv(maf: pd.DataFrame) -> pd.DataFrame:
    """Return per-sample six-class substitution counts in pyrimidine context."""
    sample_col = _get_sample_col(maf)
    _require_columns(
        maf, [sample_col, "Variant_Type", "Reference_Allele", "Tumor_Seq_Allele2"]
    )
    snp = maf.loc[maf["Variant_Type"].eq("SNP")].copy()
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}

    def normalize(row):
        ref = str(row["Reference_Allele"]).upper()
        alt = str(row["Tumor_Seq_Allele2"]).upper()
        if ref not in comp or alt not in comp or ref == alt:
            return np.nan
        if ref in {"A", "G"}:
            ref, alt = comp[ref], comp[alt]
        return f"{ref}>{alt}"

    snp["Base_Change"] = snp.apply(normalize, axis=1)
    counts = pd.crosstab(snp[sample_col], snp["Base_Change"])
    return counts.reindex(columns=BASE_CHANGE_ORDER, fill_value=0)


def plot_titv(maf: pd.DataFrame, fraction: bool = True, figsize=(10, 5)):
    """Draw Ti/Tv six-class substitution composition per sample."""
    import matplotlib.pyplot as plt

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
    ax.set_xlabel("Sample")
    ax.set_ylabel("Fraction" if fraction else "SNP count")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.margins(x=0.01)
    style.style_axes(ax)
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


def plot_rainfall(
    maf: pd.DataFrame,
    sample: str | None = None,
    figsize=(12, 4),
    point_size: float = 10,
):
    """Draw inter-mutation distance across genomic coordinates."""
    import matplotlib.pyplot as plt

    from .ColorManager import ColorManager
    from . import style

    sample_col = _get_sample_col(maf)
    _require_columns(
        maf,
        [sample_col, "Chromosome", "Start_Position", "Variant_Classification"],
    )
    data = maf.copy()
    if sample is not None:
        data = data.loc[data[sample_col].eq(sample)].copy()
    elif data[sample_col].nunique() > 1:
        sample = str(data[sample_col].iloc[0])
        data = data.loc[data[sample_col].eq(sample)].copy()
    if data.empty:
        raise ValueError("No variants available for rainfall plot.")

    data["Start_Position"] = pd.to_numeric(data["Start_Position"], errors="coerce")
    data = data.dropna(subset=["Start_Position"])
    chroms = sorted(data["Chromosome"].dropna().unique(), key=_chrom_sort_key)
    offsets = {}
    offset = 0
    for chrom in chroms:
        offsets[chrom] = offset
        max_pos = data.loc[data["Chromosome"].eq(chrom), "Start_Position"].max()
        offset += int(max_pos) + 1
    data["genomic_pos"] = data.apply(
        lambda row: offsets[row["Chromosome"]] + row["Start_Position"],
        axis=1,
    )
    data["_chrom_order"] = data["Chromosome"].map(lambda chrom: _chrom_sort_key(chrom)[0])
    data["_chrom_label"] = data["Chromosome"].map(lambda chrom: _chrom_sort_key(chrom)[1])
    data = data.sort_values(["_chrom_order", "_chrom_label", "Start_Position"])
    data["distance"] = data.groupby("Chromosome")["Start_Position"].diff()
    data = data.dropna(subset=["distance"])
    if data.empty:
        raise ValueError("Rainfall plot requires at least two variants on one chromosome.")

    fig, ax, legend_ax = style.fig_with_legend(figsize, legend_width=0.16)
    present = {}
    for cls, sub in data.groupby("Variant_Classification"):
        color = ColorManager.ALL_MUTATION_CMAP.get(cls, style.MUTED)
        present[cls] = color
        ax.scatter(
            sub["genomic_pos"],
            np.log10(sub["distance"].clip(lower=1)),
            s=point_size,
            color=color,
            alpha=0.8,
            edgecolors="none",
        )
    ax.set_title(f"Rainfall Plot{f' - {sample}' if sample else ''}")
    ax.set_xlabel("Genomic position")
    ax.set_ylabel("log10 inter-mutation distance")
    style.style_axes(ax)
    style.draw_legend_cards(legend_ax, {"Mutation": present})
    fig.tight_layout()
    return fig


def infer_vaf(maf: pd.DataFrame, vaf_col: str | None = None) -> pd.Series:
    """Infer VAF from a named column or t_alt_count / t_depth."""
    candidates = [vaf_col] if vaf_col else [
        "t_vaf",
        "VAF",
        "Tumor_VAF",
        "i_TumorVAF_WU",
        "i_TumorVAF",
        "tumor_vaf",
    ]
    for col in candidates:
        if col and col in maf.columns:
            vaf = pd.to_numeric(maf[col], errors="coerce")
            return (
                vaf / 100
                if vaf.max(skipna=True) and vaf.max(skipna=True) > 1
                else vaf
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


def somatic_interactions(table, top: int = 25, alpha: float = 0.05) -> pd.DataFrame:
    """Pairwise Fisher tests for co-occurrence / mutual exclusivity."""
    binary = table.to_binary_table().astype(bool)
    genes = binary.astype(int).sum(axis=1).sort_values(ascending=False).head(top).index
    binary = binary.loc[genes]
    rows = []
    for gene1, gene2 in combinations(binary.index, 2):
        both_mutated = int((binary.loc[gene1] & binary.loc[gene2]).sum())
        gene2_only = int((~binary.loc[gene1] & binary.loc[gene2]).sum())
        gene1_only = int((binary.loc[gene1] & ~binary.loc[gene2]).sum())
        neither_mutated = int((~binary.loc[gene1] & ~binary.loc[gene2]).sum())
        odds_ratio, p_value = fisher_exact(
            [
                [both_mutated, gene2_only],
                [gene1_only, neither_mutated],
            ]
        )
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
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["adjusted_p_value"] = multipletests(result["p_value"], method="fdr_bh")[1]
    result["interaction"] = np.where(
        result["odds_ratio"] >= 1,
        "Co-occurrence",
        "Mutual exclusivity",
    )
    result["is_significant"] = result["adjusted_p_value"] < alpha
    return result.sort_values("adjusted_p_value")


def plot_somatic_interactions(
    table,
    top: int = 25,
    alpha: float = 0.05,
    figsize=(8, 7),
):
    """Draw a signed -log10(FDR) interaction heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    from . import style

    stats = somatic_interactions(table, top=top, alpha=alpha)
    binary = table.to_binary_table().astype(bool)
    genes = (
        binary.astype(int)
        .sum(axis=1)
        .sort_values(ascending=False)
        .head(top)
        .index
        .tolist()
    )
    mat = pd.DataFrame(np.nan, index=genes, columns=genes)
    for _, row in stats.iterrows():
        if np.isinf(row["odds_ratio"]):
            value = 6.0
        elif row["odds_ratio"] == 0:
            value = -6.0
        else:
            value = np.log2(row["odds_ratio"])
        mat.loc[row["gene1"], row["gene2"]] = value
        mat.loc[row["gene2"], row["gene1"]] = value
    annotations = pd.DataFrame("", index=genes, columns=genes)
    for _, row in stats.loc[stats["is_significant"]].iterrows():
        annotations.loc[row["gene1"], row["gene2"]] = "*"
        annotations.loc[row["gene2"], row["gene1"]] = "*"
    vmax = float(np.nanmax(np.abs(mat.values))) if mat.notna().any().any() else 1.0
    vmax = max(vmax, 1.0)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        mat,
        cmap=style.DIVERGING_CMAP,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        square=True,
        mask=mat.isna(),
        annot=annotations,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar=False,
    )
    ax.set_title("Somatic Interactions  (* FDR < %.2g)" % alpha)
    ax.tick_params(length=0)
    style.add_vertical_colorbar(
        ax, style.DIVERGING_CMAP, -vmax, vmax, label="log2 OR"
    )
    fig.tight_layout()
    return fig, stats


def compare_cohorts(
    cohort1_maf: pd.DataFrame,
    cohort2_maf: pd.DataFrame,
    cohort1_name: str = "Cohort1",
    cohort2_name: str = "Cohort2",
    min_mutated: int = 1,
    *,
    name1: str | None = None,
    name2: str | None = None,
) -> pd.DataFrame:
    """Compare gene mutation frequencies between two MAF cohorts."""
    if name1 is not None:
        cohort1_name = name1
    if name2 is not None:
        cohort2_name = name2

    cohort1_sample_col = _get_sample_col(cohort1_maf)
    cohort2_sample_col = _get_sample_col(cohort2_maf)
    _require_columns(cohort1_maf, [cohort1_sample_col, "Hugo_Symbol"])
    _require_columns(cohort2_maf, [cohort2_sample_col, "Hugo_Symbol"])

    genes = sorted(
        set(cohort1_maf["Hugo_Symbol"].dropna())
        | set(cohort2_maf["Hugo_Symbol"].dropna())
    )
    cohort1_sample_count = cohort1_maf[cohort1_sample_col].nunique()
    cohort2_sample_count = cohort2_maf[cohort2_sample_col].nunique()
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
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["adjusted_p_value"] = multipletests(result["p_value"], method="fdr_bh")[1]
    return result.sort_values("adjusted_p_value")


def plot_cohort_comparison_forest(
    compare_result: pd.DataFrame,
    top: int = 20,
    figsize=(7, 6),
):
    """Draw cohort-comparison odds ratios as a forest-style plot."""
    import matplotlib.pyplot as plt

    from . import style

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
