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

    _require_columns(maf, ["Variant_Classification", "Variant_Type", "Hugo_Symbol"])
    burden = mutation_burden_by_class(maf)
    variant_class = maf["Variant_Classification"].value_counts()
    variant_type = maf["Variant_Type"].value_counts()
    top_genes = top_mutated_genes(maf, top=top)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax = axes[0, 0]
    burden.plot(kind="bar", stacked=True, ax=ax, width=0.9, legend=False)
    ax.set_title("Mutation Burden by Sample")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Mutations")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)

    ax = axes[0, 1]
    variant_class.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Variant Classification")
    ax.set_xlabel("Mutations")

    ax = axes[1, 0]
    variant_type.sort_values().plot(kind="barh", ax=ax, color="#4C78A8")
    ax.set_title("Variant Type")
    ax.set_xlabel("Mutations")

    ax = axes[1, 1]
    top_genes.sort_values().plot(kind="barh", ax=ax, color="#F58518")
    ax.set_title(f"Top {len(top_genes)} Mutated Genes")
    ax.set_xlabel("Mutated samples")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(4, len(labels)),
            title="Classification",
        )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
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

    counts = summarize_titv(maf)
    if counts.empty:
        raise ValueError("No SNP substitutions available for Ti/Tv plot.")
    plot_data = (
        counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        if fraction
        else counts
    )
    ax = plot_data.plot(kind="bar", stacked=True, figsize=figsize, width=0.9)
    ax.set_title("Transition / Transversion Composition")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Fraction" if fraction else "SNP count")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.legend(title="Base change", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.figure.tight_layout()
    return ax.figure


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

    fig, ax = plt.subplots(figsize=figsize)
    for cls, sub in data.groupby("Variant_Classification"):
        ax.scatter(
            sub["genomic_pos"],
            np.log10(sub["distance"].clip(lower=1)),
            s=point_size,
            label=cls,
            alpha=0.75,
        )
    ax.set_title(f"Rainfall Plot{f' - {sample}' if sample else ''}")
    ax.set_xlabel("Genomic position")
    ax.set_ylabel("log10 inter-mutation distance")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
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

    data = maf.copy()
    data["VAF"] = infer_vaf(data, vaf_col=vaf_col)
    data = data.dropna(subset=["VAF"])
    if data.empty:
        raise ValueError("No numeric VAF values available.")
    fig, ax = plt.subplots(figsize=figsize)
    if by and by in data.columns:
        for label, sub in data.groupby(by):
            ax.hist(sub["VAF"], bins=30, alpha=0.5, label=str(label), density=True)
        ax.legend(title=by)
    else:
        ax.hist(data["VAF"], bins=30, color="#4C78A8", alpha=0.8)
        ax.set_ylabel("Variants")
    ax.set_title("Variant Allele Frequency")
    ax.set_xlabel("VAF")
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
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        mat,
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=True,
        mask=mat.isna(),
        annot=annotations,
        fmt="",
        cbar_kws={"label": "log2 odds ratio"},
    )
    ax.set_title("Somatic Interactions")
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
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        data["plot_odds_ratio"],
        data["gene"],
        c=-np.log10(data["adjusted_p_value"].clip(lower=np.finfo(float).tiny)),
        cmap="viridis",
    )
    ax.axvline(1, color="0.4", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio")
    ax.set_ylabel("Gene")
    ax.set_title("Differentially Mutated Genes")
    fig.tight_layout()
    return fig


plot_forest = plot_cohort_comparison_forest
