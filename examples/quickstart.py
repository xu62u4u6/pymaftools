"""Five-minute quickstart: your MAF -> checked events -> gene table -> oncoplot.

Run on the bundled example::

    python examples/quickstart.py

or on your own file (GDC, cBioPortal, vcf2maf, Funcotator, maftools ...)::

    python examples/quickstart.py my_cohort.maf.gz --samples samples.txt

``samples.txt`` (optional, one ID per line) is the full list of sequenced
samples, including those with no called mutations; without it frequencies are
computed over samples that appear in the MAF only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from pymaftools import MAF, example_maf_path  # noqa: E402


def run(maf_path: Path, samples_path: Path | None, out: Path, top: int = 20) -> Path:
    # 1. Read. Problems that would silently change results (blank genes,
    #    duplicate events, mixed genome builds ...) raise here with a fix;
    #    pass validate="warn" to load anyway.
    maf = MAF.read_maf(maf_path)

    # 2. Check what the reader detected and changed.
    print(maf.ingest_report)

    # 3. Gene x sample table of protein-altering events, over the full sample
    #    universe. Filtering first would otherwise drop samples that only have
    #    silent/non-coding events and shrink every frequency's denominator.
    if samples_path is not None:
        samples = [line.strip() for line in samples_path.read_text().splitlines()]
        samples = [sample for sample in samples if sample]
    else:
        samples = maf["sample_ID"].unique()
    events = maf.filter_maf(MAF.nonsynonymous_types)
    table = (
        events.to_gene_table(samples=samples)
        .add_freq()
        .sort_features(by="freq")
        .sort_samples_by_mutations()
    )

    # 4. Oncoplot of the most frequently mutated genes.
    plot_table = table.subset(features=table.index[:top])
    oncoplot = (
        plot_table.plot.oncoplot(figsize=(12, 7))
        .main()
        .add_bar("mutations_count", side="top")
        .add_freq(side="right")
        .render()
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    oncoplot.save(out, dpi=150)
    print(f"oncoplot saved to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("maf", nargs="?", type=Path, default=example_maf_path())
    parser.add_argument("--samples", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/oncoplot.png"))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    run(args.maf, args.samples, args.out, args.top)


if __name__ == "__main__":
    main()
