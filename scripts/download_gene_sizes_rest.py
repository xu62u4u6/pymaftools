"""Fallback gene-size cache builder using the Ensembl REST API.

Produces the same ``pymaftools/data/ensembl_gene_sizes.tsv`` schema as
``download_gene_sizes.py`` (hugo_symbol, ensembl_gene_id, transcript_length,
cds_length), but sources it from the Ensembl REST API instead of BioMart. Use
this when BioMart is in maintenance (the martservice endpoint returns a
"Service unavailable" page) and the genome-wide dump cannot be fetched.

Scope: builds sizes for the genes in the bundled fixture by default (the genes
the docs/demo touch), because while BioMart is down the REST backend is under
load and only tolerates small ``/lookup/symbol`` batches — a genome-wide pass
times out. Pass ``--genes`` to cover a different set. A partial cache is safe:
``get_exon_size`` fetches any gene missing from it rather than returning NaN.
Refresh the full cache from BioMart (``download_gene_sizes.py --force``) once it
is back online.

Two lean steps (no ``expand`` — avoids pulling every exon of giant genes like
TTN, which times the request out):

1. ``POST /lookup/symbol`` per batch -> each gene's ``canonical_transcript`` id.
2. ``POST /lookup/id``  per batch -> that transcript's ``length`` (= BioMart
   ``transcript_length`` for the canonical transcript).

``cds_length`` is not available from these lean lookups, so it is written as
empty (NaN). The default ``add_exon_size`` metric is ``transcript_length``, so
this is sufficient for size-based gene grouping; refresh from BioMart (the
``cds_length`` source) once it is back online.

Usage:
    uv run python scripts/download_gene_sizes_rest.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "pymaftools" / "data"
FIXTURE = DATA_DIR / "example_tcga_lung_mutation_grouped.h5"
OUT = DATA_DIR / "ensembl_gene_sizes.tsv"
REST = "https://rest.ensembl.org"
BATCH = 25  # small: the REST backend times out on large symbol batches


def _post(path: str, payload: dict, retries: int = 4) -> dict:
    """POST JSON to the REST API, retrying on timeout / 429 / 5xx."""
    for attempt in range(retries):
        try:
            r = requests.post(
                REST + path,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                data=json.dumps(payload),
                timeout=60,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
        except requests.exceptions.RequestException:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"REST POST {path} failed after {retries} retries.")


def _batched(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--genes",
        nargs="+",
        help="Symbols to fetch (default: genes in the bundled fixture).",
    )
    args = parser.parse_args()

    if args.genes:
        symbols = args.genes
    else:
        import pymaftools

        symbols = list(pymaftools.read_h5(FIXTURE).feature_metadata.index)
    print(f"[rest] {len(symbols)} symbols")

    # Step 1: symbol -> canonical transcript id (+ gene id)
    sym_to_tx: dict[str, str] = {}
    sym_to_gene: dict[str, str] = {}
    for bi, batch in enumerate(_batched(symbols, BATCH)):
        d = _post("/lookup/symbol/homo_sapiens", {"symbols": batch})
        for sym, obj in d.items():
            if not obj:
                continue
            ct = obj.get("canonical_transcript")
            if ct:
                sym_to_tx[sym] = ct.split(".")[0]  # strip version
                sym_to_gene[sym] = obj.get("id")
        print(f"[rest] step1 batch {bi + 1}: {len(sym_to_tx)} canonical tx so far")

    # Step 2: canonical transcript id -> length
    tx_ids = list(set(sym_to_tx.values()))
    tx_len: dict[str, int] = {}
    for bi, batch in enumerate(_batched(tx_ids, BATCH)):
        d = _post("/lookup/id", {"ids": batch})
        for tid, obj in d.items():
            if obj and obj.get("length") is not None:
                tx_len[tid] = int(obj["length"])
        print(f"[rest] step2 batch {bi + 1}: {len(tx_len)} lengths so far")

    rows = [
        {
            "hugo_symbol": sym,
            "ensembl_gene_id": sym_to_gene.get(sym),
            "transcript_length": tx_len.get(tx),
            "cds_length": pd.NA,  # not available from lean REST lookups
        }
        for sym, tx in sym_to_tx.items()
        if tx in tx_len
    ]
    df = pd.DataFrame(rows).dropna(subset=["transcript_length"])
    df["transcript_length"] = df["transcript_length"].astype(int)
    df.to_csv(OUT, sep="\t", index=False)
    print(f"[rest] {len(df)} genes -> {OUT}")


if __name__ == "__main__":
    main()
