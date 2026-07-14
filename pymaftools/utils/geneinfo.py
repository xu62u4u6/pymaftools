from __future__ import annotations

import io
import time
from pathlib import Path

import requests
import pandas as pd

_NCBI_TIMEOUT_SECONDS = 30


def get_ncbi_gene_ID(gene_symbol: str) -> str | None:
    """
    Query the NCBI Entrez API for a gene symbol and return its Gene ID.

    Parameters
    ----------
    gene_symbol : str
        The gene symbol to query (e.g. ``"TP53"``).

    Returns
    -------
    str or None
        The first matching Gene ID, or ``None`` if no result is found.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "gene",
        "term": f"{gene_symbol}[gene] AND human[orgn]",
        "retmode": "json",
    }
    try:
        response = requests.get(url, params=params, timeout=_NCBI_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        gene_ids = data.get("esearchresult", {}).get("idlist", [])
        return gene_ids[0] if gene_ids else None
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving Gene ID for {gene_symbol}: {e}")
        return None


def get_ncbi_gene_IDs(gene_symbols: list[str]) -> dict[str, str | None]:
    """
    Batch-query gene symbols and return their corresponding Gene IDs.

    Parameters
    ----------
    gene_symbols : list[str]
        List of gene symbols to query.

    Returns
    -------
    dict[str, str or None]
        Mapping from gene symbol to Gene ID (or ``None``).
    """
    gene_ids = {}
    for symbol in gene_symbols:
        gene_id = get_ncbi_gene_ID(symbol)
        gene_ids[symbol] = gene_id
        print(f"Retrieved Gene ID for {symbol}: {gene_id}")
    return gene_ids


def get_gene_info_json(gene_ids: dict[str, str | None]) -> dict[str, dict]:
    """
    Retrieve detailed gene information from NCBI for multiple Gene IDs.

    Parameters
    ----------
    gene_ids : dict[str, str or None]
        Mapping from gene symbol to Gene ID.

    Returns
    -------
    dict[str, dict]
        Mapping from gene symbol to its detailed information dictionary,
        or ``None`` for symbols whose ID was missing or not found.
    """
    # Extract valid Gene IDs
    valid_gene_ids = [gene_id for gene_id in gene_ids.values() if gene_id]

    if not valid_gene_ids:
        print("No valid Gene IDs found.")
        return {}

    # Join all Gene IDs with a comma
    joined_ids = ",".join(valid_gene_ids)

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db": "gene", "id": joined_ids, "retmode": "json"}

    try:
        response = requests.get(url, params=params, timeout=_NCBI_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()

        result = data.get("result", {})
        result.pop("uids", None)

        gene_info = {}
        for symbol, gene_id in gene_ids.items():
            if gene_id and str(gene_id) in result:
                gene_info[symbol] = result[str(gene_id)]
            else:
                gene_info[symbol] = None

        print(f"Retrieved details for {len(valid_gene_ids)} genes.")
        return gene_info

    except requests.exceptions.RequestException as e:
        print(f"Error retrieving gene details: {e}")
        return {}


def parse_gene_info(gene_info: dict[str, dict]) -> dict[str, str | None]:
    """
    Extract summary descriptions from detailed gene information.

    Parameters
    ----------
    gene_info : dict[str, dict]
        Mapping from gene symbol to its detailed information dictionary.

    Returns
    -------
    dict[str, str or None]
        Mapping from gene symbol to its summary string (or ``None``).
    """
    summaries = {}
    for symbol, info in gene_info.items():
        summaries[symbol] = info.get("summary", None)
    return summaries


def get_gene_description_df(gene_symbols: list[str]) -> pd.DataFrame:
    """
    Look up gene descriptions from NCBI for a list of gene symbols.

    Combines ``get_ncbi_gene_IDs``, ``get_gene_info_json``, and
    ``parse_gene_info`` into a single convenience function.

    Parameters
    ----------
    gene_symbols : list[str]
        List of gene symbols to query.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Gene`` and ``Description``.
    """
    IDs = get_ncbi_gene_IDs(gene_symbols)
    gene_info = get_gene_info_json(IDs)
    parsed_info = parse_gene_info(gene_info)
    gene_description_df = pd.DataFrame.from_dict(
        parsed_info, orient="index", columns=["Description"]
    )
    gene_description_df.reset_index(inplace=True)
    gene_description_df.rename(columns={"index": "Gene"}, inplace=True)

    return gene_description_df

_DATA_DIR = Path(__file__).parent.parent / "data"
_ENSEMBL_CACHE = _DATA_DIR / "ensembl_gene_map.tsv"
_GENE_SIZE_CACHE = _DATA_DIR / "ensembl_gene_sizes.tsv"


def load_ensembl_map(force: bool = False) -> pd.DataFrame:
    """Load Ensembl gene ID -> HUGO symbol mapping, downloading if needed.

    Parameters
    ----------
    force:
        If True, re-download even if the cache exists.

    Returns
    -------
    pd.DataFrame
        Columns: ensembl_gene_id, hugo_symbol, chromosome_name,
        start_position, end_position, gene_biotype
    """
    if _ENSEMBL_CACHE.exists() and not force:
        return pd.read_csv(_ENSEMBL_CACHE, sep="\t")

    from pybiomart import Server  # lazy import

    server = Server(host="http://www.ensembl.org")
    dataset = (
        server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]
    )
    df = dataset.query(
        attributes=[
            "ensembl_gene_id",
            "external_gene_name",
            "chromosome_name",
            "start_position",
            "end_position",
            "gene_biotype",
        ]
    )
    # pybiomart returns display-name columns for these attributes.
    rename_map = {
        "Gene stable ID": "ensembl_gene_id",
        "Gene name": "hugo_symbol",
        "Chromosome/scaffold name": "chromosome_name",
        "Gene start (bp)": "start_position",
        "Gene end (bp)": "end_position",
        "Gene type": "gene_biotype",
    }
    df = df.rename(columns=rename_map)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_ENSEMBL_CACHE, sep="\t", index=False)
    return df


def ensembl_to_symbol(
    ensembl_ids,
    force_download: bool = False,
) -> dict:
    """Map Ensembl gene IDs to HUGO symbols.

    Parameters
    ----------
    ensembl_ids:
        Ensembl IDs to convert (with or without version suffix, e.g.
        ENSG00000141510.18).
    force_download:
        Passed to load_ensembl_map.

    Returns
    -------
    dict
        Mapping each input ID to its HUGO symbol (or None if not found).
    """
    df = load_ensembl_map(force=force_download)
    mapping = df.set_index("ensembl_gene_id")["hugo_symbol"].to_dict()
    result = {}
    for eid in ensembl_ids:
        base = eid.split(".")[0]  # strip version suffix
        result[eid] = mapping.get(base) or mapping.get(eid)
    return result


def symbol_to_ensembl(
    symbols,
    force_download: bool = False,
) -> dict:
    """Map HUGO symbols to Ensembl gene IDs.

    Parameters
    ----------
    symbols:
        HUGO symbols to convert.
    force_download:
        Passed to load_ensembl_map.

    Returns
    -------
    dict
        Mapping each symbol to its Ensembl gene ID (or None if not found).
        If a symbol maps to multiple Ensembl IDs, the first is returned.
    """
    df = load_ensembl_map(force=force_download)
    mapping = (
        df.dropna(subset=["hugo_symbol"])
        .drop_duplicates(subset=["hugo_symbol"], keep="first")
        .set_index("hugo_symbol")["ensembl_gene_id"]
        .to_dict()
    )
    return {s: mapping.get(s) for s in symbols}


# BioMart REST endpoint. We query it directly (instead of via pybiomart) so we
# can pass ``redirect=no`` — Ensembl's mirror sites are periodically down and the
# main site only serves BioMart when the mirror redirect is suppressed.
_BIOMART_URL = "https://www.ensembl.org/biomart/martservice"

# transcript_is_canonical lives on a different BioMart attribute "page" and
# cannot be queried alongside the length attributes; we represent each gene by
# its LONGEST transcript (see _reduce_to_canonical_transcript, which still
# prefers a canonical flag when one is supplied).
_GENE_SIZE_QUERY = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<!DOCTYPE Query>'
    '<Query virtualSchemaName="default" formatter="TSV" header="0" '
    'uniqueRows="1" count="" datasetConfigVersion="0.6">'
    '<Dataset name="hsapiens_gene_ensembl" interface="default">'
    '<Attribute name="ensembl_gene_id"/>'
    '<Attribute name="external_gene_name"/>'
    '<Attribute name="transcript_length"/>'
    '<Attribute name="cds_length"/>'
    '</Dataset></Query>'
)

_SIZE_ATTRS = (
    '<Attribute name="ensembl_gene_id"/>'
    '<Attribute name="transcript_length"/>'
    '<Attribute name="cds_length"/>'
)


def _biomart_query(query_xml: str, retries: int = 4) -> str:
    """Run a BioMart REST query and return its TSV text.

    Uses ``redirect=no`` (Ensembl's mirror sites are periodically down and the
    main site only serves BioMart when the mirror redirect is suppressed) and
    retries while the service returns its HTML maintenance page instead of TSV.
    """
    for attempt in range(retries):
        resp = requests.get(
            _BIOMART_URL,
            params={"query": query_xml, "redirect": "no"},
            timeout=180,
        )
        resp.raise_for_status()
        text = resp.text
        if "\t" in text.split("\n", 1)[0] and not text.lstrip().startswith("<"):
            return text
        if attempt < retries - 1:
            time.sleep(5)
    raise RuntimeError(
        "Ensembl BioMart returned a non-TSV response (the service may be down "
        "for maintenance). Try again later, or see https://www.ensembl.org."
    )


def load_gene_sizes(force: bool = False) -> pd.DataFrame:
    """Load per-gene transcript sizes, downloading from Ensembl if needed.

    Returns the cached table if present, otherwise queries the Ensembl BioMart
    REST service for transcript/CDS length, reduces to one row per gene (its
    longest transcript), caches the result to ``data/ensembl_gene_sizes.tsv``
    and returns it.

    Parameters
    ----------
    force:
        If True, re-download even if the cache exists.

    Returns
    -------
    pd.DataFrame
        Columns: hugo_symbol, ensembl_gene_id, transcript_length, cds_length.
        One row per HUGO symbol.

    Raises
    ------
    RuntimeError
        If BioMart returns a non-TSV response (e.g. Ensembl's maintenance page
        when the service is temporarily unavailable).
    """
    if _GENE_SIZE_CACHE.exists() and not force:
        return pd.read_csv(_GENE_SIZE_CACHE, sep="\t")

    text = _biomart_query(_GENE_SIZE_QUERY)
    df = pd.read_csv(
        io.StringIO(text),
        sep="\t",
        header=None,
        names=["ensembl_gene_id", "hugo_symbol", "transcript_length", "cds_length"],
    )
    df = _reduce_to_canonical_transcript(df)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_GENE_SIZE_CACHE, sep="\t", index=False)
    return df


def _fetch_gene_sizes(genes: list[str]) -> pd.DataFrame:
    """Fetch sizes for just ``genes`` via a filtered BioMart query.

    Maps symbols to Ensembl IDs (offline, via the bundled gene map), queries
    only those genes, and reduces to one row per gene. Far cheaper and more
    robust than the genome-wide dump in :func:`load_gene_sizes` — a small
    filtered query succeeds even when Ensembl rejects the full dump.
    """
    sym2ens = symbol_to_ensembl(list(genes))
    ens2sym = {ens: sym for sym, ens in sym2ens.items() if ens}
    ids = list(ens2sym)
    if not ids:
        return pd.DataFrame(
            columns=["hugo_symbol", "ensembl_gene_id", "transcript_length", "cds_length"]
        )

    query = (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="0" '
        'uniqueRows="1" datasetConfigVersion="0.6">'
        '<Dataset name="hsapiens_gene_ensembl" interface="default">'
        f'<Filter name="link_ensembl_gene_id" value="{",".join(ids)}"/>'
        f'{_SIZE_ATTRS}</Dataset></Query>'
    )
    df = pd.read_csv(
        io.StringIO(_biomart_query(query)),
        sep="\t",
        header=None,
        names=["ensembl_gene_id", "transcript_length", "cds_length"],
    )
    df["hugo_symbol"] = df["ensembl_gene_id"].map(ens2sym)
    return _reduce_to_canonical_transcript(df)


def _reduce_to_canonical_transcript(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a transcript-level BioMart frame to one row per HUGO symbol.

    Prefers the canonical transcript (``is_canonical`` truthy); otherwise falls
    back to the longest transcript for that gene.
    """
    df = df.dropna(subset=["hugo_symbol", "transcript_length"]).copy()
    df["transcript_length"] = df["transcript_length"].astype(int)

    if "is_canonical" in df:
        canonical_mask = df["is_canonical"].notna() & df["is_canonical"].ne(0)
        canonical = df.loc[canonical_mask]
    else:
        canonical = df.iloc[0:0]
    # genes without a canonical row fall back to their longest transcript
    longest = df.sort_values("transcript_length").drop_duplicates(
        subset=["hugo_symbol"], keep="last"
    )
    chosen = pd.concat([longest, canonical]).drop_duplicates(
        subset=["hugo_symbol"], keep="last"
    )
    cols = ["hugo_symbol", "ensembl_gene_id", "transcript_length", "cds_length"]
    return chosen[[c for c in cols if c in chosen.columns]].reset_index(drop=True)


def get_exon_size(
    genes,
    metric: str = "transcript_length",
    force_download: bool = False,
) -> pd.Series:
    """Return the exon size (canonical-transcript length, in bp) per gene.

    Parameters
    ----------
    genes:
        HUGO symbols to look up.
    metric:
        ``"transcript_length"`` (default; total exon length including UTRs) or
        ``"cds_length"`` (coding length only).
    force_download:
        If True, ignore the bundled cache and fetch from Ensembl.

    Returns
    -------
    pd.Series
        Indexed by the input symbols (order preserved), values are the size in
        base pairs; genes absent from Ensembl are ``NaN``.

    Notes
    -----
    Uses the bundled ``ensembl_gene_sizes.tsv`` cache when present; any
    requested genes missing from it are still queried from Ensembl (a small
    filtered query, not the genome-wide dump), so a partial cache never silently
    yields NaN and a table's genes can be annotated without a full cache.
    """
    if metric not in ("transcript_length", "cds_length"):
        raise ValueError(
            f"metric must be 'transcript_length' or 'cds_length', got {metric!r}."
        )
    genes = list(genes)
    if _GENE_SIZE_CACHE.exists() and not force_download:
        sizes = pd.read_csv(_GENE_SIZE_CACHE, sep="\t")
        # The bundled cache may be partial (e.g. a fixture-scoped fallback built
        # while BioMart was down). Fetch any genes it doesn't cover rather than
        # silently returning NaN for them.
        missing = [g for g in genes if g not in set(sizes["hugo_symbol"].dropna())]
        if missing:
            sizes = pd.concat([sizes, _fetch_gene_sizes(missing)], ignore_index=True)
    else:
        sizes = _fetch_gene_sizes(genes)
    mapping = (
        sizes.dropna(subset=["hugo_symbol"])
        .drop_duplicates(subset=["hugo_symbol"], keep="first")
        .set_index("hugo_symbol")[metric]
    )
    return pd.Series([mapping.get(g) for g in genes], index=genes, name=metric)
