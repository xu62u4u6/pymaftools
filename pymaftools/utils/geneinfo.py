from __future__ import annotations

import requests
import pandas as pd


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
        response = requests.get(url, params=params)
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
        response = requests.get(url, params=params)
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
