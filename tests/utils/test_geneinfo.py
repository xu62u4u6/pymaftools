import pandas as pd

from pymaftools.utils import geneinfo


class MockResponse:
    def __init__(self, payload, raise_error=False):
        self._payload = payload
        self._raise_error = raise_error

    def raise_for_status(self):
        if self._raise_error:
            raise geneinfo.requests.exceptions.RequestException("mock error")

    def json(self):
        return self._payload


def test_get_ncbi_gene_id_success_and_failure(monkeypatch):
    def mock_get_success(url, params=None, timeout=None):
        assert timeout == 30
        return MockResponse({"esearchresult": {"idlist": ["7157"]}})

    monkeypatch.setattr(geneinfo.requests, "get", mock_get_success)
    assert geneinfo.get_ncbi_gene_ID("TP53") == "7157"

    def mock_get_fail(url, params=None, timeout=None):
        assert timeout == 30
        return MockResponse({}, raise_error=True)

    monkeypatch.setattr(geneinfo.requests, "get", mock_get_fail)
    assert geneinfo.get_ncbi_gene_ID("TP53") is None


def test_get_gene_info_json_and_parse(monkeypatch):
    def mock_get(url, params=None, timeout=None):
        assert timeout == 30
        return MockResponse(
            {
                "result": {
                    "uids": ["7157", "3845"],
                    "7157": {"summary": "TP53 summary"},
                    "3845": {"summary": "KRAS summary"},
                }
            }
        )

    monkeypatch.setattr(geneinfo.requests, "get", mock_get)

    info = geneinfo.get_gene_info_json({"TP53": "7157", "KRAS": "3845", "X": None})
    assert info["TP53"]["summary"] == "TP53 summary"
    assert info["X"] is None
    parsed = geneinfo.parse_gene_info({"TP53": info["TP53"], "KRAS": info["KRAS"]})
    assert parsed["KRAS"] == "KRAS summary"


def test_get_gene_description_df(monkeypatch):
    monkeypatch.setattr(geneinfo, "get_ncbi_gene_IDs", lambda genes: {"TP53": "7157", "KRAS": "3845"})
    monkeypatch.setattr(
        geneinfo,
        "get_gene_info_json",
        lambda ids: {"TP53": {"summary": "p53"}, "KRAS": {"summary": "kras"}},
    )

    df = geneinfo.get_gene_description_df(["TP53", "KRAS"])

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Gene", "Description"]
    assert set(df["Gene"]) == {"TP53", "KRAS"}


def test_reduce_to_canonical_transcript_prefers_canonical():
    """When a canonical transcript is flagged, its length wins over the longest."""
    df = pd.DataFrame(
        {
            "hugo_symbol": ["TP53", "TP53", "KRAS", "KRAS"],
            "ensembl_gene_id": ["ENSG_TP53", "ENSG_TP53", "ENSG_KRAS", "ENSG_KRAS"],
            "transcript_length": [2500, 9999, 5300, 1200],
            "cds_length": [1182, 600, 567, 400],
            "is_canonical": [1, None, None, 1],
        }
    )
    out = geneinfo._reduce_to_canonical_transcript(df).set_index("hugo_symbol")

    # TP53: canonical (2500) chosen over the longer 9999 transcript
    assert out.loc["TP53", "transcript_length"] == 2500
    # KRAS: canonical is the shorter 1200 transcript
    assert out.loc["KRAS", "transcript_length"] == 1200


def test_reduce_to_canonical_transcript_falls_back_to_longest():
    """With no canonical flag, the longest transcript represents the gene."""
    df = pd.DataFrame(
        {
            "hugo_symbol": ["EGFR", "EGFR"],
            "ensembl_gene_id": ["ENSG_EGFR", "ENSG_EGFR"],
            "transcript_length": [3000, 6500],
            "cds_length": [1000, 3633],
            "is_canonical": [None, None],
        }
    )
    out = geneinfo._reduce_to_canonical_transcript(df).set_index("hugo_symbol")
    assert out.loc["EGFR", "transcript_length"] == 6500


def test_get_exon_size_maps_and_handles_missing(monkeypatch):
    """get_exon_size returns sizes per requested gene; unknown genes are NaN."""
    fake = pd.DataFrame(
        {
            "hugo_symbol": ["TP53", "TTN"],
            "ensembl_gene_id": ["ENSG_TP53", "ENSG_TTN"],
            "transcript_length": [2591, 109224],
            "cds_length": [1182, 100272],
        }
    )
    # force_download bypasses any local cache and uses the targeted fetch path
    monkeypatch.setattr(geneinfo, "_fetch_gene_sizes", lambda genes: fake)

    s = geneinfo.get_exon_size(["TTN", "TP53", "NOTAGENE"], force_download=True)
    assert list(s.index) == ["TTN", "TP53", "NOTAGENE"]  # input order preserved
    assert s["TTN"] == 109224
    assert pd.isna(s["NOTAGENE"])

    cds = geneinfo.get_exon_size(["TP53"], metric="cds_length", force_download=True)
    assert cds["TP53"] == 1182


def test_fetch_gene_sizes_maps_ids_and_reduces(monkeypatch):
    """_fetch_gene_sizes filters to requested genes, maps Ensembl IDs back to
    symbols, and keeps each gene's longest transcript."""
    monkeypatch.setattr(
        geneinfo,
        "symbol_to_ensembl",
        lambda genes: {"TP53": "ENSG_TP53", "TTN": "ENSG_TTN"},
    )
    # two TTN transcripts -> longest kept
    tsv = "ENSG_TP53\t2591\t1182\nENSG_TTN\t80000\t79000\nENSG_TTN\t109224\t100272\n"
    monkeypatch.setattr(geneinfo, "_biomart_query", lambda q, retries=4: tsv)

    out = geneinfo._fetch_gene_sizes(["TP53", "TTN"]).set_index("hugo_symbol")
    assert out.loc["TTN", "transcript_length"] == 109224
    assert out.loc["TP53", "transcript_length"] == 2591


def test_get_exon_size_invalid_metric_raises():
    import pytest

    with pytest.raises(ValueError, match="transcript_length"):
        geneinfo.get_exon_size(["TP53"], metric="bogus")
