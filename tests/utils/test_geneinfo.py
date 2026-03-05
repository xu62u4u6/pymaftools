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
    def mock_get_success(url, params=None):
        return MockResponse({"esearchresult": {"idlist": ["7157"]}})

    monkeypatch.setattr(geneinfo.requests, "get", mock_get_success)
    assert geneinfo.get_ncbi_gene_ID("TP53") == "7157"

    def mock_get_fail(url, params=None):
        return MockResponse({}, raise_error=True)

    monkeypatch.setattr(geneinfo.requests, "get", mock_get_fail)
    assert geneinfo.get_ncbi_gene_ID("TP53") is None


def test_get_gene_info_json_and_parse(monkeypatch):
    def mock_get(url, params=None):
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
