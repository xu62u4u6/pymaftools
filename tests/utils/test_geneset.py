import pandas as pd

from pymaftools.utils.geneset import read_GMT, fetch_msigdb_geneset


def test_read_gmt_parses_pathway_and_genes(tmp_path):
    gmt_path = tmp_path / "mini.gmt"
    gmt_path.write_text(
        "PATHWAY_A\thttp://a\tTP53\tKRAS\nPATHWAY_B\thttp://b\tEGFR\tPIK3CA\n",
        encoding="utf-8",
    )

    df = read_GMT(str(gmt_path))

    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["PATHWAY_A", "PATHWAY_B"]
    assert "Genes" in df.columns
    assert "TP53" in df.loc["PATHWAY_A", "Genes"]


def test_fetch_msigdb_geneset_parses_html_table(monkeypatch):
    html = """
    <div id=\"geneListing\">
      <table>
        <tr><th>source_id</th><th>entrez_id</th><th>gene_symbol</th><th>description</th></tr>
        <tr><td>S1</td><td>7157</td><td>TP53</td><td>tumor suppressor</td></tr>
        <tr><td>S2</td><td>3845</td><td>KRAS</td><td>oncogene</td></tr>
      </table>
    </div>
    """

    class MockResponse:
        def __init__(self, content: bytes):
            self.content = content

    def mock_get(url, timeout=30):
        return MockResponse(html.encode("utf-8"))

    monkeypatch.setattr("pymaftools.utils.geneset.requests.get", mock_get)

    df = fetch_msigdb_geneset("HALLMARK_TEST")

    assert list(df.columns) == ["source_id", "entrez_id", "gene_symbol", "description"]
    assert len(df) == 2
    assert set(df["gene_symbol"]) == {"TP53", "KRAS"}
