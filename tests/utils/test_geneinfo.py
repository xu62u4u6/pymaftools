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
    assert geneinfo.get_ncbi_gene_id("TP53") == "7157"

    def mock_get_fail(url, params=None, timeout=None):
        assert timeout == 30
        return MockResponse({}, raise_error=True)

    monkeypatch.setattr(geneinfo.requests, "get", mock_get_fail)
    assert geneinfo.get_ncbi_gene_id("TP53") is None


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
    monkeypatch.setattr(geneinfo, "get_ncbi_gene_ids", lambda genes: {"TP53": "7157", "KRAS": "3845"})
    monkeypatch.setattr(
        geneinfo,
        "get_gene_info_json",
        lambda ids: {"TP53": {"summary": "p53"}, "KRAS": {"summary": "kras"}},
    )

    df = geneinfo.get_gene_description_df(["TP53", "KRAS"])

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Gene", "Description"]
    assert list(df["Gene"]) == ["TP53", "KRAS"]
    assert dict(zip(df["Gene"], df["Description"])) == {"TP53": "p53", "KRAS": "kras"}


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


def test_reduce_to_canonical_transcript_prefers_coding_over_longer_noncoding():
    df = pd.DataFrame(
        {
            "hugo_symbol": ["TP53", "TP53", "LINC001"],
            "ensembl_gene_id": ["ENSG_TP53", "ENSG_TP53", "ENSG_LINC"],
            "transcript_length": [4000, 2512, 1800],
            "cds_length": [None, 1182, None],
        }
    )
    out = geneinfo._reduce_to_canonical_transcript(df).set_index("hugo_symbol")
    assert out.loc["TP53", "cds_length"] == 1182
    assert out.loc["TP53", "transcript_length"] == 2512
    assert out.loc["LINC001", "transcript_length"] == 1800
    assert pd.isna(out.loc["LINC001", "cds_length"])


def test_get_exon_size_treats_cached_nan_as_miss_and_warns_unresolved(
    tmp_path, monkeypatch
):
    import pytest

    cache = tmp_path / "partial_sizes.tsv"
    pd.DataFrame(
        {
            "hugo_symbol": ["TP53"],
            "ensembl_gene_id": ["ENSG00000141510"],
            "transcript_length": [2512],
            "cds_length": [float("nan")],
        }
    ).to_csv(cache, sep="\t", index=False)
    monkeypatch.setattr(geneinfo, "_GENE_SIZE_CACHE", cache)

    fetched_calls = []

    def fake_fetch(genes):
        fetched_calls.append(list(genes))
        return pd.DataFrame(
            {
                "hugo_symbol": ["TP53"],
                "ensembl_gene_id": ["ENSG00000141510"],
                "transcript_length": [2512],
                "cds_length": [1182],
            }
        )

    monkeypatch.setattr(geneinfo, "_fetch_gene_sizes", fake_fetch)

    with pytest.warns(UserWarning, match="NOTAGENE"):
        sizes = geneinfo.get_exon_size(["TP53", "NOTAGENE"], metric="cds_length")

    assert fetched_calls == [["TP53", "NOTAGENE"]]
    assert sizes["TP53"] == 1182
    assert pd.isna(sizes["NOTAGENE"])


def test_bundled_gene_sizes_cover_drivers_and_flags_offline(monkeypatch):
    import pytest

    def fail_if_online(genes):
        raise AssertionError(f"Expected offline cache hit, got fetch for {genes}")

    monkeypatch.setattr(geneinfo, "_fetch_gene_sizes", fail_if_online)

    cds = geneinfo.get_exon_size(["TP53", "KRAS", "TTN", "MUC16"], metric="cds_length")
    assert cds.notna().all()
    assert cds["TTN"] > cds["MUC16"] > cds["TP53"] > cds["KRAS"] > 0

    # Non-coding gene present in the bundled full cache must NOT hit the network
    with pytest.warns(UserWarning, match="ZNF73P"):
        mixed = geneinfo.get_exon_size(["TP53", "ZNF73P"], metric="cds_length")
    assert mixed["TP53"] == 1236
    assert pd.isna(mixed["ZNF73P"])

    flags = geneinfo.load_flags_genes()
    assert len(flags) == 20
    assert {"TTN", "MUC16", "OBSCN", "SYNE1"}.issubset(set(flags))
    assert "TP53" not in flags
    assert "KMT2D" not in flags


def test_pivot_table_add_mutation_density_and_flags_annotation(monkeypatch):
    import pytest
    from pymaftools.core.SmallVariationTable import SmallVariationTable

    def fail_if_online(genes):
        raise AssertionError(f"Expected offline cache hit, got fetch for {genes}")

    monkeypatch.setattr(geneinfo, "_fetch_gene_sizes", fail_if_online)

    table = SmallVariationTable(
        {
            "S1": ["Missense_Mutation", False, "Missense_Mutation", False],
            "S2": ["Missense_Mutation", "Missense_Mutation", "Missense_Mutation", "Missense_Mutation"],
            "S3": [False, False, False, False],
            "S4": [False, False, False, False],
        },
        index=["TP53", "KRAS", "TTN", "MUC16"],
    )
    # Pre-populating exon_size with transcript_length must NOT silently override
    # metric='cds_length' in add_mutation_density().
    annotated = (
        table.add_exon_size(metric="transcript_length")
        .add_mutation_density(metric="cds_length")
        .add_flags_annotation()
    )

    density = annotated.feature_metadata["mutation_density_per_kb"]
    assert density["TP53"] == pytest.approx(0.5 / (1236 / 1000.0))
    assert density["TTN"] == pytest.approx(0.5 / (107976 / 1000.0))
    assert density["KRAS"] == pytest.approx(0.25 / (570 / 1000.0))
    assert density["MUC16"] == pytest.approx(0.25 / (43710 / 1000.0))
    assert annotated.attrs["mutation_density"] == {
        "col": "mutation_density_per_kb",
        "size_col": "cds_length",
        "metric": "cds_length",
    }
    assert annotated.feature_metadata["is_flag_gene"].to_dict() == {
        "TP53": False,
        "KRAS": False,
        "TTN": True,
        "MUC16": True,
    }
