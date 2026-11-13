"""Smoke test: the five-minute quickstart runs end to end."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "quickstart.py"


def _load():
    spec = importlib.util.spec_from_file_location("quickstart", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.plot
@pytest.mark.parametrize(
    "maf_path",
    [None, ROOT / "tests" / "data" / "maf_sources" / "gdc_style_grch38.maf.gz"],
)
def test_quickstart_renders_oncoplot(tmp_path, maf_path):
    quickstart = _load()
    from pymaftools import example_maf_path

    samples = None
    if maf_path is not None:
        samples = tmp_path / "samples.txt"
        from pymaftools import MAF

        ids = list(MAF.read_maf(maf_path)["sample_ID"].unique()) + ["ZERO-EVENTS"]
        samples.write_text("\n".join(ids) + "\n")
    out = quickstart.run(
        maf_path or example_maf_path(), samples, tmp_path / "oncoplot.png", top=10
    )
    assert out.exists() and out.stat().st_size > 10_000
