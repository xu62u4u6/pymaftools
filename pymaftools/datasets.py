"""Bundled example datasets.

Small MAF files shipped inside the package so that, after ``pip install
pymaftools``, the documented ``MAF.read_maf`` / oncoplot workflow can be run with
no external download. The example barcodes are a subset of the demo cohort used
by ``scripts/download_demo_samples.py``.

Examples
--------
>>> from pymaftools import load_example_maf
>>> maf = load_example_maf("multisample")        # 6 tumor samples
>>> table = maf.to_pivot_table()
"""

from __future__ import annotations

import os
from pathlib import Path

# Bundled data lives next to this module, matching the convention used elsewhere
# (e.g. pymaftools/utils/geneinfo.py, pymaftools/io/tcga/cnv_segment.py).
_DATA_DIR = Path(__file__).parent / "data"

_EXAMPLE_MAFS = {
    "multisample": "example_multisample.maf",  # 6 tumor samples in one file
    "single_sample": "example_single_sample.maf",  # one per-aliquot file
}
_EXAMPLE_TABLE = "example_tcga_lung_mutation_grouped.h5"


def example_maf_path(name: str = "multisample") -> Path:
    """Return the path to a bundled example MAF file.

    Parameters
    ----------
    name : {"multisample", "single_sample"}, default "multisample"
        Which example to use.

    Returns
    -------
    pathlib.Path
        Path to the bundled ``.maf`` file.
    """
    if name not in _EXAMPLE_MAFS:
        raise ValueError(
            f"Unknown example MAF {name!r}; choose from {sorted(_EXAMPLE_MAFS)}."
        )
    return _DATA_DIR / _EXAMPLE_MAFS[name]


def load_example_maf(name: str = "multisample", **kwargs):
    """Read a bundled example MAF with :meth:`MAF.read_maf`.

    Parameters
    ----------
    name : {"multisample", "single_sample"}, default "multisample"
        Which example to load.
    **kwargs
        Forwarded to :meth:`pymaftools.core.MAF.MAF.read_maf` (e.g. ``sample_ID``).

    Returns
    -------
    MAF
        The parsed MAF object.
    """
    from .core.MAF import MAF

    return MAF.read_maf(os.fspath(example_maf_path(name)), **kwargs)


def example_table_path() -> Path:
    """Return the path to the bundled TCGA lung mutation HDF5 table."""
    return _DATA_DIR / _EXAMPLE_TABLE


def load_example_table():
    """Load the bundled TCGA lung mutation table from HDF5."""
    from .core.PivotTable import PivotTable

    return PivotTable.read_h5(example_table_path())
