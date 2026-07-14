"""I/O helpers for :class:`pymaftools.core.PivotTable.PivotTable`."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from ._atomic import atomic_output_path


def to_sqlite(table, db_path: str) -> None:
    """Save a PivotTable-like object to SQLite."""
    db_path = Path(db_path)
    table_to_save = table.copy().rename_index_and_columns()
    table_to_save = table_to_save.replace(False, "WT")
    with atomic_output_path(db_path) as temporary_path:
        with sqlite3.connect(str(temporary_path)) as conn:
            table_to_save.to_sql("data", conn, index=True)
            table_to_save.sample_metadata.to_sql("sample_metadata", conn, index=True)
            table_to_save.feature_metadata.to_sql(
                "feature_metadata", conn, index=True
            )
    print(f"[PivotTable] saved to {db_path}")


def to_h5(
    table,
    h5_path: str | Path,
    *,
    complib: str = "zlib",
    complevel: int = 9,
) -> None:
    """Save a PivotTable-like object to HDF5."""
    h5_path = Path(h5_path)
    table_to_save = table.copy().rename_index_and_columns()
    with atomic_output_path(h5_path) as temporary_path:
        with pd.HDFStore(
            str(temporary_path), mode="w", complib=complib, complevel=complevel
        ) as store:
            table_metadata = pd.DataFrame({"class_name": [type(table).__name__]})
            store.put("table_metadata", table_metadata)
            store.put("data", pd.DataFrame(table_to_save))
            store.put("sample_metadata", table_to_save.sample_metadata)
            store.put("feature_metadata", table_to_save.feature_metadata)

    print(f"[PivotTable] saved to {h5_path}")


def read_sqlite(table_cls, db_path: str):
    """Load a PivotTable-like object from SQLite."""
    conn = sqlite3.connect(db_path)

    data = pd.read_sql("SELECT * FROM 'data'", conn, index_col="feature")
    data.columns.name = "sample"

    sample_metadata = pd.read_sql(
        "SELECT * FROM 'sample_metadata'", conn, index_col="sample"
    )
    feature_metadata = pd.read_sql(
        "SELECT * FROM 'feature_metadata'", conn, index_col="feature"
    )

    table = table_cls(data)
    table = table.replace("WT", False)
    table.sample_metadata = sample_metadata
    table.feature_metadata = feature_metadata
    table._validate_metadata()
    conn.close()
    print(f"[PivotTable] loaded from {db_path}")
    return table


def read_h5(table_cls, base_table_cls, h5_path: str | Path):
    """Load a PivotTable-like object from HDF5."""
    h5_path = Path(h5_path)
    with pd.HDFStore(str(h5_path), mode="r") as store:
        required = {"/data", "/sample_metadata", "/feature_metadata"}
        keys = set(store.keys())
        missing = sorted(required - keys)
        if missing:
            raise ValueError(
                f"HDF5 file '{h5_path}' is missing required key(s): {missing}."
            )

        data = store.get("data")
        sample_metadata = store.get("sample_metadata")
        feature_metadata = store.get("feature_metadata")

        resolved_table_cls = table_cls
        if table_cls is base_table_cls and "/table_metadata" in keys:
            table_metadata = store.get("table_metadata")
            if "class_name" in table_metadata.columns:
                class_name = table_metadata["class_name"].iloc[0]
                resolved_table_cls = base_table_cls._subclass_registry.get(
                    class_name, base_table_cls
                )

    table = resolved_table_cls(data)
    table.sample_metadata = sample_metadata.reindex(table.columns)
    table.feature_metadata = feature_metadata.reindex(table.index)
    table._validate_metadata()
    print(f"[PivotTable] loaded from {h5_path}")
    return table


def to_anndata(table, **kwargs: Any):
    """Convert a PivotTable-like object to AnnData."""
    try:
        import anndata
    except ImportError:
        raise ImportError(
            "anndata is required for AnnData conversion. "
            "Install it with: pip install anndata"
        )

    X = table.values.T

    if X.dtype == object:
        import numpy as _np

        X = _np.array(X, dtype=object)

    return anndata.AnnData(
        X=X,
        obs=table.sample_metadata.copy(),
        var=table.feature_metadata.copy(),
        **kwargs,
    )


def from_anndata(table_cls, adata):
    """Create a PivotTable-like object from AnnData."""
    try:
        import anndata  # noqa: F401
    except ImportError:
        raise ImportError(
            "anndata is required for AnnData conversion. "
            "Install it with: pip install anndata"
        )

    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    data = pd.DataFrame(
        X.T,
        index=adata.var_names,
        columns=adata.obs_names,
    )

    table = table_cls(data)
    table.feature_metadata = adata.var.copy()
    table.sample_metadata = adata.obs.copy()
    table._validate_metadata()
    return table
