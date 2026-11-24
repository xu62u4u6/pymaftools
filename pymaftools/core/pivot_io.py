"""I/O helpers for :class:`pymaftools.core.PivotTable.PivotTable`."""

from __future__ import annotations

from contextlib import closing
import sqlite3
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from ._atomic import atomic_output_path

_SQLITE_FALSE_SENTINEL = "__PYMAFTOOLS_FALSE__"
_SQLITE_TRUE_SENTINEL = "__PYMAFTOOLS_TRUE__"
_LEGACY_SQLITE_FALSE_SENTINELS = frozenset({_SQLITE_FALSE_SENTINEL, "WT"})


def quote_sql_identifier(name: str) -> str:
    """Safely double-quote a SQLite table or column identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def warn_sqlite_unpersisted_attributes(table, *, table_name: str | None = None) -> None:
    """Warn if table carries attributes that SQLite format cannot persist."""
    dropped = [
        attr
        for attr in ("observation_mask", "sample_manifest", "value_scale")
        if getattr(table, attr, None) is not None
    ]
    if dropped:
        prefix = (
            f"Table {table_name!r}" if table_name is not None else type(table).__name__
        )
        warnings.warn(
            f"{prefix} has {', '.join(dropped)} attached, which SQLite I/O "
            "does not persist. Use to_h5() / read_h5() to keep these attributes.",
            UserWarning,
            stacklevel=3,
        )


def _is_missing_scalar(value: Any) -> bool:
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, (float, np.floating)) and bool(np.isnan(value)))
    )


def encode_sqlite_data(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Python/NumPy booleans distinctly from numeric 0/1 for SQLite.

    SQLite has no native boolean column type and pandas ``to_sql`` coerces
    ``bool`` inside mixed ``object`` columns to the text string ``'0'``/``'1'``.
    Encoding only true boolean values (never numeric ``0``/``0.0``) avoids both
    flipping wildtype ``False`` cells into truthy ``'0'`` strings and corrupting
    numeric zeros into ``False``.
    """
    encoded = pd.DataFrame(df).copy()
    for col in encoded.columns:
        series = encoded[col]
        if pd.api.types.is_bool_dtype(series.dtype):
            encoded[col] = pd.Series(
                [
                    None
                    if _is_missing_scalar(v)
                    else (_SQLITE_TRUE_SENTINEL if bool(v) else _SQLITE_FALSE_SENTINEL)
                    for v in series.tolist()
                ],
                index=series.index,
                dtype=object,
            )
        elif pd.api.types.is_object_dtype(series.dtype):
            values = series.tolist()
            if any(isinstance(v, (bool, np.bool_)) for v in values):
                encoded[col] = pd.Series(
                    [
                        (_SQLITE_TRUE_SENTINEL if bool(v) else _SQLITE_FALSE_SENTINEL)
                        if isinstance(v, (bool, np.bool_))
                        else v
                        for v in values
                    ],
                    index=series.index,
                    dtype=object,
                )
            else:
                non_null = [v for v in values if not _is_missing_scalar(v)]
                if non_null and all(
                    isinstance(v, (int, float, np.number))
                    and not isinstance(v, (bool, np.bool_))
                    for v in non_null
                ):
                    encoded[col] = pd.to_numeric(series)
    return encoded


def decode_sqlite_data(df: pd.DataFrame) -> pd.DataFrame:
    """Restore boolean sentinels (and legacy ``'WT'``) read back from SQLite."""
    decoded = df.copy()
    for col in decoded.columns:
        series = decoded[col]
        if not (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            continue
        values = series.tolist()
        if not any(
            isinstance(v, str)
            and (v in _LEGACY_SQLITE_FALSE_SENTINELS or v == _SQLITE_TRUE_SENTINEL)
            for v in values
        ):
            continue
        restored = [
            False
            if isinstance(v, str) and v in _LEGACY_SQLITE_FALSE_SENTINELS
            else True
            if isinstance(v, str) and v == _SQLITE_TRUE_SENTINEL
            else v
            for v in values
        ]
        if restored and all(isinstance(v, bool) for v in restored):
            decoded[col] = pd.Series(restored, index=series.index, dtype=bool)
        else:
            decoded[col] = pd.Series(restored, index=series.index, dtype=object)
    return decoded


def to_sqlite(table, db_path: str) -> None:
    """Save a PivotTable-like object to SQLite."""
    db_path = Path(db_path)
    warn_sqlite_unpersisted_attributes(table)
    table_to_save = table.copy().rename_index_and_columns()
    data_to_save = encode_sqlite_data(table_to_save)
    with atomic_output_path(db_path) as temporary_path:
        with closing(sqlite3.connect(str(temporary_path))) as conn:
            with conn:
                data_to_save.to_sql("data", conn, index=True)
                table_to_save.sample_metadata.to_sql(
                    "sample_metadata", conn, index=True
                )
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
            meta_payload: dict[str, list[Any]] = {"class_name": [type(table).__name__]}
            value_scale = getattr(table, "value_scale", None)
            if value_scale is not None:
                meta_payload["value_scale"] = [value_scale]
            table_metadata = pd.DataFrame(meta_payload)
            store.put("table_metadata", table_metadata)
            store.put("data", pd.DataFrame(table_to_save))
            store.put("sample_metadata", table_to_save.sample_metadata)
            store.put("feature_metadata", table_to_save.feature_metadata)
            if table_to_save.sample_manifest is not None:
                store.put("sample_manifest", table_to_save.sample_manifest.to_frame())
            if table_to_save.observation_mask is not None:
                store.put("observation_mask", table_to_save.observation_mask.to_frame())

    print(f"[PivotTable] saved to {h5_path}")


def read_sqlite(table_cls, db_path: str):
    """Load a PivotTable-like object from SQLite."""
    with closing(sqlite3.connect(db_path)) as conn:
        data = pd.read_sql("SELECT * FROM 'data'", conn, index_col="feature")
        data.columns.name = "sample"

        sample_metadata = pd.read_sql(
            "SELECT * FROM 'sample_metadata'", conn, index_col="sample"
        )
        feature_metadata = pd.read_sql(
            "SELECT * FROM 'feature_metadata'", conn, index_col="feature"
        )

    data = decode_sqlite_data(data)
    table = table_cls(data)
    table.sample_metadata = sample_metadata
    table.feature_metadata = feature_metadata
    table._validate_metadata()
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
        sample_manifest_frame = (
            store.get("sample_manifest") if "/sample_manifest" in keys else None
        )
        observation_mask_frame = (
            store.get("observation_mask") if "/observation_mask" in keys else None
        )

        resolved_table_cls = table_cls
        saved_value_scale = None
        if "/table_metadata" in keys:
            table_metadata = store.get("table_metadata")
            if table_cls is base_table_cls and "class_name" in table_metadata.columns:
                class_name = table_metadata["class_name"].iloc[0]
                resolved_table_cls = base_table_cls._subclass_registry.get(
                    class_name, base_table_cls
                )
            if "value_scale" in table_metadata.columns:
                raw_scale = table_metadata["value_scale"].iloc[0]
                if pd.notna(raw_scale):
                    saved_value_scale = str(raw_scale)

    table = resolved_table_cls(data)
    if saved_value_scale is not None and hasattr(table, "value_scale"):
        table.value_scale = saved_value_scale
    table.sample_metadata = sample_metadata.reindex(table.columns)
    table.feature_metadata = feature_metadata.reindex(table.index)
    if sample_manifest_frame is not None:
        from .SampleManifest import SampleManifest

        table.sample_manifest = SampleManifest(sample_manifest_frame)
    if observation_mask_frame is not None:
        from .ObservationMask import ObservationMask

        table.observation_mask = ObservationMask(
            observation_mask_frame.reindex(index=table.index, columns=table.columns)
        )
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
            "Install it with: pip install 'pymaftools[anndata]'"
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
            "Install it with: pip install 'pymaftools[anndata]'"
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
