"""Machine-readable results: provenance records and JSON serialisation.

Agents (and reviewers) need to know *what* a number is, *which inputs and
parameters* produced it, and *which samples* it was computed over -- without
reading a notebook. This module is the single source for that:

* :func:`session_info` -- package / interpreter / dependency versions.
* :class:`Provenance` -- analysis name, parameters, input checksums and the
  sample universe of one result.
* :func:`attach_provenance` / :func:`get_provenance` -- statistics functions
  keep returning plain :class:`pandas.DataFrame` objects (so no existing code
  or published number changes); the record rides along in ``df.attrs``.
* :func:`to_jsonable` / :func:`to_json` -- one serialiser for every result
  type. ``NaN`` becomes ``null``; ``+/-inf`` become the strings ``"inf"`` /
  ``"-inf"`` so infinite odds ratios are never silently dropped.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import platform
from collections.abc import Iterable, Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"
"""Version of the JSON layout pymaftools emits (provenance records, ingest
reports, TMB audits, figure summaries and CLI output).

Bump the minor part when fields are only added; bump the major part when a
field is renamed, removed or changes meaning, so consumers can refuse output
they do not understand.
"""

PROVENANCE_ATTR = "pymaftools_provenance"
"""Key under which :func:`attach_provenance` stores the record in ``df.attrs``."""

_DEPENDENCIES = (
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "tables",
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def session_info() -> dict[str, Any]:
    """Return the versions that determine a pymaftools result.

    Returns
    -------
    dict
        ``pymaftools``, ``python``, ``platform`` and a ``dependencies`` mapping
        (``None`` for a package that is not installed). JSON-serialisable.

    Examples
    --------
    >>> import pymaftools
    >>> info = pymaftools.session_info()
    >>> sorted(info)
    ['dependencies', 'platform', 'pymaftools', 'python']
    """
    return {
        "pymaftools": _package_version("pymaftools"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {name: _package_version(name) for name in _DEPENDENCIES},
    }


def file_fingerprint(path: str | os.PathLike, chunk_size: int = 1 << 20) -> dict:
    """Return ``{"path", "bytes", "sha256"}`` for an input file."""
    path = os.fspath(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return {
        "path": path,
        "bytes": os.path.getsize(path),
        "sha256": digest.hexdigest(),
    }


def sample_universe(
    samples: Iterable[object], source: str = "table_columns"
) -> dict[str, Any]:
    """Describe the set of samples a result was computed over.

    ``samples_sha256`` hashes the sorted IDs so two results can be checked for
    the same denominator without comparing long lists.
    """
    ids = [str(sample) for sample in samples]
    digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    return {
        "source": source,
        "n_samples": len(ids),
        "samples_sha256": digest,
        "samples": ids,
    }


def table_sample_universe(table) -> dict[str, Any]:
    """Sample universe of a matrix table: its manifest if attached, else columns."""
    manifest = getattr(table, "sample_manifest", None)
    if manifest is not None:
        return sample_universe(manifest.eligible_samples, source="sample_manifest")
    return sample_universe(table.columns, source="table_columns")


INPUTS_ATTR = "pymaftools_inputs"
"""``attrs`` key carrying input-file fingerprints from a MAF to derived tables."""


def table_inputs(obj) -> list[dict[str, Any]]:
    """Input-file fingerprints a table inherited from its source MAF (best effort)."""
    attrs = getattr(obj, "attrs", None)
    if isinstance(attrs, Mapping):
        return [dict(item) for item in attrs.get(INPUTS_ATTR, [])]
    return []


def provenance_for_table(analysis: str, table, parameters: dict) -> "Provenance":
    """Build a :class:`Provenance` for a statistic computed from a matrix table."""
    return Provenance(
        analysis=analysis,
        parameters=parameters,
        inputs=table_inputs(table),
        sample_universe=table_sample_universe(table),
    )


@dataclasses.dataclass
class Provenance:
    """Where one result came from.

    Serialise with :meth:`to_dict` or :func:`to_json`.
    """

    analysis: str
    """Producing function, e.g. ``"somatic_interactions"``."""
    parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Arguments that change the result."""
    inputs: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    """Input files as :func:`file_fingerprint` records."""
    sample_universe: dict[str, Any] | None = None
    """See :func:`sample_universe`."""
    pymaftools_version: str | None = dataclasses.field(
        default_factory=lambda: _package_version("pymaftools")
    )
    created_at: str = dataclasses.field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        )
    )
    schema_version: str = SCHEMA_VERSION
    """See :data:`SCHEMA_VERSION`."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return to_jsonable(dataclasses.asdict(self))


def attach_provenance(obj, provenance: Provenance):
    """Store ``provenance`` on a DataFrame/Series (``obj.attrs``); return ``obj``.

    Values are not touched; only metadata is added.
    """
    obj.attrs[PROVENANCE_ATTR] = provenance.to_dict()
    return obj


def get_provenance(obj) -> dict[str, Any] | None:
    """Return the provenance record carried by a result, or ``None``.

    Works for DataFrames/Series produced by pymaftools statistics and for
    result objects with a ``provenance`` attribute (e.g. ``TMBAudit``).
    """
    attrs = getattr(obj, "attrs", None)
    if isinstance(attrs, Mapping) and PROVENANCE_ATTR in attrs:
        return attrs[PROVENANCE_ATTR]
    record = getattr(obj, "provenance", None)
    if isinstance(record, Provenance):
        return record.to_dict()
    if isinstance(record, Mapping):
        return dict(record)
    return None


def _scalar(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        return value.isoformat()
    return value


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [str(column) for column in frame.columns]
    return [
        {column: to_jsonable(value) for column, value in zip(columns, row)}
        for row in frame.itertuples(index=False, name=None)
    ]


def dataframe_to_dict(frame: pd.DataFrame, *, index: bool | None = None) -> dict:
    """Serialise a DataFrame as ``{"provenance", "columns", "records"}``.

    ``index=None`` keeps the index only when it carries labels: a named index
    or non-integer labels (a positional integer index left by sorting or
    filtering is dropped).
    """
    if index is None:
        index = (
            frame.index.name is not None
            or any(name is not None for name in frame.index.names)
            or not pd.api.types.is_integer_dtype(frame.index.dtype)
        )
    plain = pd.DataFrame(frame)
    if index:
        index_name = plain.index.name or "index"
        plain = plain.reset_index(names=index_name)
    out: dict[str, Any] = {}
    record = get_provenance(frame)
    if record is not None:
        out["provenance"] = record
    out["n_rows"] = int(len(plain))
    out["columns"] = [str(column) for column in plain.columns]
    out["records"] = _frame_records(plain)
    return out


def to_jsonable(obj: Any) -> Any:
    """Convert a pymaftools result (or nested containers) to JSON-safe values.

    Handles objects with ``to_dict()``, dataclasses, DataFrames, Series, numpy
    arrays/scalars, sets and paths. ``NaN`` -> ``None``; ``inf`` -> ``"inf"``.
    """
    if isinstance(obj, pd.DataFrame):
        return dataframe_to_dict(obj)
    if isinstance(obj, pd.Series):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, pd.Index):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(value) for value in obj.tolist()]
    if isinstance(obj, Mapping):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj, key=str) if isinstance(obj, (set, frozenset)) else obj
        return [to_jsonable(value) for value in items]
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict) and not isinstance(obj, type):
        return to_jsonable(to_dict())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    return _scalar(obj)


def to_json(
    obj: Any,
    path: str | os.PathLike | None = None,
    *,
    indent: int | None = 2,
) -> str:
    """Serialise a result to JSON text; also write it to ``path`` if given.

    Parameters
    ----------
    obj
        Any pymaftools result: a statistics DataFrame (its provenance is
        included), ``IngestReport``, ``TMBAudit``, a plot summary dict, ...
    path : str or os.PathLike, optional
        File to write.
    indent : int or None, default 2
        Passed to :func:`json.dumps`.

    Returns
    -------
    str
        The JSON text (strict JSON: no ``NaN``/``Infinity`` literals).
    """
    text = json.dumps(to_jsonable(obj), indent=indent, allow_nan=False)
    if path is not None:
        with open(os.fspath(path), "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return text


__all__ = [
    "INPUTS_ATTR",
    "PROVENANCE_ATTR",
    "Provenance",
    "attach_provenance",
    "dataframe_to_dict",
    "file_fingerprint",
    "get_provenance",
    "sample_universe",
    "provenance_for_table",
    "session_info",
    "table_inputs",
    "table_sample_universe",
    "to_json",
    "to_jsonable",
]
