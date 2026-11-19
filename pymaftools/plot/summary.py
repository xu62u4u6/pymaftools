"""Machine-readable figure summaries.

An agent cannot look at a PNG, so a figure that renders but shows nothing (or
the wrong thing) is invisible to it. :func:`figure_summary` describes what was
actually drawn -- per-axes artist counts and an ``empty`` flag -- and plot
classes extend it via :meth:`BasePlot.summary` with domain facts (genes,
samples, hidden labels, top genes). ``BasePlot.save(path, summary=True)``
writes it next to the figure as ``<path stem>.json``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure


def _axes_record(ax) -> dict[str, Any]:
    n_lines = len(ax.lines)
    n_patches = len(ax.patches)
    n_collections = len(ax.collections)
    n_images = len(ax.images)
    n_texts = len([text for text in ax.texts if text.get_text().strip()])
    has_legend = ax.get_legend() is not None
    data_artists = n_lines + n_patches + n_collections + n_images
    axis_on = bool(ax.axison)
    return {
        "title": ax.get_title() or None,
        "xlabel": ax.get_xlabel() or None,
        "ylabel": ax.get_ylabel() or None,
        "axis_on": axis_on,
        "n_lines": n_lines,
        "n_patches": n_patches,
        "n_collections": n_collections,
        "n_images": n_images,
        "n_texts": n_texts,
        "has_legend": has_legend,
        "has_data": data_artists > 0,
        # A visible plotting area with nothing in it. Axis-off spacer/gutter
        # axes are layout, not content, and are never flagged.
        "empty": axis_on and data_artists == 0 and n_texts == 0 and not has_legend,
    }


def figure_summary(fig: Figure) -> dict[str, Any]:
    """Describe what a matplotlib figure actually contains.

    Works for any figure, including the ``(fig, stats)`` returned by
    ``plot_somatic_interactions`` and other ``plot/wes.py`` functions.

    Returns
    -------
    dict
        ``figsize_in``, ``n_axes``, ``axes`` (one record per axes),
        ``n_empty_axes`` and ``warnings`` (human-readable strings).
    """
    if fig is None:
        raise ValueError("figure_summary() needs a Figure; got None (render first).")
    axes = [_axes_record(ax) for ax in fig.axes]
    width, height = fig.get_size_inches()
    warnings_list = []
    n_empty = sum(record["empty"] for record in axes)
    if not axes:
        warnings_list.append("figure has no axes: nothing was drawn")
    elif not any(record["has_data"] for record in axes):
        warnings_list.append("no axes contains any drawn data: the figure is blank")
    elif n_empty:
        warnings_list.append(f"{n_empty} visible axes contain no drawn data")
    record = {
        "figsize_in": [round(float(width), 3), round(float(height), 3)],
        "n_axes": len(axes),
        "n_empty_axes": int(n_empty),
        "axes": axes,
        "warnings": warnings_list,
    }
    # Functions that return a bare Figure (plot/wes.py) attach domain facts here.
    extra = getattr(fig, "pymaftools_summary", None)
    if extra:
        extra = dict(extra)
        record["warnings"] = warnings_list + list(extra.pop("warnings", []))
        record.update(extra)
    return record


def summary_path(filename: str | os.PathLike) -> Path:
    """Sidecar path for a figure: ``plot.png`` -> ``plot.json``."""
    return Path(os.fspath(filename)).with_suffix(".json")


def write_figure_summary(
    summary: dict[str, Any], filename: str | os.PathLike, *, figure_path=None
) -> Path:
    """Write ``summary`` as JSON to the sidecar of ``filename``; return its path."""
    from ..provenance import session_info, to_json

    record = dict(summary)
    if figure_path is not None:
        record.setdefault("figure_path", os.fspath(figure_path))
    record.setdefault("pymaftools_version", session_info()["pymaftools"])
    path = summary_path(filename)
    to_json(record, path)
    return path
