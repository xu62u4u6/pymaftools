"""Auditable tumor mutation burden result records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TMBAudit:
    """Event-level audit and per-sample TMB summary.

    Serialise with :meth:`to_dict` / :meth:`to_json`.
    """

    events: pd.DataFrame
    """One row per input event with ``included`` and ``exclusion_reason``."""
    summary: pd.DataFrame
    """Per-sample numerator, callable territory, TMB, and policy metadata."""
    provenance: Any = None
    """:class:`pymaftools.provenance.Provenance` of the run (parameters, input
    checksum, sample universe), or ``None`` for a hand-built audit."""

    def __post_init__(self) -> None:
        required_event_columns = {"included", "exclusion_reason"}
        missing_event = required_event_columns - set(self.events.columns)
        if missing_event:
            raise ValueError(
                f"TMBAudit events are missing required column(s): {sorted(missing_event)}."
            )
        required_summary_columns = {
            "eligible",
            "mutation_count",
            "callable_mb",
            "TMB",
        }
        missing_summary = required_summary_columns - set(self.summary.columns)
        if missing_summary:
            raise ValueError(
                "TMBAudit summary is missing required column(s): "
                f"{sorted(missing_summary)}."
            )

        object.__setattr__(self, "events", self.events.copy())
        object.__setattr__(self, "summary", self.summary.copy())

    def included_events(self) -> pd.DataFrame:
        """Return events contributing to the TMB numerator."""
        return self.events.loc[self.events["included"]].copy()

    def exclusion_counts(self) -> pd.Series:
        """Return counts for each exclusion reason."""
        return (
            self.events.loc[~self.events["included"], "exclusion_reason"]
            .value_counts()
            .sort_index()
        )

    def to_dict(self, include_events: bool = False) -> dict:
        """Return a JSON-serialisable dict.

        Parameters
        ----------
        include_events : bool, default False
            Also include the full event ledger (can be large). By default
            only per-reason exclusion counts are included.
        """
        from ..provenance import SCHEMA_VERSION, dataframe_to_dict, get_provenance

        record: dict = {
            "schema_version": SCHEMA_VERSION,
            "provenance": get_provenance(self),
            "n_events": int(len(self.events)),
            "n_included_events": int(self.events["included"].sum()),
            "exclusion_counts": {
                str(reason): int(count)
                for reason, count in self.exclusion_counts().items()
            },
            "summary": dataframe_to_dict(self.summary, index=True),
        }
        if include_events:
            record["events"] = dataframe_to_dict(self.events)
        return record

    def to_json(
        self, path=None, *, include_events: bool = False, indent: int | None = 2
    ) -> str:
        """Serialise :meth:`to_dict` to JSON; write to ``path`` if given."""
        from ..provenance import to_json

        return to_json(self.to_dict(include_events=include_events), path, indent=indent)
