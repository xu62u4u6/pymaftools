"""Auditable tumor mutation burden result records."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TMBAudit:
    """Event-level audit and per-sample TMB summary.

    Attributes
    ----------
    events : pandas.DataFrame
        One row per input event with ``included`` and ``exclusion_reason``.
    summary : pandas.DataFrame
        Per-sample numerator, callable territory, TMB, and policy metadata.
    """

    events: pd.DataFrame
    summary: pd.DataFrame

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
