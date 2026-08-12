"""Explicit sample-universe metadata for auditable cohort analyses."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import pandas as pd
from pandas.api.types import is_bool_dtype


class SampleManifest:
    """Validate and expose the complete sample universe for an analysis.

    A mutation event table cannot represent eligible samples with zero called
    events.  ``SampleManifest`` keeps those samples explicit and records the
    patient-level analysis unit needed to detect repeated observations.

    Parameters
    ----------
    data : pandas.DataFrame
        Sample metadata. The index is the sample identifier unless
        ``sample_id_col`` is provided. Required columns are ``patient_id`` and
        boolean ``eligible``. Additional provenance and assay columns are kept.
    sample_id_col : str, optional
        Column to move to the index before validation.
    """

    REQUIRED_COLUMNS = ("patient_id", "eligible")

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        sample_id_col: str | None = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("SampleManifest data must be a pandas DataFrame.")

        frame = data.copy()
        if sample_id_col is not None:
            if sample_id_col not in frame.columns:
                raise ValueError(
                    f"sample_id_col '{sample_id_col}' not found in manifest."
                )
            frame = frame.set_index(sample_id_col, drop=True)

        missing_columns = [
            column for column in self.REQUIRED_COLUMNS if column not in frame.columns
        ]
        if missing_columns:
            raise ValueError(
                f"SampleManifest is missing required column(s): {missing_columns}."
            )
        if frame.index.hasnans:
            raise ValueError("SampleManifest sample identifiers cannot be missing.")
        if not frame.index.is_unique:
            duplicates = frame.index[frame.index.duplicated()].unique().tolist()
            raise ValueError(
                f"SampleManifest sample identifiers must be unique: {duplicates}."
            )
        if frame["patient_id"].isna().any():
            raise ValueError("SampleManifest patient_id values cannot be missing.")
        if frame["eligible"].isna().any() or not is_bool_dtype(
            frame["eligible"].dtype
        ):
            raise TypeError(
                "SampleManifest eligible must be a non-missing boolean column."
            )

        frame.index.name = "sample_ID"
        self._frame = frame

    @property
    def samples(self) -> pd.Index:
        """Return all declared sample identifiers, including excluded samples."""
        return self._frame.index.copy()

    @property
    def eligible_samples(self) -> pd.Index:
        """Return samples included in the declared analysis universe."""
        return self._frame.index[self._frame["eligible"]].copy()

    @property
    def patient_counts(self) -> pd.Series:
        """Return eligible sample counts per patient for independence audits."""
        return self.eligible_frame()["patient_id"].value_counts().sort_index()

    def eligible_frame(self) -> pd.DataFrame:
        """Return metadata for eligible samples in manifest order."""
        return self._frame.loc[self.eligible_samples].copy()

    def to_frame(self) -> pd.DataFrame:
        """Return a defensive copy of the complete manifest."""
        return self._frame.copy()

    def validate_event_samples(self, sample_ids: Iterable[object]) -> None:
        """Reject mutation events outside the declared eligible universe."""
        event_samples = pd.Index(sample_ids).unique()
        unknown = event_samples.difference(self.samples)
        if not unknown.empty:
            raise ValueError(
                "Mutation events contain sample(s) absent from SampleManifest: "
                f"{unknown.tolist()}."
            )

        ineligible = event_samples.difference(self.eligible_samples)
        if not ineligible.empty:
            raise ValueError(
                "Mutation events contain sample(s) marked ineligible in "
                f"SampleManifest: {ineligible.tolist()}."
            )

    def assert_independent(
        self,
        analysis_unit: Literal["sample", "patient"] = "patient",
    ) -> None:
        """Reject repeated eligible observations for the analysis unit.

        Sample identifiers are unique by construction. Patient-level analyses
        additionally require at most one eligible sample per patient unless a
        model explicitly accounts for repeated measures.
        """
        if analysis_unit == "sample":
            return
        if analysis_unit != "patient":
            raise ValueError("analysis_unit must be either 'sample' or 'patient'.")

        repeated = self.patient_counts[self.patient_counts > 1]
        if not repeated.empty:
            raise ValueError(
                "Patient-level analysis requires independent observations; "
                "repeated eligible patient(s): "
                f"{repeated.to_dict()}."
            )

    def copy(self) -> "SampleManifest":
        """Return an independent copy of the complete manifest."""
        return SampleManifest(self._frame)

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        return (
            f"SampleManifest(samples={len(self)}, "
            f"eligible={len(self.eligible_samples)}, "
            f"patients={self._frame['patient_id'].nunique()})"
        )
