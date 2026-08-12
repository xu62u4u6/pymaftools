"""Observed-versus-unobserved state for feature-by-sample matrices."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype


class ObservationMask:
    """Boolean feature-by-sample mask defining valid analysis denominators.

    ``True`` means the feature was observable for that sample and therefore a
    negative value can be interpreted as observed absence. ``False`` means the
    cell was unobserved or excluded and must not enter a frequency denominator.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("ObservationMask data must be a pandas DataFrame.")
        if not data.index.is_unique or not data.columns.is_unique:
            raise ValueError("ObservationMask axes must be unique.")
        if data.isna().any().any() or not all(
            is_bool_dtype(dtype) for dtype in data.dtypes
        ):
            raise TypeError(
                "ObservationMask values must be non-missing booleans."
            )
        self._data = data.copy()

    @classmethod
    def fully_observed(cls, table) -> "ObservationMask":
        """Create an explicit all-observed mask for a table."""
        return cls(pd.DataFrame(True, index=table.index, columns=table.columns))

    def to_frame(self) -> pd.DataFrame:
        """Return a defensive copy of the mask."""
        return self._data.copy()

    def validate_for(self, table) -> None:
        """Validate exact feature and sample alignment with ``table``."""
        if not self._data.index.equals(table.index):
            raise ValueError(
                "ObservationMask feature index does not match table index."
            )
        if not self._data.columns.equals(table.columns):
            raise ValueError(
                "ObservationMask sample columns do not match table columns."
            )

    def subset(self, *, features=None, samples=None) -> "ObservationMask":
        """Return an axis-aligned subset in the requested order."""
        features = self._data.index if features is None else features
        samples = self._data.columns if samples is None else samples
        return ObservationMask(self._data.loc[features, samples])

    def reindex(self, *, features=None, samples=None) -> "ObservationMask":
        """Reindex the mask, treating newly introduced cells as unobserved."""
        features = self._data.index if features is None else features
        samples = self._data.columns if samples is None else samples
        return ObservationMask(
            self._data.reindex(index=features, columns=samples, fill_value=False)
        )

    def copy(self) -> "ObservationMask":
        """Return an independent copy of the mask."""
        return ObservationMask(self._data)

    def calculate_feature_frequency(self, table) -> pd.Series:
        """Calculate event frequency using only observed cells."""
        self.validate_for(table)
        observed = self._data
        present = pd.DataFrame(table.to_binary_table()).astype(bool)

        impossible = present & ~observed
        if impossible.any().any():
            locations = list(zip(*impossible.to_numpy().nonzero()))
            first_locations = [
                (str(table.index[row]), str(table.columns[column]))
                for row, column in locations[:5]
            ]
            raise ValueError(
                "ObservationMask marks present event(s) as unobserved: "
                f"{first_locations}."
            )

        numerator = (present & observed).sum(axis=1).astype(float)
        denominator = observed.sum(axis=1).astype(float)
        return numerator.divide(denominator.where(denominator > 0))

    def __repr__(self) -> str:
        observed = int(self._data.to_numpy().sum())
        return (
            f"ObservationMask(features={self._data.shape[0]}, "
            f"samples={self._data.shape[1]}, observed_cells={observed})"
        )
