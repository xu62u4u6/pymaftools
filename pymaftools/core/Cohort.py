from __future__ import annotations

import pandas as pd
import copy
import warnings

from .PivotTable import PivotTable
import sqlite3
from pathlib import Path


class Cohort:
    def __init__(self, name: str, description: str = "") -> None:
        """
        Parameters
        ----------
        name : str
            Name of the cohort.
        description : str, optional
            Description of the cohort, by default "".

        Attributes
        ----------
        name : str
            Name of the cohort.
        description : str
            Description of the cohort.
        tables : dict[str, PivotTable]
            A dictionary mapping table names to PivotTable objects.
        sample_metadata : pandas.DataFrame or None
            Combined sample metadata for the entire cohort.
        sample_IDs : pandas.Index or None
            The primary sample index for the cohort.
        """
        self.name = name
        self.description = description
        self.tables: dict[str, PivotTable] = {}
        self.sample_metadata: pd.DataFrame | None = None
        self.sample_IDs: pd.Index | None = None

    def add_sample_metadata(self, new_metadata: pd.DataFrame, source: str = "") -> None:
        """
        Add or merge sample metadata into the cohort.

        Parameters
        ----------
        new_metadata : pd.DataFrame
            DataFrame containing sample metadata, indexed by sample ID.
        source : str, optional
            Name of the source providing the metadata, used in error messages,
            by default "".

        Raises
        ------
        TypeError
            If ``new_metadata`` is not a pandas DataFrame.
        ValueError
            If the index of ``new_metadata`` does not match the existing cohort
            index, or if shared columns have conflicting values.
        """
        if not isinstance(new_metadata, pd.DataFrame):
            raise TypeError("Sample metadata must be a pandas DataFrame.")

        if self.sample_metadata is None:
            self.sample_metadata = new_metadata.copy()
            self.sample_IDs = new_metadata.index
            return

        if not new_metadata.index.equals(self.sample_metadata.index):
            raise ValueError(
                f"Sample metadata index from '{source}' does not match existing cohort index."
            )

        shared_cols = set(self.sample_metadata.columns) & set(new_metadata.columns)
        new_cols = set(new_metadata.columns) - set(self.sample_metadata.columns)

        if shared_cols:
            cohort_shared = self.sample_metadata[list(shared_cols)]
            incoming_shared = new_metadata[list(shared_cols)]

            equal_mask = (cohort_shared == incoming_shared) | (
                cohort_shared.isna() & incoming_shared.isna()
            )
            non_equal = equal_mask.columns[~equal_mask.all()]
            if not non_equal.empty:
                raise ValueError(
                    f"Shared metadata columns have conflicting values from '{source}': {', '.join(non_equal)}"
                )

        self.sample_metadata = pd.concat(
            [self.sample_metadata, new_metadata[list(new_cols)]], axis=1
        )

        for key, table in self.tables.items():
            table.sample_metadata = self.sample_metadata.copy()

    def add_table(self, table: PivotTable, table_name: str) -> None:
        """
        Add a PivotTable to the cohort.

        Parameters
        ----------
        table : PivotTable
            The PivotTable to add.
        table_name : str
            Name to assign to the table within the cohort.

        Raises
        ------
        TypeError
            If ``table`` is not an instance of PivotTable.
        """
        if not isinstance(table, PivotTable):
            raise TypeError(
                f"Assay data for '{table_name}' must be an instance of PivotTable."
            )

        if self.sample_IDs is None:
            self.sample_IDs = table.sample_metadata.index
            self.tables[table_name] = table
        else:
            table = table.subset(samples=self.sample_IDs)
            self.tables[table_name] = table

        # Reindex sample_metadata to cohort's sample_IDs before merging
        # (table may have fewer samples after subset)
        meta = table.sample_metadata.reindex(self.sample_IDs)
        self.add_sample_metadata(meta, source=table_name)

    def _is_index_matched(self, table: PivotTable) -> bool:
        """
        Check whether a table's sample index matches the cohort's sample IDs.

        Parameters
        ----------
        table : PivotTable
            The table to check.

        Returns
        -------
        bool
            True if the table's sample metadata index matches the cohort's
            sample IDs.
        """
        return table.sample_metadata.index.equals(self.sample_IDs)

    def remove_table(self, table_name: str) -> None:
        """
        Remove a table from the cohort by name.

        Parameters
        ----------
        table_name : str
            Name of the table to remove.
        """
        if table_name in self.tables:
            del self.tables[table_name]

    def subset(self, samples: list[str] = []) -> Cohort:
        """
        Create a new Cohort containing only the specified samples.

        Parameters
        ----------
        samples : list of str, optional
            Sample IDs to keep, by default [].

        Returns
        -------
        Cohort
            A new Cohort containing only the specified samples.
        """
        cohort = self.copy()
        for table_name, table in cohort.tables.items():
            cohort.tables[table_name] = table.subset(samples=samples)

        if cohort.sample_metadata is not None:
            cohort.sample_metadata = cohort.sample_metadata.loc[samples, :].copy()
        cohort.sample_IDs = samples
        return cohort

    def copy(self, deep: bool = True) -> Cohort:
        """
        Create a copy of the Cohort.

        Parameters
        ----------
        deep : bool, optional
            If True, perform a deep copy of all tables and metadata.
            If False, perform a shallow copy, by default True.

        Returns
        -------
        Cohort
            A new Cohort instance.
        """
        new_instance = Cohort(self.name, self.description)
        new_instance.tables = copy.deepcopy(self.tables) if deep else self.tables.copy()
        new_instance.sample_metadata = (
            self.sample_metadata.copy(deep=True)
            if deep and self.sample_metadata is not None
            else self.sample_metadata
        )
        new_instance.sample_IDs = (
            self.sample_IDs.copy() if self.sample_IDs is not None else None
        )
        return new_instance

    def __getattr__(self, name: str) -> PivotTable:
        if name in self.tables:
            return self.tables[name]
        raise AttributeError(f"'Cohort' object has no attribute '{name}'")

    def info(self) -> str:
        """
        Return a summary string of the Cohort structure.

        Returns
        -------
        str
            A tree-formatted summary showing each table's dimensions and
            metadata counts.
        """
        lines = [f"Cohort('{self.name}')"]

        table_names = list(self.tables.keys())
        for i, table_name in enumerate(table_names):
            table = self.tables[table_name]
            n_samples = len(table.columns)
            n_features = len(table.index)
            n_sample_meta = (
                len(table.sample_metadata.columns)
                if hasattr(table, "sample_metadata")
                and table.sample_metadata is not None
                else 0
            )
            n_feature_meta = (
                len(table.feature_metadata.columns)
                if hasattr(table, "feature_metadata")
                and table.feature_metadata is not None
                else 0
            )

            prefix = "└──" if i == len(table_names) - 1 else "├──"
            lines.append(
                f"{prefix} {table_name}: {n_samples} samples × {n_features} features (sample_meta: {n_sample_meta}, feature_meta: {n_feature_meta})"
            )

        if not table_names:
            lines.append("└── (no tables)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a string representation of the Cohort."""
        return self.info()

    def to_sql_registry(self) -> pd.DataFrame:
        """
        Generate a registry DataFrame for SQL table mapping.

        Creates a mapping between logical table names and their corresponding
        SQL table names for data, sample metadata, and feature metadata.

        Returns
        -------
        pd.DataFrame
            Registry with columns: sql_table_name, cohort_name, table_name, type
        """
        records = [
            {
                "sql_table_name": f"{table_name}{suffix}",
                "cohort_name": self.name,
                "table_name": table_name,
                "type": type_name,
            }
            for table_name in self.tables.keys()
            for suffix, type_name in [
                ("", "data"),
                ("__sample_metadata", "sample_metadata"),
                ("__feature_metadata", "feature_metadata"),
            ]
        ]
        return pd.DataFrame(records)

    def to_sqlite(self, db_path: str) -> None:
        """
        Save Cohort to SQLite database format.

        .. deprecated:: 0.4.0
            ``to_sqlite`` will be removed in a future version.
            Use :meth:`to_hdf5` instead, which supports larger datasets
            without column limits.

        Parameters
        ----------
        db_path : str
            Path to the output SQLite database file.
        """
        warnings.warn(
            "to_sqlite is deprecated and will be removed in a future version. "
            "Use to_hdf5() instead, which supports larger datasets.",
            DeprecationWarning,
            stacklevel=2,
        )
        db_path = Path(db_path)

        if db_path.exists():
            db_path.unlink()

        conn = sqlite3.connect(str(db_path))
        registry = self.to_sql_registry()

        for _, row in registry.iterrows():
            table_name = row["table_name"]
            sql_table_name = row["sql_table_name"]
            kind = row["type"]
            table = self.tables[table_name].copy().rename_index_and_columns()

            if kind == "data":
                table.to_sql(sql_table_name, conn, if_exists="replace", index=True)
            elif kind == "sample_metadata":
                table.sample_metadata.to_sql(
                    sql_table_name, conn, if_exists="replace", index=True
                )
            elif kind == "feature_metadata":
                table.feature_metadata.to_sql(
                    sql_table_name, conn, if_exists="replace", index=True
                )

        registry.to_sql("registry", conn, if_exists="replace", index=False)
        conn.close()
        print(f"[Cohort] saved to {db_path}")

    @classmethod
    def read_sqlite(cls, db_path: str) -> Cohort:
        """
        Load Cohort from SQLite database format.

        .. deprecated:: 0.4.0
            ``read_sqlite`` will be removed in a future version.
            Use :meth:`read_hdf5` instead.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file.

        Returns
        -------
        Cohort
            Loaded Cohort object.
        """
        warnings.warn(
            "read_sqlite is deprecated and will be removed in a future version. "
            "Use read_hdf5() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        conn = sqlite3.connect(db_path)
        registry_df = pd.read_sql("SELECT * FROM registry", conn)

        cohort_name = registry_df["cohort_name"].iloc[0]
        cohort = cls(cohort_name)

        for table_name in registry_df["table_name"].unique():
            data = pd.read_sql(
                f"SELECT * FROM '{table_name}'", conn, index_col="feature"
            )
            sample_metadata = pd.read_sql(
                f"SELECT * FROM '{table_name}__sample_metadata'",
                conn,
                index_col="sample",
            )
            feature_metadata = pd.read_sql(
                f"SELECT * FROM '{table_name}__feature_metadata'",
                conn,
                index_col="feature",
            )

            pivot = PivotTable(data)
            pivot.sample_metadata = sample_metadata
            pivot.feature_metadata = feature_metadata

            cohort.add_table(pivot, table_name)

        conn.close()
        print(f"[Cohort] loaded from {db_path}")
        return cohort

    def to_hdf5(self, h5_path: str) -> None:
        """
        Save Cohort to HDF5 format.

        HDF5 format is recommended for large datasets as it doesn't have
        the column limit that SQLite has (~2000 columns).

        Parameters
        ----------
        h5_path : str
            Path to the output HDF5 file.
        """
        h5_path = Path(h5_path)

        if h5_path.exists():
            h5_path.unlink()

        with pd.HDFStore(str(h5_path), mode="w") as store:
            cohort_meta = pd.DataFrame(
                {"name": [self.name], "description": [self.description]}
            )
            store.put("cohort_metadata", cohort_meta)

            table_names = pd.DataFrame({"table_name": list(self.tables.keys())})
            store.put("table_registry", table_names)

            for table_name, table in self.tables.items():
                table_copy = table.copy().rename_index_and_columns()
                store.put(f"{table_name}/data", table_copy.T)
                store.put(f"{table_name}/sample_metadata", table_copy.sample_metadata)
                store.put(f"{table_name}/feature_metadata", table_copy.feature_metadata)

        print(f"[Cohort] saved to {h5_path}")

    @classmethod
    def read_hdf5(cls, h5_path: str) -> Cohort:
        """
        Load Cohort from HDF5 format.

        Parameters
        ----------
        h5_path : str
            Path to the HDF5 file.

        Returns
        -------
        Cohort
            Loaded Cohort object.
        """
        with pd.HDFStore(str(h5_path), mode="r") as store:
            cohort_meta = store.get("cohort_metadata")
            cohort = cls(
                name=cohort_meta["name"].iloc[0],
                description=cohort_meta["description"].iloc[0],
            )

            table_registry = store.get("table_registry")

            for table_name in table_registry["table_name"]:
                data = store.get(f"{table_name}/data").T
                sample_metadata = store.get(f"{table_name}/sample_metadata")
                feature_metadata = store.get(f"{table_name}/feature_metadata")

                pivot = PivotTable(data)
                pivot.sample_metadata = sample_metadata
                pivot.feature_metadata = feature_metadata

                cohort.add_table(pivot, table_name)

        print(f"[Cohort] loaded from {h5_path}")
        return cohort


# cohort = Cohort(name="ASC")
# cohort.add_table(cnv_table, "CNV")
# cohort.add_table(snv_table, "SNV")
# cohort.subset(samples=["AS_001_T", "AS_001_A", "AS_001_S"]).SNV
