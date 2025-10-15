import pandas as pd
import pickle
import copy
from .PivotTable import PivotTable
import sqlite3
from pathlib import Path

class Cohort:
    def __init__(self, name, description=""):
        self.name = name
        self.description = description
        self.tables = {}
        self.sample_metadata = None
        self.sample_IDs = None

    def add_sample_metadata(self, new_metadata: pd.DataFrame, source: str = ""):
        if not isinstance(new_metadata, pd.DataFrame):
            raise TypeError("Sample metadata must be a pandas DataFrame.")
        
        if self.sample_metadata is None:
            self.sample_metadata = new_metadata.copy()
            self.sample_IDs = new_metadata.index
            return

        if not new_metadata.index.equals(self.sample_metadata.index):
            raise ValueError(f"Sample metadata index from '{source}' does not match existing cohort index.")

        shared_cols = set(self.sample_metadata.columns) & set(new_metadata.columns)
        new_cols = set(new_metadata.columns) - set(self.sample_metadata.columns)

        if shared_cols:
            cohort_shared = self.sample_metadata[list(shared_cols)]
            incoming_shared = new_metadata[list(shared_cols)]

            equal_mask = (cohort_shared == incoming_shared) | (cohort_shared.isna() & incoming_shared.isna())
            non_equal = equal_mask.columns[~equal_mask.all()]
            if not non_equal.empty:
                raise ValueError(f"Shared metadata columns have conflicting values from '{source}': {', '.join(non_equal)}")

        # 安全合併：只補充新欄位
        self.sample_metadata = pd.concat(
            [self.sample_metadata, new_metadata[list(new_cols)]],
            axis=1
        )

        # 更新所有sample_metadata
        for key, table in self.tables.items():
            table.sample_metadata = self.sample_metadata.copy()
            
    def add_table(self, table: PivotTable, table_name: str):
        if not isinstance(table, PivotTable):
            raise TypeError(f"Assay data for '{table_name}' must be an instance of PivotTable.")
        
        # If this is the first table added, use its sample IDs as the reference
        if self.sample_IDs is None:
            self.sample_IDs = table.sample_metadata.index
            self.tables[table_name] = table
        else:
            # Otherwise, align the samples before adding the table
            table = table.subset(samples=self.sample_IDs)
            self.tables[table_name] = table
            
        self.add_sample_metadata(table.sample_metadata, source=table_name)

    def _is_index_matched(self, table):
        return table.sample_metadata.index.equals(self.sample_IDs)

    def remove_table(self, table_name):
        if table_name in self.tables:
            del self.tables[table_name]

    def subset(self, 
        samples: list = []
        ):
        
        cohort = self.copy()
        for table_name, table in cohort.tables.items():
            cohort.tables[table_name] = table.subset(samples=samples)
        
        # subset sample_metadata
        if cohort.sample_metadata is not None:
            cohort.sample_metadata = cohort.sample_metadata.loc[samples, :].copy()
        cohort.sample_IDs = samples
        return cohort
    
    def copy(self, deep=True):
        new_instance = Cohort(self.name, self.description)
        new_instance.tables = copy.deepcopy(self.tables) if deep else self.tables.copy()
        new_instance.sample_metadata = (
            self.sample_metadata.copy(deep=True) if deep and self.sample_metadata is not None else self.sample_metadata
        )
        return new_instance
    
    def __getattr__(self, name):
        if name in self.tables:
            return self.tables[name]
        raise AttributeError(f"'Cohort' object has no attribute '{name}'")
    
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
                "type": type_name
            }
            for table_name in self.tables.keys()
            for suffix, type_name in [("", "data"), ("__sample_metadata", "sample_metadata"), ("__feature_metadata", "feature_metadata")]
        ]
        return pd.DataFrame(records)
    
    def to_sqlite(self, db_path: str):
        """ Save Cohort to SQLite database format."""
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
                table.sample_metadata.to_sql(sql_table_name, conn, if_exists="replace", index=True)
            elif kind == "feature_metadata":
                table.feature_metadata.to_sql(sql_table_name, conn, if_exists="replace", index=True)

        registry.to_sql("registry", conn, if_exists="replace", index=False)
        conn.close()
        print(f"[Cohort] saved to {db_path}")

    @classmethod
    def read_sqlite(cls, db_path: str):
        """ Load Cohort from SQLite database format."""
        conn = sqlite3.connect(db_path)
        registry_df = pd.read_sql("SELECT * FROM registry", conn)

        cohort_name = registry_df["cohort_name"].iloc[0]
        cohort = cls(cohort_name)

        for table_name in registry_df["table_name"].unique():
            data = pd.read_sql(f"SELECT * FROM '{table_name}'", conn, index_col="feature")
            sample_metadata = pd.read_sql(f"SELECT * FROM '{table_name}__sample_metadata'", conn, index_col="sample")
            feature_metadata = pd.read_sql(f"SELECT * FROM '{table_name}__feature_metadata'", conn, index_col="feature")

            pivot = PivotTable(data)
            pivot.sample_metadata = sample_metadata
            pivot.feature_metadata = feature_metadata

            cohort.add_table(pivot, table_name)

        conn.close()
        print(f"[Cohort] loaded from {db_path}")
        return cohort
    
# cohort = Cohort(name="ASC")
# cohort.add_table(cnv_table, "CNV")
# cohort.add_table(snv_table, "SNV")
# cohort.subset(samples=["AS_001_T", "AS_001_A", "AS_001_S"]).SNV