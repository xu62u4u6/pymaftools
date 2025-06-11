import pandas as pd
import pickle
import copy
from .PivotTable import PivotTable

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
        cohort.sample_metadata = cohort.sample_metadata.loc[samples, :].copy()
        return cohort
    
    def order(self, group_col=None, group_order=None):
        cohort = self.copy()
        for table_name, table in cohort.tables.items():
            cohort.tables[table_name] = table.order(group_col=group_col, group_order=group_order)
        return cohort
    
    def copy(self, deep=True):
        new_instance = Cohort(self.name, self.description)
        new_instance.tables = copy.deepcopy(self.tables) if deep else self.tables.copy()
        new_instance.sample_metadata = (
            self.sample_metadata.copy(deep=True) if deep and self.sample_metadata is not None else self.sample_metadata
        )
        return new_instance

    def to_pickle(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Data saved to {file_path}.")

    @classmethod
    def read_pickle(cls, file_path):
        with open(file_path, 'rb') as f:
            cohort_instance = pickle.load(f)
        return cohort_instance
    
    def __getattr__(self, name):
        if name in self.tables:
            return self.tables[name]
        raise AttributeError(f"'Cohort' object has no attribute '{name}'")
    
# cohort = Cohort(name="ASC")
# cohort.add_table(cnv_table, "CNV")
# cohort.add_table(snv_table, "SNV")
# cohort.subset(samples=["AS_001_T", "AS_001_A", "AS_001_S"]).SNV