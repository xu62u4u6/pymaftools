from .PivotTable import PivotTable
import pandas as pd

class SignatureTable(PivotTable):
    """
    SignatureTable class for handling COSMIC Single Base Substitution (SBS) signature data.
    """
    @property
    def _constructor(self):
        def _new_constructor(*args, **kwargs):
            obj = SignatureTable(*args, **kwargs)
            # attempt to preserve metadata if available
            if hasattr(self, 'sample_metadata') and not self.sample_metadata.empty:
                try:
                    obj.sample_metadata = self.sample_metadata.copy()
                except:
                    pass
            if hasattr(self, 'feature_metadata') and not self.feature_metadata.empty:
                try:
                    obj.feature_metadata = self.feature_metadata.copy()
                except:
                    pass
            return obj
        return _new_constructor

    @classmethod
    def read_signature(cls, file_path):
        df = pd.read_csv(file_path, sep="\t", index_col=0).T
        return cls(df)