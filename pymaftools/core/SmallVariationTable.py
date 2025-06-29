from .PivotTable import PivotTable


class SmallVariationTable(PivotTable):
    """
    SmallVariationTable class for handling small variation(SNV/INDEL) data.

    Inherits from PivotTable and provides specific functionality for small variation analysis.
    The _constructor property ensures that pandas operations return SmallVariationTable objects.
    """

    @property
    def _constructor(self):
        """Return constructor for pandas operations that preserves CNV type."""
        def _new_constructor(*args, **kwargs):
            obj = SmallVariationTable(*args, **kwargs)
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
