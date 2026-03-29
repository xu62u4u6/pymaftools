"""TCGA Clinical data builder."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from .mapping import load_file_mapping, resolve_files


class TCGAClinicalBuilder:
    """
    Build a clinical DataFrame from TCGA BCR clinical TXT files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing clinical supplement files.
    mapping : str, Path, or pd.DataFrame
        Path to file_to_case.tsv or pre-loaded mapping DataFrame.
    file_type : str, default "patient"
        Type of clinical file: patient, drug, radiation, follow_up, nte.
    """

    def __init__(
        self,
        data_dir: str | Path,
        mapping: str | Path | pd.DataFrame,
        file_type: Literal["patient", "drug", "radiation", "follow_up", "nte"] = "patient",
    ):
        self.data_dir = Path(data_dir)
        self.file_type = file_type

        if isinstance(mapping, (str, Path)):
            self.mapping_df = load_file_mapping(mapping)
        else:
            self.mapping_df = mapping

    def build(self) -> pd.DataFrame:
        """
        Read and merge clinical TXT files.

        Returns
        -------
        pd.DataFrame
            Clinical table indexed by bcr_patient_barcode.
        """
        pattern = f"*_clinical_{self.file_type}_*.txt"
        files = resolve_files(self.data_dir, pattern, self.mapping_df)

        if not files:
            raise FileNotFoundError(
                f"No clinical {self.file_type} files found in {self.data_dir}. "
                f"Expected {pattern} inside uuid subdirectories."
            )

        frames = []
        for f in files:
            df = pd.read_csv(f["filepath"], sep="\t", skiprows=[1, 2], dtype=str)
            frames.append(df)

        clinical = pd.concat(frames, ignore_index=True)
        clinical = clinical.drop_duplicates(subset="bcr_patient_barcode", keep="first")
        clinical = clinical.set_index("bcr_patient_barcode", drop=False)

        print(
            f"[{self.__class__.__name__}] "
            f"{len(clinical)} patients from {len(files)} {self.file_type} files"
        )
        return clinical
