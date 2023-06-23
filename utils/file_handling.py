# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Union
import pandas as pd


def load_wsi_files_from_csv(csv_path: Union[Path, str], wsi_extension: str) -> List:
    """Load filenames from csv file with column name "Filename"

    Args:
        csv_path (Union[Path, str]): Path to csv file
        wsi_extension (str): WSI file ending (suffix)

    Returns:
        List: _description_
    """
    wsi_filelist = pd.read_csv(csv_path)
    wsi_filelist = wsi_filelist["Filename"].to_list()
    wsi_filelist = [f for f in wsi_filelist if Path(f).suffix == f".{wsi_extension}"]

    return wsi_filelist
