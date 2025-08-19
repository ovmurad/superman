import os
from pathlib import Path
import shutil
from typing import Type, TypeVar, Union, Dict
import numpy as np
import gdown

T = TypeVar("T")

def load_gdrive_file(fid: str, cls: Type[T], data_dir: str = "data/") -> T:
    name: str = gdown.download(id=fid, quiet=False, fuzzy=True)
    shutil.move(name, data_dir + name)
    return np.load(data_dir + name, allow_pickle=True)

def load_gdrive_folder(fid: str, cls: Type[T], data_dir: str = "data/") -> Dict[str, T]:
    files = gdown.download_folder(id=fid, skip_download=True)
    out: Dict[str, T] = {}
    for file in files:
        out[file[1]] = load_gdrive_file(file[0], cls, data_dir + file[1])
    return out
