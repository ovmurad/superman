import os
import shutil
from pathlib import Path
from typing import Any, Type, TypeVar

import gdown
import numpy as np

DEFAULT_GDRIVE_FOLDER = "1NGfGcVpBarNtFMmdvQhq28hWBlVuxdQz"

T = TypeVar("T")


def load_gdrive_file(fid: str, cls: Type[T], data_dir: str = "data/") -> T:
    name: str = gdown.download(id=fid, quiet=False, fuzzy=True)
    path = data_dir + name

    if Path(path).is_file():
        print("File already downloaded. Loading only.")
        os.remove(name)
    else:
        shutil.move(name, path)

    out = np.load(path, allow_pickle=True)
    if isinstance(out, cls):
        return out
    raise TypeError(f"Expected {cls} got {type(out)}")


def download_gdrive_folder(fid: str = DEFAULT_GDRIVE_FOLDER, data_dir: str = "data/"):
    files = gdown.download_folder(id=fid, skip_download=True)
    for file in files:
        _ = load_gdrive_file(file[0], Any, data_dir + file[1])
