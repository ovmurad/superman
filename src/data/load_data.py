import os
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, TypeVar

import gdown
import numpy as np
import sklearn.datasets

from src.geometry import Points, Coordinates

DEFAULT_GDRIVE_FOLDER = "1NGfGcVpBarNtFMmdvQhq28hWBlVuxdQz"

T = TypeVar("T")


def load_gdrive_file(fid: str, cls: Type[T], data_dir: str = "data/") -> T:
    name: str = gdown.download(id=fid, quiet=False, fuzzy=True)
    path: str = data_dir + name

    if Path(path).is_file():
        print("File already downloaded. Loading only.")
        os.remove(name)
    else:
        shutil.move(name, path)

    return T(np.load(path, allow_pickle=True))


def download_gdrive_folder(fid: str = DEFAULT_GDRIVE_FOLDER, data_dir: str = "data/"):
    files: List[gdown.GoogleDriveFileToDownload] = gdown.download_folder(id=fid, skip_download=True)
    for file in files:
        _ = load_gdrive_file(file[0], Any, data_dir + file[1])


def load_swiss_roll(n_samples: int = 100, noise: float = 0.0, random_state: Optional[int] = None, hole: Optional[bool] = False) -> Tuple[Points, Coordinates]:
    X, t = sklearn.datasets.make_swiss_roll(n_samples, noise, random_state, hole)
    return (Points(X), Coordinates(t))


def load_s_curve(n_samples: int = 100, noise: float = 0.0, random_state: Optional[int] = None) -> Tuple[Points, Coordinates]:
    X, t = sklearn.datasets.make_s_curve(n_samples, noise, random_state)
    return (Points(X), Coordinates(t))


def load_file(path: str) -> Points:
    return Points(np.load(path, allow_pickle=True))
