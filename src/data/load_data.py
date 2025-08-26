import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Type, TypeVar

import gdown  # type: ignore
import numpy as np
import sklearn.datasets  # type: ignore

from src.array.base import BaseArray
from src.geometry import Coordinates, Points

DEFAULT_GDRIVE_FOLDER = "1NGfGcVpBarNtFMmdvQhq28hWBlVuxdQz"

T = TypeVar("T", bound=BaseArray)


def load_gdrive_file(
    fid: str, cls: Type[T] | None, data_dir: str = "data/"
) -> T | None:
    name: str = gdown.download(id=fid, quiet=False, fuzzy=True)
    path: str = data_dir + name

    if Path(path).is_file():
        print("File already downloaded. Loading only.")
        os.remove(name)
    else:
        shutil.move(name, path)

    return cls(np.load(path, allow_pickle=True)) if cls is not None else None


def download_gdrive_folder(
    fid: str = DEFAULT_GDRIVE_FOLDER, data_dir: str = "data/"
) -> None:
    files: List[gdown.GoogleDriveFileToDownload] = gdown.download_folder(
        id=fid, skip_download=True
    )  # type: ignore
    for file in files:
        _ = load_gdrive_file(file[0], None, data_dir + file[1])


def load_swiss_roll(
    n_samples: int = 100,
    *,
    noise: float = 0.0,
    random_state: Optional[int] = None,
    hole: Optional[bool] = False
) -> Tuple[Points, Coordinates]:
    X, t = sklearn.datasets.make_swiss_roll(
        n_samples, noise=noise, random_state=random_state, hole=hole
    )
    t_arr = np.array(t)
    return (Points(np.array(X)), Coordinates(t_arr.reshape(t_arr.shape[0], 1)))


def load_s_curve(
    n_samples: int = 100, *, noise: float = 0.0, random_state: Optional[int] = None
) -> Tuple[Points, Coordinates]:
    X, t = sklearn.datasets.make_s_curve(
        n_samples, noise=noise, random_state=random_state
    )
    t_arr = np.array(t)
    return (Points(np.array(X)), Coordinates(t_arr.reshape(t_arr.shape[0], 1)))


def load_file(path: str) -> Points:
    return Points(np.load(path, allow_pickle=True))
