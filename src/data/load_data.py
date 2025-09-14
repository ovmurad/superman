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
    """
    Download a file from Google Drive using its file ID, move it into a data directory,
    and optionally load it into a class.

    :param fid: Google Drive file ID.
    :param cls: Class to wrap the loaded numpy array in. If None, no class is returned.
    :param data_dir: Target directory where the file will be stored. (default: "data/").

    :return: An instance of `cls` containing the loaded data, or None if `cls` is None.
    """

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
    """
    Download all files from a Google Drive folder by ID and store them in a local directory.

    :param fid: Google Drive folder ID. Defaults to `DEFAULT_GDRIVE_FOLDER`.
    :param data_dir: Target directory where files will be stored. (default: "data/").

    :return: None
    """

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
    """
    Generate a synthetic Swiss roll dataset.

    :param n_samples: Number of samples to generate. (default: 100).
    :param noise: Standard deviation of Gaussian noise added to the data. (default: 0.0).
    :param random_state: Seed for reproducibility. (default: None).
    :param hole: Whether to generate the Swiss roll with a hole. (default: False).

    :return: A tuple (Points, Coordinates) where Points are the 3D data
             and Coordinates are the unrolled 1D parameters.
    """

    X, t = sklearn.datasets.make_swiss_roll(
        n_samples, noise=noise, random_state=random_state, hole=hole
    )
    t_arr = np.array(t)
    return (Points(np.array(X)), Coordinates(t_arr.reshape(t_arr.shape[0], 1)))


def load_s_curve(
    n_samples: int = 100, *, noise: float = 0.0, random_state: Optional[int] = None
) -> Tuple[Points, Coordinates]:
    """
    Generate a synthetic S-curve dataset.

    :param n_samples: Number of samples to generate. (default: 100).
    :param noise: Standard deviation of Gaussian noise added to the data. (default: 0.0).
    :param random_state: Seed for reproducibility. (default: None).

    :return: A tuple (Points, Coordinates) where Points are the 3D data
             and Coordinates are the unrolled 1D parameters.
    """

    X, t = sklearn.datasets.make_s_curve(
        n_samples, noise=noise, random_state=random_state
    )
    t_arr = np.array(t)
    return (Points(np.array(X)), Coordinates(t_arr.reshape(t_arr.shape[0], 1)))


def load_file(path: str) -> Points:
    """
    Load a numpy file into a Points object.

    :param path: Path to the .npy file.

    :return: A Points object containing the loaded data.
    """

    return Points(np.load(path, allow_pickle=True))
