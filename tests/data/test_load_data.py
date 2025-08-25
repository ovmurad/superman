import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path
from typing import Any

# Import your functions and classes
from src.data.load_data import (
    load_gdrive_file,
    download_gdrive_folder,
    load_swiss_roll,
    load_s_curve,
    load_file,
    Points,
    Coordinates,
)

# ------------------------------
# Test load_gdrive_file
# ------------------------------
@patch("src.data.load_data.gdown.download")
@patch("src.data.load_data.Path.is_file")
@patch("src.data.load_data.shutil.move")
@patch("src.data.load_data.os.remove")
@patch("src.data.load_data.np.load")
def test_load_gdrive_file(mock_np_load, mock_os_remove, mock_shutil_move, mock_is_file, mock_gdown_download):
    mock_gdown_download.return_value = "file.npy"
    mock_is_file.return_value = True
    mock_np_load.return_value = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    result = load_gdrive_file("dummy_fid", cls=Points)

    mock_gdown_download.assert_called_once()
    mock_np_load.assert_called_once_with("data/file.npy", allow_pickle=True)
    mock_os_remove.assert_called_once_with("file.npy")
    assert isinstance(result, Points)
    np.testing.assert_array_equal(result.as_nparray(), np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))

# ------------------------------
# Test download_gdrive_folder
# ------------------------------
@patch("src.data.load_data.load_gdrive_file")
@patch("src.data.load_data.gdown.download_folder")
def test_download_gdrive_folder(mock_download_folder, mock_load_gdrive_file):
    mock_download_folder.return_value = [("file_id_1", "subdir/file1.npy")]
    mock_load_gdrive_file.return_value = None

    download_gdrive_folder("folder_id")

    mock_download_folder.assert_called_once()
    mock_load_gdrive_file.assert_called_once_with("file_id_1", Any, "data/subdir/file1.npy")

# ------------------------------
# Test load_swiss_roll
# ------------------------------
def test_load_swiss_roll():
    points, coords = load_swiss_roll(50, noise=0.1)
    assert isinstance(points, Points)
    assert isinstance(coords, Coordinates)
    assert points.shape[0] == 50
    assert coords.shape[0] == 50

# ------------------------------
# Test load_s_curve
# ------------------------------
def test_load_s_curve():
    points, coords = load_s_curve(60, noise=0.2)
    assert isinstance(points, Points)
    assert isinstance(coords, Coordinates)
    assert points.shape[0] == 60
    assert coords.shape[0] == 60

# ------------------------------
# Test load_file
# ------------------------------
@patch("src.data.load_data.np.load")
def test_load_file(mock_np_load):
    mock_np_load.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
    points = load_file("dummy_path.npy")
    mock_np_load.assert_called_once_with("dummy_path.npy", allow_pickle=True)
    assert isinstance(points, Points)
    np.testing.assert_array_equal(points.as_nparray(), np.array([[1.0, 2.0], [3.0, 4.0]]))
