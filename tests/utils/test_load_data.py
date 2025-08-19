# test_loader.py
from typing import Dict
from unittest.mock import patch

import numpy as np
import pytest
import src.utils.load_data as loader  # replace with the filename where your functions live


@pytest.fixture
def fake_numpy_array():
    return np.array([1, 2, 3])


def test_load_gdrive_file(fake_numpy_array, tmp_path):
    fid = "fake_file_id"
    fake_file = "file.npy"
    data_dir = tmp_path.as_posix() + "/"

    with patch("gdown.download", return_value=fake_file) as mock_download, patch(
        "shutil.move"
    ) as mock_move, patch("numpy.load", return_value=fake_numpy_array) as mock_npload:

        arr = loader.load_gdrive_file(fid, np.ndarray, data_dir=data_dir)

        mock_download.assert_called_once_with(id=fid, quiet=False, fuzzy=True)
        mock_move.assert_called_once_with(fake_file, data_dir + fake_file)
        mock_npload.assert_called_once_with(data_dir + fake_file, allow_pickle=True)

        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, fake_numpy_array)


def test_load_gdrive_folder(fake_numpy_array, tmp_path):
    fid = "fake_folder_id"
    fake_files = [
        ("id1", "file1.npy"),
        ("id2", "file2.npy"),
    ]
    data_dir = tmp_path.as_posix() + "/"

    with patch(
        "gdown.download_folder", return_value=fake_files
    ) as mock_download_folder, patch(
        "src.utils.load_data.load_gdrive_file", return_value=fake_numpy_array
    ) as mock_load_file:

        result: Dict[str, np.ndarray] = loader.load_gdrive_folder(
            fid, np.ndarray, data_dir=data_dir
        )

        mock_download_folder.assert_called_once_with(id=fid, skip_download=True)
        # should call load_gdrive_file twice
        assert mock_load_file.call_count == 2

        assert isinstance(result, dict)
        assert set(result.keys()) == {"file1.npy", "file2.npy"}
        for v in result.values():
            np.testing.assert_array_equal(v, fake_numpy_array)
