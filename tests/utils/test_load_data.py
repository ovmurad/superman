import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import ANY, patch

import src.utils.load_data as load_data


@pytest.fixture
def dummy_array():
    return np.array([1, 2, 3])


def test_load_gdrive_file_new_download(tmp_path, dummy_array):
    """File does not exist, should download, move, and load correctly."""
    fid = "fakeid"
    fname = "file.npy"
    data_dir = tmp_path.as_posix() + "/"
    fpath = data_dir + fname

    with patch("gdown.download", return_value=fname) as mock_download, \
         patch("shutil.move") as mock_move, \
         patch("pathlib.Path.is_file", return_value=False), \
         patch("numpy.load", return_value=dummy_array):

        out = load_data.load_gdrive_file(fid, np.ndarray, data_dir)

        mock_download.assert_called_once_with(id=fid, quiet=False, fuzzy=True)
        mock_move.assert_called_once_with(fname, fpath)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, dummy_array)


def test_load_gdrive_file_already_exists(tmp_path, dummy_array):
    """File already exists, should skip move and remove stray download."""
    fid = "fakeid"
    fname = "file.npy"
    data_dir = tmp_path.as_posix() + "/"
    fpath = data_dir + fname

    with patch("gdown.download", return_value=fname), \
         patch("shutil.move") as mock_move, \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("os.remove") as mock_remove, \
         patch("numpy.load", return_value=dummy_array):

        out = load_data.load_gdrive_file(fid, np.ndarray, data_dir)

        mock_move.assert_not_called()
        mock_remove.assert_called_once_with(fname)
        assert isinstance(out, np.ndarray)


def test_load_gdrive_file_wrong_type(tmp_path):
    """Raises TypeError if np.load returns wrong type."""
    fid = "fakeid"
    fname = "file.npy"
    data_dir = tmp_path.as_posix() + "/"

    with patch("gdown.download", return_value=fname), \
         patch("shutil.move"), \
         patch("pathlib.Path.is_file", return_value=False), \
         patch("numpy.load", return_value=123):  # wrong type

        with pytest.raises(TypeError):
            load_data.load_gdrive_file(fid, np.ndarray, data_dir)


def test_download_gdrive_folder_calls_loader(tmp_path, dummy_array):
    """Ensures folder download calls load_gdrive_file for each file."""
    fid = "folderid"
    data_dir = tmp_path.as_posix() + "/"
    files = [("id1", "f1.npy"), ("id2", "f2.npy")]

    with patch("gdown.download_folder", return_value=files) as mock_folder, \
         patch("src.utils.load_data.load_gdrive_file", return_value=dummy_array) as mock_loader:

        load_data.download_gdrive_folder(fid, data_dir)

        mock_folder.assert_called_once_with(id=fid, skip_download=True)
        assert mock_loader.call_count == len(files)
        mock_loader.assert_any_call("id1", ANY, data_dir + "f1.npy")
        mock_loader.assert_any_call("id2", ANY, data_dir + "f2.npy")
