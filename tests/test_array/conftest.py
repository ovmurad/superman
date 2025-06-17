import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_array

from src.array import SparseArray, DenseArray


@pytest.fixture
def dense_data() -> NDArray:
    return np.array([[1.0, 0.0], [0.0, 2.0]])


@pytest.fixture
def sparse_data() -> csr_array:
    return csr_array([[1.0, 0.0], [0.0, 2.0]])


@pytest.fixture
def dense_array(dense_data: NDArray) -> DenseArray:
    return DenseArray(dense_data)


@pytest.fixture
def sparse_array(sparse_data: csr_array) -> SparseArray:
    return SparseArray(sparse_data)
