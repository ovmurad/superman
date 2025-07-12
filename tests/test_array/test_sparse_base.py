import numpy as np
import pytest
from numpy.typing import DTypeLike
from scipy.sparse import csr_array
from src.array import SparseArray

from .conftest import DummyArray, dummy_array_groups

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("dummy_array", dummy_array_groups["sparse"])
def test_basic_properties_sparse(dummy_array: DummyArray[csr_array]) -> None:
    array = SparseArray(dummy_array.array)
    assert dummy_array.check(array)


@pytest.mark.parametrize("dummy_array", dummy_array_groups["sparse"])
@pytest.mark.parametrize("dtype", (np.float64, np.int64, np.bool))
def test_astype_sparse(dummy_array: DummyArray[csr_array], dtype: DTypeLike) -> None:
    array = SparseArray(dummy_array.array).astype(dtype)
    assert array.dtype == dtype


@pytest.mark.parametrize("dummy_array", dummy_array_groups["sparse"])
def test_copy_sparse(dummy_array: DummyArray[csr_array]) -> None:
    array = SparseArray(dummy_array.array).copy()

    assert array.array is not dummy_array.array
    assert not np.shares_memory(array.data, dummy_array.array.data)
    assert not np.shares_memory(array.indptr, dummy_array.array.indptr)
    assert not np.shares_memory(array.indices, dummy_array.array.indices)
