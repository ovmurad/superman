from typing import Type

import numpy as np
import pytest

from src.array import SparseArray
from .conftest import test_array_groups, TestArray


@pytest.mark.parametrize("test_array", test_array_groups["sparse"])
def test_basic_properties_sparse(test_array: TestArray) -> None:
    array = SparseArray(test_array.array)
    assert test_array.check(array)


@pytest.mark.parametrize("test_array", test_array_groups["sparse"])
@pytest.mark.parametrize("dtype", (np.float64, np.int64, np.bool))
def test_astype_sparse(test_array: TestArray, dtype: Type) -> None:
    array = SparseArray(test_array.array).astype(dtype)
    assert array.dtype == dtype


@pytest.mark.parametrize("test_array", test_array_groups["sparse"])
def test_copy_sparse(test_array: TestArray) -> None:
    array = SparseArray(test_array.array).copy()

    assert array.array is not test_array.array
    assert not np.shares_memory(array.data, test_array.array.data)
    assert not np.shares_memory(array.indptr, test_array.array.indptr)
    assert not np.shares_memory(array.indices, test_array.array.indices)
