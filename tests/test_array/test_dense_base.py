from typing import Type

import numpy as np
import pytest

from src.array import DenseArray
from .conftest import test_array_groups, TestArray


@pytest.mark.parametrize("test_array", test_array_groups["dense"])
def test_basic_properties_dense(test_array: TestArray) -> None:
    array = DenseArray(test_array.array)
    assert test_array.check(array)


@pytest.mark.parametrize("test_array", test_array_groups["dense"])
@pytest.mark.parametrize("dtype", (np.float64, np.int64, np.bool))
def test_astype_dense(test_array: TestArray, dtype: Type) -> None:
    array = DenseArray(test_array.array).astype(dtype)
    assert array.dtype == dtype


@pytest.mark.parametrize("test_array", test_array_groups["dense"])
def test_copy_dense(test_array: TestArray) -> None:
    array = DenseArray(test_array.array).copy()
    assert not np.shares_memory(array.data, test_array.array.data)
