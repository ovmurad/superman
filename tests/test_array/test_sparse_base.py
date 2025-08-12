import numpy as np
import pytest
from src.array.base import SparseArray
from src.array.typing import _DTypeBound

from .dummy_array import DummyArray
from test_objects import dummy_array_groups

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("dummy_array", dummy_array_groups["sparse"])
def test_basic_properties_sparse(dummy_array: DummyArray) -> None:
    array = SparseArray(dummy_array.array)
    assert dummy_array.check(array)


@pytest.mark.parametrize("dummy_array", dummy_array_groups["sparse"])
@pytest.mark.parametrize("dtype", (np.float64, np.int64, np.bool))
def test_astype_sparse(dummy_array: DummyArray, dtype: _DTypeBound) -> None:
    array = SparseArray(dummy_array.array).astype(dtype)
    assert array.stype == dtype


@pytest.mark.parametrize("dummy_array", dummy_array_groups["sparse"])
def test_copy_sparse(dummy_array: DummyArray) -> None:
    array = SparseArray(dummy_array.array).copy()

    assert array.data is not dummy_array.array
    assert not np.shares_memory(array.data, dummy_array.array.data)
    assert not np.shares_memory(array.indptr, dummy_array.array.indptr)
    assert not np.shares_memory(array.indices, dummy_array.array.indices)
