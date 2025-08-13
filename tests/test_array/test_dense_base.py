import numpy as np
import pytest
from src.array.base import DenseArray
from src.array.typing import _DTypeBound

from .dummy_array import DummyArray
from tests.test_utils import dummy_array_groups

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("dummy_array", dummy_array_groups["dense"])
def test_basic_properties_dense(dummy_array: DummyArray) -> None:
    array = DenseArray(dummy_array.array)
    assert dummy_array.check(array)


@pytest.mark.parametrize("dummy_array", dummy_array_groups["dense"])
@pytest.mark.parametrize("dtype", (np.float64, np.int64, np.bool))
def test_astype_dense(dummy_array: DummyArray, dtype: _DTypeBound) -> None:
    array = DenseArray(dummy_array.array).astype(dtype)
    assert array.stype == dtype


@pytest.mark.parametrize("dummy_array", dummy_array_groups["dense"])
def test_copy_dense(dummy_array: DummyArray) -> None:
    array = DenseArray(dummy_array.array).copy()
    assert not np.shares_memory(array.raw_array, dummy_array.array)
