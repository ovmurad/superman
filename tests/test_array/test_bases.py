import pytest

from src.array import SparseArray, DenseArray
from .conftest import test_data_groups, TestData


@pytest.mark.parametrize("test_data", test_data_groups["dense"])
def test_basic_properties_dense(test_data: TestData) -> None:
    test_array = DenseArray(test_data.data)
    assert test_data.check(test_array)


@pytest.mark.parametrize("test_data", test_data_groups["sparse"])
def test_basic_properties_sparse(test_data: TestData) -> None:
    test_array = SparseArray(test_data.data)
    assert test_data.check(test_array)
