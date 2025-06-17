import numpy as np

from src.array import SparseArray, DenseArray


def test_basic_properties_dense(dense_array: DenseArray) -> None:
    assert isinstance(dense_array, DenseArray)
    assert not dense_array.is_sparse
    assert dense_array.shape == (2, 2)
    assert dense_array.ndim == 2
    assert dense_array.dtype == np.float64


def test_basic_properties_sparse(sparse_array: SparseArray) -> None:
    assert isinstance(sparse_array, SparseArray)
    assert sparse_array.is_sparse
    assert sparse_array.shape == (2, 2)
    assert sparse_array.ndim == 2
    assert sparse_array.dtype == np.float64


# def test_array_get_set_item_dense() -> None:
#     arr = dense_array()
#     assert arr[0, 1] == 0.0
#     arr[0, 1] = 9.0
#     assert arr[0, 1] == 9.0
#
#
# def test_array_get_set_item_sparse() -> None:
#     arr = make_sparse()
#     assert arr[1, 0] == 0.0
#     arr[1, 0] = 5.0
#     assert arr[1, 0] == 5.0
#
#
# def test_array_matmul_dense():
#     a = DenseArray(np.array([[1.0, 2.0], [0.0, 3.0]]))
#     b = DenseArray(np.array([[1.0], [2.0]]))
#     c = a @ b
#     assert isinstance(c, DenseArray)
#     expected = np.array([[5.0], [6.0]])
#     np.testing.assert_array_almost_equal(c.data, expected)
#
#
# def test_array_matmul_sparse():
#     a = SparseArray(csr_array([[1.0, 0.0], [0.0, 2.0]]))
#     b = SparseArray(csr_array([[1.0], [3.0]]))
#     c = a @ b
#     assert isinstance(c, SparseArray)
#     expected = np.array([[1.0], [6.0]])
#     np.testing.assert_array_almost_equal(c.data.toarray(), expected)
#
#
# def test_array_copy():
#     original = dense_array()
#     clone = original.copy()
#     assert isinstance(clone, DenseArray)
#     np.testing.assert_array_equal(clone.data, original.data)
#     assert clone is not original
#
#
# def test_array_astype_conversion():
#     arr = dense_array()
#     float32_arr = arr.astype(np.float32)
#     assert float32_arr.dtype == np.float32
#
#     sparse_arr = arr.astype(np.float64, atype=csr_array)
#     assert sparse_arr.is_sparse
#     np.testing.assert_array_equal(sparse_arr.data.toarray(), arr.data)
#
#     back_to_dense = sparse_arr.astype(np.float64, atype=np.ndarray)
#     assert not back_to_dense.is_sparse
#     np.testing.assert_array_equal(back_to_dense.data, arr.data)
#
#
# def test_to_dense_and_sparse_roundtrip():
#     d = dense_array()
#     s = d.to_sparse()
#     assert isinstance(s, SparseArray)
#     roundtrip = s.to_dense()
#     assert isinstance(roundtrip, DenseArray)
#     np.testing.assert_array_equal(roundtrip.data, d.data)
