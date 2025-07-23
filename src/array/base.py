from abc import ABC, abstractmethod
from typing import Any, Generic, Self, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from .typing import _Array, _Data, _DType, _DType_, _Idx

# ------------------------------------------
# ----------------- Array ------------------
# ------------------------------------------


class Array(Generic[_Array, _DType], ABC):
    """Abstract base class for array-like objects (dense or sparse).

    Wraps the core interface shared by numpy and scipy.sparse arrays,
    and defines an abstract contract for key operations that are format-specific.
    """

    _array: _Array

    def __init__(self, array: Any) -> None:
        self._array = array

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._array.shape

    @property
    def ndim(self) -> int:
        return self._array.ndim

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    @property
    def raw_array(self) -> _Array:
        return self._array

    # ----- Magic -----

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype}, array={self._array.__repr__()})"

    # ----- Utilities -----

    def copy(self) -> Self:
        return self.__class__(self._array.copy())

    # ----- Abstract properties -----

    # ----- Abstract methods -----

    @abstractmethod
    def astype(self, dtype: _DType_) -> "Array[_Array, _DType_]": ...

    @abstractmethod
    def to_dense(self) -> "DenseArray[_DType]": ...

    @abstractmethod
    def to_sparse(self) -> "SparseArray[_DType]": ...


# ------------------------------------------
# -------------- DenseArray ----------------
# ------------------------------------------


class DenseArray(Generic[_DType], Array[NDArray[_DType], _DType]):
    _array: NDArray[_DType]

    def __init__(self, array: NDArray[_DType]) -> None:
        super().__init__(np.asarray(array))

    # ----- Utilities -----

    def astype(self, dtype: _DType_) -> "DenseArray[_DType_]":
        return DenseArray[_DType_](self._array.astype(dtype))

    def to_dense(self) -> Self:
        return self

    def to_sparse(self) -> "SparseArray[_DType]":
        return SparseArray[_DType](csr_array(self._array))


# ------------------------------------------
# ------------- SparseArray ----------------
# ------------------------------------------


class SparseArray(Generic[_DType], Array[csr_array[_DType], _DType]):
    _array: csr_array[_DType]

    def __init__(self, array: csr_array[_DType]) -> None:
        super().__init__(array)

    # ----- Properties -----

    @property
    def data(self) -> _Data[_DType]:
        return np.asarray(self._array.data)

    @property
    def indptr(self) -> _Idx:
        return self._array.indptr

    @property
    def indices(self) -> _Idx:
        return self._array.indices

    # ----- Utilities -----

    def astype(self, dtype: _DType_) -> "SparseArray[_DType_]":
        array = cast(csr_array[_DType_], self._array.astype(dtype))
        return SparseArray[_DType_](array)

    def to_dense(self) -> "DenseArray[_DType]":
        return DenseArray(self._array.todense())

    def to_sparse(self) -> Self:
        return self
