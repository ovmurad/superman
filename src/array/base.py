from abc import ABC, abstractmethod
from typing import Generic, Self, Tuple, TypeVar, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy.sparse import csr_array

DType = TypeVar("DType", bound=np.number | np.bool_)
ArrayType = TypeVar("ArrayType", np.ndarray, csr_array)


class Array(ABC, Generic[ArrayType, DType]):
    """Abstract base class for array-like objects (dense or sparse).

    Wraps the core interface shared by numpy and scipy.sparse arrays,
    and defines an abstract contract for key operations that are format-specific.
    """

    def __init__(self, array: ArrayType) -> None:
        self.array: ArrayType = array

    # ----- Properties -----

    @property
    def is_sparse(self) -> bool:
        return isinstance(self.array, csr_array)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def ndim(self) -> int:
        return self.array.ndim

    # ----- Magic -----

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype}, array={self.array.__repr__()})"

    # ----- Utilities -----

    def copy(self) -> Self:
        return self.__class__(self.array.copy())

    # ----- Abstract properties -----

    @property
    @abstractmethod
    def data(self) -> NDArray[DType]:
        pass

    # ----- Abstract methods -----

    @abstractmethod
    def astype(self, dtype: DTypeLike) -> Self: ...

    @abstractmethod
    def to_dense(self) -> "DenseArray[DType]": ...

    @abstractmethod
    def to_sparse(self) -> "SparseArray[DType]": ...


class DenseArray(Array[np.ndarray, DType]):
    def __init__(self, array: NDArray[DType]) -> None:
        if not isinstance(array, np.ndarray):
            raise ValueError(
                f"DenseArray expects a dense ndarray, but got {type(array)}"
            )
        super().__init__(array)

    # ----- Properties -----

    @property
    def data(self) -> NDArray[DType]:
        return self.array

    # ----- Utilities -----

    def astype(self, dtype: DTypeLike) -> Self:
        return self.__class__(self.array.astype(dtype))

    def to_dense(self) -> Self:
        return self

    def to_sparse(self) -> "SparseArray[DType]":
        return SparseArray(csr_array(self.array))


class SparseArray(Array[csr_array, DType]):

    def __init__(self, array: csr_array[DType]) -> None:
        if not isinstance(array, csr_array):
            raise ValueError(
                f"SparseArray expects a sparse.csr_array, but got {type(array)}"
            )
        super().__init__(array)

    # ----- Properties -----

    @property
    def data(self) -> NDArray[DType]:
        return np.asarray(self.array.data)

    @property
    def indptr(self) -> NDArray[np.int32]:
        return self.array.indptr

    @property
    def indices(self) -> NDArray[np.int32]:
        return self.array.indices

    # ----- Utilities -----

    def astype(self, dtype: DTypeLike) -> Self:
        new_array = cast(csr_array, self.array.astype(dtype))
        return self.__class__(new_array)

    def to_dense(self) -> "DenseArray[DType]":
        return DenseArray(self.array.todense())

    def to_sparse(self) -> Self:
        return self
