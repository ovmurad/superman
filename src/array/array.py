from abc import ABC, abstractmethod
from typing import Any, Union, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix, issparse

ArrayData = Union[NDArray, spmatrix]


class Array(ABC):
    """Abstract base class for array-like objects (dense or sparse).

    Wraps the core interface shared by numpy and scipy.sparse arrays,
    and defines an abstract contract for key operations that are format-specific.
    """

    def __init__(self, data: ArrayData):
        self.data = data

    @property
    def is_sparse(self) -> bool:
        return issparse(self.data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def astype(self, dtype: Any) -> "Array":
        """Return a copy of the array with a new dtype."""
        return self.__class__(self.data.astype(dtype))

    def transpose(self, *axes: Tuple[int]) -> "Array":
        return self.__class__(self.data.transpose(*axes))

    def T(self) -> "Array":
        return self.transpose()

    def __getitem__(self, idx: Any) -> Any:
        return self.data[idx]

    def __setitem__(self, idx: Any, value: Any) -> None:
        self.data[idx] = value

    def __matmul__(self, other: "Array") -> "Array":
        return self.__class__(self.data @ other.data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Array):
            return NotImplemented
        return (self.shape == other.shape) and np.allclose(
            self.to_dense(), other.to_dense()
        )

    # ----- Abstract methods -----

    @abstractmethod
    def to_dense(self) -> NDArray[Any]:
        """Convert to a dense ndarray."""
        ...

    @abstractmethod
    def to_sparse(self) -> spmatrix:
        """Convert to a sparse scipy.sparse matrix."""
        ...

    @abstractmethod
    def copy(self) -> "Array":
        """Return a (deep) copy of the array."""
        ...


# from typing import Callable, Protocol, Tuple, TypeAlias, TypeVar
#
# import numpy as np
# from numpy.typing import NDArray
# from scipy.sparse import csr_matrix
#
# # DataType
# DT = TypeVar("DT", np.bool_, np.int_, np.float_, int, float)
#
# # NumericType
# NT = TypeVar("NT", np.int_, np.float_, int, float)
#
#
# class SpArr(Protocol[DT]):
#
#     data: NDArray[DT]
#     indices: NDArray[np.int32]
#     indptr: NDArray[np.int32]
#     shape: Tuple[int, ...]
#     dtype: np.dtype[DT]
#
#     astype: Callable[..., csr_matrix]
#
#     def __matmul__(self, other: NDArray[DT] | csr_matrix) -> NDArray[DT] | csr_matrix:
#         ...
#
#
# DeArr: TypeAlias = NDArray[DT]
# Arr: TypeAlias = DeArr[DT] | SpArr[DT]
#
# BoolDeArr: TypeAlias = DeArr[np.bool_]
# BoolSpArr: TypeAlias = SpArr[np.bool_]
# BoolArr = TypeVar("BoolArr", BoolSpArr, BoolSpArr)
#
# IntDeArr: TypeAlias = DeArr[np.int_]
# IntSpArr: TypeAlias = SpArr[np.int_]
# IntArr = TypeVar("IntArr", IntDeArr, IntSpArr)
#
# RealDeArr: TypeAlias = DeArr[np.float_]
# RealSpArr: TypeAlias = SpArr[np.float_]
# RealArr = TypeVar("RealArr", RealDeArr, RealSpArr)
#
# NumDeArr: TypeAlias = DeArr[NT]
# NumSpArr: TypeAlias = SpArr[NT]
# NumArr: TypeAlias = Arr[NT]
