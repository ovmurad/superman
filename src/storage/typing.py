from typing import TypeAlias, TypeVar

import numpy as np
from scipy.sparse import csr_array

BoolType = TypeVar("BoolType", bound=np.bool_)
IntType = TypeVar("IntType", bound=np.integer)
FloatType = TypeVar("FloatType", bound=np.floating)
DType = TypeVar("DType", bound=np.bool_ | np.integer | np.floating)

Shape = TypeVar("Shape", bound=tuple[int, ...])
MatrixShape: TypeAlias = tuple[int, int]

_NumpyStorage: TypeAlias = np.ndarray[Shape, np.dtype[DType]]
_CsrStorage: TypeAlias = csr_array[DType, MatrixShape]

NumpyArray = np.ndarray[Shape, np.dtype[DType]]

RawIndex: TypeAlias = np.ndarray[tuple[int], np.dtype[IntType]]
CsrIndex: TypeAlias = tuple[
    np.ndarray[tuple[int], np.dtype[IntType]], np.ndarray[tuple[int], np.dtype[IntType]]
]

SparseData: TypeAlias = np.ndarray[tuple[int], np.dtype[DType]]
