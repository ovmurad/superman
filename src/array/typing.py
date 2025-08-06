from typing import Tuple, Type, TypeAlias, TypeVar, Union

import numpy as np
from numpy._typing._array_like import _ArrayLikeFloat_co, _ArrayLikeInt_co
from scipy.sparse import csr_array

BoolLike: TypeAlias = Union[bool, np.bool_]
IntLike: TypeAlias = Union[int, np.int32, np.int64]
FloatLike: TypeAlias = Union[float, np.float32, np.float64]
ScalarLike: TypeAlias = Union[BoolLike, IntLike, FloatLike]

ScalarLikeType: TypeAlias = Union[
    Type[np.bool_],
    Type[np.int32],
    Type[np.int64],
    Type[np.float32],
    Type[np.float64],
    Type[bool],
    Type[int],
    Type[float],
]

Bool: TypeAlias = np.bool_
Int: TypeAlias = Union[np.int32, np.int64]
Float: TypeAlias = Union[np.float32, np.float64]
Scalar: TypeAlias = Union[Bool, Int, Float]
ScalarType: TypeAlias = Union[
    Type[np.bool_], Type[np.int32], Type[np.int64], Type[np.float32], Type[np.float64]
]
ScalarTypeVar = TypeVar(
    "ScalarTypeVar", np.bool_, np.int32, np.int64, np.float32, np.float64
)
ScalarTypeVar_ = TypeVar(
    "ScalarTypeVar_", np.bool_, np.int32, np.int64, np.float32, np.float64
)

ShapeTypeVar = TypeVar("ShapeTypeVar", bound=Tuple[int, ...])

DenseStorage: TypeAlias = np.ndarray[ShapeTypeVar, np.dtype[ScalarTypeVar]]
SparseStorage: TypeAlias = csr_array[ScalarTypeVar, Tuple[int, int]]

ScalarTypeVar__ = TypeVar("ScalarTypeVar__", bound=np.generic)
Storage: TypeAlias = Union[
    csr_array[ScalarTypeVar__, Tuple[int, int]],
    np.ndarray[ShapeTypeVar, np.dtype[ScalarTypeVar__]],
]

DataLike: TypeAlias = _ArrayLikeFloat_co
IndexLike: TypeAlias = _ArrayLikeInt_co

# Sparse Array Stuff
Index: TypeAlias = np.ndarray[Tuple[int], np.dtype[np.int32]]
Data: TypeAlias = np.ndarray[Tuple[int], np.dtype[ScalarTypeVar]]
