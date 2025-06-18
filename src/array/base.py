from abc import ABC, abstractmethod
from typing import Any, Union, Tuple, Type, Self

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

ArrayInput = Union[NDArray, csr_array]


class Array(ABC):
    """Abstract base class for array-like objects (dense or sparse).

    Wraps the core interface shared by numpy and scipy.sparse arrays,
    and defines an abstract contract for key operations that are format-specific.
    """

    def __init__(self, array: ArrayInput):
        self.array = array

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

    def __getitem__(self, idx: Any) -> Any:
        return self.array[idx]

    def __setitem__(self, idx: Any, value: Any) -> None:
        self.array[idx] = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype}, array={self.array.__repr__()})"

    # ----- Utilities -----

    def astype(self, dtype: Type) -> Self:
        """Return a copy of the array with a new dtype or array type"""
        return self.__class__(self.array.astype(dtype))

    def copy(self) -> Self:
        return self.__class__(self.array.copy())

    # ----- Abstract properties -----

    @property
    @abstractmethod
    def data(self) -> NDArray:
        pass

    # ----- Abstract methods -----

    @abstractmethod
    def to_dense(self) -> "Array": ...

    @abstractmethod
    def to_sparse(self) -> "Array": ...
