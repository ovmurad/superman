from abc import ABC, abstractmethod
from typing import Any, Union, Tuple, Type, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

ArrayData = Union[NDArray, csr_array]


class Array(ABC):
    """Abstract base class for array-like objects (dense or sparse).

    Wraps the core interface shared by numpy and scipy.sparse arrays,
    and defines an abstract contract for key operations that are format-specific.
    """

    def __init__(self, data: ArrayData):
        self.data = data

    # ----- Properties -----

    @property
    def is_sparse(self) -> bool:
        return isinstance(self.data, csr_array)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def astype(self, dtype: Type, atype: Optional[Type] = None) -> "Array":
        """Return a copy of the array with a new dtype or array type"""
        data = self.data.astype(dtype)

        if atype is np.ndarray and isinstance(data, csr_array):
            data = data.todense()
        if atype is csr_array and isinstance(data, np.ndarray):
            data = csr_array(data)

        return self.__class__(data)

    def copy(self) -> "Array":
        return self.__class__(self.data.copy())

    def __getitem__(self, idx: Any) -> Any:
        return self.data[idx]

    def __setitem__(self, idx: Any, value: Any) -> None:
        self.data[idx] = value

    def __matmul__(self, other: "Array") -> "Array":
        return self.__class__(self.data @ other.data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype}, data={self.data.__repr__()})"

    # ----- Abstract methods -----

    @abstractmethod
    def to_dense(self) -> "Array": ...

    @abstractmethod
    def to_sparse(self) -> "Array": ...
