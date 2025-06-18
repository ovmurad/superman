from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from .base import Array

if TYPE_CHECKING:
    from .dense_base import DenseArray  # only imported for type hints


class SparseArray(Array):
    def __init__(self, array: csr_array):
        if not isinstance(array, csr_array):
            raise ValueError(
                f"SparseArray expects a sparse.csr_array, but got {type(array)}"
            )
        super().__init__(array)

    @property
    def data(self) -> NDArray:
        return np.asarray(self.array.data)

    @property
    def indptr(self) -> NDArray:
        return self.array.indptr

    @property
    def indices(self) -> NDArray:
        return self.array.indices

    def to_dense(self) -> "DenseArray":
        return DenseArray(self.array.todense())

    def to_sparse(self) -> "SparseArray":
        return self
