from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.sparse import csr_array

from .base import Array

if TYPE_CHECKING:
    from .dense_base import DenseArray  # only imported for type hints


class SparseArray(Array):
    def __init__(self, data: csr_array):
        if not isinstance(data, csr_array):
            raise ValueError(
                f"SparseArray expects a sparse.csr_array, but got {type(data)}"
            )
        super().__init__(data)

    def to_dense(self) -> "DenseArray":
        return DenseArray(self.data.todense())

    def to_sparse(self) -> "SparseArray":
        return self
