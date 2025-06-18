from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from .base import Array

if TYPE_CHECKING:
    from .sparse_base import SparseArray  # only imported for type hints


class DenseArray(Array):
    def __init__(self, array: NDArray):
        if not isinstance(array, np.ndarray):
            raise ValueError(
                f"DenseArray expects a dense ndarray, but got {type(array)}"
            )
        super().__init__(array)

    @property
    def data(self) -> NDArray:
        return self.array

    def to_dense(self) -> "DenseArray":
        return self

    def to_sparse(self) -> "SparseArray":
        return SparseArray(csr_array(self.array))
