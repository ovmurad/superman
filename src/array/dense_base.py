from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from .base import Array

if TYPE_CHECKING:
    from .sparse_base import SparseArray  # only imported for type hints


class DenseArray(Array):
    def __init__(self, data: NDArray):
        if not isinstance(data, np.ndarray):
            raise ValueError(
                f"DenseArray expects a dense ndarray, but got {type(data)}"
            )
        super().__init__(data)

    def to_dense(self) -> "DenseArray":
        return self

    def to_sparse(self) -> "SparseArray":
        return SparseArray(csr_array(self.data))
