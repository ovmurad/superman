from __future__ import annotations

from typing import Generic, Optional, Tuple

import numpy as np
from scipy.sparse import csr_array

from .base import BaseArray
from .typing import Data, Index, Scalar, ScalarLikeType, ScalarTypeVar, SparseStorage


class SparseArray(Generic[ScalarTypeVar], BaseArray):
    storage: SparseStorage[ScalarTypeVar]

    _is_sparse = True
    _wrap_types = (csr_array,)

    def __init__(self, storage: SparseStorage[ScalarTypeVar]) -> None:
        super().__init__(storage)

    # --------------- Properties ---------------

    @property
    def shape(self) -> Tuple[int, int]:
        return self.storage.shape

    @property
    def data(self) -> Data[ScalarTypeVar]:
        return self.storage.data

    @property
    def indices(self) -> Index:
        return self.storage.indices

    @property
    def indptr(self) -> Index:
        return self.storage.indptr

    # --------------- Numpy Interface ---------------

    def __array__(
        self, dtype: Optional[ScalarLikeType] = None
    ) -> np.ndarray[Tuple[int, int], np.dtype[Scalar]]:
        dense_storage = self.storage.todense()
        return dense_storage if dtype is None else dense_storage.astype(dtype)
