from __future__ import annotations

from typing import Generic, Optional

import numpy as np

from .base import BaseArray
from .typing import DenseStorage, Scalar, ScalarLikeType, ScalarTypeVar, ShapeTypeVar


class DenseArray(Generic[ScalarTypeVar, ShapeTypeVar], BaseArray):
    storage: DenseStorage[ShapeTypeVar, ScalarTypeVar]

    _is_sparse = False
    _wrap_types = (np.ndarray, np.generic)

    def __init__(self, storage: DenseStorage[ShapeTypeVar, ScalarTypeVar]) -> None:
        super().__init__(np.asarray(storage))

    def __array__(
        self, dtype: Optional[ScalarLikeType] = None
    ) -> np.ndarray[ShapeTypeVar, np.dtype[Scalar]]:
        return self.storage if dtype is None else self.storage.astype(dtype)
