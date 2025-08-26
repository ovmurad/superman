from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np

from src.array import BaseArray, CsrArray, DenseArray
from src.object import GeometryMatrixMixin


class AdjacencyMatrixMixin(GeometryMatrixMixin, ABC):
    fixed_dtype = np.bool_

    def __new__(cls, *args: Any, **kwargs: Any):
        if cls is AdjacencyMatrix:
            if "shape" in kwargs:
                return CsrAdjacencyMatrix(*args, **kwargs)
            return DenseAdjacencyMatrix(*args, **kwargs)
        return super().__new__(cls)


class AdjacencyMatrix(AdjacencyMatrixMixin, BaseArray, ABC):
    pass


class DenseAdjacencyMatrix(AdjacencyMatrix, DenseArray):
    pass


class CsrAdjacencyMatrix(AdjacencyMatrix, CsrArray):
    pass
