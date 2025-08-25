from __future__ import annotations
from typing import Optional, Type, TypeVar

import numpy as np

from src.array.linalg import normalize
from src.object.object_mixin import ObjectMixin

from ...object.geometry_matrix import LaplacianType

from abc import ABC, abstractmethod

import numpy as np
from src.array.base import BaseArray
from src.array.dense.dense import DenseArray
from src.array.sparse.csr import CsrArray
from src.object.geometry_matrix import GeometryMatrixMixin


class LaplacianMatrixMixin(GeometryMatrixMixin, ABC):
    def __new__(cls, *args, **kwargs):
        if cls is LaplacianMatrix:
            if 'shape' in kwargs:
                return CsrLaplacianMatrix(*args, **kwargs)
            return DenseLaplacianMatrix(*args, **kwargs)
        return super().__new__(cls)


class LaplacianMatrix(LaplacianMatrixMixin, BaseArray, ABC):
    pass


class DenseLaplacianMatrix(LaplacianMatrix, DenseArray):
    pass


class CsrLaplacianMatrix(LaplacianMatrix, CsrArray):
    pass


def eps_adjustment(eps: float) -> float:
    return 4.0 / (eps**2)
