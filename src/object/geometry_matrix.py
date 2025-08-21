from abc import ABC
from typing import Any, Generic, Optional, Tuple, TypeAlias, TypeVar, Union

import numpy as np
from scipy.sparse import csr_array

from src.object.object_mixin import ObjectMixin

from ..array.base import BaseArray
from ..array.dense import DenseArray
from ..array.sparse import SparseArray
from .metadata import AffinityType, DistanceType, LaplacianType, Metadata

class GeometryMatrixMixin(ObjectMixin, ABC):
    metadata: Metadata

    fixed_ndim = 2
    fixed_dtype = np.float64

    def __init__(self, **metadata) -> None:
        super().__init__(**metadata)

    @property
    def is_square(self) -> bool:
        return self.shape[0] == self.shape[1]

    @property
    def from_npts(self) -> int:
        return self.shape[0]

    @property
    def to_npts(self) -> int:
        return self.shape[1]

    @property
    def npts(self) -> int:
        if self.is_square:
            return self.from_npts
        raise ValueError("Matrix is not square, so `npts` is not well defined!")


class AdjacencyMatrixMixin(GeometryMatrixMixin, ABC):
    fixed_dtype = np.bool_


class DenseDistanceMatrix(DenseArray, GeometryMatrixMixin):
    pass


class DenseAdjacencyMatrix(DenseArray, AdjacencyMatrixMixin):
    pass


class DenseAffinityMatrix(DenseArray, GeometryMatrixMixin):
    pass


class DenseLaplacianMatrix(DenseArray, GeometryMatrixMixin):
    pass


class SparseDistanceMatrix(SparseArray, GeometryMatrixMixin):
    pass


class SparseAdjacencyMatrix(SparseArray, AdjacencyMatrixMixin):
    pass


class SparseAffinityMatrix(SparseArray, GeometryMatrixMixin):
    pass


class SparseLaplacianMatrix(SparseArray, GeometryMatrixMixin):
    pass
