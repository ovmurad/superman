from math import prod
from typing import FrozenSet, Literal, Optional

import numpy as np

from ..array.dense import DenseArray
from .geometry_matrix import DISTANCE_TYPES, DistanceType
from .metadata import (
    FloatLikeSeq,
    FloatTupleAttr,
    IntLikeSeq,
    IntTupleAttr,
    LiteralAttr,
)
from .object import Object

DegreeType = Literal["adjacency", "affinity"]
DEGREE_TYPES: FrozenSet[DegreeType] = frozenset(("adjacency", "affinity"))


class Function(Object):
    data: DenseArray

    ndim = 2

    def __init__(self, data: DenseArray, name: Optional[str] = None) -> None:
        super().__init__(data, name)

    @property
    def npts(self) -> int:
        return self.data.shape[0]

    @property
    def nfuncs(self) -> int:
        return prod(self.data.shape[1:])


class Coordinate(Function):
    data: DenseArray[np.float64]

    dtype = np.float64
    radii = FloatTupleAttr(key="radii")

    def __init__(
        self,
        data: DenseArray[np.float64],
        radii: Optional[FloatLikeSeq] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(data, name)
        self.radii = radii


class Degree(Function):
    data: DenseArray[np.float64]

    dtype = np.float64
    dist_type = LiteralAttr[DistanceType](
        key="dist_type", allowed_values=DISTANCE_TYPES
    )
    degree_type = LiteralAttr[DistanceType](
        key="degree_type", allowed_values=DEGREE_TYPES
    )
    radii = FloatTupleAttr(key="radii")

    def __init__(
        self,
        data: DenseArray[np.float64],
        dist_type: Optional[DistanceType] = None,
        degree_type: Optional[DegreeType] = None,
        radii: Optional[FloatLikeSeq] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(data, name)
        self.dist_type = dist_type
        self.degree_type = degree_type
        self.radii = radii


class KNNDistance(Function):
    data: DenseArray[np.float64]

    dtype = np.float64
    ks = IntTupleAttr(key="ks")

    def __init__(
        self,
        data: DenseArray[np.float64],
        ks: Optional[IntLikeSeq] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(data, name)
        self.ks = ks


class NeighborCount(Function):
    data: DenseArray[np.int64]

    dtype = np.int64
    radii = FloatTupleAttr(key="radii")

    def __init__(
        self,
        data: DenseArray[np.int64],
        radii: Optional[FloatLikeSeq] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(data, name)
        self.radii = radii


class LocalDistortion(Function):
    data: DenseArray[np.float64]

    dtype = np.float64
    radii = FloatTupleAttr(key="radii")
    ds = IntTupleAttr(key="ds")

    def __init__(
        self,
        data: DenseArray[np.float64],
        radii: Optional[FloatLikeSeq] = None,
        ds: Optional[IntLikeSeq] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(data, name)
        self.radii = radii
        self.ds = ds
