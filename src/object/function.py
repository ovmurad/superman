from abc import ABC
from math import prod
from typing import Optional

import numpy as np

from src.object.metadata import FunctionMetadata
from src.object.object_mixin import ObjectMixin

from ..array.dense import DenseArray


class FunctionMixin(ObjectMixin, ABC):
    ndim = 2
    dtype = np.float64
    
    metadata: FunctionMetadata

    def __init__(self, **metadata) -> None:
        super().__init__(cls=FunctionMetadata, **metadata)

    @property
    def npts(self) -> int:
        return self.shape[0]

    @property
    def nfuncs(self) -> int:
        return prod(self.shape[1:])


class CoordinateMixin(FunctionMixin, ABC):
    radii: np.float64

    def __init__(self, radii, **metadata) -> None:
        super().__init__(**metadata)
        self.radii = radii


class Coordinate(CoordinateMixin, ABC):
    pass


class DegreeMixin(FunctionMixin, ABC):
    radii: np.float64

    def __init__(self, radii, **metadata) -> None:
        super().__init__(**metadata)
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


class NeighborCount(FunctionMixin, DenseArray):
    dtype = np.int64

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
