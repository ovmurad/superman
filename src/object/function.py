from abc import ABC
from math import prod
from typing import Optional

import numpy as np

from src.object.metadata import Metadata
from src.object.object_mixin import ObjectMixin

from ..array.dense import DenseArray


class FunctionMixin(ObjectMixin, ABC):
    ndim = 2
    dtype = np.float64
    
    metadata: Metadata

    def __init__(self, *args, **metadata) -> None:
        super().__init__(*args, cls=Metadata, **metadata)

    @property
    def npts(self) -> int:
        return self.shape[0]

    @property
    def nfuncs(self) -> int:
        return prod(self.shape[1:])


class CoordinateMixin(FunctionMixin, ABC):
    pass


class Coordinate(CoordinateMixin, ABC):
    pass


class DegreeMixin(FunctionMixin, ABC):
    pass


class Degree(DegreeMixin, DenseArray):
    pass


class KNNDistanceMixin(FunctionMixin, ABC):
    pass


class KNNDistance(KNNDistanceMixin, DenseArray):
    pass


class NeighborCountMixin(FunctionMixin, ABC):
    dtype = np.int64


class NeighborCount(NeighborCountMixin, DenseArray):
    pass


class LocalDistortionMixin(FunctionMixin, ABC):
    pass


class LocalDistortion(LocalDistortionMixin, DenseArray):
    pass
