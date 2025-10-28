from abc import ABC
from math import prod
from typing import Any, Self, Sequence, Type

import numpy as np

from src.array import BaseArray
from src.object.metadata import Metadata
from src.object.object_mixin import ObjectMixin

from ..array import DenseArray


class FunctionMixin(ObjectMixin, BaseArray, ABC):
    fixed_ndim = 2
    fixed_dtype: Type[np.generic] = np.float64

    metadata: Metadata

    def __init__(self, *args: Any, **metadata: Any) -> None:
        super().__init__(*args, **metadata)

    @property
    def npts(self) -> int:
        return self.shape[0]

    @property
    def nfuncs(self) -> int:
        return prod(self.shape[1:])

    @classmethod
    def concat_with_metadata(cls, arrs: Sequence[Self], axis: int = 0) -> Self:
        return cls(super().concat(arrs, axis=axis), metadata=arrs[0].metadata)


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
    fixed_dtype = np.int64


class NeighborCount(NeighborCountMixin, DenseArray):
    pass


class LocalDistortionMixin(FunctionMixin, ABC):
    pass


class LocalDistortion(LocalDistortionMixin, DenseArray):
    pass
