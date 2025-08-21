from abc import ABC
from typing import Optional, Tuple, TypeAlias, Union

import numpy as np

from src.object.metadata import Metadata
from src.object.object_mixin import ObjectMixin

from ..array.dense import DenseArray


class PointsMixin(ObjectMixin, ABC):
    metadata: Metadata

    fixed_ndim = 2
    fixed_dtype = np.float64

    def __init__(self, **metadata) -> None:
        super().__init__(**metadata)

    @property
    def npts(self) -> int:
        return self.shape[0]

    @property
    def nfeats(self) -> int:
        return self.shape[1]


class Points(DenseArray, PointsMixin):
    pass


class Data(Points):

    @property
    def D(self) -> int:
        return self.nfeats


class Embedding(Points):

    @property
    def p(self) -> int:
        return self.nfeats


class Coordinates(Points):

    @property
    def d(self) -> int:
        return self.nfeats
