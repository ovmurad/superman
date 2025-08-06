from typing import Optional, Tuple, TypeAlias, Union

import numpy as np

from ..array.dense import DenseArray
from .object import Object

PointsStorage: TypeAlias = np.ndarray[Tuple[int, int], np.dtype[np.float64]]
PointsArray: TypeAlias = DenseArray[np.float64, Tuple[int, int]]


class Points(Object):
    data: PointsArray

    ndim = 2
    dtype = np.float64

    def __init__(
        self, data: Union[PointsStorage, PointsArray], name: Optional[str] = None
    ) -> None:
        if isinstance(data, DenseArray):
            super().__init__(data, name=name)
        elif isinstance(data, np.ndarray):
            super().__init__(DenseArray(data), name=name)
        else:
            raise TypeError(f"Cannot format {type(data)} as Points!")

    @property
    def npts(self) -> int:
        return self.data.shape[0]

    @property
    def nfeats(self) -> int:
        return self.data.shape[1]


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
