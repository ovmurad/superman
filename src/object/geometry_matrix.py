from abc import ABC
from typing import Any, ClassVar, Generic, Optional, Tuple, TypeAlias, TypeVar, Union

import numpy as np
from scipy.sparse import csr_array

from src.object.object_mixin import ObjectMixin

from .metadata import AffinityType, DistanceType, LaplacianType, Metadata

class GeometryMatrixMixin(ObjectMixin, ABC):
    fixed_ndim = 2
    fixed_dtype = np.float64

    def __init__(self, *args, **metadata) -> None:
        super().__init__(*args, **metadata)

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
