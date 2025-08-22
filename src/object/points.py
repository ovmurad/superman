from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple, TypeAlias, Union

import numpy as np
from scipy.sparse import csr_matrix

from src.geometry.matrix.distance import DistanceMatrix
from src.object.metadata import DistanceType, Metadata
from src.object.object_mixin import ObjectMixin

from ..array.dense import DenseArray


class PointsMixin(ObjectMixin, ABC):
    metadata: Metadata

    fixed_ndim = 2
    fixed_dtype = np.float64

    def __init__(self, *args, **metadata) -> None:
        super().__init__(*args, **metadata)

    
    @abstractmethod
    def distance(
        self,
        y_pts: Optional[Points] = None,
        dist_type: DistanceType = "euclidean",
    ) -> DistanceMatrix:
        pass

    @property
    def npts(self) -> int:
        return self.shape[0]

    @property
    def nfeats(self) -> int:
        return self.shape[1]


class Points(PointsMixin, DenseArray):
    def distance(
        self,
        y_pts: Optional[Points] = None,
        dist_type: DistanceType = "euclidean",
    ) -> DistanceMatrix:

        x_pts_name = self.metadata.name
        y_pts_name = None if y_pts is None else y_pts.metadata.name
        dist_name = (
            None
            if (x_pts_name is None or y_pts_name is None)
            else x_pts_name + "_" + y_pts_name
        )

        dist_mat = DenseArray.distance(self, self, dist_type=dist_type) if y_pts is None else DenseArray.distance(self, y_pts, dist_type=dist_type)

        return DistanceMatrix(dist_mat, dist_type=dist_type, name=dist_name)


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
