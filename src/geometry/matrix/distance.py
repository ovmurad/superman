from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, Optional, Union

import numpy as np
from sklearn.metrics import pairwise_distances_chunked

from src.array.base import BaseArray
from src.array.dense.dense import DenseArray
from src.array.sparse.sparse import SparseArray
from src.object.geometry_matrix import GeometryMatrixMixin

from ...object.metadata import DistanceType
from ...object.points import Points


class DistanceMatrix(BaseArray, GeometryMatrixMixin, ABC):
    def __new__(cls, *args, **kwargs):
        if cls is DistanceMatrix:
            if 'shape' in kwargs:
                return SparseDistanceMatrix(*args, **kwargs)
            return DenseDistanceMatrix(*args, **kwargs)
        return super().__new__(cls)

    def threshold(
        self,
        radius: float,
        in_place: bool = False,
    ) -> DistanceMatrix:
        """
        :param dist_mat:
        :param radius:
        :param in_place:
        :return:
        """

        dist_mat_radius = self.metadata.radius
        if dist_mat_radius is not None and radius > dist_mat_radius:
            raise ValueError(
                f"`radius`={radius} is greater than the radius of the input distance matrix {dist_mat_radius}!"
            )

        return self._execute_threshold(radius, in_place)

    @abstractmethod
    def _execute_threshold(self, radius: float, in_place: bool = False) -> DistanceMatrix:
        pass
    
    def threshold_distance_iter(
        self,
        radii: Union[float, Iterable[float]],
        in_place: bool = False,
    ) -> Iterator[DistanceMatrix]:
        """
        Take a distance matrix and eliminate entries at the new radii. Note that we return an iterator
        that could be used by other functions down stream without storing all matrices.
        :param dist_mat:
        :param radii:
        :param in_place:
        :return:
        """

        if isinstance(radii, float):
            radii = (radii,)

        for radius in reversed(sorted(radii)):
            yield self.threshold(radius, in_place)


class DenseDistanceMatrix(DenseArray, DistanceMatrix):
    def _execute_threshold(
        self, radius: float, in_place: bool
    ) -> DistanceMatrix:
        dist_mat = self if in_place else self.copy()
        dist_mat[dist_mat > radius] = np.inf
        return dist_mat


class SparseDistanceMatrix(SparseArray, DistanceMatrix):
    def _execute_threshold(
        self, radius: float, in_place: bool
    ) -> DistanceMatrix:
        raise NotImplementedError()
