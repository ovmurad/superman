from __future__ import annotations
from abc import ABC, abstractmethod
from functools import singledispatch
from typing import Any, Callable, ClassVar, Iterable, Iterator, Optional, Union

import numpy as np
from sklearn.metrics import pairwise_distances_chunked

from src.array.base import BaseArray
from src.array.dense.dense import DenseArray
from src.array.sparse.csr import CsrArray
from src.array.sparse.sparse import SparseArray
from src.geometry.matrix.affinity import AffinityMatrix
from src.object.geometry_matrix import GeometryMatrixMixin

from ...object.metadata import AffinityType, DistanceType


#Adds factory functionality
class DistanceMatrixMixin(GeometryMatrixMixin, ABC):
    #__new__ allows us to use DistanceMatrixMixin's constructor as a factory to create Sparse and Dense DistanceMatrices
    def __new__(cls, *args, **kwargs):
        if cls is DistanceMatrix:
            if 'shape' in kwargs:
                return CsrDistanceMatrix(*args, **kwargs)
            return DenseDistanceMatrix(*args, **kwargs)
        return super().__new__(cls)


#Adds basearray functionality shared by dense and sparse
#ABC disallows constructing instances of DistanceMatrix which have no functionality because BaseArray has no functionality
#Mixin gives factory functionality to create Sparse and Dense DistanceMatrices
class DistanceMatrix(DistanceMatrixMixin, BaseArray, ABC):
    _dispatch_affinity: ClassVar[dict[AffinityType, Callable]] = {}


    @classmethod
    def _register(cls, name: AffinityType):
        """Decorator to register a class method in the dispatch table."""
        def decorator(func: Callable):
            cls._dispatch_affinity[name] = classmethod(func).__get__(None, cls)
            return func
        return decorator


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


    def affinity(
        self,
        aff_type: AffinityType = "gaussian",
        eps: Optional[float] = None,
        in_place: bool = False,
    ) -> AffinityMatrix:
        """
        Corresponds to affinity.py in cryo_experiments. Can deduce dist_is_sq from the dist_type and set to default
        False if distance is not 'sqeuclidean'. All distances can be found at
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
        :param dist_mat:
        :param aff_type:
        :param eps:
        :param in_place:
        :return:
        """

        dist_is_sq = self.metadata.dist_type == "sqeuclidean"

        return self._dispatch_affinity[aff_type].__get__(self)(eps, dist_is_sq, in_place)


@DistanceMatrix._register("gaussian")
def gaussian(
    dists: DistanceMatrix,
    eps: Optional[float] = None,
    dist_is_sq: bool = False,
    in_place: bool = False,
) -> AffinityMatrix:
    if in_place:
        return gaussian_in_place(dists, eps, dist_is_sq)
    return gaussian_out_of_place(dists, eps, dist_is_sq)


def gaussian_in_place(
    dists: DistanceMatrix, eps: float, dist_is_sq: bool
) -> AffinityMatrix:

    if not dist_is_sq:
        dists **= 2.0

    dists /= eps**2
    dists *= -1.0
    dists.iexp()

    return AffinityMatrix(dists, eps=eps, aff_type="gaussian")


def gaussian_out_of_place(
    dists: DistanceMatrix, eps: float, dist_is_sq: bool
) -> AffinityMatrix:
    if dist_is_sq:
        return AffinityMatrix(((dists / (eps**2)) * -1.0).exp(), eps=eps, metadata=dists.metadata, aff_type="gaussian")
    return AffinityMatrix((((dists / eps) ** 2) * -1).exp(), eps=eps, metadata=dists.metadata, aff_type="gaussian")


class DenseDistanceMatrix(DistanceMatrix, DenseArray):
    def _execute_threshold(
        self, radius: float, in_place: bool
    ) -> DistanceMatrix:
        dist_mat: DistanceMatrix = self if in_place else self.copy()
        dist_mat[dist_mat > radius] = np.inf
        return DistanceMatrix(dist_mat, radius=radius) if in_place else DistanceMatrix(dist_mat, radius=radius, metadata=self.metadata)
 

class CsrDistanceMatrix(DistanceMatrix, CsrArray):
    def _execute_threshold(
        self, radius: float, in_place: bool
    ) -> DistanceMatrix:
        raise NotImplementedError()
