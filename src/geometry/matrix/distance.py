from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Iterable, Iterator, Optional, Union

import numpy as np

from src.array import CsrArray, DenseArray
from src.geometry.matrix.adjacency import AdjacencyMatrix
from src.geometry.matrix.affinity import AffinityMatrix
from src.object import AffinityType
from src.object.geometry_matrix_mixin import GeometryMatrixMixin


class DistanceMatrixMixin(GeometryMatrixMixin, ABC):
    """
    Mixin class that adds factory functionality to the DistanceMatrix
    hierarchy.

    This allows users to work with `DistanceMatrix` as an abstract
    entry point without worrying about the underlying dense/sparse
    representation.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> DistanceMatrix:
        """
        The constructor returns an instance of either `DenseLaplacianMatrix` or `CsrLaplacianMatrix` depending on if constructed in `DenseArray` format or `CsrArray` format respectively.

        :param args: Positional arguments forwarded to the chosen
                     Laplacian matrix subclass.
        :type args: Any
        :param kwargs: Keyword arguments forwarded to the chosen
                       Laplacian matrix subclass.
                       If `shape` is present, a sparse CSR-backed
                       matrix will be constructed.
        :type kwargs: Any
        :return: A new `DenseLaplacianMatrix` or `CsrLaplacianMatrix`
                 instance.
        :rtype: LaplacianMatrix
        """
        if cls is DistanceMatrix:
            return cls.create(*args, **kwargs)
        return super().__new__(cls)  # type: ignore

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> DistanceMatrix:
        if "shape" in kwargs:
            return CsrDistanceMatrix(*args, **kwargs)
        return DenseDistanceMatrix(*args, **kwargs)


# Adds basearray functionality shared by dense and sparse
# ABC disallows constructing instances of DistanceMatrix which have no functionality because BaseArray has no functionality
# Mixin gives factory functionality to create Sparse and Dense DistanceMatrices
class DistanceMatrix(DistanceMatrixMixin, ABC):
    """
    Abstract base class representing a distance matrix, with functionality to
    derive adjacency and affinity matrices, and to threshold distances.

    This class serves as a template for distance matrix operations and enforces
    the implementation of adjacency and threshold execution methods in subclasses.
    It also provides registration and dispatching mechanisms for affinity functions.
    """

    def adjacency(
        self,
        copy: bool = False,
    ) -> AdjacencyMatrix:
        """
        Convert the distance matrix into an adjacency matrix.

        :param copy: Whether to return a copy of the adjacency matrix or reuse existing data. (default: False
        :type copy: bool
        :return: The adjacency matrix corresponding to this distance matrix.
        :rtype: AdjacencyMatrix
        """

        return AdjacencyMatrix.create(
            self._execute_adjacency(copy), metadata=self.metadata
        )

    @abstractmethod
    def _execute_adjacency(self, copy: bool) -> AdjacencyMatrix:
        """
        Abstract method to execute the conversion to an adjacency matrix.
        Must be implemented by subclasses.

        :param copy: Whether to return a copy of the adjacency matrix or reuse existing data.
        :type copy: bool
        :return: The computed adjacency matrix.
        :rtype: AdjacencyMatrix
        """
        pass

    def threshold(
        self,
        radius: float,
        in_place: bool = False,
    ) -> DistanceMatrix:
        """
        Threshold the distance matrix by eliminating entries above a given radius.

        :param radius: The maximum radius to retain distances. When dense, entries larger than
                       this will be set to np.inf.
        :type radius: float
        :param in_place: If True, modify the existing distance matrix; otherwise,
                         return a new one. (default: False)
        :type in_place: bool
        :raises ValueError: If the given radius is greater than the maximum radius
                            defined in the matrix metadata.
        :return: The thresholded distance matrix.
        :rtype: DistanceMatrix
        """

        dist_mat_radius = self.metadata.radius
        if dist_mat_radius is not None and radius > dist_mat_radius:
            raise ValueError(
                f"`radius`={radius} is greater than the radius of the input distance matrix {dist_mat_radius}!"
            )

        return self._execute_threshold(radius, in_place)

    @abstractmethod
    def _execute_threshold(
        self, radius: float, in_place: bool = False
    ) -> DistanceMatrix:
        """
        Abstract method to threshold the distance matrix by radius.
        Must be implemented by subclasses.

        :param radius: The maximum radius to retain distances. Entries larger than
                       this will be eliminated.
        :type radius: float
        :param in_place: If True, modify the existing distance matrix; otherwise,
                         return a new one. (default: False)
        :type in_place: bool
        :return: The thresholded distance matrix.
        :rtype: DistanceMatrix
        """
        pass

    def threshold_distance_iter(
        self,
        radii: Union[float, Iterable[float]],
        in_place: bool = False,
    ) -> Iterator[DistanceMatrix]:
        """
        Generate an iterator over thresholded distance matrices with a single radius or sequence of radii.

        :param radii: A single radius or an iterable of radii with which to threshold
                      the distance matrix.
        :type radii: float or Iterable[float]
        :param in_place: If True, modify the existing distance matrix with each radius (This will ultimately threshold with the largest radius);
                         otherwise, return new matrices. (default: False)
        :type in_place: bool
        :return: An iterator over thresholded distance matrices.
        :rtype: Iterator[DistanceMatrix]
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
        Compute an affinity matrix from the distance matrix.

        The operation can be computed either in place or out of place depending on
        the `in_place` flag. The distance matrix is assumed to contain squared distances
        if the DistanceMatrix`s `dist_type` metadata is `sqeuclidean`.

        :param aff_type: The type of affinity to compute (e.g., `gaussian`).
        :type aff_type: AffinityType
        :param eps: Optional scaling parameter for the affinity function. If None,
                    a default is used. (default: None)
        :type eps: Optional[float]
        :param in_place: If True, modify the existing matrix; otherwise, return a new one. (default: False)
        :type in_place: bool
        :return: The computed affinity matrix.
        :rtype: AffinityMatrix
        """

        dist_is_sq = self.metadata.dist_type == "sqeuclidean"
        
        if aff_type == "gaussian":
            return gaussian(self, eps, dist_is_sq, in_place)

def gaussian(
    dists: DistanceMatrix,
    eps: Optional[float] = None,
    dist_is_sq: bool = False,
    in_place: bool = False,
) -> AffinityMatrix:
    """
    Compute a Gaussian affinity matrix from a distance matrix.

    :param dists: The distance matrix to convert into an affinity matrix.
    :type dists: DistanceMatrix
    :param eps: Scaling parameter for the Gaussian kernel. If None, chosen to be 1. (default: None)
    :type eps: Optional[float]
    :param dist_is_sq: Whether the distance matrix already contains squared
                       distances. (default: False)
    :type dist_is_sq: bool
    :param in_place: If True, compute the affinity in place by modifying the
                     given distance matrix; otherwise, compute out of place. (default: False)
    :type in_place: bool
    :return: The Gaussian affinity matrix.
    :rtype: AffinityMatrix
    """

    if in_place:
        return _gaussian_in_place(dists, 1.0 if eps is None else eps, dist_is_sq)
    return _gaussian_out_of_place(dists, 1.0 if eps is None else eps, dist_is_sq)


def _gaussian_in_place(
    dists: DistanceMatrix, eps: float, dist_is_sq: bool
) -> AffinityMatrix:

    if not dist_is_sq:
        dists **= 2.0

    dists /= eps**2
    dists *= -1.0
    dists.iexp()

    return AffinityMatrix.create(dists, eps=eps, aff_type="gaussian")


def _gaussian_out_of_place(
    dists: DistanceMatrix, eps: float, dist_is_sq: bool
) -> AffinityMatrix:
    if dist_is_sq:
        return AffinityMatrix.create(
            ((dists / (eps**2)) * -1.0).exp(),
            eps=eps,
            metadata=dists.metadata,
            aff_type="gaussian",
        )
    return AffinityMatrix.create(
        (((dists / eps) ** 2) * -1).exp(),
        eps=eps,
        metadata=dists.metadata,
        aff_type="gaussian",
    )


class DenseDistanceMatrix(DistanceMatrix, DenseArray):
    """
    Implementation of a dense (NumPy-backed) distance matrix.
    Provides thresholding and adjacency operations for dense arrays.

    Typically not instantiated directly: instead, construct an
    `DistanceMatrix` in `DenseArray` format which will return an instance.
    """

    def _execute_threshold(
        self, radius: float, in_place: bool = False
    ) -> DistanceMatrix:
        """
        Threshold a dense distance matrix by eliminating entries larger
        than a given radius.

        :param radius: The maximum radius to retain distances. Entries
                       greater than this are set to infinity.
        :type radius: float
        :param in_place: If True, modify the existing distance matrix;
                         otherwise, operate on a copy.
        :type in_place: bool
        :return: The thresholded distance matrix.
        :rtype: DistanceMatrix
        """
        dist: DistanceMatrix = (
            self
            if in_place
            else DistanceMatrix.create(self.copy(), metadata=self.metadata)
        )
        dist[dist > radius] = np.inf
        dist.metadata.radius = radius
        return dist

    def _execute_adjacency(self, copy: bool) -> AdjacencyMatrix:
        """
        Convert the dense distance matrix into an adjacency matrix by
        checking for nonzero entries. Due to numpy matrices being immutable, always copies.

        :param copy: Whether to return a copy or reuse existing data.
        :type copy: bool
        :return: The adjacency matrix.
        :rtype: AdjacencyMatrix
        """
        return AdjacencyMatrix.create(self != 0)


class CsrDistanceMatrix(DistanceMatrix, CsrArray):
    """
    Implementation of a sparse (CSR-backed) distance matrix.
    Provides thresholding and adjacency operations for sparse arrays.

    Typically not instantiated directly: instead, construct an
    `DistanceMatrix` in `CsrArray` format which will return an instance.
    """

    def _execute_threshold(
        self, radius: float, in_place: bool = False
    ) -> DistanceMatrix:
        """
        Threshold a sparse (CSR) distance matrix.

        Currently not implemented.

        :param radius: The maximum radius to retain distances.
        :type radius: float
        :param in_place: If True, modify the existing distance matrix;
                         otherwise, operate on a copy.
        :type in_place: bool
        :raises NotImplementedError: Always, since this method is not yet implemented.
        """
        raise NotImplementedError()

    def _execute_adjacency(self, copy: bool) -> AdjacencyMatrix:
        """
        Convert the sparse distance matrix into an adjacency matrix.

        Currently not implemented.

        :param copy: Whether to return a copy or reuse existing data.
        :type copy: bool
        :raises NotImplementedError: Always, since this method is not yet implemented.
        """
        raise NotImplementedError()
