from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.array import DenseArray
from src.geometry import DistanceMatrix
from src.object import FunctionMixin
from src.object import DistanceType, Metadata
from src.object import ObjectMixin


class PointsMixin(ObjectMixin, ABC):
    """
    Mixin class for point cloud objects.

    Adds fixed dimensionality and dtype constraints, metadata handling,
    and distance computation functionality to point cloud classes.
    """

    metadata: Metadata

    fixed_ndim = 2
    fixed_dtype = np.float64

    def __init__(self, *args, **metadata) -> None:
        """
        Initialize a Points object with optional metadata.

        :param args: Positional arguments forwarded to the base class.
        :param metadata: Keyword arguments representing metadata fields.
        """
        super().__init__(*args, cls=Metadata, **metadata)

    @property
    def npts(self) -> int:
        """
        Number of points in the point set.

        :return: Number of points.
        :rtype: int
        """
        return self.shape[0]

    @property
    def nfeats(self) -> int:
        """
        Number of features (dimensions) per point.

        :return: Number of features per point.
        :rtype: int
        """
        return self.shape[1]


class Points(PointsMixin, DenseArray):
    def distance(
        self,
        y_pts: Optional[Points] = None,
        dist_type: DistanceType = "euclidean",
    ) -> DistanceMatrix:
        """
        Compute the pairwise distance matrix between points.

        If `y_pts` is provided, computes distances between `self` and `y_pts`.
        Otherwise, computes distances among points in `self`.

        :param y_pts: Another `Points` object to compute distances to. If None,
                    distances are computed within `self`.
        :type y_pts: Optional[Points]
        :param dist_type: The type of distance metric to use ("euclidean", "cityblock", "sqeuclidian"). Name of metric is stored in metadata. (default: "euclidian")
        :type dist_type: DistanceType
        :return: A `DistanceMatrix` containing pairwise distances. The matrix name
                is constructed from the metadata of the points involved.
        :rtype: DistanceMatrix
        """

        x_pts_name = self.metadata.name
        y_pts_name = None if y_pts is None else y_pts.metadata.name
        dist_name = (
            None
            if (x_pts_name is None or y_pts_name is None)
            else x_pts_name + "_" + y_pts_name
        )

        dist_mat = (
            DenseArray.distance(self, self, dist_type=dist_type)
            if y_pts is None
            else DenseArray.distance(self, y_pts, dist_type=dist_type)
        )

        return DistanceMatrix(dist_mat, dist_type=dist_type, name=dist_name)


class Data(Points):

    @property
    def D(self) -> int:
        return self.nfeats


class Embedding(Points):

    @property
    def p(self) -> int:
        return self.nfeats


class Coordinates(PointsMixin, DenseArray):

    @property
    def d(self) -> int:
        return self.nfeats
