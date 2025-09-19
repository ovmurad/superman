from __future__ import annotations

from abc import ABC
from typing import Any, Iterator, Optional, Self, Sequence, Tuple

import numpy as np

from src.array import BaseArray, DenseArray
from src.geometry.matrix import DistanceMatrix, LaplacianMatrix
from src.geometry.normalize import normalize
from src.object import DistanceType, Metadata, ObjectMixin
from src.object.object_mixin import chunk


class PointsMixin(ObjectMixin, ABC):
    """
    Mixin class for point cloud objects.

    Adds fixed dimensionality and dtype constraints, metadata handling,
    and distance computation functionality to point cloud classes.
    """

    metadata: Metadata

    fixed_ndim = 2
    fixed_dtype = np.float64

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
    def pairwise_distance(
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

        return DistanceMatrix.create(dist_mat, dist_type=dist_type, name=dist_name)

    def demean(
        self,
        mean_pt: Optional[DenseArray| int | bool] = None,
        weights: Optional[BaseArray] = None,
        needs_norm: bool = True,
        in_place_demean: bool = False,
        in_place_norm: bool = False,
    ) -> Tuple[Points, DenseArray, BaseArray | None]:
        """
        Prepare inputs for single function computation: normalizes weights, computes mean if needed,
        and demeans 'x_pts'. Optionally performs demeans in-place on 'x_pts' and normalizes in-place on 'weights'.

        :param x_pts: Array of input points.
        :param mean_pt: Optional mean point. (default: None)
            - If True, compute the mean (weighted or unweighted) of 'x_pts'.
            - If an integer, use 'x_pts[mean_pt]' as the mean.
            - If an array, use as mean directly.
            - If None, do not demean.
        :param weights: Optional weights for each point. (default: None)
        :param needs_norm: Whether to normalize weights. (default: True)
        :param in_place_demean: Whether to demean in-place on 'x_pts'. (default: False)
        :param in_place_norm: Whether to normalize weights in-place on 'weights'. (default: False)

        :return: self if 'in_place_demean' otherwise a copy.
        """

        pts: Points = self if in_place_demean else self.copy()

        if needs_norm and weights is not None:
            weights = normalize(weights, axis=None, in_place=in_place_norm)

        if mean_pt is True:
            if weights is None:
                mean_pt = pts.mean(axis=0)
            else:
                mean_pt = (pts * weights.expand_dims(axis=1)).sum(axis=0)
        elif isinstance(mean_pt, int):
            mean_pt = pts[mean_pt]

        if mean_pt is not None:
            pts -= mean_pt

        return pts, mean_pt, weights


    @classmethod
    def concat_with_metadata(cls, arrs: Sequence[Self], axis: int = 0) -> Self:
        return cls(super().concat(arrs, axis=axis), metadata=arrs[0].metadata)

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
