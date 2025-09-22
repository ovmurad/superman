from __future__ import annotations

from abc import ABC
from itertools import cycle
from typing import Iterator, Optional

import numpy as np

from src.array import BaseArray, DenseArray
from src.geometry import Points, normalize
from src.linalg.func import func
from src.object import GeometryMatrixMixin, ObjectMixin, chunk
from src.object.metadata import Metadata


class CovarianceMatrixMixin(GeometryMatrixMixin, ABC):
    """
    Abstract mixin class for covariance matrices.

    Provides a common interface for covariance-related matrix objects.
    """

    pass


class CovarianceMatrix(CovarianceMatrixMixin, DenseArray):
    """
    Container class for a single Dense covariance matrix.
    """

    pass


class CovarianceMatrices(CovarianceMatrixMixin, DenseArray):
    """
    Container class for multiple covariance matrices.

    Extends :class:`DenseArray` with a fixed third dimension for stacking
    covariance matrices along axis 0.
    """

    fixed_ndim = 3


class Covariance(func[CovarianceMatrix, CovarianceMatrices], ABC):
    @classmethod
    def global_func(
        cls,
        centered_x_pts: DenseArray,
        weights: Optional[BaseArray] = None,
        md: Optional[Metadata] = None,
    ) -> CovarianceMatrix:
        """
        Computes the global covariance matrix from centered data points.

        :param centered_x_pts: Centered data points with shape ``(m, p)``,
            where ``m`` is the number of samples and ``p`` is the feature dimension.
        :param weights: Optional weight array of shape ``(m,)`` to compute a
            weighted covariance. If None, computes the unweighted covariance. (default: None).
        :param md: Metadata to attach to the returned `CovarianceMatrix`.

        :return: A `CovarianceMatrix` representing the global covariance structure.
        """
        centered_x_pts_arr = centered_x_pts.as_nparray()

        if weights is None:
            return (
                CovarianceMatrix(
                    (1.0 / centered_x_pts_arr.shape[0])
                    * (centered_x_pts_arr.T @ centered_x_pts_arr)
                )
                if md is None
                else CovarianceMatrix(
                    (1.0 / centered_x_pts_arr.shape[0])
                    * (centered_x_pts_arr.T @ centered_x_pts_arr),
                    metadata=md,
                )
            )
        return (
            CovarianceMatrix(
                np.einsum(
                    "mp,m,mq->pq",
                    centered_x_pts_arr,
                    weights.as_nparray(),
                    centered_x_pts_arr,
                )
            )
            if md is None
            else CovarianceMatrix(
                np.einsum(
                    "mp,m,mq->pq",
                    centered_x_pts_arr,
                    weights.as_nparray(),
                    centered_x_pts_arr,
                ),
                metadata=md,
            )
        )

    @classmethod
    def local_func(
        cls,
        x_pts: DenseArray,
        mean_pts: DenseArray,
        weights: Optional[BaseArray],
        needs_means: bool,
        md: Optional[Metadata] = None,
    ) -> CovarianceMatrices:
        """
        Computes the local covariance on all `x_pts` centered on `mean_pts`. If there are multiple mean points, computes covariance for each.

        :param x_pts: The neighborhood to calculate covariance with.
        :param mean_pts: The center for the covariance calculation. If there are multiple mean points, computes covariance for each.
        :param weights: Optional weight matrix to calculate weighted covariance with.
        :param needs_means: Whether `x_pts` is centered.
        :param md: Optional metadata to include in return `CovarianceMatrices`. (default: None).

        :return: Matrix where covariance matrices are joined along axis 0.
        """
        # hack for now
        x_pts_arr: np.ndarray = x_pts.as_nparray()
        mean_pts_arr: np.ndarray = mean_pts.as_nparray()

        nmeans: int = mean_pts_arr.shape[0] if weights is None else weights.shape[0]

        if weights is None:
            cov: np.ndarray = (1.0 / x_pts_arr.shape[0]) * (x_pts_arr.T @ x_pts_arr)
            cov = np.repeat(np.expand_dims(cov, axis=0), repeats=nmeans, axis=0)
        elif isinstance(weights, DenseArray):
            cov = np.einsum("mn,np,nq->mpq", weights.as_nparray(), x_pts_arr, x_pts_arr)

        if weights is None:
            means_outer_x_pts = np.einsum(
                "mp,q->mpq", mean_pts_arr, np.mean(x_pts_arr, axis=0)
            )
        else:
            means_outer_x_pts = np.einsum(
                "mp,mq->mpq", mean_pts_arr, DenseArray(weights * x_pts_arr).as_nparray()
            )

        cov -= means_outer_x_pts
        cov -= np.transpose(means_outer_x_pts, axes=(0, 2, 1))

        if needs_means:

            if weights is None:
                outer_means = np.einsum("mp,mq->mpq", mean_pts_arr, mean_pts_arr)
            else:
                weights = weights.sum(axis=1, keepdims=True)
                outer_means = np.einsum(
                    "mp,m,mq->mpq",
                    mean_pts_arr,
                    DenseArray(weights).as_nparray(),
                    mean_pts_arr,
                )

            cov += outer_means

        return (
            CovarianceMatrices(cov)
            if md is None
            else CovarianceMatrices(cov, metadata=md)
        )

    @classmethod
    def local_iter(
        cls,
        x_pts: Points,
        mean_pts: Optional[DenseArray] = None,
        weights: Optional[BaseArray] = None,
        needs_means: bool = True,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
    ) -> Iterator[CovarianceMatrices]:
        """
        Computes the local covariance in a batched manner on all `x_pts` centered on `mean_pts`. If there are multiple mean points, computes covariance for each.

        :param x_pts: The neighborhood to calculate covariance with.
        :param mean_pts: Optional center for the covariance calculation. If there are multiple mean points, computes covariance for each. If None, each point is treated as its own center. (default: None)
        :param weights: Optional weight matrix to calculate weighted covariance with. (default: None).
        :param needs_means: Whether `x_pts` is centered. (default: True)
        :param needs_norm: Whether the weights need to be normalized. (default: True)
        :param in_place_norm: If `needs_norm`, whether to do the weight normalization in-place. (default: False)
        :param bsize: Optional batch size to divide the covariance calculation. (default: None).

        :return: Iterator that gives the desired covariance matrices.
        """
        if weights is None and mean_pts is None:
            processed_mean_pts: DenseArray = x_pts[np.arange(x_pts.shape[0])]
        elif mean_pts is None:
            processed_mean_pts = (
                x_pts if weights is None else DenseArray(weights * x_pts)
            )
        else:
            processed_mean_pts = mean_pts
        if weights is not None and needs_norm:
            weights = normalize(weights, axis=None, in_place=in_place_norm)
        md: Metadata = (
            x_pts.metadata.update_with(weights.metadata)
            if isinstance(weights, ObjectMixin)
            else x_pts.metadata
        )
        return (
            cls.local_func(x_pts, mean_chunk, weight_chunk, needs_means, md)
            for mean_chunk, weight_chunk in zip(
                chunk(processed_mean_pts, bsize=bsize),
                cycle((None,)) if weights is None else chunk(weights, bsize=bsize),
            )
        )

    @classmethod
    def local(
        cls,
        x_pts: Points,
        mean_pts: Optional[DenseArray] = None,
        weights: Optional[BaseArray] = None,
        needs_means: bool = True,
        needs_norm: bool = True,
        in_place_norm: bool = False,
        bsize: Optional[int] = None,
    ) -> CovarianceMatrices:
        """
        Lazily computes the local covariance on all `x_pts` centered on `mean_pts`. If there are multiple mean points, computes covariance for each.

        :param x_pts: The neighborhood to calculate covariance with.
        :param mean_pts: Optional center for the covariance calculation. If there are multiple mean points, computes covariance for each. If None, each point is treated as its own center. (default: None)
        :param weights: Optional weight matrix to calculate weighted covariance with. (default: None).
        :param needs_means: Whether `x_pts` is centered. (default: True)
        :param needs_norm: Whether the weights need to be normalized. (default: True)
        :param in_place_norm: If `needs_norm`, whether to do the weight normalization in-place. (default: False)
        :param bsize: Optional batch size to divide the covariance calculation. (default: None).

        :return: The desired covariance matrices in a `CovarianceMatrices` object.
        """
        return super().package(
            x_pts,
            mean_pts,
            weights,
            needs_means,
            needs_norm,
            in_place_norm,
            output_cls=CovarianceMatrices,
            bsize=bsize,
        )
