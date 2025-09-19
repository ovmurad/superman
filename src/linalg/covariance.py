from __future__ import annotations

from abc import ABC
from itertools import cycle
from typing import Any, Iterator, Optional

import numpy as np
from src.array import BaseArray, CsrArray, DenseArray
from src.geometry import normalize
from src.geometry import Points
from src.linalg.func import func
from src.object import ObjectMixin
from src.object import chunk
from src.object import GeometryMatrixMixin
from src.object.metadata import Metadata


class CovarianceMatrixMixin(GeometryMatrixMixin, ABC):
    pass

class CovarianceMatrix(CovarianceMatrixMixin, DenseArray):
    pass

class CovarianceMatrices(CovarianceMatrixMixin, DenseArray):
    fixed_ndim = 3

class Covariance(func[CovarianceMatrix, CovarianceMatrices], ABC):
    @classmethod
    def global_func(*args: Any, **kwargs: Any) -> CovarianceMatrix:
        raise NotImplementedError()

    @classmethod
    def local_func(
        cls,
        x_pts: DenseArray,
        mean_pts: DenseArray,
        weights: Optional[BaseArray],
        needs_means: bool,
        md: Metadata,
    ) -> CovarianceMatrices:
        #hack for now
        x_pts: np.ndarray = x_pts.as_nparray()
        mean_pts: np.ndarray = mean_pts.as_nparray()
        weights: Optional[np.ndarray] = None if weights is None else weights.as_nparray()

        nmeans: int = mean_pts.shape[0] if weights is None else weights.shape[0]

        if weights is None:
            cov: np.ndarray = (1.0 / x_pts.shape[0]) * (x_pts.T @ x_pts)
            cov = np.repeat(np.expand_dims(cov, axis=0), repeats=nmeans, axis=0)
        elif isinstance(weights, np.ndarray):
            cov = np.einsum("mn,np,nq->mpq", weights, x_pts, x_pts)

        if weights is None:
            means_outer_x_pts = np.einsum("mp,q->mpq", mean_pts, np.mean(x_pts, axis=0))
        else:
            means_outer_x_pts = np.einsum("mp,mq->mpq", mean_pts, weights @ x_pts)

        cov -= means_outer_x_pts
        cov -= np.transpose(means_outer_x_pts, axes=(0, 2, 1))

        if needs_means:

            if weights is None:
                outer_means = np.einsum("mp,mq->mpq", mean_pts, mean_pts)
            else:
                weights = weights.sum(axis=1, keepdims=True)
                outer_means = np.einsum("mp,m,mq->mpq", mean_pts, weights, mean_pts)

            cov += outer_means

        return CovarianceMatrices(cov, metadata=md)

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
        if weights is None and mean_pts is None:
            mean_pts = x_pts[np.arange(x_pts.shape[0])]
        elif mean_pts is None:
            mean_pts = x_pts if weights is None else weights * x_pts
        if weights is not None and needs_norm:
            weights = normalize(weights, axis=None, in_place=in_place_norm)
        md: Metadata = x_pts.metadata.update_with(weights.metadata) if isinstance(weights, ObjectMixin) else x_pts.metadata
        return (cls.local_func(x_pts, mean_chunk, weight_chunk, needs_means, md) for mean_chunk, weight_chunk in chunk((mean_pts, weights), bsize=bsize))

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
        return super().package(x_pts, mean_pts, weights, needs_means, needs_norm, in_place_norm, output_cls=CovarianceMatrices, bsize=bsize)
