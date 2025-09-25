from __future__ import annotations

from abc import ABC
from typing import Iterator, Optional, Tuple, Type

import numpy as np

from src.array import DenseArray
from src.array.linalg import eigen_decomp
from src.geometry import Embedding
from src.geometry.eigen_system import EigenSystem
from src.geometry.matrix import LaplacianMatrix
from src.linalg.covariance import Covariance
from src.linalg.func import func
from src.object import chunk
from src.object.metadata import Metadata


class RMetricSystem(EigenSystem):
    """
    Container class for a single Riemannian metric eigensystem.
    """

    pass


class RMetricSystems(EigenSystem):
    """
    Container class for multiple Riemannian metric eigensystems. Index 0 is an array of eigenvalue arrays. Index 1 is an array of eigenvector matrices.
    """

    fixed_value_ndim: int = 2
    fixed_vector_ndim: int = 3
    fixed_value_type: Type = DenseArray
    fixed_vector_type: Type = DenseArray


class RMetric(func[RMetricSystem, RMetricSystems], ABC):
    @classmethod
    def global_func(
        cls,
        x_pts: Embedding,
        lap: DenseArray,
        ncomp: Optional[int] = None,
        dual: bool = True,
        md: Optional[Metadata] = None,
    ) -> RMetricSystem:
        """
        Compute the global Riemannian metric decomposition from data embeddings and a Laplacian vector.

        :param x_pts: Input data embedding.
        :param lap: Laplacian vector or equivalent weight vector.
        :param ncomp: Number of eigencomponents to retain. If None, keep all. (default: None).
        :param dual: If True, return the dual metric. If False, return the inverse. (default: True).
        :param md: Optional metadata to merge with x_pts metadata. If None, inherits from x_pts. (default: None)
        :return: An `RMetricSystems` object containing eigenvalues and eigenvectors.
        """
        md = x_pts.metadata if md is None else x_pts.metadata.update_with(md)
        dual_metric = Covariance.global_func(x_pts, lap, md)
        return RMetricSystem(
            cls._decompose_dual_metric(dual_metric, ncomp, dual), metadata=md
        )

    @staticmethod
    def _decompose_dual_metric(
        dual_metric: DenseArray, ncomp: int | None, dual: bool
    ) -> Tuple[DenseArray, DenseArray]:
        """
        Performs eigen-decomposition of a dual Riemannian metric and optionally inverts it.

        :param dual_metric: Symmetric positive semi-definite matrix (e.g., weighted covariance).
        :param ncomp: Number of components (eigenpairs) to keep. If None, keep all.
        :param dual: If True, return the dual metric. If False, return the inverse. (default: True).
        :return: Tuple of (eigenvalues, eigenvectors).
        """
        eigvals, eigvecs = eigen_decomp(
            dual_metric.as_nparray(), ncomp, is_symmetric=True
        )
        eigvals *= 0.5

        if not dual:
            zero_eigvals_mask = eigvals == 0.0
            eigvals = np.divide(1.0, eigvals, where=~zero_eigvals_mask)
            eigvals[zero_eigvals_mask] = np.inf

        return DenseArray(eigvals), DenseArray(eigvecs)

    @classmethod
    def local_func(
        cls,
        x_pts: Embedding,
        mean_pts: DenseArray,
        lap: LaplacianMatrix,
        ncomp: Optional[int],
        dual: bool,
        md: Optional[Metadata] = None,
    ) -> RMetricSystems:
        """
        Computes the local Riemannian metric decomposition using uncentered data and affinity weights.

        :param x_pts: Input data embedding.
        :param mean_pts: Local mean points for each neighborhood.
        :param lap: Laplacian weights for local neighborhoods.
        :param ncomp: Number of eigencomponents to retain. If None, keep all. (default: None).
        :param dual: If True, return the dual metric. If False, return the inverse. (default: True).
        :param md: Metadata to include in the return `RMetricSystems`.

        :return: The desired local Riemannian metric in an `RMetricSystems` object.
        """

        # hacky
        dual_metric = Covariance.local_func(
            x_pts, mean_pts, lap, needs_means=False, md=md
        )
        if md is None:
            return RMetricSystems(
            cls._decompose_dual_metric(dual_metric, ncomp, dual)
        )
        return RMetricSystems(
            cls._decompose_dual_metric(dual_metric, ncomp, dual), metadata=md
        )

    @classmethod
    def local_iter(
        cls,
        x_pts: Embedding,
        lap: LaplacianMatrix,
        ncomp: Optional[int] = None,
        mean_pts: Optional[DenseArray] = None,
        dual: bool = True,
        bsize: Optional[int] = None,
    ) -> Iterator[RMetricSystems]:
        """
        Computes the local Riemannian metric decomposition in a batched manner on all `x_pts` centered on `mean_pts`.
        If there are multiple mean points, computes metrics for each.

        :param x_pts: Input data embedding.
        :param lap: Laplacian weights for local neighborhoods.
        :param ncomp: Number of eigencomponents to retain. If None, keep all. (default: None).
        :param mean_pts: Optional local mean points. If None, defaults to each data point. (default: None).
        :param dual: If True, return the dual metric. If False, return the inverse. (default: True).
        :param bsize: Optional batch size for dividing the computation. (default: None).

        :return: Iterator that yields `RMetricSystems` objects per batch.
        """

        if mean_pts is None:
            mean_pts = x_pts[np.arange(x_pts.shape[0])]
        md: Metadata = x_pts.metadata.update_with(lap.metadata)
        return (
            cls.local_func(x_pts, mean_chunk, weight_chunk, ncomp, dual, md)
            for mean_chunk, weight_chunk in zip(
                chunk(mean_pts, bsize=bsize), chunk(lap, bsize=bsize)
            )
        )

    @classmethod
    def local(
        cls,
        x_pts: Embedding,
        lap: LaplacianMatrix,
        ncomp: Optional[int] = None,
        mean_pts: Optional[DenseArray] = None,
        dual: bool = True,
        bsize: Optional[int] = None,
    ) -> RMetricSystems:
        """
        Lazily computes the local Riemannian metric decomposition on all `x_pts` centered on `mean_pts`.
        If there are multiple mean points, computes metrics for each.

        :param x_pts: Input data embedding.
        :param lap: Laplacian weights for local neighborhoods.
        :param ncomp: Number of eigencomponents to retain. If None, keep all. (default: None).
        :param mean_pts: Optional local mean points. If None, defaults to each data point. (default: None).
        :param dual: If True, return the dual metric. If False, return the inverse. (default: True).
        :param bsize: Optional batch size for dividing the computation. (default: None).

        :return: The desired local Riemannian metric in an `RMetricSystems` object.
        """

        return super().package(
            x_pts, lap, ncomp, mean_pts, dual, output_cls=RMetricSystems, bsize=bsize
        )
