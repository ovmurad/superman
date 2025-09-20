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
    pass


class RMetricSystems(EigenSystem):
    fixed_value_ndim: int = 2
    fixed_vector_ndim: int = 3
    fixed_value_type: Type = DenseArray
    fixed_vector_type: Type = DenseArray


class RMetric(func[RMetricSystem, RMetricSystems], ABC):
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
        md: Metadata,
    ) -> RMetricSystems:
        """
        Computes the local Riemannian metric using uncentered data and affinity weights.

        :param x_pts: Input data points.
        :param mean_pts: Local means for each point.
        :param lap: Laplacian weights.
        :param ncomp: Number of eigencomponents to retain.
        :param dual: If True, return the dual metric. If False, return the inverse. (default: True).
        :return: Tuple of (eigenvalues, eigenvectors).
        """

        # hacky
        dual_metric = Covariance.local_func(
            x_pts, mean_pts, lap, needs_means=False, md=md
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
        return super().package(
            x_pts, lap, ncomp, mean_pts, dual, output_cls=RMetricSystems, bsize=bsize
        )
