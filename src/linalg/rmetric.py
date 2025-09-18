from abc import ABC
from typing import Iterator, Optional, Tuple

import numpy as np
from src.array import DenseArray
from src.array import BaseArray
from src.array.linalg import eigen_decomp
from src.geometry.matrix import LaplacianMatrix
from src.geometry import Embedding
from src.geometry import normalize
from src.linalg.covariance import local_covariance, local_covariance_func
from src.object import ObjectMixin
from src.object import chunk


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
    eigvals, eigvecs = eigen_decomp(dual_metric, ncomp, is_symmetric=True)
    eigvals *= 0.5

    if not dual:
        zero_eigvals_mask = eigvals == 0.0
        eigvals = np.divide(1.0, eigvals, where=~zero_eigvals_mask)
        eigvals[zero_eigvals_mask] = np.inf

    return eigvals, eigvecs

class EmbeddingMixin(ObjectMixin, ABC):
    pass

@staticmethod
def local_rmetric_func(
    x_pts: Embedding,
    mean_pts: DenseArray,
    lap: LaplacianMatrix,
    ncomp: int | None,
    dual: bool,
) -> Tuple[DenseArray, DenseArray]:
    """
    Computes the local Riemannian metric using uncentered data and affinity weights.

    :param x_pts: Input data points.
    :param mean_pts: Local means for each point.
    :param lap: Laplacian weights.
    :param ncomp: Number of eigencomponents to retain.
    :param dual: If True, return the dual metric. If False, return the inverse. (default: True).
    :return: Tuple of (eigenvalues, eigenvectors).
    """

    #hacky
    dual_metric = local_covariance_func(x_pts, mean_pts, lap, needs_means=False).as_nparray()
    return _decompose_dual_metric(dual_metric, ncomp, dual)

def local_rmetric_iter(
    x_pts: Embedding,
    lap: LaplacianMatrix,
    ncomp: Optional[int] = None,
    mean_pts: Optional[DenseArray] = None,
    dual: bool = True,
    bsize: Optional[int] = None,
) -> Iterator[BaseArray]:
    if mean_pts is None:
        mean_pts = x_pts[np.arange(x_pts.shape[0])]
    return (local_rmetric_func(x_pts, mean_chunk, weight_chunk, ncomp, dual) for mean_chunk, weight_chunk in chunk((mean_pts, lap), bsize=bsize))

def local_rmetric(
    x_pts: Embedding,
    lap: LaplacianMatrix,
    ncomp: Optional[int] = None,
    mean_pts: Optional[DenseArray] = None,
    dual: bool = True,
    bsize: Optional[int] = None,
) -> BaseArray:
    if bsize is None:
        return local_rmetric_func(
            x_pts, lap, ncomp, mean_pts, dual, bsize
        )

    local_data_iter = local_rmetric_iter(
        x_pts, lap, ncomp, mean_pts, dual, bsize
    )

    local_data_batches = [[] for _ in range(2)]
    for ld in local_data_iter:
        for out_ld_b, out in zip(local_data_batches, ld):
            out_ld_b.append(out)
    return tuple(np.concatenate(ld_batches) for ld_batches in local_data_batches)