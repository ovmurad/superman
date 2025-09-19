
from typing import Any
from src.array.linalg import EigenSolver, eigen_decomp
from src.geometry import EigenSystem
from src.geometry.matrix import LaplacianMatrix
from src.geometry.matrix import SYM_LAPLACIAN_TYPES
from src.object.metadata import Metadata


def diffusion_embedding(
    lap: LaplacianMatrix,
    ncomp: int,
    eigen_solver: EigenSolver = "dense",
    drop_first: bool = True,
    **kwargs: Any,
) -> EigenSystem:
    """
    Compute the Laplacian embedding given a precomputed Laplacian matrix.

    :param lap: A LaplacianMatrix object.
    :param ncomp: Number of components (eigenvectors) to compute.
    :param eigen_solver: Eigenvalue solver to use. (default: "dense").
    :param drop_first: Whether to drop the first eigenvector (typically trivial). (default: True).
    :param kwargs: Additional keyword arguments passed to the eigen decomposition function.

    :return: A tuple of eigenvalues and eigenvectors.
    """

    md: Metadata = lap.metadata

    eigvals, eigvecs = eigen_decomp(
        arr=lap.as_nparray() * -1.0 if lap.metadata.aff_minus_id else lap.as_nparray(),
        ncomp=ncomp + int(drop_first),
        eigen_solver=eigen_solver,
        is_symmetric=lap.metadata.lap_type in SYM_LAPLACIAN_TYPES,
        largest=False,
        **kwargs,
    )

    if drop_first:
        eigvals = eigvals[1:]
        eigvecs = eigvecs[:, 1:]

    return EigenSystem((eigvals, eigvecs), metadata=md)