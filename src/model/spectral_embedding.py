import warnings
from typing import Any, Final, Optional, Set

import numpy as np

from src.array.linalg import EigenSolver
from src.geometry import EigenSystem, Embedding
from src.geometry.embedding import laplacian_embedding
from src.geometry.matrix import AffinityMatrix, DistanceMatrix, LaplacianMatrix
from src.geometry.points import Points
from src.model.embedding import GeometryType
from src.object import LaplacianType
from src.object.metadata import Metadata

DIFFUSION_TYPES: Final[Set[LaplacianType]] = {"geometric"}
EPS_RADIUS_RATIO: Final[int] = 1 / 3


def compute_diffusion_maps(eig: EigenSystem, diffusion_time: float) -> EigenSystem:
    """Credit to Satrajit Ghosh (http://satra.cogitatum.org/) for final steps"""
    md: Metadata = eig.metadata
    # Check that diffusion maps is using the correct laplacian, warn otherwise
    if eig.metadata.lap_type not in DIFFUSION_TYPES:
        warnings.warn(
            "for correct diffusion maps embedding use laplacian type 'geometric' or 'renormalized'."
        )
    # Step 5 of diffusion maps:
    vectors = eig.eigenvectors.as_nparray()
    lambdas = eig.eigenvalues.as_nparray()
    if diffusion_time == 0:
        lambdas = np.abs(lambdas)
        lambdas = lambdas / (1 - lambdas)
    else:
        lambdas = np.abs(lambdas)
        lambdas = lambdas ** float(diffusion_time)
    return EigenSystem((lambdas, vectors), metadata=md)


class SpectralEmbedding(Embedding):
    def __init__(
        self,
        radius: float,
        n_components: int = 2,
        eps: Optional[float] = None,
        eigen_solver: EigenSolver = "amg",
        drop_first: bool = True,
        lap_type: LaplacianType = "geometric",
        diffusion_time: float = 0.0,
        diffusion_maps: bool = False,
        **solver_kwds: Any,
    ) -> None:
        """
        Spectral embedding for non-linear dimensionality reduction.

        Forms an affinity matrix given by the specified function and
        applies spectral decomposition to the corresponding graph Laplacian.
        The resulting transformation is given by the eigenvectors for each data point.

        :param radius: Radius for adjacency and affinity calculations.
        :param n_components: Number of coordinates for the manifold.
        :param eps: Optional epsilon to use for adjacency and affinity calculations. If None, uses `EPS_RADIUS_RATIO` to calculate it from the radius.
        :param eigen_solver: Eigenvalue decomposition method.
            - ``'auto'``: algorithm will attempt to choose the best method for input data
            - ``'dense'``: use standard dense matrix operations (avoid for large problems)
            - ``'arpack'``: use Arnoldi iteration in shift-invert mode (can be unstable)
            - ``'lobpcg'``: Locally Optimal Block Preconditioned Conjugate Gradient Method
            - ``'amg'``: Algebraic Multigrid (requires pyamg; may be faster but less stable)
        :param drop_first: Whether to drop the first eigenvector.
            For spectral embedding, this should be ``True`` (the first eigenvector is a constant vector
            for a connected graph). For spectral clustering, this should be ``False``.
        :param lap_type: The type of graph laplacian to use.
        :param diffusion_time: The diffusion time used if `diffusion_maps` is True.
        :param diffusion_maps: Whether to return the diffusion maps version by
            re-scaling the embedding by the eigenvalues.
        :param solver_kwds: Additional keyword arguments to pass to the selected eigen_solver.

        :returns: A configured `SpectralEmbedding` object.

        :references:
            - Ulrike von Luxburg. *A Tutorial on Spectral Clustering*, 2007
            http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

            - Andrew Y. Ng, Michael I. Jordan, Yair Weiss.
            *On Spectral Clustering: Analysis and an Algorithm*, 2011
            http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.19.8100

            - Jianbo Shi, Jitendra Malik.
            *Normalized Cuts and Image Segmentation*, 2000
            http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324
        """

        self.n_components = n_components
        self.radius = radius
        self.eps = radius * EPS_RADIUS_RATIO if eps is None else eps
        self.eigen_solver = eigen_solver
        self.drop_first = drop_first
        self.solver_kwds = solver_kwds
        self.lap_type = lap_type
        self.diffusion_time = diffusion_time
        self.diffusion_maps = diffusion_maps

    def fit(self, data: GeometryType) -> EigenSystem:
        """
        Fit the model from geometry matrix data.

        :param data: Any GeometryType geometry matrix. If a `LaplacianMatrix`, overrides the `lap_type` specified in the constructor.

        :returns: The `EigenSystem` corresponding to the spectral embedding.
        """
        if isinstance(data, Points):
            data = data.pairwise_distance()
        if isinstance(data, DistanceMatrix):
            data = data.affinity(eps=self.eps, in_place=True)
        if isinstance(data, AffinityMatrix) or isinstance(data, LaplacianMatrix):
            lap_emb = laplacian_embedding(
                data,
                self.n_components + int(self.drop_first),
                self.lap_type,
                drop_first=False,
                in_place=True,
                **self.solver_kwds,
            )

            if self.diffusion_maps:
                lap_emb = compute_diffusion_maps(lap_emb, self.diffusion_time)

            return lap_emb.drop_first() if self.drop_first else lap_emb
        raise ValueError(f"`data` type {type(data)} not in {GeometryType}!")
