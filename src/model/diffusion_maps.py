import warnings
from typing import Any, Optional

import numpy as np

from src.array.linalg import EigenSolver
from src.geometry import EigenSystem, Embedding
from src.geometry.embedding import laplacian_embedding
from src.geometry.matrix import AffinityMatrix, DistanceMatrix, LaplacianMatrix
from src.geometry.points import Points
from src.model.embedding import GeometryType
from src.object import LaplacianType
from src.object.metadata import Metadata


def compute_diffusion_maps(eig: EigenSystem, diffusion_time: float) -> EigenSystem:
    """Credit to Satrajit Ghosh (http://satra.cogitatum.org/) for final steps"""
    md: Metadata = eig.metadata
    # Check that diffusion maps is using the correct laplacian, warn otherwise
    if eig.metadata.lap_type not in ["geometric"]:
        warnings.warn(
            "for correct diffusion maps embedding use laplacian type 'geometric' or 'renormalized'."
        )
    # Step 5 of diffusion maps:
    vectors = eig.eigenvectors.as_nparray()
    lambdas = eig.eigenvalues.as_nparray()
    psi = vectors / vectors[:, [0]]
    if diffusion_time == 0:
        lambdas = np.abs(lambdas)
        lambdas = lambdas / (1 - lambdas)
    else:
        lambdas = np.abs(lambdas)
        lambdas = lambdas ** float(diffusion_time)
    vectors = psi * lambdas
    return EigenSystem((lambdas, vectors), metadata=md)


class DiffusionMaps(Embedding):
    def __init__(
        self,
        radius: float,
        n_components: int = 2,
        eps: Optional[float] = None,
        eigen_solver: EigenSolver = "amg",
        drop_first: bool = True,
        lap_type: LaplacianType = "geometric",
        diffusion_time: float = 0.0,
        **solver_kwds: Any,
    ) -> None:
        self.n_components = n_components
        self.radius = radius
        self.eps = radius / 3 if eps is None else eps
        self.eigen_solver = eigen_solver
        self.drop_first = drop_first
        self.solver_kwds = solver_kwds
        self.lap_type = lap_type
        self.diffusion_time = diffusion_time

    def fit(self, data: GeometryType) -> EigenSystem:
        if isinstance(data, Points):
            data = data.pairwise_distance()
        if isinstance(data, DistanceMatrix):
            data = data.affinity(eps=self.eps, in_place=True)
        if isinstance(data, AffinityMatrix) or isinstance(data, LaplacianMatrix):
            lap_emb = laplacian_embedding(
                data,
                self.n_components + int(self.drop_first),
                self.lap_type,
                self.eigen_solver,
                False,
                in_place=True,
                **self.solver_kwds,
            )
            return (
                compute_diffusion_maps(lap_emb, self.diffusion_time).drop_first()
                if self.drop_first
                else compute_diffusion_maps(lap_emb, self.diffusion_time)
            )
        raise ValueError(f"`data` type {type(data)} not in {GeometryType}!")
