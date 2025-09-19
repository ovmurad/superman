
from typing import Any, Optional, Union

from src.array.linalg import EigenSolver
from src.geometry.embedding import laplacian_embedding
from src.geometry.matrix import AffinityMatrix
from src.geometry.matrix import DistanceMatrix
from src.geometry.matrix import LaplacianMatrix
from src.geometry import Points
from src.geometry import EigenSystem
from src.model.embedding import Embedding, GeometryType
from src.object import LaplacianType

class LaplacianEigenmaps(Embedding):
    def __init__(self, radius: float, n_components: int = 2, eps: Optional[float] = None,
                 eigen_solver: EigenSolver = "amg",
                 drop_first: bool = True, lap_type: LaplacianType = "geometric", **solver_kwds: Any) -> None:
        self.n_components = n_components
        self.radius = radius
        self.eps = radius / 3 if eps is None else eps
        self.eigen_solver = eigen_solver
        self.drop_first = drop_first
        self.solver_kwds = solver_kwds
        self.lap_type = lap_type

    def fit(self, data: GeometryType) -> EigenSystem:
        if isinstance(data, Points):
            data = data.pairwise_distance()
            data = data.threshold(self.radius, in_place=True)
        if isinstance(data, DistanceMatrix):
            data = data.affinity(eps=self.eps, in_place=True)
        if isinstance(data, AffinityMatrix) or isinstance(data, LaplacianMatrix):
            return laplacian_embedding(data, self.n_components, self.lap_type, self.eigen_solver, self.drop_first, in_place=True, **self.solver_kwds)
        raise ValueError(f"`data` type {type(data)} not in {GeometryType}!")
