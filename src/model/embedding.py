from abc import ABC
from typing import Union

from src.geometry import Points
from src.geometry.matrix import AffinityMatrix, DistanceMatrix, LaplacianMatrix

GeometryType = Union[Points, AffinityMatrix, DistanceMatrix, LaplacianMatrix]


class Embedding(ABC):
    pass
