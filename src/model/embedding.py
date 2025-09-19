from abc import ABC
from typing import Union

from src.geometry.matrix import AffinityMatrix
from src.geometry.matrix import DistanceMatrix
from src.geometry.matrix import LaplacianMatrix
from src.geometry import Points

GeometryType = Union[Points, AffinityMatrix, DistanceMatrix, LaplacianMatrix]

class Embedding(ABC):
    pass
