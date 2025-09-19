from typing import Any
from src.array import DenseArray
from src.object.eigen_mixin import EigenMixin


class EigenSystem(EigenMixin, tuple[DenseArray]):
    pass
