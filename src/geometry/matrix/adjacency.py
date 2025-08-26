from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np

from src.array import BaseArray, CsrArray, DenseArray
from src.object import GeometryMatrixMixin


class AdjacencyMatrixMixin(GeometryMatrixMixin, ABC):
    fixed_dtype = np.bool_

    def __new__(cls, *args: Any, **kwargs: Any) -> AdjacencyMatrix:
        """
        The constructor returns an instance of either `DenseLaplacianMatrix` or `CsrLaplacianMatrix` depending on if constructed in `DenseArray` format or `CsrArray` format respectively.

        :param args: Positional arguments forwarded to the chosen
                     Laplacian matrix subclass.
        :type args: Any
        :param kwargs: Keyword arguments forwarded to the chosen
                       Laplacian matrix subclass.
                       If `shape` is present, a sparse CSR-backed
                       matrix will be constructed.
        :type kwargs: Any
        :return: A new `DenseLaplacianMatrix` or `CsrLaplacianMatrix`
                 instance.
        :rtype: LaplacianMatrix
        """
        if cls is AdjacencyMatrix:
            return cls.create(*args, **kwargs)
        return super().__new__(cls)  #type: ignore

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> AdjacencyMatrix:
        if "shape" in kwargs:
            return CsrAdjacencyMatrix(*args, **kwargs)
        return DenseAdjacencyMatrix(*args, **kwargs)


class AdjacencyMatrix(AdjacencyMatrixMixin, ABC):
    pass


class DenseAdjacencyMatrix(AdjacencyMatrix, DenseArray):
    pass


class CsrAdjacencyMatrix(AdjacencyMatrix, CsrArray):
    pass
