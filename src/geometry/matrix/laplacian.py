from __future__ import annotations

from abc import ABC
from typing import Any, Final, Set

from src.array import BaseArray, CsrArray, DenseArray
from src.object import GeometryMatrixMixin, LaplacianType

SYM_LAPLACIAN_TYPES: Final[Set[LaplacianType]] = {"symmetric"}
NON_SYM_LAPLACIAN_TYPES: Final[Set[LaplacianType]] = {"geometric", "random_walk"}


class LaplacianMatrixMixin(GeometryMatrixMixin, ABC):
    """
    Mixin class that adds factory functionality to the LaplacianMatrix
    hierarchy.

    When instantiating `LaplacianMatrix` directly, this mixin intercepts
    construction and returns either a `DenseLaplacianMatrix` or a
    `CsrLaplacianMatrix` depending on the provided arguments.

    This allows users to work with `LaplacianMatrix` as an abstract entry
    point without explicitly choosing the dense or sparse representation.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> LaplacianMatrix:
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
        if cls is LaplacianMatrix:
            return cls.create(*args, **kwargs)
        return super().__new__(cls)  # type: ignore

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> LaplacianMatrix:
        if "shape" in kwargs:
            return CsrLaplacianMatrix(*args, **kwargs)
        return DenseLaplacianMatrix(*args, **kwargs)


class LaplacianMatrix(LaplacianMatrixMixin, BaseArray):
    """
    Abstract base class representing a Laplacian matrix.
    """

    pass


class DenseLaplacianMatrix(LaplacianMatrix, DenseArray):
    """
    Implementation of a dense (NumPy-backed) Laplacian matrix.

    This class provides a dense representation, offering fast
    element-wise operations at the cost of memory usage.

    Typically not instantiated directly; instead, construct a
    `LaplacianMatrix` and let the `LaplacianMatrixMixin` factory
    return a `DenseLaplacianMatrix` if constructed in a `DenseArray` format.
    """

    pass


class CsrLaplacianMatrix(LaplacianMatrix, CsrArray):
    """
    Concrete implementation of a sparse (CSR-backed) Laplacian matrix.

    This class provides a memory-efficient sparse representation,
    especially useful for large, sparse graphs.

    Typically not instantiated directly; instead, construct a
    `LaplacianMatrix` and let the `LaplacianMatrixMixin` factory
    return a `CsrLaplacianMatrix` if constructed in a `CsrArray` format.
    """

    pass


def eps_adjustment(eps: float) -> float:
    return 4.0 / (eps**2)
