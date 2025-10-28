from abc import ABC
from typing import Any, Self, Sequence, Type

import numpy as np

from src.array.base import BaseArray
from src.object.object_mixin import ObjectMixin


class GeometryMatrixMixin(ObjectMixin, BaseArray, ABC):
    """
    Mixin class providing common functionality for geometry-related matrices.

    Adds default dimensionality of 2 of and default dtype of np.float64, along with convenience properties
    for querying matrix shape and point counts. Typically used as a base
    for distance, affinity, and Laplacian matrices.
    """

    fixed_ndim = 2
    fixed_dtype: Type[np.generic] = np.float64

    def __init__(self, *args: Any, **metadata: Any) -> None:
        """
        Initialize a GeometryMatrixMixin object, forwarding arguments
        to ObjectMixin.

        :param args: Positional arguments forwarded to ObjectMixin.
        :param metadata: Keyword arguments representing metadata fields.
        """
        super().__init__(*args, **metadata)

        if self.ndim != self.fixed_ndim:
            raise ValueError(
                f"{self.__class__.__name__} object has `ndim`={self.ndim}, but expected {self.fixed_ndim}!"
            )
        if self.dtype != self.fixed_dtype:
            raise ValueError(
                f"{self.__class__.__name__} object has `dtype`={self.dtype}, but expected {self.fixed_dtype}!"
            )

    @property
    def is_square(self) -> bool:
        """
        Check if the matrix is square (number of rows equals number of columns).

        :return: True if the matrix is square, False otherwise.
        :rtype: bool
        """
        return self.shape[0] == self.shape[1]

    @property
    def from_npts(self) -> int:
        """
        Number of points corresponding to the rows of the matrix.

        :return: Number of rows in the matrix.
        :rtype: int
        """
        return self.shape[0]

    @property
    def to_npts(self) -> int:
        """
        Number of points corresponding to the columns of the matrix.

        :return: Number of columns in the matrix.
        :rtype: int
        """
        return self.shape[1]

    @property
    def npts(self) -> int:
        """
        Number of points in the matrix, assuming it is square.

        :return: Number of points (rows or columns) if the matrix is square.
        :rtype: int
        :raises ValueError: If the matrix is not square.
        """
        if self.is_square:
            return self.from_npts
        raise ValueError("Matrix is not square, so `npts` is not well defined!")

    @classmethod
    def concat_with_metadata(cls, arrs: Sequence[Self], axis: int = 0) -> Self:
        return cls(super().concat(arrs, axis=axis), metadata=arrs[0].metadata)
