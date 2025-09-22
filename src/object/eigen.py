from __future__ import annotations

from typing import Any, Iterable, Self, Sequence, Tuple, Type

import numpy as np

from src.array.dense.dense import DenseArray
from src.object.object_mixin import ObjectMixin


class Eigen(ObjectMixin, Tuple[DenseArray, DenseArray]):
    """
    This class represents a pair of eigenvalues and eigenvectors.
    It is a Tuple of two `DenseArray` with additional checks and methods.
    Ensures correct eigenvalue and eigenvector lengths and dimensions.
    """

    fixed_value_ndim: int = 1
    fixed_vector_ndim: int = 2
    fixed_tuple_length: int = 2
    fixed_value_type: Type = DenseArray
    fixed_vector_type: Type = DenseArray

    def __new__(cls, iterable: Iterable[Any], **kwargs: Any) -> Eigen:
        obj: Eigen = super().__new__(cls, (DenseArray(item) for item in iterable))

        if len(obj) != cls.fixed_tuple_length:
            raise ValueError(
                f"{obj.__class__.__name__} object has `length`={len(obj)}, but expected {obj.fixed_tuple_length}!"
            )
        if not isinstance(obj[0], cls.fixed_value_type):  # type: ignore
            raise ValueError(
                f"{obj.__class__.__name__} object has `value_type`={obj[0].__class__.__name__}, but expected {obj.fixed_value_type}!"
            )
        if not isinstance(obj[1], cls.fixed_value_type):  # type: ignore
            raise ValueError(
                f"{obj.__class__.__name__} object has `value_type`={obj[1].__class__.__name__}, but expected {obj.fixed_vector_type}!"
            )
        if obj[0].ndim != cls.fixed_value_ndim:  # type: ignore
            raise ValueError(
                f"{obj.__class__.__name__} object has `value_ndim`={obj[0].ndim}, but expected {obj.fixed_value_ndim}!"
            )
        if obj[1].ndim != cls.fixed_vector_ndim:  # type: ignore
            raise ValueError(
                f"{obj.__class__.__name__} object has `vector_ndim`={obj[1].ndim}, but expected {obj.fixed_vector_ndim}!"
            )

        return obj

    def __init__(self, iterable: Iterable[Any], **metadata: Any) -> None:
        super().__init__(**metadata)

    @property
    def eigenvalues(self) -> DenseArray:
        return self[0]

    @property
    def eigenvectors(self) -> DenseArray:
        return self[1]

    def sort_by_eigval(self, largest: bool = False) -> Self:
        """
        Sorts the eigenvalues and eigenvectors by eigenvalue.

        :param largest: If True, sorts with largest first. If False, sorts with smallest first.

        :return: A sorted instance of this class.
        """
        idx = np.argsort(self[0])[::-1] if largest else np.argsort(self[0])
        return self.__class__(self[0][idx], self[1][:, idx])

    @classmethod
    def concat_with_metadata(cls, arrs: Sequence[Self], axis: int = 0) -> Self:
        """
        Concatenates eigenvalues and eigenvectors along an axis and returns an instance of this class with metadata of the first object in `arrs`.

        :param arrs: Sequence of objects of this class to concatenate.
        :param axis: Axis along which to concatenate arrays. (default: 0).

        :return: A new instance of this class with concatenated data and
             metadata taken from the first element in `arrs`.
        """
        return arrs[0].__class__(
            (DenseArray.concat(arrs, axis=axis) for arrs in zip(*arrs)),
            metadata=arrs[0].metadata,
        )
