from __future__ import annotations

from typing import Any, Iterable, Self, Sequence, Tuple, Type

from src.array.dense.dense import DenseArray
from src.object.object_mixin import ObjectMixin


class Eigen(ObjectMixin, Tuple[DenseArray, DenseArray]):
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

    @classmethod
    def concat_with_metadata(cls, arrs: Sequence[Self], axis: int = 0) -> Self:
        return arrs[0].__class__(
            (DenseArray.concat(arrs, axis=axis) for arrs in zip(*arrs)),
            metadata=arrs[0].metadata,
        )
