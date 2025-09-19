from __future__ import annotations

from abc import ABC
from typing import Any, Iterable, Self, Sequence, Tuple, Type
from src.array.dense.dense import DenseArray
from src.object.object_mixin import ObjectMixin

class EigenMixin(ObjectMixin, tuple[DenseArray]):
    fixed_value_ndim: int = 1
    fixed_vector_ndim: int = 2
    fixed_tuple_length: int = 2
    fixed_value_type: Type = DenseArray
    fixed_vector_type: Type = DenseArray

    def __new__(cls, iterable: Iterable[Any], **kwargs: Any) -> EigenMixin:
        obj: EigenMixin = super().__new__(cls, iterable)

        if not isinstance(obj[0], obj.fixed_value_type):
            raise ValueError(f"{obj.__class__.__name__} object has `value_type`={obj[0].__class__.__name__}, but expected {obj.fixed_value_type}!")
        if not isinstance(obj[1], obj.fixed_value_type):
            raise ValueError(f"{obj.__class__.__name__} object has `value_type`={obj[1].__class__.__name__}, but expected {obj.fixed_vector_type}!")
        if len(obj) != obj.fixed_tuple_length:
            raise ValueError(f"{obj.__class__.__name__} object has `length`={len(obj)}, but expected {obj.fixed_tuple_length}!")
        if obj[0].ndim != obj.fixed_value_ndim:
            raise ValueError(f"{obj.__class__.__name__} object has `value_ndim`={obj[0].ndim}, but expected {obj.fixed_value_ndim}!")
        if obj[1].ndim != obj.fixed_vector_ndim:
            raise ValueError(f"{obj.__class__.__name__} object has `vector_ndim`={obj[1].ndim}, but expected {obj.fixed_vector_ndim}!")
        if obj[0].shape[0] != obj[1].shape[0]:
            raise ValueError(f"`value` has number of rows={obj[0].shape[0]}, but expected `vector` number of rows={obj[1].shape[0]}!")

        return obj

    def __init__(self, iterable: Iterable[Any], **metadata):
        super().__init__(**metadata)

    @property
    def eigenvalues(self) -> DenseArray:
        return self[0]

    @property
    def eigenvectors(self) -> DenseArray:
        return self[1]

    @classmethod
    def concat_with_metadata(cls, arrs: Sequence[Self], axis: int = 0) -> Self:
        return arrs[0].__class__((DenseArray.concat(arrs, axis=axis) for arrs in zip(*arrs)), metadata=arrs[0].metadata)
