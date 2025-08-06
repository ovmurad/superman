from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Tuple

from ..array.base import BaseArray
from ..array.typing import ScalarType
from .metadata import Metadata


class Object(ABC):
    data: BaseArray
    metadata: Metadata

    ndim: ClassVar[int]
    dtype: ClassVar[ScalarType]

    def __init__(self, data: BaseArray, **metadata: Any) -> None:

        self.data = data

        if self.data.ndim != self.ndim:
            raise ValueError(
                f"{self.__class__.__name__} object has `ndim`={self.data.ndim}, but expected {self.ndim}!"
            )
        if self.data.dtype != self.dtype:
            raise ValueError(
                f"{self.__class__.__name__} object has `dtype`={self.data.dtype}, but expected {self.dtype}!"
            )

        self.metadata = Metadata(**metadata)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def is_dense(self) -> bool:
        return self.data.is_dense

    @property
    def is_sparse(self) -> bool:
        return self.data.is_sparse

    @property
    def is_scalar(self) -> bool:
        return self.is_dense and self.data.ndim == 0
