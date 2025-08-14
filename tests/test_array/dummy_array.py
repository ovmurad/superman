from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar

import numpy as np
from src.array.base import BaseArray

T = TypeVar("T")


@dataclass
class DummyArray(Generic[T]):
    array: T
    properties: Tuple[str, ...] = ("shape", "ndim", "dtype")

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return len(self.array.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.array.stype

    def check(self, array: BaseArray) -> bool:
        return (self.array is array.raw_array) and all(
            getattr(self, prop) == getattr(array, prop) for prop in self.properties
        )
