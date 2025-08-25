from typing import Optional, TypeVar

import numpy as np

from src.array.base import BaseArray
from src.object import ObjectMixin

T = TypeVar("T")


def normalize(
    arr: BaseArray[np.float64],
    axis: Optional[int] = 1,
    degree_exp: float = 1.0,
    sym_norm: bool = False,
    in_place: bool = False,
) -> BaseArray[np.float64]:

    object = False
    if isinstance(arr, ObjectMixin):
        object = True
        cls = arr.__class__
        metadata = arr.metadata

    degrees = arr.sum(axis=axis, keepdims=True) ** degree_exp

    if in_place:
        arr /= degrees
    else:
        arr = arr / degrees

    if sym_norm and axis is not None:
        arr /= degrees.reshape(shape=degrees.shape[::-1])

    return cls(arr, metadata=metadata) if object and not in_place else arr
