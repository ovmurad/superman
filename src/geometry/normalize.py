from typing import Optional, TypeVar

from src.array.base import BaseArray
from src.object import ObjectMixin
from src.object.metadata import Metadata

T = TypeVar("T", bound=BaseArray)


def normalize(
    arr: T,
    axis: Optional[int] = 1,
    degree_exp: float = 1.0,
    sym_norm: bool = False,
    in_place: bool = False,
) -> T:
    md: Optional[Metadata] = None
    if isinstance(arr, ObjectMixin):
        md = arr.metadata

    degrees = arr.sum(axis=axis, keepdims=True) ** degree_exp

    if in_place:
        arr /= degrees  # type: ignore
    else:
        arr = arr / degrees  # type: ignore

    if sym_norm and axis is not None:
        arr /= degrees.reshape(shape=degrees.shape[::-1])  # type: ignore

    return arr.__class__(arr) if md is None else arr.__class__(arr, metadata=md)
