from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Self, Sequence, Tuple, Type, Union

import attr
import numpy as np

from src.array.base import BaseArray
from src.object.metadata import Metadata


class ObjectMixin(ABC):
    """
    Abstract base class providing metadata handling and type/dimension
    validation.
    """

    metadata: Metadata

    fixed_ndim: int
    fixed_dtype: Type[np.generic]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize an object with metadata and enforce
        fixed dimensionality and dtype.

        :param args: Positional arguments for object initialization.
                     If the first argument is another ObjectMixin instance,
                     its metadata is merged into the new instance.
        :type args: Any
        :param kwargs: Keyword arguments for object initialization, including
                       metadata fields. The `metadata` key can also be used
                       to update metadata after object creation.
        :type kwargs: Any
        :raises ValueError: If the object's ndim or dtype does not match
                            the fixed expected values.
        """

        metadata_args = tuple(
            self._give_arg_and_consume(f.name, kwargs) if f.name in kwargs else None for f in attr.fields(Metadata)
        )

        self.metadata = Metadata(*metadata_args)

        if "metadata" in kwargs:
            self.metadata = self.metadata.update_with(kwargs["metadata"])
            del kwargs["metadata"]

        if len(args) > 0 and isinstance(args[0], ObjectMixin):
            self.metadata = self.metadata.update_with(args[0].metadata)

        super().__init__(*args, **kwargs)  # type: ignore

    @staticmethod
    def _give_arg_and_consume(name: str, kwargs: Any) -> Any:
        temp: Any = kwargs[name]
        del kwargs[name]
        return temp

    @classmethod
    @abstractmethod
    def concat_with_metadata(cls, arrs: Sequence[Self], axis: int = 0) -> Self:
        ...


BaseArrayLike = Union[BaseArray, Tuple[Union[BaseArray, None], ...]]

def _nbatches(arr_len: int, batch_size: int) -> int:
    return int(np.ceil(arr_len / batch_size))

def chunk(arr: BaseArrayLike, bsize: Optional[int] = None) -> Iterator[BaseArrayLike]:
    if isinstance(arr, tuple):
        # Find the first non-None array to get the dimension
        non_none_arrays = [a for a in arr if a is not None]
        if not non_none_arrays:
            yield arr  # All None tuple
            return

        n = non_none_arrays[0].shape[0]
        if not all(a.shape[0] == n for a in non_none_arrays):
            raise ValueError("All non-None arrays in the tuple must have the same first dimension")
    else:
        n = arr.shape[0]
    
    if bsize is None or n <= bsize:
        yield arr
    else:
        for b in range(_nbatches(n, bsize)):
            sl = slice(b * bsize, (b + 1) * bsize)
            if isinstance(arr, tuple):
                yield tuple(a[sl] if a is not None else None for a in arr)
            else:
                yield arr[sl]
