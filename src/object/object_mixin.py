from __future__ import annotations

from abc import ABC
from typing import ClassVar

import attr
import numpy as np

from src.array.base import BaseArray
from src.object.metadata import Metadata


class ObjectMixin(BaseArray, ABC):
    """
    Abstract base class providing metadata handling and type/dimension
    validation.
    """

    metadata: Metadata

    fixed_ndim: ClassVar[int]
    fixed_dtype: ClassVar[np.dtype]

    def __init__(self, *args, **kwargs) -> None:
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
        super().__init__(*args, **kwargs)

        if self.ndim != self.fixed_ndim:
            raise ValueError(
                f"{self.__class__.__name__} object has `ndim`={self.ndim}, but expected {self.fixed_ndim}!"
            )
        if self.dtype != self.fixed_dtype:
            raise ValueError(
                f"{self.__class__.__name__} object has `dtype`={self.dtype}, but expected {self.fixed_dtype}!"
            )

        metadata_args = tuple(
            kwargs[f.name] if f.name in kwargs else None for f in attr.fields(Metadata)
        )

        self.metadata = Metadata(*metadata_args)

        if "metadata" in kwargs:
            self.metadata = self.metadata.update_with(kwargs["metadata"])

        if isinstance(args[0], ObjectMixin):
            self.metadata = self.metadata.update_with(args[0].metadata)
