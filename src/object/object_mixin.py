from __future__ import annotations
from dataclasses import replace, fields
from abc import ABC
from typing import Any, ClassVar, Optional, Tuple

import attr
import numpy as np
from src.array.base import BaseArray
from src.object.metadata import Metadata


class ObjectMixin(ABC):
    metadata: Metadata

    fixed_ndim: ClassVar[int]
    fixed_dtype: ClassVar[np.dtype]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.ndim != self.fixed_ndim:
            raise ValueError(
                f"{self.__class__.__name__} object has `ndim`={self.ndim}, but expected {self.fixed_ndim}!"
            )
        if self.dtype != self.fixed_dtype:
            raise ValueError(
                f"{self.__class__.__name__} object has `dtype`={self.dtype}, but expected {self.fixed_dtype}!"
            )

        metadata_args = tuple(kwargs[f.name] if f.name in kwargs else None for f in attr.fields(Metadata))

        self.metadata = Metadata(*metadata_args)

        if "metadata" in kwargs:
            self.metadata = self.metadata.update_with(kwargs["metadata"])
            
        if isinstance(args[0], ObjectMixin):
            self.metadata = self.metadata.update_with(args[0].metadata)
