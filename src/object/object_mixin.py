from dataclasses import replace
from abc import ABC
from typing import Any, ClassVar, Optional, Tuple

import numpy as np
from src.array.base import BaseArray
from src.object.metadata import Metadata


class ObjectMixin(BaseArray, ABC):
    metadata: Metadata

    fixed_ndim: ClassVar[int]
    fixed_dtype: ClassVar[np.dtype]

    def __init__(self, metadata: Optional[Metadata] = None, **kwargs) -> None:
        if self.ndim != self.fixed_ndim:
            raise ValueError(
                f"{self.__class__.__name__} object has `ndim`={self.ndim}, but expected {self.fixed_ndim}!"
            )
        if self.dtype != self.fixed_dtype:
            raise ValueError(
                f"{self.__class__.__name__} object has `dtype`={self.dtype}, but expected {self.fixed_dtype}!"
            )

        if metadata is None:
            self.metadata = Metadata(**kwargs)
        else:
            print(**kwargs)
            self.metadata = replace(metadata, **kwargs)
