from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, Optional, Tuple, Type, TypeVar

import numpy as np

from src.array.base import BaseArray
from src.array.dense.dense import DenseArray
from src.array.sparse.csr import CsrArray
from src.linalg.eigen_system import EigenSystem
from src.object import ObjectMixin
from src.object.geometry_matrix_mixin import GeometryMatrixMixin

GlobalType = TypeVar("GlobalType", bound=ObjectMixin)
LocalType = TypeVar("LocalType", bound=ObjectMixin)

class func(ABC, Generic[GlobalType, LocalType]):
    local_type: Type
    global_type : Type

    @classmethod
    @abstractmethod
    def global_func(cls, *args: Any, **kwargs: Any) -> GlobalType:
        ...

    @classmethod
    @abstractmethod
    def local_func(cls, *args: Any, **kwargs: Any) -> LocalType:
        ...
    
    @classmethod
    @abstractmethod
    def local_iter(cls, *args: Any, **kwargs: Any) -> Iterable[LocalType]:
        ...

    @classmethod
    def package(cls, *args: Any, output_cls: Type[LocalType], bsize: Optional[int] = None, **kwargs: Any) -> LocalType:
        return output_cls.concat_with_metadata(list(cls.local_iter(*args, bsize, **kwargs)))