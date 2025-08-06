from __future__ import annotations

from abc import ABC, abstractmethod
from math import prod
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Optional,
    Tuple,
)

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from .typing import Scalar, ScalarLikeType, ScalarType


class BaseArray(ABC, NDArrayOperatorsMixin):
    storage: Any

    _is_sparse: ClassVar[bool]
    _wrap_types: ClassVar[Tuple[type, ...]] = tuple[type]()

    @classmethod
    def _wrap(cls, out: Any) -> Any:

        if isinstance(out, cls._wrap_types):
            return cls(out)
        elif isinstance(out, (tuple, list)):
            return out.__class__((cls._wrap(o) for o in out))
        elif isinstance(out, dict):
            return {k: cls._wrap(o) for k, o in out.items()}
        else:
            return out

    @classmethod
    def _unwrap(cls, inp: Any) -> Any:

        if isinstance(inp, cls):
            return inp.storage
        elif isinstance(inp, (tuple, list)):
            return inp.__class__((cls._unwrap(r) for r in inp))
        elif isinstance(inp, dict):
            return {k: cls._unwrap(r) for k, r in inp.items()}
        else:
            return inp

    @classmethod
    def _wrapped_call(cls, func: Callable, *args: Any, **kwargs: Any) -> Any:
        args = cls._unwrap(args)
        kwargs = cls._unwrap(kwargs)
        result = func(*args, **kwargs)
        return cls._wrap(result)

    def __init__(self, storage: Any) -> None:
        self.storage = storage

    def __getattr__(self, attr: str) -> Any:
        """Delegate attributes/methods to storage."""
        attr_value = getattr(self.storage, attr)

        # If the attribute is a function call and wrap the function
        if callable(attr_value):

            def _wrapped_attr(*args: Any, **kwargs: Any) -> Any:
                return self._wrapped_call(attr_value, *args, **kwargs)

            return _wrapped_attr
        return attr_value

    # --------------- Properties ---------------
    # Reproduced for type hinting and IDE purpose(__getattr__ would handle everything)
    # or for the purpose of implementing a common interface
    @property
    def is_dense(self) -> bool:
        return not self._is_sparse

    @property
    def is_sparse(self) -> bool:
        return self._is_sparse

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.storage.shape

    @property
    def ndim(self) -> int:
        return self.storage.ndim

    @property
    def dtype(self) -> ScalarType:
        return self.storage.dtype.type

    @property
    def nnz(self) -> int:
        return self.storage.size

    @property
    def size(self) -> int:
        return prod(self.shape)

    # --------------- Numpy Interface ---------------

    @abstractmethod
    def __array__(
        self, dtype: Optional[ScalarLikeType] = None
    ) -> np.ndarray[Tuple[int, ...], np.dtype[Scalar]]: ...

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *args: Any, **kwargs: Any
    ) -> Any:
        func = getattr(ufunc, method)
        return self._wrapped_call(func, *args, **kwargs)

    def __array_function__(
        self,
        func: Callable,
        types: Tuple[type, ...],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        return self._wrapped_call(func, *args, **kwargs)
