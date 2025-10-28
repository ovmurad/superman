from __future__ import annotations

from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    Iterator,
    NoReturn,
    Optional,
    ParamSpec,
    Sequence,
)

import numpy as np

from src.storage import Data, Storage

from ..base import BaseArray, unwrap, unwrap_args, unwrap_kwargs

_OpArgs = ParamSpec("_OpArgs")


def _dense_op(
    stg_func: Callable[Concatenate[Storage[Data], _OpArgs], Storage[Data]],
    *,
    in_place: bool = False,
) -> Callable[Concatenate[DenseArray[Data], _OpArgs], DenseArray[Data]]:
    def _wrapper(
        self: DenseArray[Data], *args: _OpArgs.args, **kwargs: _OpArgs.kwargs
    ) -> DenseArray[Data]:
        if in_place:
            self._values = stg_func(
                self._values, *unwrap_args(args), **unwrap_kwargs(kwargs)
            )
            return self

        return DenseArray(
            stg_func(self._values, *unwrap_args(args), **unwrap_kwargs(kwargs))
        )

    return _wrapper


def _lazy_dense_op(
    stg_func: Callable[Concatenate[Storage[Data], _OpArgs], Iterator[Storage[Data]]],
) -> Callable[Concatenate[DenseArray[Data], _OpArgs], Iterator[DenseArray[Data]]]:
    def _wrapper(
        self: DenseArray[Data], *args: _OpArgs.args, **kwargs: _OpArgs.kwargs
    ) -> Iterator[DenseArray[Data]]:
        for stg in stg_func(self._values, *unwrap_args(args), **unwrap_kwargs(kwargs)):
            yield DenseArray(stg)

    return _wrapper


class DenseArray(Generic[Data], BaseArray[Data]):
    # ======================================================================
    # Instance Vars
    # ======================================================================
    _values: Storage[Data]
    __slots__ = ("_values",)

    # ======================================================================
    # Class Vars
    # ======================================================================
    _format = "dense"

    # ======================================================================
    # Initialization
    # ======================================================================
    def __init__(
        self,
        arr_like: Any,
        /,
        *,
        dtype: Optional[type] = None,
        copy: bool = False,
        **kwargs: Any,
    ) -> None:
        self._values = Storage.as_values(unwrap(arr_like), dtype=dtype, copy=copy)

    # ======================================================================
    # Constructors
    # ======================================================================
    @staticmethod
    def as_array(
        arr_like: Any, /, *, dtype: Optional[type] = None, copy: bool = False
    ) -> DenseArray[Data]:
        return DenseArray(arr_like, dtype=dtype, copy=copy)

    @staticmethod
    def zeros(
        *, shape: int | Sequence[int], dtype: Optional[type] = None
    ) -> DenseArray[Data]:
        return DenseArray(Storage[Data].zeros(shape=shape, dtype=dtype))

    @staticmethod
    def ones(
        *, shape: int | Sequence[int], dtype: Optional[type] = None
    ) -> DenseArray[Data]:
        return DenseArray(Storage[Data].ones(shape=shape, dtype=dtype))

    @staticmethod
    def full(
        fill_value: Any, /, *, shape: int | Sequence[int], dtype: Optional[type] = None
    ) -> DenseArray[Data]:
        return DenseArray(Storage[Data].full(fill_value, shape=shape, dtype=dtype))

    # ======================================================================
    # Introspection
    # ======================================================================
    @property
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    # ======================================================================
    # Dunder
    # ======================================================================

    # ----------------------------------------------------------------------
    # Get & Set items
    # ----------------------------------------------------------------------
    def __getitem__(self, key: Any) -> DenseArray[Data]:
        key = unwrap_args(key) if isinstance(key, tuple) else unwrap(key)
        return DenseArray(self._values[key])

    def __setitem__(self, key: Any, values_like: Any) -> None:
        key = unwrap_args(key) if isinstance(key, tuple) else unwrap(key)
        self._values[key] = unwrap(values_like)

    # ----------------------------------------------------------------------
    # Arithmetic
    # ----------------------------------------------------------------------
    __add__ = _dense_op(Storage.__add__, in_place=False)
    __iadd__ = _dense_op(Storage.__iadd__, in_place=True)
    __sub__ = _dense_op(Storage.__sub__, in_place=False)
    __isub__ = _dense_op(Storage.__isub__, in_place=True)
    __mul__ = _dense_op(Storage.__mul__, in_place=False)
    __imul__ = _dense_op(Storage.__imul__, in_place=True)
    __truediv__ = _dense_op(Storage.__truediv__, in_place=False)
    __itruediv__ = _dense_op(Storage.__itruediv__, in_place=True)
    __mod__ = _dense_op(Storage.__mod__, in_place=False)
    __imod__ = _dense_op(Storage.__imod__, in_place=True)
    __floordiv__ = _dense_op(Storage.__floordiv__, in_place=False)
    __ifloordiv__ = _dense_op(Storage.__ifloordiv__, in_place=True)
    __pow__ = _dense_op(Storage.__pow__, in_place=False)
    __ipow__ = _dense_op(Storage.__ipow__, in_place=True)

    # ----------------------------------------------------------------------
    # Logic & Comparison
    # ----------------------------------------------------------------------
    __lt__ = _dense_op(Storage.__lt__, in_place=False)
    __le__ = _dense_op(Storage.__le__, in_place=False)
    __gt__ = _dense_op(Storage.__gt__, in_place=False)
    __ge__ = _dense_op(Storage.__ge__, in_place=False)
    __eq__ = _dense_op(Storage.__eq__, in_place=False)  # type: ignore
    __ne__ = _dense_op(Storage.__ne__, in_place=False)  # type: ignore
    __invert__ = _dense_op(Storage.__invert__, in_place=False)  # type: ignore

    # ----------------------------------------------------------------------
    # Utils
    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"DenseArray: \n"
            f"  - shape: {self.shape} \n"
            f"  - dtype: {self.dtype} \n"
            f"  - data: {self._values.__repr__()}"
        )

    __copy__ = _dense_op(Storage.copy, in_place=False)

    # ======================================================================
    # Shape Manipulation
    # ======================================================================
    reshape = _dense_op(Storage.reshape, in_place=False)
    expand_dims = _dense_op(Storage.expand_dims, in_place=False)
    squeeze = _dense_op(Storage.squeeze, in_place=False)
    broadcast_to = _dense_op(Storage.broadcast_to, in_place=False)
    diagonal = _dense_op(Storage.diagonal, in_place=False)
    fill_diagonal = _dense_op(Storage.fill_diagonal, in_place=False)

    @staticmethod
    def concat(
        arrs: Sequence[DenseArray[Data]], /, *, axis: int = 0
    ) -> DenseArray[Data]:
        return DenseArray(Storage.concat([arr.values for arr in arrs], axis=axis))

    @staticmethod
    def stack(
        arrs: Sequence[DenseArray[Data]], /, *, axis: int = 0
    ) -> DenseArray[Data]:
        return DenseArray(Storage.stack([arr.values for arr in arrs], axis=axis))

    # ======================================================================
    # Type Casting
    # ======================================================================
    def as_type(self, *, dtype: type, copy: bool = False) -> DenseArray[Data]:
        return DenseArray(self, dtype=dtype, copy=copy)

    def as_dense(self, copy: bool = False) -> DenseArray[Data]:
        return self.copy() if copy else self

    def as_csr(self) -> NoReturn:
        raise NotImplementedError(
            "Cannot not transform a DenseArray to CsrArray directly!"
        )

    def as_nparray(self) -> np.ndarray:
        return self.values.data

    # ======================================================================
    # Elementwise math functions (ufuncs)
    # ======================================================================
    exp = _dense_op(Storage.exp, in_place=False)
    iexp = _dense_op(Storage.iexp, in_place=True)
    log = _dense_op(Storage.log, in_place=False)
    ilog = _dense_op(Storage.ilog, in_place=True)
    sqrt = _dense_op(Storage.sqrt, in_place=False)
    isqrt = _dense_op(Storage.isqrt, in_place=True)
    sin = _dense_op(Storage.sin, in_place=False)
    isin = _dense_op(Storage.isin, in_place=True)
    cos = _dense_op(Storage.cos, in_place=False)
    icos = _dense_op(Storage.icos, in_place=True)

    # ======================================================================
    # Reductions
    # ======================================================================
    sum = _dense_op(Storage.sum, in_place=False)
    mean = _dense_op(Storage.mean, in_place=False)
    min = _dense_op(Storage.min, in_place=False)
    max = _dense_op(Storage.max, in_place=False)
    count_nonzero = _dense_op(Storage.count_nonzero, in_place=False)

    # ======================================================================
    # Compression
    # ======================================================================
    compress_axis = _dense_op(Storage.compress_axis, in_place=True)
    compress = _dense_op(Storage.compress, in_place=True)

    # ======================================================================
    # Miscellaneous
    # ======================================================================
    copy = _dense_op(Storage.copy, in_place=False)

    # ======================================================================
    # Linear Algebra
    # ======================================================================
    distance = _dense_op(Storage.distance, in_place=False)
    distance_lazy = _lazy_dense_op(Storage.distance_lazy)
