from __future__ import annotations

from math import prod
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    Iterator,
    Optional,
    ParamSpec,
    Sequence,
)

from ..backend import BACKEND, Data

_OpArgs = ParamSpec("_OpArgs")


def _unwrap(stg_like: Any) -> Any:
    if isinstance(stg_like, Storage):
        return stg_like.data
    return stg_like


def _unwrap_args(args: tuple) -> tuple:
    return tuple(_unwrap(arg) for arg in args)


def _unwrap_kwargs(kwargs: dict) -> dict[str, Any]:
    return {kw: _unwrap(arg) for kw, arg in kwargs.items()}


def _storage_op(
    backend_func: Callable[Concatenate[Data, _OpArgs], Data],
    *,
    in_place: bool = False,
) -> Callable[Concatenate[Storage[Data], _OpArgs], Storage[Data]]:
    def _wrapper(
        self: Storage[Data], *args: _OpArgs.args, **kwargs: _OpArgs.kwargs
    ) -> Storage[Data]:
        if in_place:
            self._data = backend_func(
                self._data,
                *_unwrap_args(args),
                out=self._data,
                **_unwrap_kwargs(kwargs),
            )
            return self
        return Storage(
            backend_func(self._data, *_unwrap_args(args), **_unwrap_kwargs(kwargs))
        )

    return _wrapper


def _lazy_storage_op(
    backend_func: Callable[Concatenate[Data, _OpArgs], Iterator[Data]],
) -> Callable[Concatenate[Storage[Data], _OpArgs], Iterator[Storage[Data]]]:
    def _wrapper(
        self: Storage[Data], *args: _OpArgs.args, **kwargs: _OpArgs.kwargs
    ) -> Iterator[Storage[Data]]:
        for data in backend_func(
            self._data, *_unwrap_args(args), **_unwrap_kwargs(kwargs)
        ):
            yield Storage(data)

    return _wrapper


class Storage(Generic[Data]):
    # ======================================================================
    # Instance Vars
    # ======================================================================
    __slots__ = ("_data",)
    _data: Data

    # ======================================================================
    # Class Vars
    # ======================================================================

    # ======================================================================
    # Initialization
    # ======================================================================
    def __init__(
        self, stg_like: Any, /, *, dtype: Optional[type] = None, copy: bool = False
    ) -> None:
        self._data = BACKEND.as_data(_unwrap(stg_like), dtype=dtype, copy=copy)

    # ======================================================================
    # Constructors
    # ======================================================================
    @staticmethod
    def as_storage(
        stg_like: Any, /, *, dtype: Optional[type] = None, copy: bool = False
    ) -> Storage[Data]:
        return Storage[Data](stg_like, dtype=dtype, copy=copy)

    @staticmethod
    def as_values(
        values_like: Any,
        /,
        *,
        dtype: Optional[type] = None,
        copy: bool = False,
        assert_flat: bool = False,
    ) -> Storage[Data]:
        values = Storage[Data](values_like, dtype=dtype, copy=copy)
        if assert_flat and values.ndim != 1:
            raise ValueError(
                f"`values` must be 1 dimensional, got {values.ndim} instead!"
            )
        return values

    @staticmethod
    def as_index(
        index_like: Any, /, *, copy: bool = False, assert_flat: bool = True
    ) -> Storage[Data]:
        axis_index = Storage[Data](index_like, dtype=BACKEND.index_dtype, copy=copy)
        if assert_flat and axis_index.ndim != 1:
            raise ValueError(
                f"`axis_index` must be 1 dimensional, got {axis_index.ndim} instead!"
            )
        return axis_index

    @staticmethod
    def as_mask(
        mask_like: Any, /, *, copy: bool = False, assert_flat: bool = False
    ) -> Storage[Data]:
        mask = Storage[Data](mask_like, dtype=BACKEND.bool_dtype, copy=copy)
        if assert_flat and mask.ndim != 1:
            raise ValueError(f"`mask` must be 1 dimensional, got {mask.ndim} instead!")
        return mask

    @staticmethod
    def zeros(
        *, shape: int | Sequence[int], dtype: Optional[type] = None
    ) -> Storage[Data]:
        return Storage[Data](BACKEND.zeros(shape=shape, dtype=dtype))

    @staticmethod
    def ones(
        *, shape: int | Sequence[int], dtype: Optional[type] = None
    ) -> Storage[Data]:
        return Storage[Data](BACKEND.ones(shape=shape, dtype=dtype))

    @staticmethod
    def full(
        fill_value: Any, /, *, shape: int | Sequence[int], dtype: Optional[type] = None
    ) -> Storage[Data]:
        return Storage[Data](BACKEND.full(fill_value, shape=shape, dtype=dtype))

    @staticmethod
    def arange(
        *,
        start: int,
        stop: int,
        step: Optional[int] = None,
        dtype: Optional[type] = None,
    ) -> Storage[Data]:
        return Storage[Data](
            BACKEND.arange(start=start, stop=stop, step=step, dtype=dtype)
        )

    # ======================================================================
    # Introspection
    # ======================================================================
    @property
    def data(self) -> Data:
        return self._data

    @property
    def dtype(self) -> type:
        return self._data.dtype.type

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return prod(self.shape)

    # ======================================================================
    # Dunder
    # ======================================================================

    # ----------------------------------------------------------------------
    # Get & Set items
    # ----------------------------------------------------------------------
    def __getitem__(self, key: Any) -> Storage[Data]:
        key = _unwrap_args(key) if isinstance(key, tuple) else _unwrap(key)
        return Storage(self._data[key])

    def __setitem__(self, key: Any, stg_like: Any) -> None:
        key = _unwrap_args(key) if isinstance(key, tuple) else _unwrap(key)
        self._data[key] = _unwrap(stg_like)

    # ----------------------------------------------------------------------
    # Arithmetic
    # ----------------------------------------------------------------------
    __add__ = _storage_op(BACKEND.add, in_place=False)
    __radd__ = _storage_op(BACKEND.add, in_place=False)
    __iadd__ = _storage_op(BACKEND.add, in_place=True)
    __sub__ = _storage_op(BACKEND.subtract, in_place=False)
    __rsub__ = _storage_op(BACKEND.subtract, in_place=False)
    __isub__ = _storage_op(BACKEND.subtract, in_place=True)
    __mul__ = _storage_op(BACKEND.multiply, in_place=False)
    __rmul__ = _storage_op(BACKEND.multiply, in_place=False)
    __imul__ = _storage_op(BACKEND.multiply, in_place=True)
    __truediv__ = _storage_op(BACKEND.divide, in_place=False)
    __rtruediv__ = _storage_op(BACKEND.divide, in_place=False)
    __itruediv__ = _storage_op(BACKEND.divide, in_place=True)
    __mod__ = _storage_op(BACKEND.mod, in_place=False)
    __rmod__ = _storage_op(BACKEND.mod, in_place=False)
    __imod__ = _storage_op(BACKEND.mod, in_place=True)
    __floordiv__ = _storage_op(BACKEND.floor_divide, in_place=False)
    __rfloordiv__ = _storage_op(BACKEND.floor_divide, in_place=False)
    __ifloordiv__ = _storage_op(BACKEND.floor_divide, in_place=True)
    __pow__ = _storage_op(BACKEND.power, in_place=False)
    __rpow__ = _storage_op(BACKEND.power, in_place=False)
    __ipow__ = _storage_op(BACKEND.power, in_place=True)
    __abs__ = _storage_op(BACKEND.absolute_value, in_place=False)

    # ----------------------------------------------------------------------
    # Logic & Comparison
    # ----------------------------------------------------------------------
    __lt__ = _storage_op(BACKEND.less, in_place=False)
    __le__ = _storage_op(BACKEND.less_equal, in_place=False)
    __gt__ = _storage_op(BACKEND.greater, in_place=False)
    __ge__ = _storage_op(BACKEND.greater_equal, in_place=False)
    __eq__ = _storage_op(BACKEND.equal, in_place=False)  # type: ignore
    __ne__ = _storage_op(BACKEND.not_equal, in_place=False)  # type: ignore
    __invert__ = _storage_op(BACKEND.invert, in_place=False)  # type: ignore

    # ----------------------------------------------------------------------
    # Utils
    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Storage({self._data.__repr__()})"

    __copy__ = _storage_op(BACKEND.copy, in_place=False)

    # ======================================================================
    # Shape Manipulation
    # ======================================================================

    reshape = _storage_op(BACKEND.reshape, in_place=False)
    expand_dims = _storage_op(BACKEND.expand_dims, in_place=False)
    squeeze = _storage_op(BACKEND.squeeze, in_place=False)
    broadcast_to = _storage_op(BACKEND.broadcast_to, in_place=False)
    diagonal = _storage_op(BACKEND.diagonal, in_place=False)
    fill_diagonal = _storage_op(BACKEND.fill_diagonal, in_place=False)

    @staticmethod
    def concat(stgs: Sequence[Storage[Data]], /, *, axis: int = 0) -> Storage[Data]:
        return BACKEND.concat([stg.data for stg in stgs], axis=axis)

    @staticmethod
    def stack(stgs: Sequence[Storage[Data]], /, *, axis: int = 0) -> Storage[Data]:
        return BACKEND.stack([stg.data for stg in stgs], axis=axis)

    # ======================================================================
    # Type Casting
    # ======================================================================

    as_type = _storage_op(BACKEND.as_type, in_place=False)

    # ======================================================================
    # Elementwise math functions (ufuncs)
    # ======================================================================

    exp = _storage_op(BACKEND.exp, in_place=False)
    iexp = _storage_op(BACKEND.exp, in_place=True)
    log = _storage_op(BACKEND.log, in_place=False)
    ilog = _storage_op(BACKEND.log, in_place=True)
    sqrt = _storage_op(BACKEND.sqrt, in_place=False)
    isqrt = _storage_op(BACKEND.sqrt, in_place=True)
    abs = _storage_op(BACKEND.abs, in_place=False)
    iabs = _storage_op(BACKEND.abs, in_place=True)
    sin = _storage_op(BACKEND.sin, in_place=False)
    isin = _storage_op(BACKEND.sin, in_place=True)
    cos = _storage_op(BACKEND.cos, in_place=False)
    icos = _storage_op(BACKEND.cos, in_place=True)

    # ======================================================================
    # Reductions
    # ======================================================================

    sum = _storage_op(BACKEND.sum, in_place=False)
    mean = _storage_op(BACKEND.mean, in_place=False)
    min = _storage_op(BACKEND.min, in_place=False)
    max = _storage_op(BACKEND.max, in_place=False)
    count_nonzero = _storage_op(BACKEND.count_nonzero, in_place=False)

    # ======================================================================
    # Segment Reductions
    # ======================================================================

    segment_sum = _storage_op(BACKEND.segment_sum, in_place=False)
    segment_mean = _storage_op(BACKEND.segment_mean, in_place=False)
    segment_min = _storage_op(BACKEND.segment_min, in_place=False)
    segment_max = _storage_op(BACKEND.segment_max, in_place=False)

    # ======================================================================
    # Cumulative
    # ======================================================================
    cumsum = _storage_op(BACKEND.cumsum, in_place=False)

    # ======================================================================
    # Indexing
    # ======================================================================
    diff = _storage_op(BACKEND.diff, in_place=False)
    repeat = _storage_op(BACKEND.repeat, in_place=False)
    flat_nonzero = _storage_op(BACKEND.flat_nonzero, in_place=False)
    bin_count = _storage_op(BACKEND.bin_count, in_place=False)

    # ======================================================================
    # Compression
    # ======================================================================
    compress_axis = _storage_op(BACKEND.compress_axis, in_place=True)

    def compress(
        self,
        /,
        *,
        keep: Any | Sequence[Any],
        axis: Optional[int | Sequence[int]] = 0,
    ) -> Storage[Data]:

        if isinstance(axis, int) or axis is None:
            return self.compress_axis(keep=keep, axis=axis)

        if len(axis) != len(keep):
            raise ValueError(
                f"`keep` and `axis` must have the same length, got {len(axis)} and {len(keep)} instead!"
            )

        stg = self
        for k, a in zip(keep, axis):
            stg = stg.compress_axis(keep=k, axis=a)
        return stg

    # ======================================================================
    # Miscellaneous
    # ======================================================================
    copy = _storage_op(BACKEND.copy, in_place=False)

    # ======================================================================
    # Linear Algebra
    # ======================================================================
    distance = _storage_op(BACKEND.distance, in_place=False)
    distance_lazy = _lazy_storage_op(BACKEND.distance_lazy)

    # ======================================================================
    # Operate at
    # ======================================================================

    add_at = _storage_op(BACKEND.add_at, in_place=False)

    # ======================================================================
    # Operate between
    # ======================================================================

    add_between = _storage_op(BACKEND.add_between, in_place=False)
