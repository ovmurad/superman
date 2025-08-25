from __future__ import annotations

from typing import Any, Iterator, Literal, Optional, Protocol, Sequence, TypeVar

BackendStr = Literal["numpy"]
Data = TypeVar("Data", bound=Any)


class Backend(Protocol[Data]):
    """
    Protocol describing the Array API standard subset.
    Backends like numpy, jax.numpy, cupy, torch (with array-api mode)
    can be typed against this.
    """

    # ======================================================================
    # Class Vars
    # ======================================================================
    name: BackendStr

    data_type: type[Data]

    index_dtype: type

    bool_dtype: type
    counting_dtype: type
    real_dtype: type

    # ======================================================================
    # Creation
    # ======================================================================
    @staticmethod
    def as_data(
        x: Any, /, *, dtype: Optional[type] = None, copy: bool = False
    ) -> Data: ...

    @staticmethod
    def zeros(*, shape: int | Sequence[int], dtype: Optional[type] = None) -> Data: ...

    @staticmethod
    def ones(*, shape: int | Sequence[int], dtype: Optional[type] = None) -> Data: ...

    @staticmethod
    def full(
        x: Any, /, *, shape: int | Sequence[int], dtype: Optional[type] = None
    ) -> Data: ...

    @staticmethod
    def arange(
        *,
        start: int,
        stop: int,
        step: Optional[int] = None,
        dtype: Optional[type] = None,
    ) -> Data: ...

    # ======================================================================
    # Shape manipulation
    # ======================================================================
    @staticmethod
    def reshape(x: Data, /, *, shape: int | Sequence[int]) -> Data: ...

    @staticmethod
    def expand_dims(x: Data, /, *, axis: int | Sequence[int]) -> Data: ...

    @staticmethod
    def squeeze(x: Data, /, *, axis: int | Sequence[int]) -> Data: ...

    @staticmethod
    def transpose(x: Data, /) -> Data: ...

    @staticmethod
    def broadcast_to(x: Data, /, *, shape: int | Sequence[int]) -> Data: ...

    @staticmethod
    def concat(xs: Sequence[Data], /, *, axis: int = 0) -> Data: ...

    @staticmethod
    def stack(xs: Sequence[Data], /, *, axis: int = 0) -> Data: ...

    @staticmethod
    def diagonal(x: Data, /) -> Data: ...

    @staticmethod
    def fill_diagonal(x: Data, y: Data) -> Data: ...

    # ======================================================================
    # Type casting
    # ======================================================================

    @staticmethod
    def as_type(x: Data, /, *, dtype: type, copy: bool = False) -> Data: ...

    # ======================================================================
    # Elementwise arithmetic
    # ======================================================================
    @staticmethod
    def add(x: Data, y: Data, /, *, out: Optional[Data] = None) -> Data: ...

    @staticmethod
    def subtract(x: Data, y: Data, /, *, out: Optional[Data] = None) -> Data: ...

    @staticmethod
    def multiply(x: Data, y: Data, /, *, out: Optional[Data] = None) -> Data: ...

    @staticmethod
    def divide(x: Data, y: Data, /, *, out: Optional[Data] = None) -> Data: ...

    @staticmethod
    def floor_divide(x: Data, y: Data, /, *, out: Optional[Data] = None) -> Data: ...

    @staticmethod
    def mod(x: Data, y: Data, /, *, out: Optional[Data] = None) -> Data: ...

    @staticmethod
    def power(x: Data, y: Data, /, *, out: Optional[Data] = None) -> Data: ...

    @staticmethod
    def absolute_value(x: Data, /, *, out: Optional[Data] = None) -> Data: ...

    # ======================================================================
    # Comparisons & logical ops
    # ======================================================================
    @staticmethod
    def less(x: Data, y: Data, /) -> Data: ...

    @staticmethod
    def less_equal(x: Data, y: Data, /) -> Data: ...

    @staticmethod
    def greater(x: Data, y: Data, /) -> Data: ...

    @staticmethod
    def greater_equal(x: Data, y: Data, /) -> Data: ...

    @staticmethod
    def equal(x: Data, y: Data, /) -> Data: ...

    @staticmethod
    def not_equal(x: Data, y: Data, /) -> Data: ...

    @staticmethod
    def invert(x: Data, /) -> Data: ...

    # ======================================================================
    # Elementwise math functions (ufuncs)
    # ======================================================================
    @staticmethod
    def exp(x: Data, /, *, out: Optional[Data] = None) -> Data: ...

    @staticmethod
    def log(x: Data, /, *, out: Optional[Data] = None) -> Data: ...

    @staticmethod
    def sqrt(x: Data, /, *, out: Optional[Data] = None) -> Data: ...

    @staticmethod
    def abs(x: Data, /, *, out: Optional[Data] = None) -> Data: ...

    @staticmethod
    def sin(x: Data, /, *, out: Optional[Data] = None) -> Data: ...

    @staticmethod
    def cos(x: Data, /, *, out: Optional[Data] = None) -> Data: ...

    # ======================================================================
    # Reductions
    # ======================================================================
    @staticmethod
    def sum(
        x: Data,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Data: ...

    @staticmethod
    def mean(
        x: Data,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Data: ...

    @staticmethod
    def min(
        x: Data,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Data: ...

    @staticmethod
    def max(
        x: Data,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Data: ...

    @staticmethod
    def count_nonzero(
        x: Data,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Data: ...

    # ======================================================================
    # Segment
    # ======================================================================
    @staticmethod
    def segment_sum(
        x: Data,
        /,
        *,
        indices: Any,
        are_slices: bool = False,
        axis: Optional[int] = 0,
        nbins: Optional[int] = None,
    ) -> Data: ...

    @staticmethod
    def segment_mean(
        x: Data,
        /,
        *,
        indices: Any,
        are_slices: bool = False,
        axis: Optional[int] = 0,
        nbins: Optional[int] = None,
    ) -> Data: ...

    @staticmethod
    def segment_min(
        x: Data,
        /,
        *,
        indices: Any,
        are_slices: bool = False,
        axis: Optional[int] = 0,
        nbins: Optional[int] = None,
    ) -> Data: ...

    @staticmethod
    def segment_max(
        x: Data,
        /,
        *,
        indices: Any,
        are_slices: bool = False,
        axis: Optional[int] = 0,
        nbins: Optional[int] = None,
    ) -> Data: ...

    # ======================================================================
    # Cumulative
    # ======================================================================
    @staticmethod
    def cumsum(
        x: Data,
        /,
        *,
        axis: Optional[int] = None,
        dtype: Optional[type] = None,
        out: Optional[Data] = None,
    ) -> Data: ...

    # ======================================================================
    # Indexing
    # ======================================================================
    @staticmethod
    def diff(x: Data, /, *, n: int = 1, axis: int = -1) -> Data: ...

    @staticmethod
    def repeat(x: Data, /, *, repeats: Any, axis: Optional[int] = None) -> Data: ...

    @staticmethod
    def flat_nonzero(x: Data, /) -> Data: ...

    @staticmethod
    def bin_count(x: Data, /, *, nbins: int = 0) -> Data: ...

    # ======================================================================
    # Compression
    # ======================================================================
    @staticmethod
    def compress_axis(
        x: Data, /, *, keep: Any, axis: Optional[int] = 0, out: Optional[Data] = None
    ) -> Data: ...

    # ======================================================================
    # Miscellaneous
    # ======================================================================
    @staticmethod
    def copy(x: Data) -> Data: ...

    # ======================================================================
    # Linear Algebra
    # ======================================================================
    @staticmethod
    def distance(x: Data, y: Data, /, *, dist_type: str) -> Data: ...

    @staticmethod
    def distance_lazy(x: Data, y: Data, /, *, dist_type: str) -> Iterator[Data]: ...

    # ======================================================================
    # Operate at
    # ======================================================================

    @staticmethod
    def add_at(x: Data, /, *, indices: Data, out: Optional[Data] = None) -> Data: ...
