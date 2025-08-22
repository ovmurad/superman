from typing import Any, Iterator, Optional, Sequence

import numba as nb  # type: ignore
import numpy as np
from sklearn.metrics import (  # type: ignore
    pairwise_distances,
    pairwise_distances_chunked,
)

from .backend import Backend


def _mask_len(keep: np.ndarray[tuple[int], np.dtype[np.bool_]]) -> int:
    return int(np.sum(keep))


def _coo_len(keep: np.ndarray[tuple[int], np.dtype[np.integer]]) -> int:
    return len(keep)


def _slice_len(keep: slice, axis_len: int) -> int:
    """
    Calculate the number of elements that a slice would extract.
    The slice must have `start`, `stop`, and `step` defined.
    """
    start = 0 if keep.start is None else keep.start
    stop = axis_len if keep.stop is None else min(axis_len, keep.stop)
    if stop is None:
        raise ValueError("Slice 'stop' and 'ax_len' are both None!")
    step = 1 if keep.step is None else keep.step
    return (stop - start + step - 1) // step


class NumpyBackend(Backend[np.ndarray]):
    # ======================================================================
    # Class Vars
    # ======================================================================
    name = "numpy"

    data_type = np.ndarray

    index_dtype = np.int32

    bool_dtype = np.bool
    counting_dtype = np.int32
    real_dtype = np.float64

    # ======================================================================
    # Creation
    # ======================================================================
    @staticmethod
    def as_data(
        x: Any, /, *, dtype: Optional[type] = None, copy: bool = False
    ) -> np.ndarray:
        if isinstance(x, np.ndarray):
            if dtype is not None and x.dtype.type != dtype:
                return np.astype(x, dtype)
            else:
                return np.asarray(x, copy=copy)
        return np.array(x, dtype=dtype)

    @staticmethod
    def zeros(
        *, shape: int | Sequence[int], dtype: Optional[type] = None
    ) -> np.ndarray:
        return np.zeros(shape=shape, dtype=dtype)

    @staticmethod
    def ones(*, shape: int | Sequence[int], dtype: Optional[type] = None) -> np.ndarray:
        return np.ones(shape=shape, dtype=dtype)

    @staticmethod
    def full(
        x: Any, /, *, shape: int | Sequence[int], dtype: Optional[type] = None
    ) -> np.ndarray:
        return np.full(fill_value=x, shape=shape, dtype=dtype)

    @staticmethod
    def arange(
        *,
        start: int,
        stop: int,
        step: Optional[int] = None,
        dtype: Optional[type] = None,
    ) -> np.ndarray:
        return np.arange(start, stop, step, dtype=dtype)

    # ======================================================================
    # Shape manipulation
    # ======================================================================
    @staticmethod
    def reshape(x: np.ndarray, /, *, shape: int | Sequence[int]) -> np.ndarray:
        return np.reshape(x, shape=shape)

    @staticmethod
    def expand_dims(x: np.ndarray, /, *, axis: int | Sequence[int]) -> np.ndarray:
        return np.expand_dims(x, axis=axis)

    @staticmethod
    def squeeze(x: np.ndarray, /, *, axis: int | Sequence[int]) -> np.ndarray:
        return np.squeeze(x, axis=axis)

    @staticmethod
    def transpose(x: np.ndarray, /) -> np.ndarray:
        return np.transpose(x)

    @staticmethod
    def broadcast_to(x: np.ndarray, /, *, shape: int | Sequence[int]) -> np.ndarray:
        return np.broadcast_to(x, shape=shape)

    @staticmethod
    def concat(xs: Sequence[np.ndarray], /, *, axis: int = 0) -> np.ndarray:
        return np.concat(xs, axis=axis)

    @staticmethod
    def stack(xs: Sequence[np.ndarray], /, *, axis: int = 0) -> np.ndarray:
        return np.stack(xs, axis=axis)

    # ======================================================================
    # Type casting
    # ======================================================================

    @staticmethod
    def as_type(x: np.ndarray, /, *, dtype: type, copy: bool = False) -> np.ndarray:
        return np.astype(x, dtype, copy=copy)

    # ======================================================================
    # Elementwise arithmetic
    # ======================================================================
    @staticmethod
    def add(
        x: np.ndarray, y: np.ndarray, /, *, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.add(x, y, out=out)

    @staticmethod
    def subtract(
        x: np.ndarray, y: np.ndarray, /, *, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.subtract(x, y, out=out)

    @staticmethod
    def multiply(
        x: np.ndarray, y: np.ndarray, /, *, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.multiply(x, y, out=out)

    @staticmethod
    def divide(
        x: np.ndarray, y: np.ndarray, /, *, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.divide(x, y, out=out)

    @staticmethod
    def floor_divide(
        x: np.ndarray, y: np.ndarray, /, *, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.floor_divide(x, y, out=out)

    @staticmethod
    def mod(
        x: np.ndarray, y: np.ndarray, /, *, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.mod(x, y, out=out)

    @staticmethod
    def power(
        x: np.ndarray, y: np.ndarray, /, *, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.power(x, y, out=out)

    # ======================================================================
    # Comparisons & logical ops
    # ======================================================================

    @staticmethod
    def less(x: np.ndarray, y: np.ndarray, /) -> np.ndarray:
        return np.less(x, y)

    @staticmethod
    def less_equal(x: np.ndarray, y: np.ndarray, /) -> np.ndarray:
        return np.less_equal(x, y)

    @staticmethod
    def greater(x: np.ndarray, y: np.ndarray, /) -> np.ndarray:
        return np.greater(x, y)

    @staticmethod
    def greater_equal(x: np.ndarray, y: np.ndarray, /) -> np.ndarray:
        return np.greater_equal(x, y)

    @staticmethod
    def equal(x: np.ndarray, y: np.ndarray, /) -> np.ndarray:
        return np.equal(x, y)

    @staticmethod
    def not_equal(x: np.ndarray, y: np.ndarray, /) -> np.ndarray:
        return np.not_equal(x, y)

    @staticmethod
    def invert(x: np.ndarray, /) -> np.ndarray:
        return np.invert(x)

    # ======================================================================
    # Elementwise math functions (ufuncs)
    # ======================================================================

    @staticmethod
    def exp(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
        return np.exp(x, out=out)

    @staticmethod
    def log(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
        return np.log(x, out=out)

    @staticmethod
    def sqrt(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
        return np.sqrt(x, out=out)

    @staticmethod
    def abs(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
        return np.abs(x, out=out)

    @staticmethod
    def sin(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
        return np.sin(x, out=out)

    @staticmethod
    def cos(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
        return np.cos(x, out=out)

    # ======================================================================
    # Reductions
    # ======================================================================
    @staticmethod
    def sum(
        x: np.ndarray,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.sum(x, axis, keepdims=keepdims)

    @staticmethod
    def mean(
        x: np.ndarray,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.mean(x, axis, keepdims=keepdims)

    @staticmethod
    def min(
        x: np.ndarray,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.min(x, axis, keepdims=keepdims)

    @staticmethod
    def max(
        x: np.ndarray,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.max(x, axis, keepdims=keepdims)

    @staticmethod
    def count_nonzero(
        x: np.ndarray,
        /,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.count_nonzero(x, axis, keepdims=keepdims)

    # ======================================================================
    # Segment Reductions
    # ======================================================================
    @staticmethod
    def _segment_ufunc(
        ufunc: type[np.ufunc],
        x: np.ndarray,
        indices: Any,
        are_slices: bool,
        axis: int | None,
        nbins: int | None,
        fill_value: Any,
    ) -> np.ndarray:

        if axis is None:
            xm = x.reshape(-1)
        else:
            xm = np.moveaxis(x, source=axis, destination=0)

        indices = np.asarray(indices, dtype=NumpyBackend.index_dtype)

        if are_slices:
            nbins = indices.shape[0] - 1
        elif nbins is None:
            nbins = int(indices.max()) + 1

        shape = (nbins,) + xm.shape[1:]
        out = np.full(fill_value=fill_value, shape=shape, dtype=xm.dtype)

        if are_slices:
            indices = np.repeat(np.arange(nbins), repeats=np.diff(indices))

        ufunc.at(out, indices, xm)
        return out

    @staticmethod
    def segment_sum(
        x: np.ndarray,
        /,
        *,
        indices: Any,
        are_slices: bool = False,
        axis: Optional[int] = 0,
        nbins: Optional[int] = None,
    ) -> np.ndarray:
        return NumpyBackend._segment_ufunc(
            np.add,
            x,
            indices=indices,
            are_slices=are_slices,
            axis=axis,
            nbins=nbins,
            fill_value=0,
        )

    @staticmethod
    def segment_mean(
        x: np.ndarray,
        /,
        *,
        indices: Any,
        are_slices: bool = False,
        axis: Optional[int] = 0,
        nbins: Optional[int] = None,
    ) -> np.ndarray:

        indices = np.asarray(indices, dtype=NumpyBackend.index_dtype)
        segment_sums = NumpyBackend._segment_ufunc(
            np.add,
            x,
            indices=indices,
            are_slices=are_slices,
            axis=axis,
            nbins=nbins,
            fill_value=0,
        )

        if are_slices:
            diffs = np.diff(indices)
        else:
            diffs = np.bincount(indices, minlength=segment_sums.shape[0])

        diffs = np.expand_dims(diffs, axis=tuple(range(1, segment_sums.ndim)))

        segment_sums /= diffs
        return segment_sums

    @staticmethod
    def segment_min(
        x: np.ndarray,
        /,
        *,
        indices: Any,
        are_slices: bool = False,
        axis: Optional[int] = 0,
        nbins: Optional[int] = None,
    ) -> np.ndarray:
        return NumpyBackend._segment_ufunc(
            np.minimum,
            x,
            indices=indices,
            are_slices=are_slices,
            axis=axis,
            nbins=nbins,
            fill_value=np.inf,
        )

    @staticmethod
    def segment_max(
        x: np.ndarray,
        /,
        *,
        indices: Any,
        are_slices: bool = False,
        axis: Optional[int] = 0,
        nbins: Optional[int] = None,
    ) -> np.ndarray:
        return NumpyBackend._segment_ufunc(
            np.maximum,
            x,
            indices=indices,
            are_slices=are_slices,
            axis=axis,
            nbins=nbins,
            fill_value=-np.inf,
        )

    # ======================================================================
    # Cumulative
    # ======================================================================
    @staticmethod
    def cumsum(
        x: np.ndarray,
        /,
        *,
        axis: Optional[int] = None,
        dtype: Optional[type] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.cumsum(x, axis=axis, dtype=dtype, out=out)

    # ======================================================================
    # Indexing
    # ======================================================================
    @staticmethod
    def diff(x: np.ndarray, /, *, n: int = 1, axis: int = -1) -> np.ndarray:
        return np.diff(x, n=n, axis=axis)

    @staticmethod
    def repeat(
        x: np.ndarray, /, *, repeats: Any, axis: Optional[int] = None
    ) -> np.ndarray:
        return np.repeat(x, repeats=repeats, axis=axis)

    @staticmethod
    def flat_nonzero(x: np.ndarray, /) -> np.ndarray:
        return np.flatnonzero(x)

    @staticmethod
    def bin_count(x: np.ndarray, /, *, nbins: int = 0) -> np.ndarray:
        return np.bincount(x, minlength=nbins)

    # ======================================================================
    # Compression
    # ======================================================================
    @staticmethod
    def compress_axis(
        x: np.ndarray,
        /,
        *,
        keep: Any,
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = 0,
    ) -> np.ndarray:

        out = x if out is None else out
        out = out.reshape(-1) if axis is None else out

        if not isinstance(keep, slice):
            keep = np.asarray(keep)

        if isinstance(keep, np.ndarray) and np.issubdtype(keep.dtype, np.bool_):
            axis_sl = slice(0, _mask_len(keep))
        elif isinstance(keep, np.ndarray) and np.issubdtype(keep.dtype, np.integer):
            axis_sl = slice(0, _coo_len(keep))
        elif isinstance(keep, slice):
            axis_len = x.size if axis is None else x.shape[axis]
            axis_sl = slice(0, _slice_len(keep, axis_len))
        else:
            raise ValueError(f"Cannot compress axis using index of type {type(keep)}!")

        sl = (axis_sl,) if axis is None else ((slice(None),) * axis + (axis_sl,))
        out = out[sl]

        if isinstance(keep, np.ndarray) and np.issubdtype(keep.dtype, np.bool_):
            return np.compress(keep, x, axis=axis, out=out)
        elif isinstance(keep, np.ndarray) and np.issubdtype(keep.dtype, np.integer):
            return np.take(x, keep, axis=axis, out=out)
        elif isinstance(keep, slice):
            sl = (keep,) if axis is None else ((slice(None),) * axis + (keep,))
            if out is None:
                return x[sl]
            else:
                out[:] = x[sl]
                return out

        raise ValueError(f"Cannot compress axis using index of type {type(keep)}!")

    # ======================================================================
    # Miscellaneous
    # ======================================================================
    @staticmethod
    def copy(x: np.ndarray) -> np.ndarray:
        return np.copy(x)

    # ======================================================================
    # Linear Algebra
    # ======================================================================
    @staticmethod
    def distance(
        x: np.ndarray,
        y: np.ndarray,
        /,
        *,
        dist_type: str,
    ) -> np.ndarray:
        return pairwise_distances(x, y, metric=dist_type)

    @staticmethod
    def distance_lazy(
        x: np.ndarray,
        y: np.ndarray,
        /,
        *,
        dist_type: str,
    ) -> Iterator[np.ndarray]:
        return pairwise_distances_chunked(x, y, metric=dist_type)

    # ======================================================================
    # Operate at
    # ======================================================================

    @staticmethod
    def add_at(
        x: np.ndarray, /, *, indices: np.ndarray, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.add.at(x, indices=indices, out=out)

    # ======================================================================
    # Operate between
    # ======================================================================

    @staticmethod
    @nb.njit
    def add_between(
        x: np.ndarray,
        /,
        *,
        start: np.ndarray,
        stop: np.ndarray,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if out is None:
            out = np.zeros(shape=(start.shape[0],), dtype=x.dtype)

        for i in range(out.shape[0]):
            for j in range(start[i], stop[i]):
                out[i] += x[j]
        return out
