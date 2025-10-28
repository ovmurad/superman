from __future__ import annotations

from typing import Any, Generic, NoReturn, Optional, Sequence

import numpy as np

from src.storage import BACKEND, Data, Storage

from ..base import BaseArray, unwrap
from ..dense import DenseArray
from .sparse import SparseArray


class CsrArray(Generic[Data], SparseArray[Data]):
    # ======================================================================
    # Instance Vars
    # ======================================================================
    _values: Storage[Data]
    _index: tuple[Storage[Data], ...]
    _shape: tuple[int, int]
    __slots__ = ("_values", "_index", "_shape")

    # ======================================================================
    # Class Vars
    # ======================================================================
    _format = "csr"

    # ======================================================================
    # Initialization
    # ======================================================================
    def __init__(
        self,
        values_like: Any,
        index_like: Sequence[Any],
        /,
        *,
        shape: Sequence[int],
        dtype: Optional[type] = None,
        copy_values: bool = False,
        copy_index: bool = False,
    ) -> None:

        super().__init__(
            values_like,
            index_like,
            shape=shape,
            dtype=dtype,
            copy_values=copy_values,
            copy_index=copy_index,
        )
        if self.ndim != 2:
            raise ValueError(
                f"CsrArray can only have exactly 2 dimensions, got {self.ndim} instead!"
            )
        if self.indptr.size != self.nrows + 1:
            raise ValueError(
                f"`indptr` and `nrows` sizes disagree, got {self.indptr.size} and {self.nrows}!"
            )
        if self.indices.size != self._values.size:
            raise ValueError(
                f"`cols` and `data` sizes disagree, got {self.indices.size} and {self._values.size}!"
            )

    # ======================================================================
    # Constructors
    # ======================================================================

    @staticmethod
    def as_array(
        values_like: Any,
        index_like: Sequence[Any],
        /,
        *,
        shape: Optional[Sequence[int]] = None,
        dtype: Optional[type] = None,
        copy_values: bool = False,
        copy_index: bool = False,
    ) -> CsrArray[Data]:

        if not isinstance(index_like, tuple):
            *index_like, shape = CsrArray.mask_as_csr_index(index_like)

        if shape is None:
            raise ValueError("`shape` could not be inferred and wasn't set!")

        return CsrArray(
            values_like,
            index_like,
            shape=shape,
            dtype=dtype,
            copy_values=copy_values,
            copy_index=copy_index,
        )

    # ======================================================================
    # Introspection
    # ======================================================================
    @property
    def indptr(self) -> Storage[Data]:
        return self._index[0]

    @property
    def indices(self) -> Storage[Data]:
        return self._index[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def nrows(self) -> int:
        return self._shape[0]

    @property
    def ncols(self) -> int:
        return self._shape[1]

    # ======================================================================
    # Dunder
    # ======================================================================

    # ----------------------------------------------------------------------
    # Get & Set items
    # ----------------------------------------------------------------------
    def __getitem__(self, key: Any) -> DenseArray[Data] | CsrArray[Data]:
        raise NotImplementedError("TODO")

    def __setitem__(self, key: Any, values_like: Any) -> None:
        raise NotImplementedError("TODO")

    def diagonal(self) -> BaseArray[Data]:
        raise NotImplementedError("TODO")

    def fill_diagonal(self, other: Any) -> CsrArray[Data]:
        raise NotImplementedError("TODO")

    # ----------------------------------------------------------------------
    # Arithmetic
    # ----------------------------------------------------------------------
    def __add__(self, other: Any) -> BaseArray[Data]:
        raise NotImplementedError("TODO")

    def __iadd__(self, other: Any) -> CsrArray[Data]:
        raise NotImplementedError("TODO")

    def __sub__(self, other: Any) -> BaseArray[Data]:
        raise NotImplementedError("TODO")

    def __isub__(self, other: Any) -> CsrArray[Data]:
        raise NotImplementedError("TODO")

    def __mul__(self, other: Any) -> BaseArray[Data]:
        raise NotImplementedError("TODO")

    def __imul__(self, other: Any) -> CsrArray[Data]:
        raise NotImplementedError("TODO")

    def __truediv__(self, other: Any) -> BaseArray[Data]:
        raise NotImplementedError("TODO")

    def __itruediv__(self, other: Any) -> CsrArray[Data]:
        raise NotImplementedError("TODO")

    def __mod__(self, other: Any) -> BaseArray[Data]:
        raise NotImplementedError("TODO")

    def __imod__(self, other: Any) -> CsrArray[Data]:
        raise NotImplementedError("TODO")

    def __floordiv__(self, other: Any) -> BaseArray[Data]:
        raise NotImplementedError("TODO")

    def __ifloordiv__(self, other: Any) -> CsrArray[Data]:
        raise NotImplementedError("TODO")

    def __pow__(self, other: Any) -> BaseArray[Data]:
        raise NotImplementedError("TODO")

    def __ipow__(self, other: Any) -> CsrArray[Data]:
        raise NotImplementedError("TODO")

    # ----------------------------------------------------------------------
    # Logic & Comparison
    # ----------------------------------------------------------------------

    def __lt__(self, other: Any) -> BaseArray[Data]:
        raise NotImplementedError("TODO")

    def __le__(self, other: Any) -> BaseArray[Data]:
        raise NotImplementedError("TODO")

    def __ge__(self, other: Any) -> BaseArray[Data]:
        raise NotImplementedError("TODO")

    def __gt__(self, other: Any) -> BaseArray[Data]:
        raise NotImplementedError("TODO")

    def __eq__(self, other: Any) -> BaseArray[Data]:  # type: ignore
        raise NotImplementedError("TODO")

    def __ne__(self, other: Any) -> BaseArray[Data]:  # type: ignore
        raise NotImplementedError("TODO")

    # ----------------------------------------------------------------------
    # Utils
    # ----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CsrArray: \n"
            f"  - shape: {self._shape} \n"
            f"  - dtype: {self.dtype} \n"
            f"  - indptr: {self.indptr.__repr__()} \n"
            f"  - cols: {self.indices.__repr__()} \n"
            f"  - values: {self._values.__repr__()} \n"
            f"  - nnz: {self.nnz} \n"
            f"  - sparsity: {self.sparsity}"
        )

    # ======================================================================
    # Shape Manipulation
    # ======================================================================

    @staticmethod
    def reshape(*, shape: int | Sequence[int]) -> NoReturn:
        raise NotImplementedError("Csr Array does not implement `reshape`!")

    @staticmethod
    def expand_dims(*, axis: int | Sequence[int]) -> NoReturn:
        raise NotImplementedError("Csr Array does not implement `expand_dims`!")

    @staticmethod
    def squeeze(*, axis: int | Sequence[int]) -> NoReturn:
        raise NotImplementedError("Csr Array does not implement `squeeze`!")

    @staticmethod
    def broadcast_to(*, shape: int | Sequence[int]) -> NoReturn:
        raise NotImplementedError("Csr Array does not implement `broadcast_to`!")

    @staticmethod
    def concat(arrs: Sequence[CsrArray[Data]], /, *, axis: int = 0) -> CsrArray[Data]:

        if axis == 0:

            if any(arrs[0].ncols != arr.ncols for arr in arrs[1:]):
                raise ValueError("Csr Arrays have mismatching `ncols`!")
            nrows, ncols = sum(arr.nrows for arr in arrs), arrs[0].ncols

            start, stop, offset = 0, arrs[0].nrows + 1, arrs[0].nnz

            indptr = Storage[Data].zeros(shape=nrows + 1, dtype=BACKEND.index_dtype)
            indptr[:stop] = arrs[0].indptr

            for arr in arrs[1:]:
                start, stop = stop, stop + arr.nrows
                indptr[start:stop] = offset
                indptr[start:stop] += arr.indptr[1:]
                offset += arr.nnz

            indices = Storage[Data].concat([arr.indices for arr in arrs])
            values = Storage[Data].concat([arr.values for arr in arrs])

        elif axis == 1:
            raise NotImplementedError("TODO")
        else:
            raise ValueError(f"Unrecognized axis {axis}! Must be either 0 or 1!")

        return CsrArray(values, (indptr, indices), shape=(nrows, ncols))

    @staticmethod
    def stack(arrs: Sequence[CsrArray[Data]], /, *, axis: int = 0) -> NoReturn:
        raise NotImplementedError("Csr Array does not implement `stack`!")

    # ======================================================================
    # Type Casting
    # ======================================================================

    as_csr = SparseArray.copy

    def as_nparray(self) -> np.ndarray:
        raise NotImplementedError("TODO")

    # ======================================================================
    # Elementwise math functions (ufuncs)
    # ======================================================================

    # ======================================================================
    # Reductions
    # ======================================================================

    def _format_reduce_out(
        self, stg: Storage[Data], axis: int | Sequence[int] | None, keepdims: bool
    ) -> DenseArray[Data]:
        if axis is None or axis == (0, 1):
            shape = (1, 1)
        elif axis == 0:
            shape = (1, self.ncols)
        elif axis == 1:
            shape = (self.nrows, 1)
        else:
            raise ValueError(f"Unrecognized axis {axis}!")

        arr = DenseArray[Data](stg)
        if keepdims:
            return arr.reshape(shape=shape)
        return arr

    def sum(
        self,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> DenseArray[Data]:

        if axis is None or axis == (0, 1):
            stg = self._values.sum()
        elif axis == 0:
            stg = self._values.segment_sum(indices=self.indices, nbins=self.ncols)
        elif axis == 1:
            stg = self._values.segment_sum(indices=self.indptr, are_slices=True)
        else:
            raise ValueError(f"Unrecognized axis {axis}!")
        return self._format_reduce_out(stg, axis, keepdims)

    def mean(
        self,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> DenseArray[Data]:

        if axis is None or axis == (0, 1):
            stg = self._values.mean()
        elif axis == 0:
            stg = self._values.segment_mean(indices=self.indices, nbins=self.ncols)
        elif axis == 1:
            stg = self._values.segment_mean(indices=self.indptr, are_slices=True)
        else:
            raise ValueError(f"Unrecognized axis {axis}!")
        return self._format_reduce_out(stg, axis, keepdims)

    def min(
        self,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> DenseArray[Data]:
        if axis is None or axis == (0, 1):
            stg = self._values.min()
        elif axis == 0:
            stg = self._values.segment_min(indices=self.indices, nbins=self.ncols)
        elif axis == 1:
            stg = self._values.segment_min(indices=self.indptr, are_slices=True)
        else:
            raise ValueError(f"Unrecognized axis {axis}!")
        return self._format_reduce_out(stg, axis, keepdims)

    def max(
        self,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> DenseArray[Data]:

        if axis is None or axis == (0, 1):
            stg = self._values.max()
        elif axis == 0:
            stg = self._values.segment_max(indices=self.indices, nbins=self.ncols)
        elif axis == 1:
            stg = self._values.segment_max(indices=self.indptr, are_slices=True)
        else:
            raise ValueError(f"Unrecognized axis {axis}!")
        return self._format_reduce_out(stg, axis, keepdims)

    def count_nonzero(
        self,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> DenseArray[Data]:

        if axis is None or axis == (0, 1):
            stg = self.nnz
        elif axis == 0:
            stg = self.indices.bin_count(nbins=self.ncols)
        elif axis == 1:
            stg = self.indptr.diff()
        else:
            raise ValueError(f"Unrecognized axis {axis}!")
        return self._format_reduce_out(stg, axis, keepdims)

    # ======================================================================
    # Indexing
    # ======================================================================

    def get_dense_index(self) -> tuple[Storage[Data], Storage[Data]]:
        rows = Storage[Data].arange(start=0, stop=self.nrows)
        rows = rows.repeat(repeats=self.indptr.diff())
        return rows, self.indices

    @staticmethod
    def mask_as_csr_index(
        mask_like: Any,
    ) -> tuple[Storage[Data], Storage[Data], tuple[int, int]]:

        mask = Storage[Data].as_mask(unwrap(mask_like))

        if (ndim := mask.ndim) != 2:
            raise ValueError(f"A 2 dimensional mask is needed, got {ndim} instead!")
        nrows, ncols = mask.shape

        indptr = Storage[Data].zeros(shape=nrows + 1, dtype=BACKEND.index_dtype)
        Storage[Data].cumsum(mask.count_nonzero(axis=1), out=indptr[1:])

        indices = mask.flat_nonzero()
        indices %= ncols

        return indptr, indices, (nrows, ncols)

    # ======================================================================
    # Compression
    # ======================================================================

    def compress_axis(
        self,
        /,
        *,
        keep: Any,
        axis: Optional[int] = 0,
    ) -> CsrArray[Data]:

        # TODO: Assumes mask, but wouldn't work for coos and slice(there is code in cryo)
        keep = Storage[Data].as_mask(keep, assert_flat=True)

        if axis == 0:

            diffs = self.indptr.diff()

            values_mask = keep.repeat(repeats=diffs)

            self._values.compress_axis(keep=values_mask)
            self.indices.compress_axis(keep=values_mask)

            diffs[~keep] = 0
            diffs.cumsum(out=self.indptr[1:])

        elif axis == 1:

            values_mask = keep[self.indices]

            self._values.compress_axis(keep=values_mask)
            self.indices.compress_axis(keep=values_mask)

            self.indptr[1:] = values_mask.segment_sum(
                indices=self.indptr, are_slices=True
            )
            self.indptr.cumsum(out=self.indptr)

        elif axis is None:

            self._values.compress_axis(keep=keep)
            self.indices.compress_axis(keep=keep)

            self.indptr[1:] = keep.segment_sum(indices=self.indptr, are_slices=True)
            self.indptr.cumsum(out=self.indptr)

        else:
            raise ValueError(f"Unrecognized axis {axis}! Must be either 0, 1, None!")

        return self

    def compress(
        self,
        /,
        *,
        keep: Any | Sequence[Any],
        axis: Optional[int | Sequence[int]] = 0,
    ) -> CsrArray[Data]:
        raise NotImplementedError("TODO")

    # ======================================================================
    # Miscellaneous
    # ======================================================================

    # ======================================================================
    # Linear Algebra
    # ======================================================================
