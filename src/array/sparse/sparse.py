from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Generic, Optional, ParamSpec, Self, Sequence

from ...storage import Data, Storage
from ..base import BaseArray, unwrap
from ..dense import DenseArray

_OpArgs = ParamSpec("_OpArgs")


def _values_op_out_of_place(
    stg_func: Callable[[Storage[Data]], Storage[Data]],
) -> Callable[[SparseArray[Data], bool], SparseArray[Data]]:
    def _wrapper(
        self: SparseArray[Data], copy_index: bool = False
    ) -> SparseArray[Data]:
        values = stg_func(self._values)
        return self.__class__(
            values, self.index, shape=self.shape, copy_index=copy_index
        )

    return _wrapper


def _values_op_in_place(
    stg_func: Callable[[Storage[Data]], Storage[Data]],
) -> Callable[[SparseArray[Data]], SparseArray[Data]]:
    def _wrapper(self: SparseArray[Data]) -> SparseArray[Data]:
        self._values = stg_func(self._values)
        return self

    return _wrapper


class SparseArray(Generic[Data], BaseArray[Data]):
    # ======================================================================
    # Instance Vars
    # ======================================================================
    _values: Storage[Data]
    _index: tuple[Storage[Data], ...]
    _shape: tuple[int, ...]

    # ======================================================================
    # Class Vars
    # ======================================================================

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
        self._values = Storage.as_values(
            unwrap(values_like), dtype=dtype, copy=copy_values, assert_flat=True
        )
        self._index = tuple(
            Storage.as_index(unwrap(axis_idx_like), copy=copy_index, assert_flat=True)
            for axis_idx_like in index_like
        )
        self._shape = tuple(shape)

    # ======================================================================
    # Constructors
    # ======================================================================

    # ======================================================================
    # Introspection
    # ======================================================================
    @property
    def index(self) -> tuple[Storage[Data], ...]:
        return self._index

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def sparsity(self) -> float:
        return round(self.nnz / self.size, 4)

    # ======================================================================
    # Dunder
    # ======================================================================

    # ----------------------------------------------------------------------
    # Get & Set items
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Arithmetic
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Logic & Comparison
    # ----------------------------------------------------------------------
    __invert__ = _values_op_out_of_place(Storage.__invert__)

    # ----------------------------------------------------------------------
    # Utils
    # ----------------------------------------------------------------------
    def __copy__(self) -> Self:
        return self.copy()

    # ======================================================================
    # Shape Manipulation
    # ======================================================================

    # ======================================================================
    # Type Casting
    # ======================================================================

    def as_type(
        self, *, dtype: type, copy_values: bool = False, copy_index: bool = False
    ) -> Self:
        return self.__class__(
            self._values,
            self._index,
            shape=self.shape,
            dtype=dtype,
            copy_values=copy_values,
            copy_index=copy_index,
        )

    def as_dense(self, fill_value: Any) -> DenseArray[Data]:
        values = DenseArray[Data].full(fill_value, shape=self._shape, dtype=self.dtype)
        values[self.get_dense_index()] = self._values
        return values

    # ======================================================================
    # Elementwise math functions (ufuncs)
    # ======================================================================

    exp = _values_op_out_of_place(Storage.exp)
    iexp = _values_op_in_place(Storage.iexp)
    log = _values_op_out_of_place(Storage.log)
    ilog = _values_op_in_place(Storage.ilog)
    sqrt = _values_op_out_of_place(Storage.sqrt)
    isqrt = _values_op_in_place(Storage.isqrt)
    sin = _values_op_out_of_place(Storage.sin)
    isin = _values_op_in_place(Storage.isin)
    cos = _values_op_out_of_place(Storage.cos)
    icos = _values_op_in_place(Storage.icos)

    # ======================================================================
    # Reductions
    # ======================================================================

    # ======================================================================
    # Indexing
    # ======================================================================
    @abstractmethod
    def get_dense_index(self) -> tuple[Storage[Data], ...]: ...

    # ======================================================================
    # Compression
    # ======================================================================

    # ======================================================================
    # Miscellaneous
    # ======================================================================

    def copy(self, copy_values: bool = True, copy_index: bool = True) -> Self:
        return self.__class__(
            self._values,
            self._index,
            shape=self.shape,
            copy_values=copy_values,
            copy_index=copy_index,
        )

    # ======================================================================
    # Linear Algebra
    # ======================================================================
