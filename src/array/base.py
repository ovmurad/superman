from __future__ import annotations

from abc import ABC, abstractmethod
from math import prod
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    Optional,
    Self,
    Sequence,
)

import numpy as np

from ..storage import Data, Storage

if TYPE_CHECKING:
    from .dense import DenseArray
    from .sparse import CsrArray

ArrayFormat = Literal["dense", "csr"]


# ======================================================================
# Utils
# ======================================================================
def unwrap(stg_like: Any) -> Any:
    if isinstance(stg_like, BaseArray) and stg_like.is_dense:
        return stg_like.values
    return stg_like


def unwrap_args(args: tuple) -> tuple:
    return tuple(unwrap(arg) for arg in args)


def unwrap_kwargs(kwargs: dict) -> dict[str, Any]:
    return {kw: unwrap(arg) for kw, arg in kwargs.items()}


class BaseArray(Generic[Data], ABC):
    # ======================================================================
    # Instance Vars
    # ======================================================================
    _values: Storage[Data]

    # ======================================================================
    # Class Vars
    # ======================================================================
    _format: ClassVar[ArrayFormat]

    # ======================================================================
    # Initialization
    # ======================================================================
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    # ======================================================================
    # Constructors
    # ======================================================================
    @classmethod
    @abstractmethod
    def as_array(cls, *args: Any, **kwargs: Any) -> Self: ...

    # ======================================================================
    # Introspection
    # ======================================================================
    @property
    def values(self) -> Storage[Data]:
        return self._values

    @property
    def format(self) -> ArrayFormat:
        return self._format

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> type:
        return self._values.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def nnz(self) -> int:
        return self._values.size

    @property
    def is_dense(self) -> bool:
        return self._format == "dense"

    @property
    def is_sparse(self) -> bool:
        return not self.is_dense

    # ======================================================================
    # Dunder
    # ======================================================================

    # ----------------------------------------------------------------------
    # Get & Set items
    # ----------------------------------------------------------------------
    @abstractmethod
    def __getitem__(self, key: Any) -> BaseArray[Data]: ...

    @abstractmethod
    def __setitem__(self, key: Any, values_like: Any) -> None: ...

    # ----------------------------------------------------------------------
    # Arithmetic
    # ----------------------------------------------------------------------
    @abstractmethod
    def __add__(self, other: Any) -> BaseArray[Data]: ...

    @abstractmethod
    def __iadd__(self, other: Any) -> Self: ...

    @abstractmethod
    def __sub__(self, other: Any) -> BaseArray[Data]: ...

    @abstractmethod
    def __isub__(self, other: Any) -> Self: ...

    @abstractmethod
    def __mul__(self, other: Any) -> BaseArray[Data]: ...

    @abstractmethod
    def __imul__(self, other: Any) -> Self: ...

    @abstractmethod
    def __truediv__(self, other: Any) -> BaseArray[Data]: ...

    @abstractmethod
    def __itruediv__(self, other: Any) -> Self: ...

    @abstractmethod
    def __mod__(self, other: Any) -> BaseArray[Data]: ...

    @abstractmethod
    def __imod__(self, other: Any) -> Self: ...

    @abstractmethod
    def __floordiv__(self, other: Any) -> BaseArray[Data]: ...

    @abstractmethod
    def __ifloordiv__(self, other: Any) -> Self: ...

    @abstractmethod
    def __pow__(self, other: Any) -> BaseArray[Data]: ...

    @abstractmethod
    def __ipow__(self, other: Any) -> Self: ...

    # ----------------------------------------------------------------------
    # Logic & Comparison
    # ----------------------------------------------------------------------
    @abstractmethod
    def __lt__(self, other: Any) -> BaseArray[Data]: ...

    @abstractmethod
    def __le__(self, other: Any) -> BaseArray[Data]: ...

    @abstractmethod
    def __gt__(self, other: Any) -> BaseArray[Data]: ...

    @abstractmethod
    def __ge__(self, other: Any) -> BaseArray[Data]: ...

    @abstractmethod
    def __eq__(self, other: Any) -> BaseArray[Data]:  # type: ignore
        ...

    @abstractmethod
    def __ne__(self, other: Any) -> BaseArray[Data]:  # type: ignore
        ...

    @abstractmethod
    def __invert__(self, other: Any) -> BaseArray[Data]:  # type: ignore
        ...

    # ----------------------------------------------------------------------
    # Utils
    # ----------------------------------------------------------------------
    @abstractmethod
    def __copy__(self) -> Self: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    # ======================================================================
    # Shape Manipulation
    # ======================================================================

    @abstractmethod
    def reshape(self, /, *, shape: int | Sequence[int]) -> Self: ...

    @abstractmethod
    def expand_dims(self, /, *, axis: int | Sequence[int]) -> Self: ...

    @abstractmethod
    def squeeze(self, /, *, axis: int | Sequence[int]) -> Self: ...

    @abstractmethod
    def broadcast_to(self, /, *, shape: int | Sequence[int]) -> Self: ...

    @abstractmethod
    def diagonal(self) -> BaseArray[Data]: ...

    @abstractmethod
    def fill_diagonal(self, *args: Any, **kwargs: Any) -> Self: ...

    @classmethod
    @abstractmethod
    def concat(cls, arrs: Sequence[Self], /, *, axis: int = 0) -> Self: ...

    @classmethod
    @abstractmethod
    def stack(cls, arrs: Sequence[Self], /, *, axis: int = 0) -> Self: ...

    # ======================================================================
    # Type Casting
    # ======================================================================
    @abstractmethod
    def as_type(self, *args: Any, **kwargs: Any) -> Self: ...

    @abstractmethod
    def as_dense(self, *args: Any, **kwargs: Any) -> DenseArray[Data]: ...

    @abstractmethod
    def as_csr(self, *args: Any, **kwargs: Any) -> CsrArray[Data]: ...

    @abstractmethod
    def as_nparray(
        self, *args: Any, **kwargs: Any
    ) -> np.ndarray[tuple[int, ...], Data]: ...

    # ======================================================================
    # Elementwise math functions (ufuncs)
    # ======================================================================
    @abstractmethod
    def exp(self, *args: Any, **kwargs: Any) -> Self: ...

    @abstractmethod
    def iexp(self) -> Self: ...

    @abstractmethod
    def log(self, *args: Any, **kwargs: Any) -> Self: ...

    @abstractmethod
    def ilog(self) -> Self: ...

    @abstractmethod
    def sqrt(self, *args: Any, **kwargs: Any) -> Self: ...

    @abstractmethod
    def isqrt(self) -> Self: ...

    @abstractmethod
    def sin(self, *args: Any, **kwargs: Any) -> Self: ...

    @abstractmethod
    def isin(self) -> Self: ...

    @abstractmethod
    def cos(self, *args: Any, **kwargs: Any) -> Self: ...

    @abstractmethod
    def icos(self) -> Self: ...

    # ======================================================================
    # Reductions
    # ======================================================================
    @abstractmethod
    def sum(
        self,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> BaseArray[Data]: ...

    @abstractmethod
    def mean(
        self,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> BaseArray[Data]: ...

    @abstractmethod
    def min(
        self,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> BaseArray[Data]: ...

    @abstractmethod
    def max(
        self,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> BaseArray[Data]: ...

    @abstractmethod
    def count_nonzero(
        self,
        /,
        *,
        axis: Optional[int | Sequence[int]] = None,
        keepdims: bool = False,
    ) -> BaseArray[Data]: ...

    # ======================================================================
    # Compression
    # ======================================================================
    @abstractmethod
    def compress_axis(
        self,
        /,
        *,
        keep: Any,
        axis: Optional[int] = 0,
    ) -> BaseArray[Data]: ...

    @abstractmethod
    def compress(
        self,
        /,
        *,
        keep: Any | Sequence[Any],
        axis: Optional[int | Sequence[int]] = 0,
    ) -> BaseArray[Data]: ...

    # ======================================================================
    # Miscellaneous
    # ======================================================================

    @abstractmethod
    def copy(self, *args: Any, **kwargs: Any) -> Self: ...

    # ======================================================================
    # Linear Algebra
    # ======================================================================
