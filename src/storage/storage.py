from __future__ import annotations

from math import prod
from typing import ClassVar, Generic, Protocol, runtime_checkable

from .typing import DType, Shape, _NumpyStorage


@runtime_checkable
class Storage(Protocol[DType, Shape]):
    is_dense: ClassVar[bool]
    is_sparse: ClassVar[bool]

    # ----------- Array API -----------
    # See: https://data-apis.org/array-api/2024.12/API_specification/index.html
    @property
    def shape(self) -> Shape: ...

    @property
    def dtype(self) -> type[DType]: ...

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return prod(self.shape)

    # ----------- Specific to package -----------
    @property
    def nnz(self) -> int: ...

    @property
    def is_scalar(self) -> bool: ...

    def to_dense(self) -> Storage[DType, Shape]: ...

    def to_sparse(self) -> Storage[DType, Shape]: ...

    # ----------- numpy support -----------

    def to_numpy(self) -> _NumpyStorage[Shape, DType]: ...


class DenseStorageMixin(Generic[DType, Shape], Storage[DType, Shape]):
    is_dense = True
    is_sparse = False

    @property
    def nnz(self) -> int:
        return self.size

    @property
    def is_scalar(self) -> bool:
        return self.ndim == 0

    def to_dense(self) -> Storage[DType, Shape]:
        return self

    def to_sparse(self) -> Storage[DType, Shape]:
        raise NotImplementedError("Conversion to sparse not implemented!")


class SparseStorageMixin(Generic[DType, Shape], Storage[DType, Shape]):
    is_dense = False
    is_sparse = True

    @property
    def is_scalar(self) -> bool:
        return False

    def to_dense(self) -> Storage[DType, Shape]:
        raise NotImplementedError("Conversion to dense not implemented!")

    def to_sparse(self) -> Storage[DType, Shape]:
        return self
