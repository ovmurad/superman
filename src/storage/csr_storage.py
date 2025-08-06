from typing import Generic

import numpy as np

from .storage import SparseStorageMixin
from .typing import DType, MatrixShape, RawIndex, SparseData, _CsrStorage, _NumpyStorage


class CsrStorage(Generic[DType], SparseStorageMixin[DType, MatrixShape]):
    _storage: _CsrStorage[DType]

    def __init__(self, storage: _CsrStorage[DType]):
        self._storage = storage

    # ----------- Array API -----------
    # See: https://data-apis.org/array-api/2024.12/API_specification/index.html

    @property
    def shape(self) -> MatrixShape:
        return self._storage.shape

    @property
    def dtype(self) -> type[DType]:
        return self._storage.dtype.type

    # ----------- Specific to package -----------

    @property
    def nnz(self) -> int:
        return self._storage.indices.shape[0]

    @property
    def data(self) -> SparseData[DType]:
        return self._storage.data

    @property
    def indices(self) -> RawIndex[np.int32]:
        return self._storage.indices

    @property
    def indptr(self) -> RawIndex[np.int32]:
        return self._storage.indptr

    # ----------- numpy support -----------

    def to_numpy(self) -> _NumpyStorage[MatrixShape, DType]:
        return self._storage.todense()
