from typing import Generic

from .storage import DenseStorageMixin
from .typing import DType, Shape, _NumpyStorage


class NumpyStorage(Generic[DType, Shape], DenseStorageMixin[DType, Shape]):
    _storage: _NumpyStorage[Shape, DType]

    def __init__(self, storage: _NumpyStorage[Shape, DType]):
        self._storage = storage

    # ----------- Array API -----------
    # See: https://data-apis.org/array-api/2024.12/API_specification/index.html

    @property
    def shape(self) -> Shape:
        return self._storage.shape

    @property
    def dtype(self) -> type[DType]:
        return self._storage.dtype.type

    # ----------- Specific to package -----------

    # ----------- numpy support -----------

    def to_numpy(self) -> _NumpyStorage[Shape, DType]:
        return self._storage
