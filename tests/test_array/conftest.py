from dataclasses import dataclass
from typing import Generic, Tuple

import numpy as np
from scipy.sparse import csr_array
from src.array.base import Array, ArrayType


@dataclass
class DummyArray(Generic[ArrayType]):
    array: ArrayType
    properties: Tuple[str, ...] = ("is_sparse", "shape", "ndim", "dtype")

    @property
    def is_sparse(self) -> bool:
        return not isinstance(self.array, np.ndarray)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return len(self.array.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    def check(self, array: Array) -> bool:
        return (self.array is array.array) and all(
            getattr(self, prop) == getattr(array, prop) for prop in self.properties
        )


dense_dummy_arrays = {
    "dense_1d_float": DummyArray(
        np.random.rand(30),
    ),
    "dense_1d_int": DummyArray(
        np.random.randint(low=0, high=10, size=(30,)),
    ),
    "dense_1d_bool": DummyArray(
        np.random.randint(low=0, high=10, size=(30,)) < 5,
    ),
    "dense_2d_float": DummyArray(
        np.random.rand(30, 40) * (np.random.rand(30, 40) < 0.5),
    ),
    "dense_2d_int": DummyArray(
        np.random.randint(low=0, high=10, size=(30, 40))
        * (np.random.rand(30, 40) < 0.5),
    ),
    "dense_2d_bool": DummyArray(
        np.random.randint(low=0, high=10, size=(30, 40)) < 5,
    ),
}

sparse_dummy_arrays = {}
for dense_key, v in dense_dummy_arrays.items():
    if "2d" in dense_key:
        sparse_key = dense_key.replace("dense", "sparse")
        sparse_dummy_arrays[sparse_key] = DummyArray(csr_array(v.array))

dummy_array = {**dense_dummy_arrays, **sparse_dummy_arrays}

dummy_array_groups = {
    "all": tuple(dummy_array.values()),
    "dense": tuple(dense_dummy_arrays.values()),
    "sparse": tuple(sparse_dummy_arrays.values()),
}
