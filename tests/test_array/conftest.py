from dataclasses import dataclass
from typing import Tuple, Type

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from src.array.base import Array


@dataclass
class TestArray:
    array: NDArray | csr_array
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
    def dtype(self) -> Type:
        return self.array.dtype

    def check(self, array: Array) -> bool:
        return (self.array is array.array) and all(
            getattr(self, prop) == getattr(array, prop) for prop in self.properties
        )


dense_test_array = {
    "dense_1d_float": TestArray(
        np.random.rand(30),
    ),
    "dense_1d_int": TestArray(
        np.random.randint(low=0, high=10, size=(30,)),
    ),
    "dense_1d_bool": TestArray(
        np.random.randint(low=0, high=10, size=(30,)) < 5,
    ),
    "dense_2d_float": TestArray(
        np.random.rand(30, 40) * (np.random.rand(30, 40) < 0.5),
    ),
    "dense_2d_int": TestArray(
        np.random.randint(low=0, high=10, size=(30, 40))
        * (np.random.rand(30, 40) < 0.5),
    ),
    "dense_2d_bool": TestArray(
        np.random.randint(low=0, high=10, size=(30, 40)) < 5,
    ),
}

sparse_test_array = {}
for k, v in dense_test_array.items():
    if "2d" in k:
        sparse_test_array[k.replace("dense", "sparse")] = TestArray(csr_array(v.array))

test_array = {**dense_test_array, **sparse_test_array}

test_array_groups = {
    "all": tuple(test_array.values()),
    "dense": tuple(dense_test_array.values()),
    "sparse": tuple(sparse_test_array.values()),
}
