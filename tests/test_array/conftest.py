from dataclasses import dataclass
from typing import Tuple, Type

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from src.array.base import Array


@dataclass
class TestData:
    data: NDArray | csr_array
    properties: Tuple[str, ...] = ("is_sparse", "shape", "ndim", "dtype")

    @property
    def is_sparse(self) -> bool:
        return not isinstance(self.data, np.ndarray)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return len(self.data.shape)

    @property
    def dtype(self) -> Type:
        return self.data.dtype

    def check(self, array: Array) -> bool:
        return (self.data is array.data) and all(
            getattr(self, prop) == getattr(array, prop) for prop in self.properties
        )


dense_test_data = {
    "dense_1d_float": TestData(
        np.random.rand(30),
    ),
    "dense_1d_int": TestData(
        np.random.randint(low=0, high=10, size=(30,)),
    ),
    "dense_1d_bool": TestData(
        np.random.randint(low=0, high=10, size=(30,)) < 5,
    ),
    "dense_2d_float": TestData(
        np.random.rand(30, 40) * (np.random.rand(30, 40) < 0.5),
    ),
    "dense_2d_int": TestData(
        np.random.randint(low=0, high=10, size=(30, 40))
        * (np.random.rand(30, 40) < 0.5),
    ),
    "dense_2d_bool": TestData(
        np.random.randint(low=0, high=10, size=(30, 40)) < 5,
    ),
}

sparse_test_data = {}
for k, v in dense_test_data.items():
    if "2d" in k:
        sparse_test_data[k.replace("dense", "sparse")] = TestData(csr_array(v.data))

test_data = {**dense_test_data, **sparse_test_data}

test_data_groups = {
    "all": tuple(test_data.values()),
    "dense": tuple(dense_test_data.values()),
    "sparse": tuple(sparse_test_data.values()),
}
