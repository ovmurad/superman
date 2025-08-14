from collections import defaultdict
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.sparse import csr_array

from src.array.dense import DenseArray
from src.object.geometry_matrix import DistanceMatrix
from src.object.points import Points
from tests.test_array.dummy_array import DummyArray

def load_test_npy():
    data = defaultdict(dict)
    
    for root, _, files in os.walk("./tests/data"):
        for f in files:
            if f.endswith('.npy') and '-' in f:
                prefix = f.split('-')[0]
                if f.startswith(prefix + '-'):
                    file_key = f[f.find("-") + 1:f.find(".npy")]
                    full_path = os.path.join(root, f)
                    data[prefix][file_key] = np.load(full_path)
    return data

npy_dict: Dict[str, Dict[str, np.ndarray]] = load_test_npy()

test_rtol = 1e-5
test_atol = 1e-08

dense_dummy_arrays: Dict[str, DummyArray] = {
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

sparse_dummy_arrays: Dict[str, DummyArray] = {}
for dense_key, v in dense_dummy_arrays.items():
    if "2d" in dense_key:
        sparse_key = dense_key.replace("dense", "sparse")
        sparse_dummy_arrays[sparse_key] = DummyArray(csr_array(v.array))

dummy_arrays: Dict[str, DummyArray] = {**dense_dummy_arrays, **sparse_dummy_arrays}

dummy_array_groups: Dict[str, Tuple[DummyArray, ...]] = {
    "all": tuple(dummy_arrays.values()),
    "dense": tuple(dense_dummy_arrays.values()),
    "sparse": tuple(sparse_dummy_arrays.values()),
}

dummy_points: Points = [
    Points(np.random.rand(1, 1)),
    Points(np.random.rand(30, 10)),
    Points(np.random.rand(1, 10)),
    Points(np.random.rand(10, 1)),
]

dist_points: Dict[str, Points] = {key: Points(arr) for key, arr in npy_dict["points"].items()}

rand_dist_matrices = [
    DistanceMatrix(np.random.rand(30, 30)),
    DistanceMatrix(np.random.rand(20, 10))
]

rand_dense_arrays = [
    DenseArray(np.random.rand(30, 30)),
    DenseArray(np.random.rand(20, 10))
]
