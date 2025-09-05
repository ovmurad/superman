import os
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar

import numpy as np
from scipy.sparse import csr_array
from src.array.dense import DenseArray
from src.geometry.matrix.affinity import AffinityMatrix
from src.geometry.matrix.distance import DistanceMatrix
from src.geometry.matrix.laplacian import LaplacianMatrix
from src.geometry.points import Points

from tests.test_array.dummy_array import DummyArray


def load_test_npy():
    data = defaultdict(dict)

    for root, _, files in os.walk("./tests/test_data"):
        for f in files:
            if f.endswith(".npy") and "-" in f:
                prefix = f.split("-")[0]
                if f.startswith(prefix + "-"):
                    file_key = f[f.find("-") + 1 : f.find(".npy")]
                    full_path = os.path.join(root, f)
                    data[prefix][file_key] = np.load(full_path, allow_pickle=True)
    return data

def default_process_args(k: Any, d: Any, **kwargs)-> Dict[str, Any]:
    return kwargs

T = TypeVar("T")
def load_metadata_dict(name: str, cls: Callable[..., Any], ret: Type[T], process_args: Callable[..., Any] = default_process_args, **kwargs: Any) -> Dict[str, Dict[str, T]]:
    if cls is None:
        return {key: {k: d for k, d in dict.item().items()} for key, dict in npy_dict[name].items()}
    return {key: {k: cls(d, **process_args(k, d, **kwargs)) for k, d in dict.item().items()} for key, dict in npy_dict[name].items()}

def aff_process_args(k: Any, d: Any, **kwargs) -> Dict[str, Any]:
    kwargs["eps"] = k
    return kwargs

def lap_process_args(k: Any, d: Any, **kwargs) -> Dict[str, Any]:
    kwargs["eps"] = k
    return kwargs

def load_dict(name: str, cls: Callable[..., Any], ret: Type[T]) -> Dict[str, T]:
    if cls is None:
        return {key: arr for key, arr in npy_dict[name].items()}
    return {key: cls(arr) for key, arr in npy_dict[name].items()}

npy_dict: Dict[str, Dict[str, np.ndarray]] = load_test_npy()

laplacian_embedding_sol: Dict[str, Dict[str, np.ndarray]] = load_metadata_dict("laplacian_embedding_sol", None, np.ndarray)
affinity_sol: Dict[str, Dict[str, AffinityMatrix]] = load_metadata_dict("affinity_sol", AffinityMatrix.create, AffinityMatrix, aff_process_args)
adj_test: DistanceMatrix = DistanceMatrix(
    np.array([[0, 0, 3.12], [2.0, 0, 1], [0, 5, 0]]), dist_type="euclidean"
)
adj_sol: DenseArray[np.float64] = DenseArray(
    [[False, False, True], [True, False, True], [False, True, False]]
)
single_dist_sol: Dict[str, DistanceMatrix] = load_dict("single_dist_sol", DistanceMatrix.create, DistanceMatrix)
double_dist_sol: Dict[str, DistanceMatrix] = load_dict("double_dist_sol", DistanceMatrix.create, DistanceMatrix)
threshold_sol_radius: Dict[str, Dict[str, DistanceMatrix]] = load_metadata_dict("threshold_sol", DistanceMatrix.create, DistanceMatrix, dist_type="euclidean")
threshold_iter_sol: Dict[str, DenseArray[np.float64]] = load_dict("threshold_iter_sol", None, np.ndarray)
geometric_lap_sol: Dict[str, Dict[str, LaplacianMatrix]] = load_metadata_dict("geometric_lap_sol", LaplacianMatrix.create, LaplacianMatrix, lap_process_args, lap_type="geometric", aff_minus_id=True)
random_walk_lap_sol: Dict[str, Dict[str, LaplacianMatrix]] = load_metadata_dict("random_walk_lap_sol", LaplacianMatrix.create, LaplacianMatrix, lap_process_args, lap_type="random_walk", aff_minus_id=True)
symmetric_lap_sol: Dict[str, Dict[str, LaplacianMatrix]] = load_metadata_dict("symmetric_lap_sol", LaplacianMatrix.create, LaplacianMatrix, lap_process_args, lap_type="symmetric", aff_minus_id=True)
lap_sol = {
    "geometric": geometric_lap_sol,
    "random_walk": random_walk_lap_sol,
    "symmetric": symmetric_lap_sol,
}
dense_square_float: DenseArray[np.float64] = DenseArray(np.random.rand(30, 30))


test_rtol = 1e-5
test_atol = 1e-8

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

dist_points: Dict[str, Points] = {
    key: Points(arr) for key, arr in npy_dict["points"].items()
}

rand_dist_matrices = [
    DistanceMatrix(np.random.rand(30, 30)),
    DistanceMatrix(np.random.rand(20, 10)),
]

rand_dense_arrays = [
    DenseArray(np.random.rand(30, 30)),
    DenseArray(np.random.rand(20, 10)),
]
