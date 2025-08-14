
from typing import Dict
import numpy as np
import pytest

from src.geometry.matrix.affinity import adjacency, affinity
from src.object.geometry_matrix import AffinityMatrix, DistanceMatrix, MatrixArray

from tests.test_utils import npy_dict, test_rtol, test_atol

threshold_sol: Dict[str, DistanceMatrix] = {key: DistanceMatrix(arr, "euclidean") for key, arr in npy_dict["threshold_sol"].items()}
affinity_sol: Dict[str, AffinityMatrix] = {key: AffinityMatrix(arr) for key, arr in npy_dict["affinity_sol"].items()}
adj_test: DistanceMatrix = DistanceMatrix(np.array([[0, 0, 3.12], [2.0, 0, 1], [0, 5, 0]]), "euclidean")
adj_sol: MatrixArray[np.float64] = np.array([[False, False, True], [True, False, True], [False, True, False]])

eps = [3.67, 0.71, 0.57, 0.41]

@pytest.mark.parametrize("key", threshold_sol.keys())
def test__distance__single_points_no_radius_output(key: str):
    if key in affinity_sol.keys():
        aff = affinity(dist_mat=threshold_sol[key], eps=eps[1])
        assert np.allclose(aff.data, affinity_sol[key].data, rtol=test_rtol, atol=test_atol)
        
@pytest.mark.parametrize("key", threshold_sol.keys())
def test__distance__single_points_in_place_zero(key: str):
    if key in affinity_sol.keys():
        dummy_dist = threshold_sol[key]
        dummy_dist_threshold = affinity(dummy_dist, eps=eps[1], in_place=False)
        assert not np.shares_memory(dummy_dist.data, dummy_dist_threshold.data)
        affinity(dummy_dist, eps=eps[1], in_place=True)
        assert np.allclose(dummy_dist.data, dummy_dist_threshold.data, rtol=test_rtol, atol=test_atol)

def test__adjacency__output():
    assert adjacency(adj_test).data == adj_sol
