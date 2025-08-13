from typing import Dict, Sequence
from src.array.base import BaseArray
from src.geometry.matrix.distance import distance, threshold_distance, threshold_distance_iter
from src.object.geometry_matrix import DistanceMatrix, MatrixArray
from tests.test_utils import dist_points, npy_dict, test_rtol, test_atol, rand_dense_arrays

import numpy as np
import pytest

pytestmark = pytest.mark.slow

single_dist_sol: Dict[str, DistanceMatrix] = {key: DistanceMatrix(arr) for key, arr in npy_dict["single_dist_sol"].items()}
double_dist_sol: Dict[str, DistanceMatrix] = {key: DistanceMatrix(arr) for key, arr in npy_dict["double_dist_sol"].items()}
threshold_sol: Dict[str, DistanceMatrix] = {key: DistanceMatrix(arr) for key, arr in npy_dict["threshold_sol"].items()}

radii: Sequence[float] = [2.13, 2.0, 1.8]

@pytest.mark.parametrize("key", dist_points.keys())
def test__distance__single_points_no_radius_output(key: str):
    if key in single_dist_sol.keys():
        assert np.allclose(distance(x_pts=dist_points[key], return_sp=False).data, single_dist_sol[key].data, rtol=test_rtol, atol=test_atol)

@pytest.mark.parametrize("key1", dist_points.keys())
@pytest.mark.parametrize("key2", dist_points.keys())
def test__distance__double_points_no_radius_output(key1: str, key2: str):
    key1_to_second_underscore = key1[:key1[:key1.find("_")].find("_")]
    key2_to_second_underscore = key2[:key2[:key2.find("_")].find("_")]
    if key1_to_second_underscore == key2_to_second_underscore and f"{key1} {key2}" in double_dist_sol.keys():
        assert np.allclose(
            distance(x_pts=dist_points[key1], y_pts=dist_points[key2], return_sp=False).data,
            double_dist_sol[f"{key1} {key2}"].data,
            rtol=test_rtol,
            atol=test_atol
        )

@pytest.mark.parametrize("dist_key", single_dist_sol.keys())
def test__threshold_distance__radius_output(dist_key: str):
    if dist_key in threshold_sol.keys():
        assert np.allclose(threshold_distance(single_dist_sol[dist_key], radii[0]).data, threshold_sol[dist_key].data, rtol=test_rtol, atol=test_atol)

@pytest.mark.parametrize("rand_arr", rand_dense_arrays)
def test__threshold_distance__in_place_behavior(rand_arr: MatrixArray):
    arr_mean = np.mean(rand_arr.data)
    dummy_dist = DistanceMatrix(rand_arr)
    dummy_dist_threshold = threshold_distance(dummy_dist, arr_mean)
    assert not np.shares_memory(dummy_dist.data, dummy_dist_threshold.data)
    threshold_distance(dummy_dist, arr_mean, True)
    assert np.allclose(dummy_dist.data, dummy_dist_threshold.data, rtol=test_rtol, atol=test_atol)

"""
@pytest.mark.parametrize("radius", radii)
def test_threshold_distance_iter__radii_output(radius: float): 
"""