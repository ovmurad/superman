from typing import Sequence

import numpy as np
import pytest
from src.array import DenseArray
from src.geometry.matrix import DistanceMatrix
from tests.test_utils import (
    dist_points,
    double_dist_sol,
    rand_dense_arrays,
    single_dist_sol,
    test_atol,
    test_rtol,
    threshold_iter_sol,
    threshold_sol_radius,
)

pytestmark = pytest.mark.slow

radii: Sequence[float] = [2.13, 1.23, 11.0, 1.72]


@pytest.mark.parametrize("key", dist_points.keys())
def test__distance__single_points_no_radius_output(key: str):
    if key in single_dist_sol.keys():
        assert np.allclose(
            dist_points[key].pairwise_distance().as_nparray(),
            single_dist_sol[key].as_nparray(),
            rtol=test_rtol,
            atol=test_atol,
        )


@pytest.mark.parametrize("key1", dist_points.keys())
@pytest.mark.parametrize("key2", dist_points.keys())
def test__distance__double_points_no_radius_output(key1: str, key2: str):
    key1_to_second_underscore = key1[: key1[: key1.find("_")].find("_")]
    key2_to_second_underscore = key2[: key2[: key2.find("_")].find("_")]
    if (
        key1_to_second_underscore == key2_to_second_underscore
        and f"{key1} {key2}" in double_dist_sol.keys()
    ):
        assert np.allclose(
            dist_points[key1].pairwise_distance(dist_points[key2]).as_nparray(),
            double_dist_sol[f"{key1} {key2}"].as_nparray(),
            rtol=test_rtol,
            atol=test_atol,
        )


@pytest.mark.parametrize("dist_key", single_dist_sol.keys())
def test__threshold_distance__radius_output(dist_key: str):
    if dist_key in threshold_sol_radius.keys():
        for radius in threshold_sol_radius[dist_key].keys():
            assert np.allclose(
                single_dist_sol[dist_key].threshold(radius).as_nparray(),
                threshold_sol_radius[dist_key][radius].as_nparray(),
                rtol=test_rtol,
                atol=test_atol,
            )


@pytest.mark.parametrize("rand_arr", rand_dense_arrays)
def test__threshold_distance__in_place_behavior(rand_arr: DenseArray[np.float64]):
    arr_mean = rand_arr.mean().as_nparray()
    dummy_dist = DistanceMatrix(rand_arr)
    dummy_dist_threshold = dummy_dist.threshold(arr_mean)
    assert not np.shares_memory(
        dummy_dist.as_nparray(), dummy_dist_threshold.as_nparray()
    )
    dummy_dist.threshold(arr_mean, True)
    assert np.allclose(
        dummy_dist.as_nparray(),
        dummy_dist_threshold.as_nparray(),
        rtol=test_rtol,
        atol=test_atol,
    )


@pytest.mark.parametrize("key", single_dist_sol.keys())
def test__threshold_distance_iter__radii_single_output(key: str):
    if key in threshold_iter_sol.keys():
        arr = np.array(
            [
                d_mat.as_nparray()
                for d_mat in single_dist_sol[key].threshold_distance_iter(radii)
            ]
        )
        assert np.allclose(arr, threshold_iter_sol[key], rtol=test_rtol, atol=test_atol)
