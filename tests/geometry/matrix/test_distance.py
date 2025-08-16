from typing import Sequence

import numpy as np
import pytest
from src.geometry.matrix.distance import (
    distance,
    threshold_distance,
    threshold_distance_iter,
)
from src.object.geometry_matrix import DistanceMatrix, MatrixArray
from tests.test_utils import (
    dist_points,
    double_dist_sol,
    rand_dense_arrays,
    single_dist_sol,
    test_atol,
    test_rtol,
    threshold_iter_sol,
    threshold_sol,
)

pytestmark = pytest.mark.slow

radii: Sequence[float] = [2.13, 1.23, 11.0, 1.72]


@pytest.mark.parametrize("key", dist_points.keys())
def test__distance__single_points_no_radius_output(key: str):
    if key in single_dist_sol.keys():
        assert np.allclose(
            distance(x_pts=dist_points[key], return_sp=False).data,
            single_dist_sol[key].data,
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
            distance(
                x_pts=dist_points[key1], y_pts=dist_points[key2], return_sp=False
            ).data,
            double_dist_sol[f"{key1} {key2}"].data,
            rtol=test_rtol,
            atol=test_atol,
        )


@pytest.mark.parametrize("dist_key", single_dist_sol.keys())
def test__threshold_distance__radius_output(dist_key: str):
    if dist_key in threshold_sol.keys():
        assert np.allclose(
            threshold_distance(single_dist_sol[dist_key], radii[0]).data,
            threshold_sol[dist_key].data,
            rtol=test_rtol,
            atol=test_atol,
        )


@pytest.mark.parametrize("rand_arr", rand_dense_arrays)
def test__threshold_distance__in_place_behavior(rand_arr: MatrixArray):
    arr_mean = np.mean(rand_arr.data)
    dummy_dist = DistanceMatrix(rand_arr)
    dummy_dist_threshold = threshold_distance(dummy_dist, arr_mean)
    assert not np.shares_memory(dummy_dist.data, dummy_dist_threshold.data)
    threshold_distance(dummy_dist, arr_mean, True)
    assert np.allclose(
        dummy_dist.data, dummy_dist_threshold.data, rtol=test_rtol, atol=test_atol
    )


@pytest.mark.parametrize("key", single_dist_sol.keys())
def test__threshold_distance_iter__radii_single_output(key: str):
    if key in threshold_iter_sol.keys():
        arr = np.array(
            [
                d_mat.data
                for d_mat in threshold_distance_iter(
                    dist_mat=single_dist_sol[key], radii=radii, in_place=False
                )
            ]
        )
        assert np.allclose(arr, threshold_iter_sol[key], rtol=test_rtol, atol=test_atol)
