import numpy as np
import pytest
from src.geometry.matrix.affinity import adjacency, affinity
from tests.test_utils import (
    adj_sol,
    adj_test,
    affinity_sol,
    test_atol,
    test_rtol,
    threshold_sol,
)

pytestmark = pytest.mark.slow

eps = [3.67, 0.71, 0.57, 0.41]


@pytest.mark.parametrize("key", threshold_sol.keys())
def test__distance__single_points_no_radius_output(key: str):
    if key in affinity_sol.keys():
        aff = affinity(dist_mat=threshold_sol[key], eps=eps[1])
        assert np.allclose(
            aff.data, affinity_sol[key].data, rtol=test_rtol, atol=test_atol
        )


@pytest.mark.parametrize("key", threshold_sol.keys())
def test__distance__single_points_in_place_zero(key: str):
    if key in affinity_sol.keys():
        dummy_dist = threshold_sol[key]
        dummy_dist_threshold = affinity(dummy_dist, eps=eps[1], in_place=False)
        assert not np.shares_memory(dummy_dist.data, dummy_dist_threshold.data)
        affinity(dummy_dist, eps=eps[1], in_place=True)
        assert np.allclose(
            dummy_dist.data, dummy_dist_threshold.data, rtol=test_rtol, atol=test_atol
        )


def test__adjacency__output():
    assert adjacency(adj_test).data == adj_sol
