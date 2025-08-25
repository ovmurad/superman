import numpy as np
import pytest
from tests.test_utils import (
    adj_sol,
    adj_test,
    affinity_sol,
    test_atol,
    test_rtol,
    threshold_sol_radius,
)

pytestmark = pytest.mark.slow

eps = [3.67, 0.71, 0.57, 0.41]


@pytest.mark.parametrize("key", threshold_sol_radius.keys())
def test__affinity__single_points_no_radius_output(key: str):
    if key in affinity_sol.keys():
        for radius in threshold_sol_radius[key].keys():
            aff = threshold_sol_radius[key][radius].affinity(eps=radius / 3)
            assert np.allclose(
                aff.as_nparray(), affinity_sol[key].as_nparray(), rtol=test_rtol, atol=test_atol
            )


@pytest.mark.parametrize("key", threshold_sol_radius.keys())
def test__affinity__single_points_in_place_zero(key: str):
    if key in affinity_sol.keys():
        for radius in threshold_sol_radius[key].keys():
            dummy_dist = threshold_sol_radius[key][radius]
            dummy_dist_threshold = dummy_dist.affinity(eps=radius / 3, in_place=False)
            assert not np.shares_memory(dummy_dist.as_nparray(), dummy_dist_threshold.as_nparray())
            dummy_dist.affinity(eps=radius / 3, in_place=True)
            assert np.allclose(
                dummy_dist.as_nparray(),
                dummy_dist_threshold.as_nparray(),
                rtol=test_rtol,
                atol=test_atol,
            )


def test__adjacency__output():
    assert adj_test.adjacency() == adj_sol
