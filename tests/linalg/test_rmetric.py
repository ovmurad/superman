import numpy as np
import pytest
from tests.test_utils import (
    rmetric_sol,
    dist_points,
    double_dist_sol,
    rand_dense_arrays,
    single_dist_sol,
    test_atol,
    test_rtol,
    threshold_iter_sol,
    threshold_sol_radius,
    lap_sol,
)
from src.linalg.rmetric import RMetric

@pytest.mark.parametrize("key", dist_points.keys())
def test__local__output_30x30(key: str):
    if key in rmetric_sol.keys() and key in lap_sol.keys():
        rm = RMetric.local(dist_points[key], lap_sol[key][0.71], ncomp = 15, bsize = 10)
        assert np.allclose(rm.eigenvalues.as_nparray(), rmetric_sol[key]["eigval"], rtol=test_rtol, atol=test_atol)
        assert np.allclose(rm.eigenvectors.as_nparray(), rmetric_sol[key]["eigvec"], rtol=test_rtol, atol=test_atol)
