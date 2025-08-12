from src.geometry.matrix.distance import distance
from tests.test_objects import dist_points, single_dist_sol, test_rtol, test_atol

import numpy as np
import pytest

pytestmark = pytest.mark.slow

def test__distance__single_points_no_radius_correct_output():
    for key in dist_points.keys():
        assert np.allclose(distance(x_pts=dist_points[key], return_sp=False).data, single_dist_sol[key].data, rtol=test_rtol, atol=test_atol)

