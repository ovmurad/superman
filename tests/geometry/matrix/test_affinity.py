
from typing import Dict
import numpy as np
import pytest

from src.geometry.matrix.affinity import affinity
from src.object.geometry_matrix import AffinityMatrix, DistanceMatrix

from tests.test_utils import npy_dict, test_rtol, test_atol

threshold_sol: Dict[str, DistanceMatrix] = {key: DistanceMatrix(arr, "euclidean") for key, arr in npy_dict["threshold_sol"].items()}
affinity_sol: Dict[str, AffinityMatrix] = {key: AffinityMatrix(arr) for key, arr in npy_dict["affinity_sol"].items()}

eps = [0.71, 0.41, 3.67, 0.57]

@pytest.mark.parametrize("key", threshold_sol.keys())
def test__distance__single_points_no_radius_output(key: str):
    if key in affinity_sol.keys():
        aff = affinity(dist_mat=threshold_sol[key], eps=eps[0])
        assert np.allclose(aff.data, affinity_sol[key].data, rtol=test_rtol, atol=test_atol)