from typing import Dict, get_args
import numpy as np
import pytest
from src.geometry.matrix.laplacian import laplacian
from tests.test_utils import (
    affinity_sol,
    test_rtol,
    test_atol,
    npy_dict,
)
from src.object.geometry_matrix import LaplacianMatrix, LaplacianType


@pytest.mark.parametrize("key", affinity_sol.keys())
@pytest.mark.parametrize("type", get_args(LaplacianType))
def test__laplacian__output(key: str, type: str):
    type_dict = npy_dict[f"{type}_lap_sol"]
    if key in type_dict.keys():
        sym_lap = laplacian(affinity_sol[key], lap_type=type)
        assert np.allclose(type_dict[key], sym_lap.data, rtol=test_rtol, atol=test_atol)


