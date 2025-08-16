from typing import get_args

import numpy as np
import pytest
from src.geometry.matrix.laplacian import laplacian
from src.object.geometry_matrix import AffinityMatrix, LaplacianType
from tests.test_utils import (
    affinity_sol,
    dense_square_float,
    npy_dict,
    test_atol,
    test_rtol,
)


@pytest.mark.parametrize("key", affinity_sol.keys())
@pytest.mark.parametrize("type", get_args(LaplacianType))
def test__laplacian__output(key: str, type: str):
    type_dict = npy_dict[f"{type}_lap_sol"]
    if key in type_dict.keys():
        sym_lap = laplacian(affinity_sol[key], lap_type=type)
        assert np.allclose(type_dict[key], sym_lap.data, rtol=test_rtol, atol=test_atol)


@pytest.mark.parametrize("type", get_args(LaplacianType))
def test__laplacian__in_place_behavior(type: str):
    aff = AffinityMatrix(dense_square_float, eps=0.71)
    dummy_aff = AffinityMatrix(np.copy(dense_square_float), eps=0.71)
    dummy_lap = laplacian(dummy_aff, lap_type=type, in_place=False)
    assert np.allclose(dummy_aff.data, aff.data, rtol=test_rtol, atol=test_atol)
    laplacian(dummy_aff, lap_type=type, in_place=True)
    assert np.allclose(dummy_lap.data, dummy_aff.data, rtol=test_rtol, atol=test_atol)
