from typing import get_args

import numpy as np
import pytest
from src.geometry.matrix.affinity import AffinityMatrix
from src.object.metadata import LaplacianType
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
        lap = affinity_sol[key].laplacian(type)
        assert np.allclose(type_dict[key], lap.as_nparray(), rtol=test_rtol, atol=test_atol)


@pytest.mark.parametrize("type", get_args(LaplacianType))
def test__laplacian__in_place_behavior(type: str):
    aff = AffinityMatrix(dense_square_float, eps=0.71)
    dummy_aff = AffinityMatrix(dense_square_float.copy(), eps=0.71)
    dummy_lap = dummy_aff.laplacian(type, in_place=False)
    assert np.allclose(dummy_aff.as_nparray(), aff.as_nparray(), rtol=test_rtol, atol=test_atol)
    dummy_aff.laplacian(type, in_place=True)
    assert np.allclose(dummy_lap.as_nparray(), dummy_aff.as_nparray(), rtol=test_rtol, atol=test_atol)
