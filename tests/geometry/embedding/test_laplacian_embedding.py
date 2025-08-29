import numpy as np
import pytest
from src.geometry.matrix.distance import DistanceMatrix
from src.geometry.embedding.laplacian_embedding import laplacian_embedding
from tests.test_utils import (
    affinity_sol,
    test_atol,
    test_rtol,
    laplacian_embedding_sol,
)

pytestmark = pytest.mark.slow

@pytest.mark.parametrize("key", affinity_sol.keys())
def test__laplacian_embedding__output(key: str):
    if key in laplacian_embedding_sol.keys():
        for eps in affinity_sol[key].keys():
            for lap_type in laplacian_embedding_sol[key].keys():
                embedding = laplacian_embedding(affinity_sol[key][eps], ncomp=30, lap_type=lap_type, eigen_solver="dense", drop_first=True)
                assert np.allclose(
                    embedding[0],
                    laplacian_embedding_sol[key][lap_type][0],
                    rtol=test_rtol,
                    atol=test_atol,
                )
                assert np.allclose(
                    embedding[1],
                    laplacian_embedding_sol[key][lap_type][1],
                    rtol=test_rtol,
                    atol=test_atol,
                )
