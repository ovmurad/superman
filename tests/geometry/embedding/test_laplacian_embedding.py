from typing import get_args

import numpy as np
import pytest
from src.geometry.embedding.laplacian_embedding import laplacian_embedding
from src.object.metadata import LaplacianType
from tests.test_utils import (
    affinity_sol,
    lap_sol,
    laplacian_embedding_sol,
    test_atol,
    test_rtol,
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("key", affinity_sol.keys())
def test__aff_laplacian_embedding__output(key: str):
    if key in laplacian_embedding_sol.keys():
        for eps in affinity_sol[key].keys():
            for lap_type in laplacian_embedding_sol[key].keys():
                embedding = laplacian_embedding(
                    affinity_sol[key][eps],
                    ncomp=30,
                    lap_type=lap_type,
                    eigen_solver="dense",
                    drop_first=True,
                )
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


@pytest.mark.parametrize("type", get_args(LaplacianType))
def test__lap_laplacian_embedding__output(type: str):
    type_dict = lap_sol[type]
    for key in type_dict.keys():
        for eps in type_dict[key].keys():
            if key in laplacian_embedding_sol.keys():
                embedding = laplacian_embedding(
                    type_dict[key][eps],
                    ncomp=30,
                    eigen_solver="dense",
                    drop_first=True,
                )
                sort_idx_emb = np.argsort(embedding[0])[::-1]
                sort_idx_sol = np.argsort(laplacian_embedding_sol[key][type][0])[::-1]
                assert np.allclose(
                    embedding[0][sort_idx_emb],
                    laplacian_embedding_sol[key][type][0][sort_idx_sol],
                    rtol=test_rtol,
                    atol=test_atol,
                )
                assert np.allclose(
                    np.abs(embedding[1][:, sort_idx_emb]),
                    np.abs(laplacian_embedding_sol[key][type][1][:, sort_idx_sol]),
                    rtol=test_rtol,
                    atol=test_atol,
                )
