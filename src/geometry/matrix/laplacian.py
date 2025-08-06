import numpy as np

from ...object.geometry_matrix import (
    AffinityMatrix,
    LaplacianMatrix,
    LaplacianType,
    MatrixArray,
)


# TODO
def laplacian(
    aff_mat: AffinityMatrix,
    lap_type: LaplacianType = "geometric",
    in_place: bool = False,
) -> LaplacianMatrix:
    """
    Corresponds to laplacian.py in cryo_experiments. Note that eps will be store in aff_mat.metadata. If eps
    is None, raise Error, so can ignore the branches in the previous code where eps is None. I don't think aff_minus_id
    is ever set to False, so ignore that branch as well.
    Papers:
        - geometric and symmetric: https://www.sciencedirect.com/science/article/pii/S1063520306000546
        - random walk: https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf

    :param aff_mat:
    :param lap_type:
    :return:
    """

    eps = aff_mat.metadata.eps
    if eps is None:
        raise ValueError("Affinity matrix does not have a bandwidth `eps`!")

    lap_arr: MatrixArray[np.float64] = ...  # type: ignore

    return LaplacianMatrix(
        lap_arr,
        aff_mat.metadata.dist_type,
        aff_mat.metadata.aff_type,
        lap_type,
        aff_mat.metadata.radius,
        eps,
        aff_mat.metadata.name,
    )
