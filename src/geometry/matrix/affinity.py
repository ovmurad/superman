from typing import Optional

import numpy as np

from ...object.geometry_matrix import (
    AdjacencyMatrix,
    AffinityMatrix,
    DistanceMatrix,
    MatrixArray,
)
from ...object.metadata import AffinityType


def _de_affinity_in_place(
        dists: MatrixArray,
        eps: float,
        dist_is_sq: bool
) -> MatrixArray[np.float64]:
    #

    if not dist_is_sq:
        np.power(dists, 2, out=dists)

    np.divide(dists, (eps**2), out=dists)
    np.multiply(dists, -1, out=dists)
    np.exp(dists, out=dists)

    return dists


def _de_affinity_out_of_place(
        dists: MatrixArray,
        eps: float,
        dist_is_sq: bool
) -> MatrixArray[np.float64]:

    if dist_is_sq:
        return np.exp(-(dists / (eps**2)))
    return np.exp(-((dists / eps) ** 2))


# TODO
def adjacency(
    dist_mat: DistanceMatrix,
    copy: bool = False,
) -> AdjacencyMatrix:
    """
    Will simply cast to dist_mat.data to bool. Same as neigh in func_of_dist.py in cryo_experiments
    :param dist_mat:
    :param copy: Whether to copy the indices of dist_mat.
    :return:
    """
    adj_data: MatrixArray[np.bool_] = ...  # type: ignore
    return AdjacencyMatrix(
        adj_data,
        dist_mat.metadata.dist_type,
        dist_mat.metadata.radius,
        dist_mat.metadata.name,
    )


def affinity(
    dist_mat: DistanceMatrix,
    aff_type: AffinityType = "gaussian",
    eps: Optional[float] = None,
    in_place: bool = False,
) -> AffinityMatrix:
    """
    Corresponds to affinity.py in cryo_experiments. Can deduce dist_is_sq from the dist_type and set to default
    False if distance is not 'sqeuclidean'. All distances can be found at
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
    :param dist_mat:
    :param aff_type:
    :param eps:
    :param in_place:
    :return:
    """

    dist_is_sq = dist_mat.metadata.dist_type == "sqeuclidean"
    dist_data = dist_mat.data

    if in_place:
        aff_data = _de_affinity_in_place(dist_data, eps, dist_is_sq)

    aff_data: MatrixArray[np.float64] = _de_affinity_out_of_place(dist_data, eps, dist_is_sq)

    return AffinityMatrix(
        aff_data,
        dist_mat.metadata.dist_type,
        aff_type,
        dist_mat.metadata.radius,
        eps,
        dist_mat.metadata.name,
    )
