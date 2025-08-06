from typing import Iterable, Optional, Union

import numpy as np

from ..array.base import DenseArray
from ..object.function import Degree, DegreeType
from ..object.geometry_matrix import AdjacencyMatrix, AffinityMatrix, DistanceType

NeighborMatrix = Union[AffinityMatrix, AdjacencyMatrix]


def degree(
    neigh_mats: Union[NeighborMatrix, Iterable[NeighborMatrix]],
    axis: int = 1,
) -> Degree:
    """
    This will take in an iterator of adjacency or affinity matrices and will sum over `axis` to obtain the degrees.
    All `neigh_mats` have to be of the same type and they all correspond to a certain threshold radius which has to be
    set for each of them. The iterator could be something that repeatedly thresholds a distance matrix followed
    by computing either the affinity or adjacency at that radius
    :param neigh_mats:
    :return:
    """
    if isinstance(neigh_mats, (AffinityMatrix, AdjacencyMatrix)):
        neigh_mats = (neigh_mats,)

    degrees = list[DenseArray[np.float64]]()
    radii = list[Optional[float]]()

    dist_types = set[Optional[DistanceType]]()
    degree_types = set[DegreeType]()
    names = set[Optional[str]]()

    for neigh_mat in neigh_mats:
        dist_types.add(neigh_mat.dist_type)
        degree_types.add(
            ("adjacency" if isinstance(neigh_mat, AdjacencyMatrix) else "affinity")
        )
        names.add(neigh_mat.name)

        radii.append(neigh_mat.radius)

        degree_data: DenseArray[np.float64] = ...
        degrees.append(degree_data)

    if len(dist_types) > 1:
        raise ValueError("More than one type of distance type found!")
    dist_type = dist_types.pop()

    if not degree_types:
        raise ValueError("No neighbor matrices were provided!")
    elif len(degree_types) > 1:
        raise ValueError("More than one type of degree type found!")
    degree_type = degree_types.pop()

    if len(names) > 1:
        raise ValueError("More than one name found!")
    name = names.pop()

    return Degree(np.stack(degrees, axis=1), dist_type, degree_type, radii, name)
