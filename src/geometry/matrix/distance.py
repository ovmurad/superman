from typing import Iterable, Iterator, Optional, Union

import numpy as np
from sklearn.metrics import pairwise_distances_chunked

from ...object.geometry_matrix import DistanceMatrix, MatrixArray
from ...object.metadata import DistanceType
from ...object.points import Points
from ...array.dense import DenseArray
from ...array.sparse import SparseArray

from threshold import threshold

def distance(
    x_pts: Points,
    y_pts: Optional[Points] = None,
    dist_type: DistanceType = "euclidean",
    radius: Optional[float] = None,
    return_sp: bool = True,
) -> DistanceMatrix:
    """
    Corresponds to dist in func_of_dist in the cryo_experiments. Will change the structure a bit because that
    was a bit complex without much time saved. For now, we will just implement batched distance computation
    without adding other functions on top like in func_of_dist
    :param x_pts:
    :param y_pts:
    :param dist_type:
    :param radius:
    :param return_sp:
    :return:
    """
    #chunked pairwise distance
    #ndarray boolean mask

    if return_sp:
        raise NotImplementedError()
    else:
        dist_data: MatrixArray[np.float64] = DenseArray(np.fromiter(pairwise_distances_chunked(x_pts, y_pts), dtype=np.float64))

        if radius is not None:
            threshold(dist_data, radius, True)

    x_pts_name = x_pts.metadata.name
    y_pts_name = None if y_pts is None else y_pts.metadata.name
    dist_name = (
        None
        if (x_pts_name is None or y_pts_name is None)
        else x_pts_name + "_" + y_pts_name
    )

    return DistanceMatrix(dist_data, dist_type, radius, dist_name)


# TODO
def threshold_distance(
    dist_mat: DistanceMatrix,
    radius: float,
    in_place: bool = False,
) -> DistanceMatrix:
    """
    :param dist_mat:
    :param radius:
    :param in_place:
    :return:
    """

    dist_mat_radius = dist_mat.metadata.radius
    if dist_mat_radius is not None and radius > dist_mat_radius:
        raise ValueError(
            f"`radius`={radius} is greater than the radius of the input distance matrix {dist_mat_radius}!"
        )

    dist_data: MatrixArray[np.float64] = ...  # type: ignore

    return DistanceMatrix(
        dist_data, dist_mat.metadata.dist_type, radius, dist_mat.metadata.name
    )


# TODO
def threshold_distance_iter(
    dist_mat: DistanceMatrix,
    radii: Union[float, Iterable[float]],
    in_place: bool = False,
) -> Iterator[DistanceMatrix]:
    """
    Take a distance matrix and eliminate entries at the new radii. Note that we return an iterator
    that could be used by other functions down stream without storing all matrices.
    :param dist_mat:
    :param radii:
    :param in_place:
    :return:
    """

    if isinstance(radii, float):
        radii = (radii,)

    for radius in reversed(sorted(radii)):
        yield threshold_distance(dist_mat, radius, in_place)
