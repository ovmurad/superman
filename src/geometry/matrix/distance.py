from typing import Iterable, Iterator, Optional, Union

import numpy as np
from sklearn.metrics import pairwise_distances_chunked

from src.array.dense import DenseArray

from ...object.geometry_matrix import DistanceMatrix, MatrixArray
from ...object.metadata import DistanceType
from ...object.points import Points

from .threshold import threshold

def row_chunks_to_matrix(chunks: Iterator[np.ndarray], n_x: int, n_y: int) -> MatrixArray[np.float64]:
    full_dist_matrix = np.zeros((n_x, n_y))
    start_idx = 0

    for chunk in chunks:
        chunk_rows = chunk.shape[0]
        full_dist_matrix[start_idx:start_idx + chunk_rows, :] = chunk
        start_idx += chunk_rows

    return full_dist_matrix

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
    
    n_x = x_pts.npts
    n_y = n_x

    if y_pts is not None:
        n_y = y_pts.npts

    if return_sp:
        raise NotImplementedError()
    else:
        dist_data: MatrixArray[np.float64] = row_chunks_to_matrix(pairwise_distances_chunked(x_pts.data, y_pts if y_pts is None else y_pts.data), n_x, n_y)
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


def _threshold_de(data: MatrixArray[np.float64], radius: float, in_place: bool) -> MatrixArray[np.float64]:
    if in_place:
        data.storage[data.storage > radius] = np.inf
        return data
    return DenseArray(np.where(data.storage > radius, np.inf, data.storage))


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

    if dist_mat.data.is_sparse:
        raise NotImplementedError()
    else:
        dist_data: MatrixArray[np.float64] = _threshold_de(dist_mat.data, radius, in_place)

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
