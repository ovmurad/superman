import numpy as np

from ...object.geometry_matrix import MatrixArray


def threshold(
    arr: MatrixArray[np.float64], radius: float, in_place: bool = False
) -> MatrixArray[np.float64]:
    if in_place:
        arr[arr > radius] = np.inf
        return arr
    return np.where(arr > radius, np.inf, arr)
