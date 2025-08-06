from typing import Any, Generic, Optional, Tuple, TypeAlias, Union

import numpy as np
from scipy.sparse import csr_array

from ..array.base import BaseArray
from ..array.dense import DenseArray
from ..array.sparse import SparseArray
from ..array.typing import ScalarTypeVar
from .metadata import AffinityType, DistanceType, LaplacianType
from .object import Object

MatrixStorage: TypeAlias = Union[
    csr_array[ScalarTypeVar, Tuple[int, int]],
    np.ndarray[Tuple[int, int], np.dtype[ScalarTypeVar]],
]
MatrixArray: TypeAlias = Union[
    DenseArray[ScalarTypeVar, Tuple[int, int]],
    SparseArray[ScalarTypeVar],
]


class GeometryMatrix(Generic[ScalarTypeVar], Object):
    data: MatrixArray[ScalarTypeVar]

    ndim = 2

    def __init__(
        self,
        data: Union[MatrixStorage[ScalarTypeVar], MatrixArray[ScalarTypeVar]],
        name: Optional[str] = None,
        **metadata: Any,
    ) -> None:

        if isinstance(data, BaseArray):
            super().__init__(data, name=name, **metadata)
        elif isinstance(data, np.ndarray):
            super().__init__(DenseArray(data), name=name, **metadata)
        elif isinstance(data, csr_array):
            super().__init__(SparseArray(data), name=name, **metadata)
        else:
            raise TypeError(f"Cannot format {type(data)} as a Geometry matrix!")

    @property
    def is_square(self) -> bool:
        return self.shape[0] == self.shape[1]

    @property
    def from_npts(self) -> int:
        return self.shape[0]

    @property
    def to_npts(self) -> int:
        return self.shape[1]

    @property
    def npts(self) -> int:
        if self.is_square:
            return self.from_npts
        raise ValueError("Matrix is not square, so `npts` is not well defined!")


class DistanceMatrix(GeometryMatrix[np.float64]):
    data: MatrixArray[np.float64]
    dtype = np.float64

    def __init__(
        self,
        data: Union[MatrixStorage[np.float64], MatrixArray[np.float64]],
        dist_type: Optional[DistanceType] = None,
        radius: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(data, name, dist_type=dist_type, radius=radius)


class AdjacencyMatrix(GeometryMatrix[np.bool_]):
    data: MatrixArray[np.bool_]
    dtype = np.bool_

    def __init__(
        self,
        data: Union[MatrixStorage[np.bool_], MatrixArray[np.bool_]],
        dist_type: Optional[DistanceType] = None,
        radius: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(data, name, dist_type=dist_type, radius=radius)


class AffinityMatrix(GeometryMatrix[np.float64]):
    data: MatrixArray[np.float64]
    dtype = np.float64

    def __init__(
        self,
        data: Union[MatrixStorage[np.float64], MatrixArray[np.float64]],
        dist_type: Optional[DistanceType] = None,
        aff_type: Optional[AffinityType] = None,
        radius: Optional[float] = None,
        eps: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            data, name, dist_type=dist_type, aff_type=aff_type, radius=radius, eps=eps
        )


class LaplacianMatrix(GeometryMatrix[np.float64]):
    data: MatrixArray[np.float64]
    eigvals: Optional[DenseArray[np.float64, Tuple[int]]]
    eigvecs: Optional[DenseArray[np.float64, Tuple[int]]]

    dtype = np.float64

    def __init__(
        self,
        data: Union[MatrixStorage[np.float64], MatrixArray[np.float64]],
        dist_type: Optional[DistanceType] = None,
        aff_type: Optional[AffinityType] = None,
        lap_type: Optional[LaplacianType] = None,
        radius: Optional[float] = None,
        eps: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            data,
            name,
            dist_type=dist_type,
            aff_type=aff_type,
            lap_type=lap_type,
            radius=radius,
            eps=eps,
        )
        self.eigvals = None
        self.eigvecs = None
