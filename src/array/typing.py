from typing import TypeAlias, TypeVar

import numpy as np
from numpy._typing._array_like import (
    _ArrayLikeBool_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
)
from numpy.typing import NDArray
from scipy.sparse import csr_array

# numpy type vars
_DTypeBound: TypeAlias = np.bool_ | np.integer | np.floating
_DType = TypeVar("_DType", bound=_DTypeBound)
_DType_ = TypeVar("_DType_", bound=_DTypeBound)

# types of inputs for data and index parsing via numpy functions such as np.asarray, np.broadcast
_BoolDataLike: TypeAlias = _ArrayLikeBool_co
_IntDataLike: TypeAlias = _ArrayLikeInt_co
_FloatDataLike: TypeAlias = _ArrayLikeFloat_co
_DataLike = TypeVar("_DataLike", _BoolDataLike, _IntDataLike, _FloatDataLike)

_BoolData: TypeAlias = NDArray[np.bool_]
_IntData: TypeAlias = NDArray[np.integer]
_FloatData: TypeAlias = NDArray[np.floating]
_Data: TypeAlias = NDArray[_DType]

_IdxLike: TypeAlias = _ArrayLikeInt_co
_Idx: TypeAlias = NDArray[np.integer]

# Types of arrays that can be wrapped by Array subclasses
_Array = TypeVar("_Array", bound=NDArray[_DTypeBound] | csr_array[_DTypeBound])
