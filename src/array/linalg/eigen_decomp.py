from typing import Literal, Optional, TypeAlias

import numpy as np

from ..typing import Storage

EigenSolver: TypeAlias = Literal["dense", "arpack", "lobpcg", "amg"]

SYM_EIGEN_SOLVERS: frozenset[EigenSolver] = frozenset({"amg", "lobpcg"})
NON_SYM_EIGEN_SOLVERS: frozenset[EigenSolver] = frozenset({"dense", "arpack"})


# TODO
def eigen_decomp(
    arr: Storage[np.floating, tuple[int, int, *tuple[int, ...]]],
    ncomp: Optional[int] = None,
    eigen_solver: EigenSolver = "dense",
    is_symmetric: bool = False,
    largest: bool = True,
) -> tuple[
    Storage[np.floating, tuple[int, *tuple[int, ...]]],
    Storage[np.floating, tuple[int, int, *tuple[int, ...]]],
]:
    """
    This is in linalg eigen_decomp. Returns (eigvals, eigvecs)
    :param arr:
    :param ncomp:
    :param eigen_solver:
    :param is_symmetric:
    :param largest:
    :return:
    """
    ...
