# import numpy as np
#
# from ...object.geometry_matrix import AffinityMatrix, LaplacianType, LaplacianMatrix, MatrixArray
#
# def laplacian_embedding(
#     lap: LaplacianMatrix,
#     ncomp: int,
#     eigen_solver: EigenSolver = "amg",
#     drop_first: bool = True,
#     check_connected: bool = True,
#     in_place: bool = False,
#     **kwargs,
# ) -> Tuple[RealDeArr, RealDeArr]:
