from __future__ import annotations

from abc import ABC
from typing import Any

from src.array import BaseArray, CsrArray, DenseArray
from src.geometry.matrix.laplacian import LaplacianMatrix, eps_adjustment
from src.geometry.normalize import normalize
from src.object import GeometryMatrixMixin, LaplacianType


class AffinityMatrixMixin(GeometryMatrixMixin, ABC):
    """
    Mixin class that adds factory functionality to the AffinityMatrix
    hierarchy.

    This allows users to work with `AffinityMatrix` as an abstract
    entry point without needing to explicitly choose between dense or
    sparse representations.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        """
        Factory constructor for AffinityMatrix subclasses.

        The constructor returns an instance of either `DenseAffinityMatrix` or `CsrAffinityMatrix` if constructed in `DenseArray` format or `CsrArray` format respectively.

        :param args: Positional arguments forwarded to the chosen
                     affinity matrix subclass.
        :type args: Any
        :param kwargs: Keyword arguments forwarded to the chosen
                       affinity matrix subclass.
        :type kwargs: Any
        :return: A new `DenseAffinityMatrix` or `CsrAffinityMatrix`
                 instance.
        :rtype: AffinityMatrix
        """
        if cls is AffinityMatrix:
            if "shape" in kwargs:
                return CsrAffinityMatrix(*args, **kwargs)
            return DenseAffinityMatrix(*args, **kwargs)
        return super().__new__(cls)


class AffinityMatrix(AffinityMatrixMixin, BaseArray, ABC):
    def laplacian(
        aff: AffinityMatrix,
        lap_type: LaplacianType = "geometric",
        diag_add: float = 1.0,
        aff_minus_id: bool = True,
        in_place: bool = False,
    ) -> LaplacianMatrix:
        """
        Construct a graph Laplacian from an affinity matrix using the specified type. Uses aff's eps metadata for scaling.
        Papers:
            - geometric and symmetric: https://www.sciencedirect.com/science/article/pii/S1063520306000546
            - random walk: https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf


        :param affs: Affinity matrix.
        :param eps: Optional scaling parameter. If provided, the resulting Laplacian is multiplied by 4 / eps^2. (default: None).
        :param lap_type: Type of Laplacian to compute. (default: "geometric").
                        - "geometric": normalized symmetric followed by random-walk-like normalization.
                        - "random_walk": standard random walk Laplacian.
                        - "symmetric": symmetric normalized Laplacian.
        :param diag_add: Value to add to the diagonal after constructing the Laplacian.
                        If `aff_minus_id` is True, this value is negated before being added.
        :param aff_minus_id: Whether to subtract the identity matrix from the affinity matrix. (default: True)
                            If True, computes `I - normalized_affs`. If False, negates the Laplacian directly.
        :param in_place: Whether to modify the input affinity matrix in place during normalization. (default: False).

        :return: The constructed Laplacian matrix.
        """

        eps = aff.metadata.eps
        if eps is None:
            raise ValueError("Affinity matrix does not have a bandwidth `eps`!")

        match lap_type:
            case "geometric":
                aff = normalize(aff, sym_norm=True, in_place=in_place)
                lap: LaplacianMatrix = LaplacianMatrix(
                    normalize(aff, sym_norm=False, in_place=True), lap_type="geometric"
                )
            case "random_walk":
                lap: LaplacianMatrix = LaplacianMatrix(
                    normalize(aff, sym_norm=False, in_place=in_place),
                    lap_type="random_walk",
                )
            case "symmetric":
                lap: LaplacianMatrix = LaplacianMatrix(
                    normalize(aff, sym_norm=True, in_place=in_place, degree_exp=0.5),
                    lap_type="symmetric",
                )
            case _:
                raise ValueError(f"Unknown laplacian type: {lap_type}!")

        if not aff_minus_id:
            lap *= -1.0
        else:
            diag_add *= -1.0

        lap.fill_diagonal(lap.diagonal() + diag_add)

        lap *= eps_adjustment(eps)

        return lap


class DenseAffinityMatrix(AffinityMatrix, DenseArray):
    """
    Implementation of a dense (NumPy-backed) affinity matrix.

    This class represents an affinity matrix stored in dense format,
    providing fast element-wise operations at the cost of memory usage.

    Typically not instantiated directly: instead, construct an
    `AffinityMatrix` in `DenseArray` format which will return an instance.
    """

    pass


class CsrAffinityMatrix(AffinityMatrix, CsrArray):
    """
    Implementation of a sparse (Csr-backed) affinity matrix.

    This class represents an affinity matrix stored in sparse format,
    providing fast element-wise operations at the cost of memory usage.

    Typically not instantiated directly: instead, construct an
    `AffinityMatrix` in `CsrArray` format which will return an instance.
    """

    pass
