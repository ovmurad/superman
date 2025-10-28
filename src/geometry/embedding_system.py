from __future__ import annotations
from typing import Type

from src.array.dense.dense import DenseArray
from src.geometry.points import Embedding
from src.object import Eigen


class EmbeddingSystem(Eigen):
    """
    This class represents a pair of eigenvalues and an embedding.
    EigenSystem behaves as a Tuple with extra methods. An eigenvalue's corresponding embedding direction is stored at the same index.
    """
    
    fixed_type = (DenseArray, Embedding)

    @property
    def eigenvectors(self) -> Embedding:
        return self[1]

    def get_comp(self, ncomp: int) -> EmbeddingSystem:
        """
        Gets the first `ncomp` number of eigenvalues and eigenvectors.

        :param ncomp: The number of eigenvalues and eigenvectors to return.

        :return: An `EigenSystem` with the `ncomp` number of eigenvalues and eigenvectors.
        """
        return EmbeddingSystem(
            (self[0][:ncomp], self[1][:, :ncomp]), metadata=self.metadata
        )

    def drop_first(self) -> EmbeddingSystem:
        """
        Drops the first eigenvalue and eigenvector of the system.

        :return: An `EigenSystem` with the first eigenvalue and eigenvector dropped.
        """
        return EmbeddingSystem((self[0][1:], self[1][:, 1:]), metadata=self.metadata)
