from __future__ import annotations

from src.object.eigen import Eigen


class EigenSystem(Eigen):
    """
    This class represents a pair of eigenvalues and eigenvectors aka an eigensystem.
    EigenSystem behaves as a Tuple with extra methods. An eigenvalue's corresponding eigenvector is stored at the same index.
    """
    def get_comp(self, ncomp: int) -> EigenSystem:
        """
        Gets the first `ncomp` number of eigenvalues and eigenvectors.

        :param ncomp: The number of eigenvalues and eigenvectors to return.

        :return: An `EigenSystem` with the `ncomp` number of eigenvalues and eigenvectors.
        """
        return EigenSystem(
            (self[0][:ncomp], self[1][:, :ncomp]), metadata=self.metadata
        )

    def drop_first(self) -> EigenSystem:
        """
        Drops the first eigenvalue and eigenvector of the system.

        :return: An `EigenSystem` with the first eigenvalue and eigenvector dropped.
        """
        return EigenSystem((self[0][1:], self[1][:, 1:]), metadata=self.metadata)
