from __future__ import annotations

from src.object.eigen_mixin import Eigen


class EigenSystem(Eigen):
    def get_comp(self, ncomp: int) -> EigenSystem:
        return EigenSystem(
            (self[0][:ncomp], self[1][:, :ncomp]), metadata=self.metadata
        )

    def drop_first(self) -> EigenSystem:
        return EigenSystem((self[0][1:], self[1][:, 1:]), metadata=self.metadata)
