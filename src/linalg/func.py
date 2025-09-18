from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class func(ABC):
    nouts: int

    @abstractmethod
    def global_func(self, *args: Any, **kwargs: Any) -> Any:
        ...

    @abstractmethod
    def local_func(self, *args: Any, **kwargs: Any) -> Any:
        ...
    
    @abstractmethod
    def local_iter(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def local(self, *args: Any, bsize: Optional[int] = None, **kwargs: Any):
        if bsize is None:
            return self.local_func(
                *args, bsize, **kwargs
            )

        local_data_iter = self.local_iter(
            *args, bsize, **kwargs
        )

        local_data_batches = [[] for _ in range(self.nouts)]
        for ld in local_data_iter:
            for out_ld_b, out in zip(local_data_batches, ld):
                out_ld_b.append(out)
        return tuple(np.concatenate(ld_batches) for ld_batches in local_data_batches)