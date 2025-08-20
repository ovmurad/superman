from .backend import Backend, BackendStr, Data
from .numpy import NumpyBackend

BACKEND_STR: BackendStr = "numpy"
BACKEND_STR_TO_OBJ: dict[BackendStr, type[Backend]] = {
    "numpy": NumpyBackend,
}
BACKEND = BACKEND_STR_TO_OBJ[BACKEND_STR]

__all__ = ["BACKEND", "Backend", "Data"]
