from abc import ABC
from typing import Any, Callable, Generic, TypeVar, TypeVarTuple

T = TypeVar("T")
Ts = TypeVarTuple("Ts")


class ArrayOp(Generic[*Ts, T], ABC):

    def __init__(self, func: Callable) -> None:
        self.func = func

    def __call__(self, *args: *Ts, **kwargs: Any) -> T:
        return self.func(*args, **kwargs)


@ArrayOp[int, int, int, float]
def f(x: int, y: int, *args: int, k: float = 2.0) -> int:
    return x + y + sum(args)


t = f(0, 1, 2, k=2.0)
