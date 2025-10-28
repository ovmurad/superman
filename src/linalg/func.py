from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, Optional, Type, TypeVar

from src.object import ObjectMixin

GlobalType = TypeVar("GlobalType", bound=ObjectMixin)
LocalType = TypeVar("LocalType", bound=ObjectMixin)


class func(ABC, Generic[GlobalType, LocalType]):
    """
    Abstract base class for functions that produce both global and local outputs.

    Subclasses must implement `global_func`, `local_func`, and `local_iter`
    to define how to compute global and local structures. The `package`
    helper aggregates local outputs into a single object.

    :param GlobalType: Type returned by `global_func`.
    :param LocalType: Type returned by `local_func` and `local_iter`.
    """

    local_type: Type
    global_type: Type

    @classmethod
    @abstractmethod
    def global_func(cls, *args: Any, **kwargs: Any) -> GlobalType:
        """
        Computes the global object for the function.

        :param args: Positional arguments specific to the subclass implementation.
        :param kwargs: Keyword arguments specific to the subclass implementation.

        :return: A global object of type `GlobalType`.
        """
        ...

    @classmethod
    @abstractmethod
    def local_func(cls, *args: Any, **kwargs: Any) -> LocalType:
        """
        Computes a local object for the function.

        :param args: Positional arguments specific to the subclass implementation.
        :param kwargs: Keyword arguments specific to the subclass implementation.

        :return: A local object of type `LocalType`.
        """
        ...

    @classmethod
    @abstractmethod
    def local_iter(cls, *args: Any, **kwargs: Any) -> Iterable[LocalType]:
        """
        Iterates over local computations in a batched manner.

        :param args: Positional arguments specific to the subclass implementation.
        :param kwargs: Keyword arguments specific to the subclass implementation.

        :return: An iterable of local objects of type `LocalType`.
        """
        ...

    @classmethod
    def package(
        cls,
        *args: Any,
        output_cls: Type[LocalType],
        bsize: Optional[int] = None,
        **kwargs: Any
    ) -> LocalType:
        """
        Aggregates all local computations into a single object based on the object's `concat_with_metadata` method.

        :param args: Positional arguments forwarded to `local_iter`.
        :param output_cls: The output class used to aggregate local objects.
        :param bsize: Optional batch size for chunked iteration. (default: None).
        :param kwargs: Keyword arguments forwarded to `local_iter`.

        :return: A concatenated `LocalType` object with metadata.
        """

        return output_cls.concat_with_metadata(
            list(cls.local_iter(*args, bsize, **kwargs))
        )
