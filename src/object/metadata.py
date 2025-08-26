from typing import FrozenSet, Literal, Optional, Tuple

import attr
from attr.converters import optional as optional_conv
from attr.setters import convert, validate
from attr.validators import in_, instance_of
from attr.validators import optional as optional_val

DistanceType = Literal["euclidean", "cityblock", "sqeuclidean"]
AffinityType = Literal["gaussian"]
LaplacianType = Literal["geometric", "random_walk", "symmetric"]
DegreeType = Literal["adjacency", "affinity"]

DISTANCE_TYPES: FrozenSet[DistanceType] = frozenset(
    {"euclidean", "cityblock", "sqeuclidean"}
)
AFFINITY_TYPES: FrozenSet[AffinityType] = frozenset({"gaussian"})
LAPLACIAN_TYPES: FrozenSet[LaplacianType] = frozenset(
    {"geometric", "random_walk", "symmetric"}
)
DEGREE_TYPES: FrozenSet[DegreeType] = frozenset(("adjacency", "affinity"))


def _convert_to_int(value: int) -> int:
    return int(value)


def _convert_to_float(value: float, ndigits: int = 5) -> float:
    return round(float(value), ndigits)


def _convert_to_tuple_of_floats(value: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple(val for val in value)


def _convert_to_tuple_of_ints(value: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(val for val in value)


def optional_tuple_of_floats(instance: object, attribute: attr.Attribute, value: Optional[Tuple[float, ...]]) -> None:
    if value is None:
        return
    if not isinstance(value, tuple):
        raise TypeError(f"{attribute.name} must be a tuple or None")
    if not all(isinstance(v_, float) for v_ in value):
        raise TypeError(f"All elements of {attribute.name} must be floats")


def optional_tuple_of_ints(instance: object, attribute: attr.Attribute, value: Optional[Tuple[int, ...]]) -> None:
    if value is None:
        return
    if not isinstance(value, tuple):
        raise TypeError(f"{attribute.name} must be a tuple or None")
    if not all(isinstance(v_, int) for v_ in value):
        raise TypeError(f"All elements of {attribute.name} must be floats")


@attr.s(auto_attribs=True, slots=True)
class Metadata:
    name: Optional[str] = attr.ib(
        default=None,
        validator=optional_val(instance_of(str)),
        on_setattr=[validate],
    )

    dist_type: Optional[DistanceType] = attr.ib(
        default=None,
        validator=optional_val(in_(DISTANCE_TYPES)),
        on_setattr=[validate],
    )

    aff_type: Optional[AffinityType] = attr.ib(
        default=None,
        validator=optional_val(in_(AFFINITY_TYPES)),
        on_setattr=[validate],
    )

    lap_type: Optional[LaplacianType] = attr.ib(
        default=None,
        validator=optional_val(in_(LAPLACIAN_TYPES)),
        on_setattr=[validate],
    )

    radius: Optional[float] = attr.ib(
        default=None,
        validator=optional_val(instance_of(float)),
        converter=optional_conv(_convert_to_float),
        on_setattr=[convert, validate],
    )

    eps: Optional[float] = attr.ib(
        default=None,
        validator=optional_val(instance_of(float)),
        converter=optional_conv(_convert_to_float),
        on_setattr=[convert, validate],
    )

    ks: Optional[Tuple[int, ...]] = attr.ib(
        default=None,
        validator=optional_tuple_of_ints,
        converter=optional_conv(_convert_to_tuple_of_ints),
        on_setattr=[convert, validate],
    )

    radii: Optional[Tuple[float, ...]] = attr.ib(
        default=None,
        validator=optional_tuple_of_floats,
        converter=optional_conv(_convert_to_tuple_of_floats),
        on_setattr=[convert, validate],
    )

    ds: Optional[Tuple[int, ...]] = attr.ib(
        default=None,
        validator=optional_tuple_of_ints,
        converter=optional_conv(_convert_to_tuple_of_ints),
        on_setattr=[convert, validate],
    )

    degree_type: Optional[DegreeType] = attr.ib(
        default=None,
        validator=optional_val(in_(DEGREE_TYPES)),
        on_setattr=[validate],
    )

    def update_with(self, other: "Metadata") -> "Metadata":
        """Return a new Metadata with non-None values from 'other' overriding self."""
        updated_values = {k: v for k, v in attr.asdict(self).items()}
        for k, v in attr.asdict(other).items():
            if v is not None:
                updated_values[k] = v
        return Metadata(**updated_values)
