from abc import ABC
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


def _convert_to_float(value: float, ndigits=5) -> float:
    return round(float(value), ndigits)


def _convert_to_tuple_of_floats(value: Tuple[float]):
    return (val for val in value)


@attr.s(auto_attribs=True, slots=True)
class Metadata():
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
    
    ks: Optional[Tuple[int]] = attr.ib(
        default=(),
        validator=optional_val(instance_of(Tuple[int])),
        converter=optional_conv(_convert_to_tuple_of_floats),
        on_setattr=[convert, validate],
    )

    radii: Optional[float] = attr.ib(
        default=None,
        validator=optional_val(instance_of(float)),
        converter=optional_conv(_convert_to_float),
        on_setattr=[convert, validate],
    )

    ds: Optional[Tuple[int]] = attr.ib(
        default=(),
        validator=optional_val(instance_of(Tuple[int])),
        converter=optional_conv(_convert_to_tuple_of_floats),
        on_setattr=[convert, validate],
    )

    dist_type: Optional[DistanceType] = attr.ib(
        default=None,
        validator=optional_val(in_(DISTANCE_TYPES)),
        on_setattr=[validate],
    )

    degree_type: Optional[DegreeType] = attr.ib(
        default=None,
        validator=optional_val(in_(DEGREE_TYPES)),
        on_setattr=[validate],
    )

    ks: Optional[Tuple[int]] = attr.ib(
        default=(),
        validator=optional_val(instance_of(Tuple[int])),
        converter=optional_conv(_convert_to_tuple_of_floats),
        on_setattr=[convert, validate],
    )

    radii: Optional[float] = attr.ib(
        default=None,
        validator=optional_val(instance_of(float)),
        converter=optional_conv(_convert_to_float),
        on_setattr=[convert, validate],
    )

    ds: Optional[Tuple[int]] = attr.ib(
        default=(),
        validator=optional_val(instance_of(Tuple[int])),
        converter=optional_conv(_convert_to_tuple_of_floats),
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
