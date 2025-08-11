from dataclasses import dataclass
from typing import Dict, Generic, Tuple

from test_array.conftest import DummyArray, dummy_array_groups

import numpy as np
import pytest

pytestmark = pytest.mark.slow

@pytest.mark.parametrize("dummy_array", dummy_array_groups["dense"])
def test__de_affinity_in_place__correct_output():
