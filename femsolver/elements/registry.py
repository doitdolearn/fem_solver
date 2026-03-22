from __future__ import annotations

from typing import Dict, Type

from femsolver.elements.base import BaseElement
from femsolver.elements.bar1d import Bar1DElement
from femsolver.elements.tri3 import Tri3Element
from femsolver.elements.bar2d import Bar2DElement
from femsolver.elements.quad4 import Quad4Element
from femsolver.elements.rod1d_thermal import Rod1DThermalElement
from femsolver.elements.tri3_thermal import Tri3ThermalElement
from femsolver.elements.bar1d_nl import Bar1DNLElement
from femsolver.elements.quad4_axisym import Quad4AxisymElement

ELEMENT_REGISTRY: Dict[str, Type[BaseElement]] = {
    # Register the elements newly added
    "bar1d": Bar1DElement,
    "bar2d": Bar2DElement,
    "tri3": Tri3Element,
    "quad4": Quad4Element,
    "rod1d_thermal": Rod1DThermalElement,
    "tri3_thermal": Tri3ThermalElement,
    "bar1d_nl": Bar1DNLElement,
    "quad4_axisym": Quad4AxisymElement,
}

def get_element_class(element_type: str) -> Type[BaseElement]:
    if element_type not in ELEMENT_REGISTRY:
        raise KeyError(
            f"Unknown element type: '{element_type}'.\n Available: {sorted(ELEMENT_REGISTRY.keys())}"
        )
    return ELEMENT_REGISTRY[element_type]
