from __future__ import annotations

from abc import ABC

class BaseMaterial(ABC):
    def __init__(self, material_id: str):
        self.id = material_id