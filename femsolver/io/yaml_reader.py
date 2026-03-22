from __future__ import annotations

import yaml


def load_yaml(path: str) -> dict:
    """Load a YAML file and return the parsed dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
