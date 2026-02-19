"""Load configuration from config.yaml."""
import os
import yaml


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)
