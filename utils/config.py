"""
utils/config.py — Loads config.yaml once at import time.

Usage:
    from utils.config import cfg
    room = cfg["hue"]["room"]
"""

import pathlib
import yaml

_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config.yaml"

def _load() -> dict:
    try:
        return yaml.safe_load(_CONFIG_PATH.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"config.yaml not found at {_CONFIG_PATH}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid config.yaml: {e}")

cfg: dict = _load()
