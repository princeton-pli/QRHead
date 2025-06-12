"""Configuration utilities for QRRetriever."""

import os
from pathlib import Path
from typing import Dict, Optional, Union

import yaml


def get_default_config_path() -> Path:
    """Get the path to the default config file."""
    return Path(__file__).parent / "configs" / "default.yaml"


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to a custom config file. If None, loads the default config.

    Returns:
        Dict containing the configuration.
    """
    if config_path is None:
        config_path = get_default_config_path()
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def merge_configs(default_config: Dict, custom_config: Dict) -> Dict:
    """
    Merge a custom config with the default config.
    Custom config values override default config values.

    Args:
        default_config: Default configuration dictionary
        custom_config: Custom configuration dictionary to override defaults

    Returns:
        Merged configuration dictionary
    """
    merged = default_config.copy()
    
    def _merge_dicts(base: Dict, override: Dict) -> None:
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                _merge_dicts(base[k], v)
            else:
                base[k] = v

    _merge_dicts(merged, custom_config)
    return merged 