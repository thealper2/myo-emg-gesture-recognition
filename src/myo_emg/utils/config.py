"""YAML configuration loading with environment-variable interpolation and CLI overrides."""

from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_FILES = ("configs/dataset.yaml", "configs/training.yaml")

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` inside string values, recursively."""
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            env = os.environ.get(name)
            if env:
                return env
            if default is None:
                raise KeyError(f"Environment variable {name} is not set and has no default")
            return default

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict where ``update`` is recursively merged into ``base``."""
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_override(config: dict[str, Any], override: str) -> None:
    """Apply a ``dotted.key=value`` override in place; the value is parsed as YAML."""
    if "=" not in override:
        raise ValueError(f"Override must look like key=value, got {override!r}")
    key, raw = override.split("=", 1)
    parts = key.strip().split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override {key!r}: {part!r} is not a mapping")
    node[parts[-1]] = yaml.safe_load(raw)


def resolve_path(path: str | Path) -> Path:
    """Resolve a path from the config; relative paths are taken from the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_config(files: list[str | Path] | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load and merge YAML config files, then apply ``key=value`` overrides and env expansion."""
    config: dict[str, Any] = {}
    for f in files or DEFAULT_CONFIG_FILES:
        path = resolve_path(f)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            config = deep_merge(config, yaml.safe_load(fh) or {})
    for override in overrides or []:
        apply_override(config, override)
    return _expand_env(config)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add the ``--config``, ``--set`` and ``--log-level`` options shared by all scripts."""
    parser.add_argument("--config", action="append", help="YAML config file (repeatable; default: configs/*.yaml)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="override, e.g. neural.epochs=5")
    parser.add_argument("--log-level", default="INFO")


def config_from_args(args: argparse.Namespace, default_files: tuple[str, ...] = DEFAULT_CONFIG_FILES) -> dict[str, Any]:
    """Build the configuration from parsed CLI arguments."""
    return load_config(args.config or list(default_files), args.overrides)


def artifacts_dir(config: dict[str, Any]) -> Path:
    """Root directory for checkpoints, scalers, splits and reports."""
    return resolve_path(config["paths"]["artifacts_dir"])


def processed_dir(config: dict[str, Any]) -> Path:
    """Directory holding the prepared (parsed) dataset cache."""
    return resolve_path(config["paths"]["processed_dir"])
