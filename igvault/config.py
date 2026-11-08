from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "paths": {
        "raw": "data/raw",
        "work": "data/work",
        "library": "library",
    },
    "collect": {
        "login": "",
        "sessionfile": "",
        "dirname_pattern": "saved",
        "filename_pattern": "{date_utc:%Y-%m-%d}_{shortcode}",
        "videos_only": True,
        "limit": 0,
        "fast_update": False,
    },
    "transcribe": {
        "enabled": "auto",
        "model": "small",
        "device": "auto",
        "compute_type": "int8",
        "language": "auto",
        "min_chars": 20,
    },
    "llm": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "chat_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "temperature": 0.2,
        "timeout": 90,
    },
    "classify": {
        "clusters": "auto",
        "max_clusters": 12,
        "min_cluster_size": 3,
        "naming_samples": 8,
        "naming_language": "English",
        "outlier_similarity": 0.0,
    },
    "organize": {
        "mode": "copy",
        "unsorted_name": "Unsorted",
        "write_info_md": True,
    },
}

# environment variable -> config key
ENV_OVERRIDES = {
    "IGVAULT_LOGIN": "collect.login",
    "IGVAULT_LLM_BASE_URL": "llm.base_url",
    "IGVAULT_LLM_MODEL": "llm.chat_model",
    "IGVAULT_EMBED_MODEL": "llm.embedding_model",
    "IGVAULT_LIBRARY": "paths.library",
    "IGVAULT_RAW": "paths.raw",
    "IGVAULT_WORK": "paths.work",
}


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class Config:
    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self.data = data
        self.path = path
        self.root = path.resolve().parent if path else Path.cwd()

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def path_for(self, dotted: str) -> Path:
        raw = self.get(dotted)
        if raw is None:
            raise KeyError(f"missing path config: {dotted}")
        candidate = Path(os.path.expanduser(str(raw)))
        return candidate if candidate.is_absolute() else (self.root / candidate)

    def ensure_dirs(self) -> None:
        for key in ("paths.raw", "paths.work", "paths.library"):
            self.path_for(key).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = "config.yaml") -> Config:
    cfg_path: Path | None = None
    data = copy.deepcopy(DEFAULTS)

    if path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.is_file():
            cfg_path = candidate
            with open(candidate, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{candidate} must contain a YAML mapping")
            _deep_merge(data, loaded)

    cfg = Config(data, cfg_path)

    for env_name, dotted in ENV_OVERRIDES.items():
        value = os.getenv(env_name)
        if value:
            cfg.set(dotted, value)

    return cfg
