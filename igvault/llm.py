from __future__ import annotations

import json
import os
import time
from typing import Any

from .utils import chunked, warn


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class LLM:
    """Thin wrapper around any OpenAI-compatible chat/embeddings endpoint."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.api_key = os.getenv(cfg.get("llm.api_key_env", "OPENAI_API_KEY"), "").strip()
        self.base_url = cfg.get("llm.base_url", "https://api.openai.com/v1")
        self.chat_model = cfg.get("llm.chat_model", "gpt-4o-mini")
        self.embedding_model = cfg.get("llm.embedding_model", "text-embedding-3-small")
        self.temperature = float(cfg.get("llm.temperature", 0.2) or 0)
        self.timeout = float(cfg.get("llm.timeout", 90) or 90)
        self._client = None
        self._init_error = ""

        if not self.api_key:
            self._init_error = f"missing API key (env var {cfg.get('llm.api_key_env')})"
            return
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            self._init_error = str(exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def is_reasoning_model(self) -> bool:
        name = (self.chat_model or "").lower().split("/")[-1]
        return name.startswith(("gpt-5", "o1", "o3", "o4"))

    def explain_unavailable(self) -> str:
        return self._init_error or "not configured"

    # ------------------------------------------------------------------ #
    # embeddings
    # ------------------------------------------------------------------ #
    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self.available or not self.embedding_model:
            return None

        vectors: list[list[float]] = []
        for batch in chunked(texts, 64):
            payload = [text if text.strip() else "(empty)" for text in batch]
            result = self._embed_batch(payload)
            if result is None:
                return None
            vectors.extend(result)
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]] | None:
        last_error = ""
        for attempt in range(3):
            try:
                response = self._client.embeddings.create(model=self.embedding_model, input=batch)
                ordered = sorted(response.data, key=lambda item: item.index)
                return [list(item.embedding) for item in ordered]
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        warn(f"embeddings failed ({last_error}); falling back to TF-IDF")
        return None

    # ------------------------------------------------------------------ #
    # chat
    # ------------------------------------------------------------------ #
    def chat_json(self, system: str, user: str, *, max_tokens: int = 700) -> dict[str, Any] | None:
        if not self.available:
            return None

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        base_kwargs: dict[str, Any] = {"model": self.chat_model, "messages": messages}
        if self.is_reasoning_model:
            # GPT-5 / o-series: no temperature, and the budget must also cover hidden reasoning.
            base_kwargs["max_completion_tokens"] = max_tokens * 3 + 2000
            effort = self.cfg.get("llm.reasoning_effort", "low")
            if effort:
                base_kwargs["reasoning_effort"] = effort
        else:
            base_kwargs["temperature"] = self.temperature
            base_kwargs["max_tokens"] = max_tokens

        attempts = [
            {**base_kwargs, "response_format": {"type": "json_object"}},
            base_kwargs,
        ]

        last_error = ""
        for kwargs in attempts:
            for attempt in range(2):
                try:
                    response = self._client.chat.completions.create(**kwargs)
                    text = response.choices[0].message.content or ""
                    parsed = _parse_json_object(text)
                    if parsed is not None:
                        return parsed
                    last_error = "model did not return valid JSON"
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    if attempt == 0:
                        time.sleep(1.0)
            # try next variant (without response_format)
        warn(f"LLM request failed: {last_error}")
        return None
