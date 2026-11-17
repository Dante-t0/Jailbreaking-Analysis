"""Clients for the target (Qwen) and attacker models on Together.

Both run on Together's OpenAI-compatible API. The target client is what every
attack ultimately queries; it applies the server-side chat template (we send
plain messages) and uses fixed decoding so only the model size changes between
arms.
"""

from __future__ import annotations

import os
from typing import Sequence

from openai import OpenAI

from . import config


def _require(env: str) -> str:
    val = os.environ.get(env)
    if not val:
        raise RuntimeError(
            f"Environment variable {env} is not set. "
            "See jbscale/config.py for the keys this pipeline needs."
        )
    return val


def together_client() -> OpenAI:
    """OpenAI SDK client pointed at Together."""
    return OpenAI(base_url=config.TOGETHER_BASE_URL, api_key=_require(config.TOGETHER_API_KEY_ENV))


class TargetClient:
    """A single Qwen size behind a Together endpoint.

    ``complete`` takes a conversation (list of role/content messages) and
    returns the final assistant reply as a string. Multi-turn attacks build the
    conversation; single-turn attacks pass one user message.
    """

    def __init__(self, model: config.TargetModel, client: OpenAI | None = None) -> None:
        self.model = model
        self.model_string = config.target_model_string(model)
        self._client = client or together_client()

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_string,
            messages=list(messages),
            temperature=config.TARGET_TEMPERATURE,
            max_tokens=config.TARGET_MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()

    def complete_prompt(self, prompt: str) -> str:
        """Convenience for single-turn arms (none, dan, pap)."""
        return self.complete([{"role": "user", "content": prompt}])
