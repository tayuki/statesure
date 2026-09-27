"""Judge through an OpenAI-compatible ``chat/completions`` endpoint.

Works with local servers such as vLLM, llama.cpp or Ollama. When ``json_schema``
is enabled the request asks the server to constrain the reply to the recipe's
schema; the reply is validated again on our side either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..install import JudgeConfig
from ..normalize import normalize_vlm_json
from ..recipe import ABSTAIN_WORDS, Recipe
from .base import JudgeError, JudgeResult, Transport, data_url, post_json

MAX_IMAGES = 5  # full frame plus up to four crops


class OpenAIVisionJudge:
    def __init__(self, config: JudgeConfig, transport: Transport) -> None:
        self.config = config
        self._transport = transport

    def judge(self, recipe: Recipe, prompt: str, images: Sequence[bytes]) -> JudgeResult:
        if not images or len(images) > MAX_IMAGES:
            raise JudgeError("invalid_image_count")
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": data_url(image)}} for image in images
        ]
        content.append({"type": "text", "text": prompt + "\n\n" + format_instructions(recipe)})
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if self.config.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": recipe.id,
                    "strict": True,
                    "schema": response_schema(recipe),
                },
            }
        body = post_json(self._transport, self.config, payload)
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise JudgeError("invalid_response_shape") from None
        quality, signals = normalize_vlm_json(recipe, text)
        return JudgeResult(signals=signals, quality=quality)


def format_instructions(recipe: Recipe) -> str:
    """Output format for VLMs, generated from the recipe (not written in it)."""
    lines = ["Reply with one JSON object only, with exactly these keys:"]
    for field, values in recipe.quality.items():
        lines.append(f"- {field}: one of {', '.join(values)}")
    abstain = " or ".join(f'"{word}"' for word in sorted(ABSTAIN_WORDS))
    for spec in recipe.signals:
        if spec.type == "count":
            allowed = f"an integer from 0 to {max(spec.values)}"
        else:
            allowed = "one of " + ", ".join(str(v) for v in spec.values)
        lines.append(f"- {spec.name}: {allowed}, or {abstain} if you cannot tell")
    return "\n".join(lines)


def response_schema(recipe: Recipe) -> dict[str, Any]:
    """JSON Schema for the reply: quality fields plus every signal, nothing else."""
    abstain = sorted(ABSTAIN_WORDS)
    properties: dict[str, Any] = {
        field: {"type": "string", "enum": list(values)} for field, values in recipe.quality.items()
    }
    for spec in recipe.signals:
        if spec.type == "count":
            properties[spec.name] = {
                "anyOf": [
                    {"type": "integer", "minimum": 0, "maximum": max(spec.values)},
                    {"type": "string", "enum": abstain},
                ]
            }
        else:
            properties[spec.name] = {"type": "string", "enum": [*spec.values, *abstain]}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
