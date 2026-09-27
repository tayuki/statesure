"""Judge adapters: send images and a recipe to a model and normalize the reply."""

from __future__ import annotations

from .base import Judge, JudgeError, JudgeResult, Transport, build_judge

__all__ = ["Judge", "JudgeError", "JudgeResult", "Transport", "build_judge"]
