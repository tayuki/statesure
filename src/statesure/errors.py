"""Error types.

Every error carries a short, fixed code. Messages never include input content,
image data, request bodies or secrets, so they are safe to print and log.
"""

from __future__ import annotations


class StatesureError(Exception):
    """Base class for all statesure errors."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RecipeError(StatesureError):
    """A recipe file is malformed or unsafe."""


class InputError(StatesureError):
    """An observation, label or other input record is invalid."""


class StoreError(StatesureError):
    """The label store rejected an operation."""


class CardError(StatesureError):
    """An evaluation card would contain data outside the allowlist."""
