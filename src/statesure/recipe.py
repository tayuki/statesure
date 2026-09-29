"""Recipe loading and validation.

A recipe is one YAML file that defines a fixed-choice question about a home
state. Recipes are shared by other people, so validation is strict: unknown
keys are rejected, there is no expression language, and prompts may not ask
the judge to describe people or to produce free text.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import yaml

from .errors import RecipeError

SCHEMA = "statesure.recipe/v1"

# Words a judge may use to decline an answer. They are never valid values.
ABSTAIN_WORDS = frozenset({"uncertain", "not_visible"})

SignalType = Literal["presence", "enum", "count"]
PRESENCE_VALUES = ("present", "absent")

_ID = re.compile(r"^[a-z][a-z0-9_]{2,47}$")
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
_VALUE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
_PLACEHOLDER = re.compile(r"\{\{\s*([^}]*?)\s*\}\}")
_ALLOWED_PLACEHOLDERS = frozenset({"zone.description", "target.hint"})
_DURATION = re.compile(r"^(\d+)(s|m|h|d)$")

# Terms that would make a prompt ask about people or for free text. Prompts
# are checked with no exceptions after removing the fixed safety phrases below.
_PERSON_TERMS = re.compile(
    r"\b(person|persons|people|faces?|man|men|woman|women|boys?|girls?|child|children|"
    r"kids?|clothing|clothes|emotions?|behaviou?rs?|identity|identities|gender)\b"
    r"|人物|人の|顔|男性|女性|男の子|女の子|子ども|子供|服装|感情|行動|身元",
    re.IGNORECASE,
)
_FREE_TEXT_REQUEST = re.compile(
    r"\b(describe|description|explain|explanation|narrate|summari[sz]e)\b"
    r"|説明して|記述して|描写して|要約して",
    re.IGNORECASE,
)

# Safety phrases a prompt may use to tell the judge what NOT to do. Only these
# exact shapes are exempt; a free-form negation elsewhere is not.
_EN_TERM = (
    r"(?:people|persons|faces?|identity|identities|clothing|clothes|emotions?|"
    r"behaviou?rs?|gender|appearance)"
)
_JA_TERM = r"(?:人|人物|顔|個人|服装|感情|行動|身元)"
_SAFETY_PHRASES = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:people|persons)(?:\s+(?:entirely|completely))?", re.I),
    re.compile(
        rf"\b(?:never|do\s+not|don't|must\s+not)\s+(?:describe|mention|report|identify|infer)"
        rf"\s+(?:any\s+)?{_EN_TERM}(?:\s*(?:,\s*(?:or\s+|and\s+)?|\s+or\s+|\s+and\s+){_EN_TERM})*",
        re.I,
    ),
    re.compile(rf"{_JA_TERM}は(?:完全に)?無視(?:して|する|し)"),
    re.compile(rf"{_JA_TERM}(?:[、・や]{_JA_TERM})*(?:を|は)(?:記述|説明|判定|推測|記録)?しない"),
)
# "Do not ignore people" or "無視しない" reverses a safety phrase.
_REVERSED_SAFETY = re.compile(
    r"\b(?:not|never|don't|no\s+longer)\s+ignore\b|無視しない", re.IGNORECASE
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|(?<=[。！？])|\n\s*\n")

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


@dataclass(frozen=True)
class SignalSpec:
    """One answer slot of a recipe."""

    name: str
    type: SignalType
    values: tuple[str | int, ...]
    question: Mapping[str, str]
    zones_from: str | None = None

    def accepts(self, value: object) -> bool:
        """Return True when ``value`` is a valid decided value for this signal."""
        if self.type == "count":
            return type(value) is int and value in self.values
        return isinstance(value, str) and value in self.values


@dataclass(frozen=True)
class Condition:
    """A fixed-form condition on one signal (no expression language)."""

    signal: str
    equals: str | int | None = None
    abstained: bool = False


@dataclass(frozen=True)
class FixRule:
    """Replace a decided value that contradicts the condition."""

    if_equals: str | int
    set: str | int | None = None
    abstain: bool = False


@dataclass(frozen=True)
class Implication:
    """Consistency rule between signals, applied once in file order."""

    when: Condition
    set: Mapping[str, str | int]
    fix: Mapping[str, FixRule]
    abstain: tuple[str, ...]


@dataclass(frozen=True)
class AbstainRule:
    """Abstain on every signal when a quality field has this value."""

    field: str
    equals: str


@dataclass(frozen=True)
class Confirmation:
    """Temporal confirmation settings."""

    samples: int
    max_gap: timedelta


@dataclass(frozen=True)
class Promotion:
    """Thresholds a signal must meet before it may be approved for automation."""

    signal: str
    value: str
    min_positive: int
    min_negative: int
    recall_lower_bound: float
    precision_lower_bound: float


@dataclass(frozen=True)
class Recipe:
    """A validated, immutable recipe."""

    id: str
    version: int
    title: Mapping[str, str]
    signals: tuple[SignalSpec, ...]
    quality: Mapping[str, tuple[str, ...]]
    abstain_when: tuple[AbstainRule, ...]
    implications: tuple[Implication, ...]
    prompt: Mapping[str, str]
    confirmation: Confirmation
    min_probability: float | None
    free_text: bool
    promotion: Promotion | None
    fingerprint: str

    def signal(self, name: str) -> SignalSpec:
        """Return the signal called ``name``."""
        for spec in self.signals:
            if spec.name == name:
                return spec
        raise KeyError(name)

    @property
    def signal_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.signals)

    def conflicts(
        self, values: Mapping[str, str | int | None], *, ignore: frozenset[object] = frozenset()
    ) -> tuple[str, ...]:
        """Signals whose values contradict an implication, e.g. present with count 0.

        Values in ``ignore`` (such as "cannot tell") never conflict.
        """
        found: list[str] = []
        for rule in self.implications:
            if rule.when.abstained or values.get(rule.when.signal) != rule.when.equals:
                continue
            for target, expected in rule.set.items():
                value = values.get(target)
                if target in values and value not in ignore and value != expected:
                    found.append(target)
            for target, fix in rule.fix.items():
                if values.get(target) == fix.if_equals:
                    found.append(target)
        return tuple(dict.fromkeys(found))


def load(path: str | Path) -> Recipe:
    """Load and validate a recipe file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise RecipeError("unreadable_recipe") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        raise RecipeError("invalid_yaml") from None
    return parse(data)


def parse(data: object) -> Recipe:
    """Validate an already-parsed recipe document."""
    doc = _mapping(data, "recipe_not_mapping")
    _only_keys(
        doc,
        required={"schema", "id", "version", "title", "signals", "prompt", "confirmation"},
        optional={
            "quality",
            "abstain_when",
            "implications",
            "min_probability",
            "free_text",
            "promotion",
        },
    )
    if doc["schema"] != SCHEMA:
        raise RecipeError("unsupported_schema")
    recipe_id = doc["id"]
    if not isinstance(recipe_id, str) or not _ID.fullmatch(recipe_id):
        raise RecipeError("invalid_id")
    version = doc["version"]
    if type(version) is not int or version < 1:
        raise RecipeError("invalid_version")
    title = _text_map(doc["title"], "invalid_title")

    signals = _parse_signals(doc["signals"])
    by_name = {spec.name: spec for spec in signals}
    quality = _parse_quality(doc.get("quality", {}))
    abstain_when = _parse_abstain_when(doc.get("abstain_when", []), quality)
    implications = _parse_implications(doc.get("implications", []), by_name)
    free_text = doc.get("free_text", False)
    if type(free_text) is not bool:
        raise RecipeError("invalid_free_text")
    if free_text:
        # Free text may leak details about people; not supported in this version.
        raise RecipeError("free_text_not_supported")
    prompt = _text_map(doc["prompt"], "invalid_prompt")
    if "en" not in prompt:
        raise RecipeError("prompt_en_required")
    _check_prompt(prompt)
    for spec in signals:
        _check_prompt(spec.question)
    confirmation = _parse_confirmation(doc["confirmation"])
    min_probability = doc.get("min_probability")
    if min_probability is not None and not _is_probability(min_probability):
        raise RecipeError("invalid_min_probability")
    promotion = _parse_promotion(doc.get("promotion"), by_name)

    return Recipe(
        id=recipe_id,
        version=version,
        title=title,
        signals=signals,
        quality=quality,
        abstain_when=abstain_when,
        implications=implications,
        prompt=prompt,
        confirmation=confirmation,
        min_probability=float(min_probability) if min_probability is not None else None,
        free_text=free_text,
        promotion=promotion,
        fingerprint=_fingerprint(doc),
    )


def parse_duration(value: object) -> timedelta:
    """Parse durations such as ``30m``, ``2h`` or ``1d``."""
    if not isinstance(value, str):
        raise RecipeError("invalid_duration")
    match = _DURATION.fullmatch(value.strip())
    if not match:
        raise RecipeError("invalid_duration")
    seconds = int(match.group(1)) * _UNIT_SECONDS[match.group(2)]
    if seconds <= 0:
        raise RecipeError("invalid_duration")
    return timedelta(seconds=seconds)


def _parse_signals(raw: object) -> tuple[SignalSpec, ...]:
    signals_doc = _mapping(raw, "invalid_signals")
    if not signals_doc:
        raise RecipeError("no_signals")
    specs: list[SignalSpec] = []
    for name, body in signals_doc.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise RecipeError("invalid_signal_name")
        spec_doc = _mapping(body, "invalid_signal")
        kind = spec_doc.get("type")
        question = _text_map(spec_doc.get("question", {}), "invalid_question")
        if kind == "presence":
            _only_keys(spec_doc, required={"type"}, optional={"question"})
            values: tuple[str | int, ...] = PRESENCE_VALUES
            zones_from = None
        elif kind == "enum":
            _only_keys(spec_doc, required={"type", "values"}, optional={"question", "zones_from"})
            raw_values = spec_doc["values"]
            if not isinstance(raw_values, list) or len(raw_values) < 2:
                raise RecipeError("invalid_enum_values")
            for value in raw_values:
                if not isinstance(value, str) or not _VALUE.fullmatch(value):
                    raise RecipeError("invalid_enum_value")
                if value in ABSTAIN_WORDS:
                    raise RecipeError("abstain_word_as_value")
            if len(set(raw_values)) != len(raw_values):
                raise RecipeError("duplicate_enum_value")
            values = tuple(raw_values)
            zones_from = spec_doc.get("zones_from")
            if zones_from is not None and zones_from != "install":
                raise RecipeError("invalid_zones_from")
        elif kind == "count":
            _only_keys(spec_doc, required={"type", "max"}, optional={"question"})
            maximum = spec_doc["max"]
            if type(maximum) is not int or not 1 <= maximum <= 100:
                raise RecipeError("invalid_count_max")
            values = tuple(range(maximum + 1))
            zones_from = None
        else:
            raise RecipeError("invalid_signal_type")
        specs.append(
            SignalSpec(
                name=name, type=kind, values=values, question=question, zones_from=zones_from
            )
        )
    return tuple(specs)


def _parse_quality(raw: object) -> dict[str, tuple[str, ...]]:
    quality_doc = _mapping(raw, "invalid_quality")
    quality: dict[str, tuple[str, ...]] = {}
    for field, values in quality_doc.items():
        if not isinstance(field, str) or not _NAME.fullmatch(field):
            raise RecipeError("invalid_quality_field")
        if not isinstance(values, list) or len(values) < 2:
            raise RecipeError("invalid_quality_values")
        for value in values:
            if not isinstance(value, str) or not _VALUE.fullmatch(value):
                raise RecipeError("invalid_quality_value")
        if len(set(values)) != len(values):
            raise RecipeError("duplicate_quality_value")
        quality[field] = tuple(values)
    return quality


def _parse_abstain_when(
    raw: object, quality: Mapping[str, tuple[str, ...]]
) -> tuple[AbstainRule, ...]:
    if not isinstance(raw, list):
        raise RecipeError("invalid_abstain_when")
    rules: list[AbstainRule] = []
    for item in raw:
        rule = _mapping(item, "invalid_abstain_rule")
        _only_keys(rule, required={"field", "equals"}, optional=set())
        field, equals = rule["field"], rule["equals"]
        if field not in quality or equals not in quality[field]:
            raise RecipeError("abstain_rule_unknown_reference")
        rules.append(AbstainRule(field=field, equals=equals))
    return tuple(rules)


def _parse_implications(raw: object, signals: Mapping[str, SignalSpec]) -> tuple[Implication, ...]:
    if not isinstance(raw, list):
        raise RecipeError("invalid_implications")
    result: list[Implication] = []
    for item in raw:
        rule = _mapping(item, "invalid_implication")
        _only_keys(rule, required={"when"}, optional={"set", "fix", "abstain"})
        if not ({"set", "fix", "abstain"} & set(rule)):
            raise RecipeError("empty_implication")
        when = _parse_condition(rule["when"], signals)

        set_doc = _mapping(rule.get("set", {}), "invalid_implication_set")
        sets: dict[str, str | int] = {}
        for target, value in set_doc.items():
            _require_value(signals, target, value)
            sets[target] = value

        fix_doc = _mapping(rule.get("fix", {}), "invalid_implication_fix")
        fixes: dict[str, FixRule] = {}
        for target, body in fix_doc.items():
            fix = _mapping(body, "invalid_fix")
            _only_keys(fix, required={"if_equals"}, optional={"set", "abstain"})
            _require_value(signals, target, fix["if_equals"])
            has_set = "set" in fix
            abstain = fix.get("abstain", False)
            if type(abstain) is not bool or has_set == abstain:
                raise RecipeError("fix_needs_set_or_abstain")
            if has_set:
                _require_value(signals, target, fix["set"])
            fixes[target] = FixRule(if_equals=fix["if_equals"], set=fix.get("set"), abstain=abstain)

        abstain_list = rule.get("abstain", [])
        if not isinstance(abstain_list, list):
            raise RecipeError("invalid_implication_abstain")
        for target in abstain_list:
            if target not in signals:
                raise RecipeError("implication_unknown_signal")
        result.append(Implication(when=when, set=sets, fix=fixes, abstain=tuple(abstain_list)))
    return tuple(result)


def _parse_condition(raw: object, signals: Mapping[str, SignalSpec]) -> Condition:
    cond = _mapping(raw, "invalid_condition")
    _only_keys(cond, required={"signal"}, optional={"equals", "abstained"})
    signal = cond["signal"]
    if signal not in signals:
        raise RecipeError("condition_unknown_signal")
    has_equals = "equals" in cond
    abstained = cond.get("abstained", False)
    if type(abstained) is not bool or has_equals == abstained:
        raise RecipeError("condition_needs_equals_or_abstained")
    if has_equals:
        _require_value(signals, signal, cond["equals"])
    return Condition(signal=signal, equals=cond.get("equals"), abstained=abstained)


def _parse_confirmation(raw: object) -> Confirmation:
    doc = _mapping(raw, "invalid_confirmation")
    _only_keys(doc, required={"samples", "max_gap"}, optional=set())
    samples = doc["samples"]
    if type(samples) is not int or not 1 <= samples <= 10:
        raise RecipeError("invalid_confirmation_samples")
    return Confirmation(samples=samples, max_gap=parse_duration(doc["max_gap"]))


def _parse_promotion(raw: object, signals: Mapping[str, SignalSpec]) -> Promotion | None:
    if raw is None:
        return None
    doc = _mapping(raw, "invalid_promotion")
    _only_keys(
        doc,
        required={"positive", "min_positive", "min_negative"},
        optional={"recall_lower_bound", "precision_lower_bound"},
    )
    positive = _mapping(doc["positive"], "invalid_promotion_positive")
    _only_keys(positive, required={"signal", "value"}, optional=set())
    signal, value = positive["signal"], positive["value"]
    if signal not in signals or signals[signal].type != "presence":
        raise RecipeError("promotion_requires_presence_signal")
    _require_value(signals, signal, value)
    counts = (doc["min_positive"], doc["min_negative"])
    if any(type(count) is not int or count < 1 for count in counts):
        raise RecipeError("invalid_promotion_count")
    recall = doc.get("recall_lower_bound", 0.80)
    precision = doc.get("precision_lower_bound", 0.90)
    if not _is_probability(recall) or not _is_probability(precision):
        raise RecipeError("invalid_promotion_bound")
    return Promotion(
        signal=signal,
        value=value,
        min_positive=counts[0],
        min_negative=counts[1],
        recall_lower_bound=float(recall),
        precision_lower_bound=float(precision),
    )


def _check_prompt(texts: Mapping[str, str]) -> None:
    for text in texts.values():
        for placeholder in _PLACEHOLDER.findall(text):
            if placeholder not in _ALLOWED_PLACEHOLDERS:
                raise RecipeError("unknown_placeholder")
        for sentence in _SENTENCE_SPLIT.split(text):
            # Placeholder names such as zone.description are not requests.
            sentence = " ".join(_PLACEHOLDER.sub(" ", sentence or "").split())
            if not sentence:
                continue
            if _REVERSED_SAFETY.search(sentence):
                raise RecipeError("prompt_targets_people")
            for phrase in _SAFETY_PHRASES:
                sentence = phrase.sub(" ", sentence)
            if _PERSON_TERMS.search(sentence):
                raise RecipeError("prompt_targets_people")
            if _FREE_TEXT_REQUEST.search(sentence):
                raise RecipeError("prompt_requests_free_text")


def _require_value(signals: Mapping[str, SignalSpec], name: object, value: object) -> None:
    if not isinstance(name, str) or name not in signals:
        raise RecipeError("implication_unknown_signal")
    if not signals[name].accepts(value):
        raise RecipeError("implication_invalid_value")


def _fingerprint(doc: Mapping[str, Any]) -> str:
    canonical = json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _mapping(value: object, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecipeError(code)
    return value


def _only_keys(doc: Mapping[str, Any], *, required: set[str], optional: set[str]) -> None:
    keys = set(doc)
    if not required <= keys:
        raise RecipeError("missing_key")
    if keys - required - optional:
        raise RecipeError("unknown_key")


def _text_map(value: object, code: str) -> dict[str, str]:
    doc = _mapping(value, code)
    for lang, text in doc.items():
        if not isinstance(lang, str) or not re.fullmatch(r"[a-z]{2}(-[A-Z]{2})?", lang):
            raise RecipeError(code)
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise RecipeError(code)
    return dict(doc)


def _is_probability(value: object) -> bool:
    return type(value) in (int, float) and 0.0 <= float(value) <= 1.0
