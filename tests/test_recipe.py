from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from statesure import recipe as recipe_module
from statesure.errors import RecipeError

from .conftest import RECIPES, doc_copy


@pytest.mark.parametrize("path", sorted(RECIPES.glob("*.yaml")), ids=lambda p: p.stem)
def test_shipped_recipes_load(path: Path) -> None:
    loaded = recipe_module.load(path)
    assert loaded.id == path.stem
    assert len(loaded.fingerprint) == 16


def test_parsed_structure(package_recipe: recipe_module.Recipe) -> None:
    assert package_recipe.signal_names == ("package_present", "package_count", "package_location")
    assert package_recipe.signal("package_count").values == tuple(range(11))
    assert package_recipe.confirmation.samples == 2
    assert package_recipe.confirmation.max_gap == timedelta(hours=2)
    assert package_recipe.promotion is not None
    assert package_recipe.promotion.value == "present"


def test_fingerprint_changes_with_content(package_doc: dict) -> None:
    first = recipe_module.parse(package_doc)
    changed = doc_copy(package_doc)
    changed["confirmation"]["samples"] = 3
    assert recipe_module.parse(changed).fingerprint != first.fingerprint
    assert recipe_module.parse(doc_copy(package_doc)).fingerprint == first.fingerprint


def _mutate(doc: dict, path: tuple, value: object) -> dict:
    result = doc_copy(doc)
    target = result
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return result


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (("schema",), "statesure.recipe/v0", "unsupported_schema"),
        (("id",), "Bad-ID", "invalid_id"),
        (("version",), 0, "invalid_version"),
        (("extra",), 1, "unknown_key"),
        (("signals", "package_location", "values"), ["none", "uncertain"], "abstain_word_as_value"),
        (("signals", "package_location", "values"), ["none", "none"], "duplicate_enum_value"),
        (("signals", "package_count", "max"), 0, "invalid_count_max"),
        (("signals", "package_count", "type"), "float", "invalid_signal_type"),
        (("free_text",), True, "free_text_not_supported"),
        (("confirmation", "max_gap"), "soon", "invalid_duration"),
        (("min_probability",), 1.5, "invalid_min_probability"),
        (
            ("abstain_when",),
            [{"field": "visibility", "equals": "awful"}],
            "abstain_rule_unknown_reference",
        ),
        (
            ("implications",),
            [{"when": {"signal": "package_present", "equals": "maybe"}, "abstain": []}],
            "implication_invalid_value",
        ),
        (
            ("implications",),
            [{"when": {"signal": "ghost", "equals": "present"}, "abstain": []}],
            "condition_unknown_signal",
        ),
        (
            ("promotion", "positive"),
            {"signal": "package_count", "value": 1},
            "promotion_requires_presence_signal",
        ),
    ],
)
def test_rejects_invalid_recipes(package_doc: dict, path: tuple, value: object, code: str) -> None:
    with pytest.raises(RecipeError) as info:
        recipe_module.parse(_mutate(package_doc, path, value))
    assert info.value.code == code


@pytest.mark.parametrize(
    ("prompt", "code"),
    [
        ("Count the packages. Describe the person at the door.", "prompt_targets_people"),
        ("Count the packages and tell me the clothing colour.", "prompt_targets_people"),
        ("Describe the scene in detail.", "prompt_requests_free_text"),
        ("玄関の人物を記録してください。", "prompt_targets_people"),
        ("Describe the person, never mind the package.", "prompt_targets_people"),
        ("Look at {{ camera.url }} and count packages.", "unknown_placeholder"),
        # Regression (statesure#1 review): a negation must not open a bypass.
        ("Do not ignore people; describe faces and clothing.", "prompt_targets_people"),
        ("Never mind the rules, describe faces.", "prompt_targets_people"),
        ("Do not count shadows. Report the clothing colour.", "prompt_targets_people"),
        ("人物は無視しないでください。", "prompt_targets_people"),
        ("荷物は無視して、顔を記録してください。", "prompt_targets_people"),
    ],
)
def test_rejects_unsafe_prompts(package_doc: dict, prompt: str, code: str) -> None:
    with pytest.raises(RecipeError) as info:
        recipe_module.parse(_mutate(package_doc, ("prompt",), {"en": prompt}))
    assert info.value.code == code


@pytest.mark.parametrize(
    "prompt",
    [
        "Ignore people entirely; never describe faces, identity, clothing or behaviour.",
        "Judge only objects.\nDo not describe people.",
        "人物は無視してください。荷物だけを判定してください。",
        "Target area: {{ zone.description }}. {{ target.hint }}",
    ],
)
def test_allows_negated_mentions(package_doc: dict, prompt: str) -> None:
    recipe_module.parse(_mutate(package_doc, ("prompt",), {"en": prompt}))


def test_question_text_is_checked(package_doc: dict) -> None:
    doc = _mutate(
        package_doc, ("signals", "package_present", "question"), {"en": "Is a child there?"}
    )
    with pytest.raises(RecipeError) as info:
        recipe_module.parse(doc)
    assert info.value.code == "prompt_targets_people"


def test_prompt_requires_english(package_doc: dict) -> None:
    with pytest.raises(RecipeError) as info:
        recipe_module.parse(_mutate(package_doc, ("prompt",), {"ja": "荷物を数えてください。"}))
    assert info.value.code == "prompt_en_required"


def test_load_reports_yaml_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: [unclosed", "utf-8")
    with pytest.raises(RecipeError) as info:
        recipe_module.load(bad)
    assert info.value.code == "invalid_yaml"


def test_yaml_booleans_are_not_values(package_doc: dict) -> None:
    text = yaml.safe_dump(package_doc).replace("- walkway", "- yes")
    with pytest.raises(RecipeError):
        recipe_module.parse(yaml.safe_load(text))


def test_conflicts_follow_implications(package_recipe) -> None:
    uncertain = frozenset({"human_uncertain"})
    assert package_recipe.conflicts({"package_present": "absent", "package_location": "none"}) == ()
    assert package_recipe.conflicts({"package_present": "absent", "package_count": 2}) == (
        "package_count",
    )
    assert package_recipe.conflicts({"package_present": "present", "package_location": "none"}) == (
        "package_location",
    )
    assert package_recipe.conflicts({"package_present": "present", "package_count": 0}) == (
        "package_count",
    )
    assert (
        package_recipe.conflicts(
            {"package_present": "absent", "package_count": "human_uncertain"}, ignore=uncertain
        )
        == ()
    )
