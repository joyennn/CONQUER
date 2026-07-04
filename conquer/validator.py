"""Validation for Query Plans and Plan Sets.

This module validates the structure of Query Plans independently of any corpus.

It checks:
- whether the object is a Query Plan or Plan Set
- whether required keys exist
- whether include/exclude conditions are well-formed
- whether fields and operators are valid according to schemas.py

Corpus-specific checks, such as whether a parsed DataFrame has the required
columns, should be handled later by compiler.py or executor.py.
"""

from __future__ import annotations

import json
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from .schemas import VALID_FIELDS, VALID_OPERATORS
from .planner import _flatten_plan_set


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _suggest(value: str, candidates: set[str] | list[str]) -> str | None:
    """Return the closest valid candidate for a possibly invalid string."""
    if not value:
        return None

    matches = get_close_matches(
        value,
        list(candidates),
        n=1,
        cutoff=0.6,
    )
    return matches[0] if matches else None


def _add_suggestion(
    suggestions: list[str],
    label: str,
    invalid_value: str,
    valid_candidates: set[str] | list[str],
    category: str,
) -> None:
    """Append a correction suggestion."""
    suggestion = _suggest(invalid_value, valid_candidates)
    if suggestion:
        suggestions.append(
            f"{label}: unknown {category} '{invalid_value}'. Did you mean '{suggestion}'?"
        )


def _validate_condition(
    cond: dict[str, Any],
    label: str,
    errors: list[str],
    warnings: list[str],
    suggestions: list[str],
) -> None:
    """Validate one include/exclude condition."""
    if not isinstance(cond, dict):
        errors.append(f"{label} must be an object.")
        return

    ctype = cond.get("type")
    if ctype not in {"token_condition", "dependency_condition"}:
        errors.append(f"{label}: unsupported condition type: {ctype}")

        if isinstance(ctype, str):
            _add_suggestion(
                suggestions,
                label,
                ctype,
                {"token_condition", "dependency_condition"},
                "condition type",
            )

    field = cond.get("field")
    if not field:
        errors.append(f"{label}: missing field.")
    elif field not in VALID_FIELDS:
        errors.append(f"{label}: unknown field '{field}'.")
        _add_suggestion(
            suggestions,
            label,
            field,
            VALID_FIELDS,
            "field",
        )

    operator = cond.get("operator")
    if operator not in VALID_OPERATORS:
        errors.append(f"{label}: unsupported operator '{operator}'.")
        if isinstance(operator, str):
            _add_suggestion(
                suggestions,
                label,
                operator,
                VALID_OPERATORS,
                "operator",
            )

    value = cond.get("value")
    list_file = cond.get("list_file")

    if operator in {"equals", "not_equals", "contains", "starts_with", "ends_with", "regex"}:
        if value is None:
            warnings.append(f"{label}: operator '{operator}' usually requires a value.")

    if operator in {"in", "not_in"}:
        if value is None and not list_file:
            errors.append(f"{label}: operator '{operator}' requires either value or list_file.")

    if operator == "exists" and value is not None:
        warnings.append(f"{label}: operator 'exists' ignores value.")

    # This is only a warning because list_file may be resolved relative to
    # another project directory later.
    if list_file and not Path(list_file).exists():
        warnings.append(
            f"{label}: list_file '{list_file}' was not found in the current directory."
        )

    if ctype == "dependency_condition":
        head = cond.get("head")
        head_field = cond.get("head_field")
        head_operator = cond.get("head_operator")
        head_value = cond.get("head_value")

        if head is None and head_field is None:
            warnings.append(
                f"{label}: dependency_condition has no head/head_field constraint."
            )

        if head_operator and head_operator not in VALID_OPERATORS:
            errors.append(f"{label}: unsupported head_operator '{head_operator}'.")
            if isinstance(head_operator, str):
                _add_suggestion(
                    suggestions,
                    label,
                    head_operator,
                    VALID_OPERATORS,
                    "head_operator",
                )

        if head_field and head_field not in VALID_FIELDS:
            errors.append(f"{label}: unknown head_field '{head_field}'.")
            if isinstance(head_field, str):
                _add_suggestion(
                    suggestions,
                    label,
                    head_field,
                    VALID_FIELDS,
                    "head_field",
                )

        if head_operator in {"equals", "not_equals", "contains", "starts_with", "ends_with", "regex"}:
            if head_value is None:
                warnings.append(
                    f"{label}: head_operator '{head_operator}' usually requires head_value."
                )


def _validate_plan_set_structure(
    obj: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate the outer structure of a Plan Set before flattening."""
    if obj.get("type") != "plan_set":
        return

    if not obj.get("name"):
        warnings.append("plan_set: missing name.")

    if obj.get("mode", "and") != "and":
        errors.append("plan_set: only mode='and' is currently supported.")

    plans = obj.get("plans")
    if not isinstance(plans, list):
        errors.append("plan_set: 'plans' must be a list.")
        return

    if not plans:
        errors.append("plan_set: 'plans' must contain at least one Query Plan.")

    for i, plan in enumerate(plans):
        if not isinstance(plan, dict):
            errors.append(f"plan_set.plans[{i}] must be an object.")
            continue

        if plan.get("type", "plan") != "plan":
            errors.append(f"plan_set.plans[{i}] must have type='plan'.")

        if "include" not in plan:
            errors.append(f"plan_set.plans[{i}]: missing include.")
        if "exclude" not in plan:
            errors.append(f"plan_set.plans[{i}]: missing exclude.")


# ---------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------

def validate(
    obj: dict[str, Any],
    compiled: Any = None,
) -> dict[str, Any]:
    """Validate a Query Plan or Plan Set without requiring a corpus.

    Parameters
    ----------
    obj:
        Query Plan or Plan Set.
    compiled:
        Optional compiled object. Currently unused, kept for compatibility.

    Returns
    -------
    dict
        Validation report.
    """
    errors: list[str] = []
    warnings: list[str] = []
    suggestions: list[str] = []

    if not isinstance(obj, dict):
        return {
            "valid": False,
            "errors": ["Plan or Plan Set must be a dictionary."],
            "warnings": warnings,
            "suggestions": suggestions,
        }

    object_type = obj.get("type", "plan")

    if object_type not in {"plan", "plan_set"}:
        errors.append("Object type must be either 'plan' or 'plan_set'.")

    _validate_plan_set_structure(obj, errors, warnings)

    try:
        plan = _flatten_plan_set(obj)
    except Exception as exc:
        return {
            "valid": False,
            "errors": errors + [f"Failed to normalize plan object: {exc}"],
            "warnings": warnings,
            "suggestions": suggestions,
            "object_type": object_type,
        }

    if plan.get("target") != "sentence":
        errors.append("Only target='sentence' is currently supported.")

    for section in ["include", "exclude"]:
        conditions = plan.get(section)

        if conditions is None:
            errors.append(f"Missing required section: '{section}'.")
            continue

        if not isinstance(conditions, list):
            errors.append(f"'{section}' must be a list.")
            continue

        for i, cond in enumerate(conditions):
            _validate_condition(
                cond=cond,
                label=f"{section}[{i}]",
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
            )

    if not plan.get("include"):
        warnings.append(
            "No include conditions were specified; the query may return many sentences."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "object_type": object_type,
        "normalized_type": plan.get("type", "plan"),
        "n_include": len(plan.get("include", [])),
        "n_exclude": len(plan.get("exclude", [])),
    }


# ---------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------

def show_report(report: dict[str, Any]) -> None:
    """Print a human-readable validation report."""
    status = "VALID" if report.get("valid") else "INVALID"

    print("=" * 60)
    print("Validation Report")
    print("=" * 60)
    print(f"Status          : {status}")
    print(f"Object type     : {report.get('object_type')}")
    print(f"Normalized type : {report.get('normalized_type')}")
    print(f"Include rules   : {report.get('n_include')}")
    print(f"Exclude rules   : {report.get('n_exclude')}")
    print("-" * 60)

    errors = report.get("errors", [])
    warnings = report.get("warnings", [])
    suggestions = report.get("suggestions", [])

    print("Errors:")
    if errors:
        for error in errors:
            print(f"  - {error}")
    else:
        print("  None")

    print()
    print("Warnings:")
    if warnings:
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("  None")

    print()
    print("Suggestions:")
    if suggestions:
        for suggestion in suggestions:
            print(f"  - {suggestion}")
    else:
        print("  None")

    print("=" * 60)


def show_report_json(report: dict[str, Any]) -> None:
    """Print the raw validation report as JSON."""
    print(json.dumps(report, indent=2, ensure_ascii=False))