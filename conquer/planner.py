"""LLM-based natural-language-to-query-plan conversion.

This module handles:
- natural-language query -> Query Plan
- saving/loading Query Plans and Plan Sets
- merging multiple Query Plans into a reusable Plan Set
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .schemas import QUERY_PLAN_SCHEMA, SYSTEM_INSTRUCTIONS


# ---------------------------------------------------------------------
# API key utilities
# ---------------------------------------------------------------------

def set_api_key(api_key: str) -> None:
    """Set OpenAI API key for the current Python session."""
    if not api_key or not isinstance(api_key, str):
        raise ValueError("api_key must be a non-empty string.")
    os.environ["OPENAI_API_KEY"] = api_key


# ---------------------------------------------------------------------
# LLM response utilities
# ---------------------------------------------------------------------

def _extract_output_text(response: Any) -> str:
    """Extract text from OpenAI response objects."""
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    try:
        return response.output[0].content[0].text
    except Exception:
        return str(response)


def _ensure_plan_type(plan: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a Query Plan with type='plan'."""
    if not isinstance(plan, dict):
        raise TypeError("plan must be a dictionary.")

    plan = dict(plan)
    plan.setdefault("type", "plan")
    return plan


# ---------------------------------------------------------------------
# Query planning
# ---------------------------------------------------------------------

def plan_query(
    user_query: str,
    api_key: str | None = None,
    model: str = "gpt-5.4-mini",
    temperature: float = 0,
    schema: dict[str, Any] = QUERY_PLAN_SCHEMA,
    allow_fallback: bool = True,
    files: list[str] | dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a natural-language corpus query into a structured Query Plan.

    GPT is used only in this step. The returned plan is a JSON-compatible
    dictionary that can be inspected, edited, saved, validated, compiled,
    and executed outside the model.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI SDK with: pip install openai") from exc

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Provide api_key=..., call set_api_key(...), or set OPENAI_API_KEY."
        )

    client = OpenAI(api_key=api_key)

    file_instructions = _format_external_files(files)
    system_prompt = SYSTEM_INSTRUCTIONS + file_instructions

    try:
        response = client.responses.create(
            model=model,
            input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
            temperature=temperature,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "dp_gpt_query_plan",
                    "schema": schema,
                    "strict": True,
                }
            },
        )

        plan = json.loads(_extract_output_text(response))
        return _ensure_plan_type(plan)

    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError("Structured-output query planning failed.") from exc

        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            response_format={"type": "json_object"},
        )

        plan = json.loads(response.choices[0].message.content)
        return _ensure_plan_type(plan)


def _format_external_files(files: list[str] | dict[str, str] | None) -> str:
    if not files:
        return ""

    if isinstance(files, dict):
        lines = [f"- {name}: {path}" for name, path in files.items()]
    else:
        lines = [f"- {path}" for path in files]

    return (
        "\n\nAvailable external list files:\n"
        + "\n".join(lines)
        + "\nUse only these files when setting list_file. "
        "Do not invent file names."
    )

# ---------------------------------------------------------------------
# Display utilities
# ---------------------------------------------------------------------

def show(obj: dict[str, Any]) -> None:
    """Pretty-print a Query Plan or Plan Set."""
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def show_plan(plan: dict[str, Any]) -> None:
    """Backward-compatible alias for show()."""
    show(plan)


# ---------------------------------------------------------------------
# Save/load utilities
# ---------------------------------------------------------------------

def save(obj: dict[str, Any], path: str | Path) -> None:
    """Save a Query Plan or Plan Set as JSON.

    The object should contain:
    - type='plan' for a single Query Plan
    - type='plan_set' for a Plan Set
    """
    if not isinstance(obj, dict):
        raise TypeError("obj must be a dictionary.")

    if "type" not in obj:
        obj = _ensure_plan_type(obj)

    if obj["type"] not in {"plan", "plan_set"}:
        raise ValueError("obj['type'] must be either 'plan' or 'plan_set'.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load(path: str | Path) -> dict[str, Any]:
    """Load a Query Plan or Plan Set from JSON."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(obj, dict):
        raise TypeError("Loaded object must be a dictionary.")

    if "type" not in obj:
        obj = _ensure_plan_type(obj)

    if obj["type"] not in {"plan", "plan_set"}:
        raise ValueError("Loaded object must have type='plan' or type='plan_set'.")

    return obj


def save_plan(plan: dict[str, Any], path: str | Path) -> None:
    """Backward-compatible alias for save()."""
    save(_ensure_plan_type(plan), path)


def load_plan(path: str | Path) -> dict[str, Any]:
    """Backward-compatible alias for load()."""
    obj = load(path)
    if obj.get("type") != "plan":
        raise ValueError("Loaded object is not a single Query Plan.")
    return obj


# ---------------------------------------------------------------------
# Plan-set utilities
# ---------------------------------------------------------------------

def plan_set(
    name: str,
    plans: list[dict[str, Any]],
    description: str | None = None,
    mode: str = "and",
) -> dict[str, Any]:
    """Combine multiple Query Plans into one reusable Plan Set.

    A Plan Set is deterministic. It is not generated by GPT; it simply groups
    multiple Query Plans that together define a complex extraction target.
    """
    if mode != "and":
        raise ValueError("Only mode='and' is currently supported.")

    if not plans:
        raise ValueError("plans must contain at least one Query Plan.")

    normalized_plans = []

    for i, plan in enumerate(plans):
        if not isinstance(plan, dict):
            raise TypeError(f"plans[{i}] must be a dictionary.")

        plan = _ensure_plan_type(plan)

        if plan.get("type") != "plan":
            raise ValueError(f"plans[{i}] must have type='plan'.")

        if "include" not in plan or "exclude" not in plan:
            raise ValueError(
                f"plans[{i}] does not look like a Query Plan. "
                "Expected keys: include, exclude."
            )

        normalized_plans.append(plan)

    return {
        "type": "plan_set",
        "name": name,
        "description": description or "",
        "mode": mode,
        "target": "sentence",
        "plans": normalized_plans,
    }


def _flatten_plan_set(obj: dict[str, Any]) -> dict[str, Any]:
    """Flatten a Plan Set into a single Query Plan.

    If a single Query Plan is passed, it is returned unchanged.
    """
    if not isinstance(obj, dict):
        raise TypeError("obj must be a dictionary.")

    obj_type = obj.get("type", "plan")

    if obj_type == "plan":
        return _ensure_plan_type(obj)

    if obj_type != "plan_set":
        raise ValueError("obj must have type='plan' or type='plan_set'.")

    if obj.get("mode", "and") != "and":
        raise ValueError("Only plan_set mode='and' can be flattened.")

    include = []
    exclude = []
    notes = []

    for plan in obj.get("plans", []):
        include.extend(plan.get("include", []))
        exclude.extend(plan.get("exclude", []))

        desc = plan.get("description")
        if desc:
            notes.append(desc)

        for note in plan.get("notes", []):
            notes.append(note)

    return {
        "type": "plan",
        "target": obj.get("target", "sentence"),
        "description": obj.get("description", obj.get("name", "")),
        "include": include,
        "exclude": exclude,
        "notes": notes,
    }


def save_plan_set(plan_set: dict[str, Any], path: str | Path) -> None:
    """Backward-compatible alias for save()."""
    if plan_set.get("type") != "plan_set":
        raise ValueError("Object is not a Plan Set.")
    save(plan_set, path)


def load_plan_set(path: str | Path) -> dict[str, Any]:
    """Backward-compatible alias for load()."""
    obj = load(path)
    if obj.get("type") != "plan_set":
        raise ValueError("Loaded object is not a Plan Set.")
    return obj