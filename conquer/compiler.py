"""Deterministic compiler from Query Plans to executable predicate objects.

This module converts a Query Plan or Plan Set into:
- deterministic sentence-level predicate functions
- human-readable generated Python query code

GPT is not used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .planner import _flatten_plan_set


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class CompiledQuery:
    """Compiled sentence-level query object."""

    plan: dict[str, Any]
    include: list[Callable[[pd.DataFrame], bool]]
    exclude: list[Callable[[pd.DataFrame], bool]]
    code: str
    n_include: int
    n_exclude: int


# ---------------------------------------------------------------------
# Corpus / DataFrame handling
# ---------------------------------------------------------------------

def _get_dataframe(corpus_or_df: Any) -> pd.DataFrame:
    """Accept either a Corpus object or a pandas DataFrame."""
    if isinstance(corpus_or_df, pd.DataFrame):
        return corpus_or_df

    if hasattr(corpus_or_df, "df"):
        df = corpus_or_df.df
        if isinstance(df, pd.DataFrame):
            return df

    raise TypeError(
        "compile_plan() expects a pandas DataFrame or a Corpus object with a .df attribute."
    )


def _check_dataframe_columns(df: pd.DataFrame, plan: dict[str, Any]) -> None:
    """Check whether the parsed DataFrame can support this Query Plan."""
    required = {"sent_id", "id", "head", "deprel"}

    for section in ["include", "exclude"]:
        for cond in plan.get(section, []):
            field = cond.get("field")
            if field:
                required.add(field)

            if cond.get("type") == "dependency_condition":
                required.update({"id", "head"})

                head_field = cond.get("head_field")
                if head_field:
                    required.add(head_field)

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Parsed DataFrame is missing required column(s): {missing}")


# ---------------------------------------------------------------------
# List file handling
# ---------------------------------------------------------------------

def _load_list(path: str | None) -> list[str]:
    """Load lexical items from a plain-text list file."""
    if not path:
        return []

    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(
            f"List file not found: {path}. "
            "Check the path or pass the correct file through plan_query(..., files=[...])."
        )

    return [
        line.strip()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------

def _compare(
    series: pd.Series,
    operator: str,
    value: Any = None,
    list_file: str | None = None,
) -> pd.Series:
    """Apply an operator to a pandas Series and return a boolean mask."""
    s = series.fillna("")

    if operator == "equals":
        return s.astype(str) == str(value)

    if operator == "not_equals":
        return s.astype(str) != str(value)

    if operator == "in":
        values = _load_list(list_file) if list_file else value
        if isinstance(values, str):
            values = [values]
        return s.astype(str).isin([str(v) for v in (values or [])])

    if operator == "not_in":
        values = _load_list(list_file) if list_file else value
        if isinstance(values, str):
            values = [values]
        return ~s.astype(str).isin([str(v) for v in (values or [])])

    if operator == "contains":
        return s.astype(str).str.contains(str(value), regex=False, na=False)

    if operator == "starts_with":
        return s.astype(str).str.startswith(str(value), na=False)

    if operator == "ends_with":
        return s.astype(str).str.endswith(str(value), na=False)

    if operator == "regex":
        return s.astype(str).str.contains(str(value), regex=True, na=False)

    if operator == "exists":
        return series.notna()

    raise ValueError(f"Unsupported operator: {operator}")


# ---------------------------------------------------------------------
# Condition compilation
# ---------------------------------------------------------------------

def _token_condition(cond: dict[str, Any]) -> Callable[[pd.DataFrame], bool]:
    """Compile a token-level condition into a sentence predicate."""
    field = cond["field"]
    operator = cond["operator"]
    value = cond.get("value")
    list_file = cond.get("list_file")

    def predicate(sent_df: pd.DataFrame) -> bool:
        return bool(_compare(sent_df[field], operator, value, list_file).any())

    return predicate


def _dependency_condition(cond: dict[str, Any]) -> Callable[[pd.DataFrame], bool]:
    """Compile a dependency condition into a sentence predicate."""
    field = cond["field"]
    operator = cond["operator"]
    value = cond.get("value")
    list_file = cond.get("list_file")

    head = cond.get("head")
    head_field = cond.get("head_field")
    head_operator = cond.get("head_operator") or "equals"
    head_value = cond.get("head_value")

    def predicate(sent_df: pd.DataFrame) -> bool:
        dep_mask = _compare(sent_df[field], operator, value, list_file)
        dependents = sent_df.loc[dep_mask]

        if dependents.empty:
            return False

        if head is None and head_field is None:
            return True

        for _, dep in dependents.iterrows():
            head_rows = sent_df.loc[sent_df["id"] == dep["head"]]

            if head_rows.empty:
                continue

            if head == "root":
                if (head_rows["deprel"].astype(str) == "root").any():
                    return True

            if head_field:
                if _compare(
                    head_rows[head_field],
                    head_operator,
                    head_value,
                ).any():
                    return True

        return False

    return predicate


def _compile_condition(cond: dict[str, Any]) -> Callable[[pd.DataFrame], bool]:
    """Compile one condition dictionary into a predicate."""
    if cond.get("type") == "dependency_condition":
        return _dependency_condition(cond)

    return _token_condition(cond)


# ---------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------

def _condition_label(cond: dict[str, Any]) -> str:
    """Generate a deterministic human-readable condition label."""
    ctype = cond.get("type", "condition")
    field = cond.get("field", "")
    operator = cond.get("operator", "")
    value = cond.get("value")
    list_file = cond.get("list_file")

    rhs = list_file if list_file else value

    if cond.get("head") == "root":
        return f"{ctype}: {field} {operator} {rhs}; head=root"

    if cond.get("head_field"):
        return (
            f"{ctype}: {field} {operator} {rhs}; "
            f"head.{cond.get('head_field')} "
            f"{cond.get('head_operator', 'equals')} "
            f"{cond.get('head_value')}"
        )

    return f"{ctype}: {field} {operator} {rhs}"


def _condition_expression(cond: dict[str, Any]) -> str:
    """Generate readable Python-like code for one condition."""
    field = cond.get("field")
    operator = cond.get("operator")
    value = cond.get("value")
    list_file = cond.get("list_file")
    head = cond.get("head")
    head_field = cond.get("head_field")
    head_operator = cond.get("head_operator") or "equals"
    head_value = cond.get("head_value")

    rhs = f'load_list("{list_file}")' if list_file else repr(value)

    if operator == "equals":
        expr = f'(sent_df["{field}"].astype(str) == {repr(str(value))})'
    elif operator == "not_equals":
        expr = f'(sent_df["{field}"].astype(str) != {repr(str(value))})'
    elif operator == "in":
        expr = f'(sent_df["{field}"].astype(str).isin({rhs}))'
    elif operator == "not_in":
        expr = f'(~sent_df["{field}"].astype(str).isin({rhs}))'
    elif operator == "contains":
        expr = f'(sent_df["{field}"].astype(str).str.contains({repr(str(value))}, regex=False, na=False))'
    elif operator == "starts_with":
        expr = f'(sent_df["{field}"].astype(str).str.startswith({repr(str(value))}, na=False))'
    elif operator == "ends_with":
        expr = f'(sent_df["{field}"].astype(str).str.endswith({repr(str(value))}, na=False))'
    elif operator == "regex":
        expr = f'(sent_df["{field}"].astype(str).str.contains({repr(str(value))}, regex=True, na=False))'
    elif operator == "exists":
        expr = f'(sent_df["{field}"].notna())'
    else:
        expr = f'# Unsupported operator: {operator}'

    # Token-level condition: at least one token in the sentence satisfies the expression.
    if cond.get("type") != "dependency_condition":
        return f"{expr}.any()"

    # Dependency condition: dependent token must satisfy expr, optionally with head constraint.
    if head is None and head_field is None:
        return f"{expr}.any()"

    lines = [
        "(",
        f"    dep_rows := sent_df.loc[{expr}]",
        ")",
        "# and at least one dependent has a matching head",
    ]

    if head == "root":
        lines.append(
            'head condition: sent_df.loc[sent_df["id"] == dep["head"], "deprel"] == "root"'
        )

    if head_field:
        lines.append(
            f'head condition: sent_df.loc[sent_df["id"] == dep["head"], "{head_field}"] '
            f"{head_operator} {repr(head_value)}"
        )

    return "\n".join(lines)


def generate_code(plan_or_plan_set: dict[str, Any]) -> str:
    """Generate human-readable Python query code for inspection.

    This code is for transparency and debugging.
    It is not executed with eval/exec.
    """
    plan = _flatten_plan_set(plan_or_plan_set)

    include_conditions = plan.get("include", [])
    exclude_conditions = plan.get("exclude", [])

    lines: list[str] = []
    lines.append("# --------------------------------------------------")
    lines.append("# CONQUER Generated Python Query")
    lines.append("# --------------------------------------------------")
    lines.append(f"# Description: {plan.get('description', '')}")
    lines.append("")
    lines.append("# Input:")
    lines.append("#   sent_df = one sentence-level slice of the parsed corpus")
    lines.append("")
    lines.append("# Include conditions")
    lines.append("# ------------------")

    condition_index = 1
    include_names = []
    exclude_names = []

    for cond in include_conditions:
        name = f"condition{condition_index}"
        include_names.append(name)

        lines.append(f"# {name}: {_condition_label(cond)}")
        lines.append(f"{name} = (")
        expr = _condition_expression(cond)
        for line in expr.splitlines():
            lines.append(f"    {line}")
        lines.append(")")
        lines.append("")

        condition_index += 1

    if include_names:
        lines.append("include = (")
        for i, name in enumerate(include_names):
            op = "&" if i < len(include_names) - 1 else ""
            lines.append(f"    {name} {op}".rstrip())
        lines.append(")")
    else:
        lines.append("include = True")

    lines.append("")
    lines.append("# Exclude conditions")
    lines.append("# ------------------")

    for cond in exclude_conditions:
        name = f"condition{condition_index}"
        exclude_names.append(name)

        lines.append(f"# {name}: {_condition_label(cond)}")
        lines.append(f"{name} = (")
        expr = _condition_expression(cond)
        for line in expr.splitlines():
            lines.append(f"    {line}")
        lines.append(")")
        lines.append("")

        condition_index += 1

    if exclude_names:
        lines.append("exclude = (")
        for i, name in enumerate(exclude_names):
            op = "|" if i < len(exclude_names) - 1 else ""
            lines.append(f"    {name} {op}".rstrip())
        lines.append(")")
    else:
        lines.append("exclude = False")

    lines.append("")
    lines.append("result = include & ~exclude")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main compiler
# ---------------------------------------------------------------------

def compile_plan(
    corpus_or_df: Any,
    plan_or_plan_set: dict[str, Any],
) -> CompiledQuery:
    """Compile a Query Plan or Plan Set into deterministic sentence predicates."""
    df = _get_dataframe(corpus_or_df)
    plan = _flatten_plan_set(plan_or_plan_set)

    _check_dataframe_columns(df, plan)

    include = [_compile_condition(cond) for cond in plan.get("include", [])]
    exclude = [_compile_condition(cond) for cond in plan.get("exclude", [])]

    code = generate_code(plan)

    return CompiledQuery(
        plan=plan,
        include=include,
        exclude=exclude,
        code=code,
        n_include=len(include),
        n_exclude=len(exclude),
    )


def show_code(code_or_compiled: str | CompiledQuery) -> None:
    """Print generated Python query code."""
    if isinstance(code_or_compiled, CompiledQuery):
        print(code_or_compiled.code)
    else:
        print(code_or_compiled)
