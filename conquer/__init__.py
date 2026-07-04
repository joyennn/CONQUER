"""CONQUER: Natural language interface for construction-based corpus querying."""

from .parser import (
    Corpus,
    dp,
    load_parsed,
    cleanup_workspace,
)

from .planner import (
    set_api_key,
    plan_query,
    plan_set,
    save,
    load,
    show,
    show_plan,
    save_plan,
    load_plan,
    save_plan_set,
    load_plan_set,
)

from .validator import (
    validate,
    show_report,
    show_report_json,
)

from .compiler import (
    CompiledQuery,
    compile_plan,
    generate_code,
    show_code,
)

from .executor import (
    QueryResults,
    apply,
    show_results,
)

__all__ = [
    "Corpus",
    "dp",
    "load_parsed",
    "cleanup_workspace",
    "set_api_key",
    "plan_query",
    "plan_set",
    "save",
    "load",
    "show",
    "show_plan",
    "save_plan",
    "load_plan",
    "save_plan_set",
    "load_plan_set",
    "validate",
    "show_report",
    "show_report_json",
    "CompiledQuery",
    "compile_plan",
    "generate_code",
    "show_code",
    "QueryResults",
    "apply",
    "show_results",
]
