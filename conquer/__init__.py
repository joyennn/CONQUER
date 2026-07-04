"""DP-GPT 2.0: auditable dependency-based corpus querying."""

from .parser import dp, preview, size, save_parquet, load_parquet
from .planner import plan_query, show_plan, save_plan, load_plan
from .compiler import compile_plan, generate_code, show_code
from .validator import validate, show_report
from .executor import apply
from .diagnosis import diagnose, suggest_fix

__all__ = [
    "dp", "preview", "size", "save_parquet", "load_parquet",
    "plan_query", "show_plan", "save_plan", "load_plan",
    "compile_plan", "generate_code", "show_code",
    "validate", "show_report", "apply", "diagnose", "suggest_fix",
]
