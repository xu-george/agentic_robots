from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "configs" / "api_schema.json"


def load_api_schema(path: str | Path | None = None) -> dict[str, Any]:
    """Load the API schema JSON file."""
    schema_path = Path(path) if path else DEFAULT_SCHEMA_PATH
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_api_description(
    schema: dict[str, Any],
    *,
    function_names: list[str] | None = None,
    include_runtime: bool = True,
    include_types: bool = True,
    include_tips: bool = True,
) -> str:
    """Format an API schema into a clear prompt string for the LLM.

    Args:
        schema: The loaded API schema dictionary.
        function_names: If provided, only include these functions. If None, include all.
        include_runtime: Include runtime environment restrictions.
        include_types: Include return type definitions.
        include_tips: Include usage tips.

    Returns:
        Formatted string suitable for passing as api_description to CodeAgentConfig.
    """
    sections: list[str] = []

    if include_runtime:
        sections.append(_format_runtime(schema["runtime"]))

    api_sections: list[str] = []
    for api_name, api_def in schema["apis"].items():
        formatted = _format_api_section(api_name, api_def, function_names)
        if formatted:
            api_sections.append(formatted)
    if api_sections:
        sections.append("--- AVAILABLE FUNCTIONS ---\n\n" + "\n\n".join(api_sections))

    if include_types:
        used_types = _collect_used_types(schema, function_names)
        if used_types:
            sections.append("--- RETURN TYPES ---\n\n" + _format_types(schema["types"], used_types))

    if include_tips and "usage_tips" in schema:
        tips = "\n".join(f"- {tip}" for tip in schema["usage_tips"])
        sections.append("--- USAGE TIPS ---\n\n" + tips)

    return "\n\n".join(sections)


def build_api_description(
    *,
    function_names: list[str] | None = None,
    schema_path: str | Path | None = None,
    extra_functions: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Convenience: load schema and format in one call.

    Args:
        function_names: Only include these functions. None = all.
        schema_path: Path to schema JSON. None = default.
        extra_functions: Additional function definitions to merge (e.g. env-specific
            functions like get_object_position). Keys are function names, values are
            function definition dicts matching the schema format.

    Returns:
        Formatted API description string.
    """
    schema = load_api_schema(schema_path)
    if extra_functions:
        schema = _merge_extra_functions(schema, extra_functions)
    return format_api_description(schema, function_names=function_names)


def _format_runtime(runtime: dict[str, Any]) -> str:
    lines = ["--- RUNTIME ENVIRONMENT ---", "", runtime["description"], ""]
    lines.append("Pre-imported modules:")
    for name, desc in runtime["pre_imported"].items():
        lines.append(f"  - {name}: {desc}")
    lines.append("")
    lines.append(f"Available builtins: {', '.join(runtime['builtins'])}")
    lines.append("")
    lines.append("Restrictions:")
    for restriction in runtime["restrictions"]:
        lines.append(f"  - {restriction}")
    return "\n".join(lines)


def _format_api_section(
    api_name: str,
    api_def: dict[str, Any],
    function_names: list[str] | None,
) -> str:
    functions = api_def.get("functions", {})
    if function_names is not None:
        functions = {name: fn for name, fn in functions.items() if name in function_names}
    if not functions:
        return ""

    lines = [f"[{api_name.upper()}] {api_def.get('description', '')}", ""]
    for fn_name, fn_def in functions.items():
        lines.append(_format_function(fn_name, fn_def))
        lines.append("")

    return "\n".join(lines).rstrip()


def _format_function(name: str, fn_def: dict[str, Any]) -> str:
    signature = _build_signature(name, fn_def)
    lines = [signature, f"  Description: {fn_def['description']}"]

    params = fn_def.get("params", [])
    if params:
        lines.append("  Parameters:")
        for param in params:
            required = param.get("required", True)
            default = param.get("default")
            type_str = param["type"]
            shape = param.get("shape")
            dtype = param.get("dtype")

            type_parts = [type_str]
            if shape:
                type_parts.append(f"shape={shape}")
            if dtype:
                type_parts.append(f"dtype={dtype}")

            req_str = "required" if required else f"optional, default={default}"
            lines.append(f"    - {param['name']}: {', '.join(type_parts)} ({req_str})")
            lines.append(f"      {param['description']}")

    returns = fn_def.get("returns", {})
    if returns:
        lines.append("  Returns:")
        ret_type = returns.get("type", "None")
        ret_shape = returns.get("shape")
        ret_dtype = returns.get("dtype")
        type_parts = [ret_type]
        if ret_shape:
            type_parts.append(f"shape={ret_shape}")
        if ret_dtype:
            type_parts.append(f"dtype={ret_dtype}")
        lines.append(f"    Type: {', '.join(type_parts)}")
        lines.append(f"    {returns.get('description', '')}")

        fields = returns.get("fields", {})
        if fields:
            lines.append("    Fields (use dot access):")
            for field_name, field_def in fields.items():
                field_type = field_def["type"]
                field_shape = field_def.get("shape", "")
                field_dtype = field_def.get("dtype", "")
                type_info = field_type
                if field_shape:
                    type_info += f" {field_shape}"
                if field_dtype:
                    type_info += f" ({field_dtype})"
                lines.append(f"      .{field_name}: {type_info} - {field_def['description']}")

    kwargs = fn_def.get("kwargs", {})
    if kwargs:
        lines.append("  Optional kwargs:")
        for section_name, section_kwargs in kwargs.items():
            if isinstance(section_kwargs, dict) and section_name != "description":
                for kw_name, kw_def in section_kwargs.items():
                    default = kw_def.get("default", "")
                    lines.append(f"    - {kw_name}: {kw_def['type']} (default={default}) - {kw_def['description']}")

    return "\n".join(lines)


def _build_signature(name: str, fn_def: dict[str, Any]) -> str:
    params = fn_def.get("params", [])
    parts: list[str] = []
    for param in params:
        if param.get("required", True):
            parts.append(param["name"])
        else:
            default = param.get("default", "None")
            parts.append(f"{param['name']}={default}")
    return f"  {name}({', '.join(parts)})"


def _collect_used_types(schema: dict[str, Any], function_names: list[str] | None) -> set[str]:
    type_names: set[str] = set()
    apis = schema.get("apis", {})
    for api_def in apis.values():
        for fn_name, fn_def in api_def.get("functions", {}).items():
            if function_names is not None and fn_name not in function_names:
                continue
            returns = fn_def.get("returns", {})
            ret_type = returns.get("type", "")
            _extract_type_names(ret_type, type_names)
            for field_def in returns.get("fields", {}).values():
                _extract_type_names(field_def.get("type", ""), type_names)

    types_def = schema.get("types", {})
    expanded: set[str] = set()
    for type_name in list(type_names):
        _expand_type(type_name, types_def, expanded)
    return expanded


def _extract_type_names(type_str: str, result: set[str]) -> None:
    known_types = {
        "ObjectPose",
        "OrientedBoundingBox",
        "GraspPose",
        "GraspPlan",
        "Sam3Mask",
        "FrankaPose",
        "PointPromptResult",
    }
    for type_name in known_types:
        if type_name in type_str:
            result.add(type_name)


def _expand_type(type_name: str, types_def: dict[str, Any], result: set[str]) -> None:
    if type_name in result:
        return
    if type_name not in types_def:
        return
    result.add(type_name)
    for field_def in types_def[type_name].get("fields", {}).values():
        field_type = field_def.get("type", "")
        _extract_type_names(field_type, result)
        _expand_type(field_type, types_def, result)


def _format_types(types_def: dict[str, Any], used_types: set[str]) -> str:
    sections: list[str] = []
    for type_name in sorted(used_types):
        if type_name not in types_def:
            continue
        type_def = types_def[type_name]
        lines = [f"{type_name}: {type_def.get('description', '')}"]
        for field_name, field_def in type_def.get("fields", {}).items():
            field_type = field_def["type"]
            field_shape = field_def.get("shape", "")
            field_dtype = field_def.get("dtype", "")
            type_info = field_type
            if field_shape:
                type_info += f" {field_shape}"
            if field_dtype:
                type_info += f" ({field_dtype})"
            lines.append(f"  .{field_name}: {type_info} - {field_def['description']}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _merge_extra_functions(
    schema: dict[str, Any],
    extra_functions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    import copy

    schema = copy.deepcopy(schema)
    if "env" not in schema["apis"]:
        schema["apis"]["env"] = {
            "description": "Environment-specific functions.",
            "functions": {},
        }
    schema["apis"]["env"]["functions"].update(extra_functions)
    return schema
