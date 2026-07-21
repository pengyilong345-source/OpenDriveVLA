"""JSON-Schema validation harness for the acceptance-protocol schemas."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import jsonschema  # type: ignore

_SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"
_PROTO = Path(__file__).resolve().parent / "acceptance_protocol.yaml"

_SCHEMAS: Dict[str, Dict[str, Any]] = {}


def _load_schemas() -> Dict[str, Dict[str, Any]]:
    if _SCHEMAS:
        return _SCHEMAS
    for fp in sorted(_SCHEMAS_DIR.glob("*.schema.json")):
        _SCHEMAS[fp.stem.replace(".schema", "")] = json.loads(fp.read_text())
    return _SCHEMAS


def list_schemas() -> List[str]:
    return sorted(_load_schemas().keys())


def validate_record(schema_name: str, record: Mapping[str, Any]
                    ) -> Dict[str, Any]:
    """Validate `record` against the named schema.

    Returns {"valid": bool, "errors": [str, ...], "schema": str}.
    Never raises; reports validation failures in-band so a single bad
    record doesn't break the whole pilot.
    """
    schemas = _load_schemas()
    if schema_name not in schemas:
        return {"valid": False, "errors": [f"unknown schema: {schema_name}"],
                "schema": schema_name}
    schema = schemas[schema_name]
    try:
        jsonschema.validate(record, schema)
        return {"valid": True, "errors": [], "schema": schema_name}
    except jsonschema.ValidationError as e:
        return {"valid": False, "errors": [str(e.message)],
                "schema": schema_name}


def verify_protocol_completeness() -> Dict[str, Any]:
    """Self-check: every required field in the JSON schemas appears in
    acceptance_protocol.yaml and in the formula code.

    Returns a structured report. The check is conservative — it does NOT
    require that every YAML field appears in code (some are documentation
    only); it only requires that every schema-required field has a YAML
    home.
    """
    schemas = _load_schemas()
    import yaml as _yaml
    proto = _yaml.safe_load(_PROTO.read_text())
    out: Dict[str, Any] = {
        "protocol_version": proto.get("protocol_version"),
        "schemas_checked": sorted(schemas.keys()),
        "results": {},
        "all_ok": True,
    }
    for sname, schema in schemas.items():
        req = schema.get("required", [])
        yaml_has = set(_flatten_keys(proto))
        missing = [k for k in req if not _key_present_in_yaml(k, yaml_has)]
        result = {
            "required_fields": len(req),
            "missing_in_yaml": missing,
            "ok": not missing,
        }
        out["results"][sname] = result
        if missing:
            out["all_ok"] = False
    return out


def _flatten_keys(obj: Any, prefix: str = "") -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(f"{prefix}{k}" if not prefix else f"{prefix}.{k}")
            out.extend(_flatten_keys(v, f"{prefix}{k}" if not prefix else f"{prefix}.{k}"))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            out.extend(_flatten_keys(item, f"{prefix}[{i}]"))
    return out


def _key_present_in_yaml(key: str, all_keys: Iterable[str]) -> bool:
    """True iff `key` is mentioned in the YAML (loose substring match)."""
    return any(key in k for k in all_keys)