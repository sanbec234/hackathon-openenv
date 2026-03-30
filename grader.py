"""
DataBridgeEnv — Deterministic grader.

Compares the agent's transformed_payload against correct_output field by field.
Score is in [0.0, 1.0].  Every check is a pure function — no randomness.

Grading rules per field:
  - Field present with correct type AND correct value → 1.0
  - Field present with correct type but wrong value  → 0.5
  - Field present with wrong type                    → 0.25
  - Field present under wrong name (camelCase/alias) → 0.15  ← NEW
  - Field missing entirely                           → 0.0
  - Extra unexpected field → small penalty applied to overall score

The 0.15 "wrong name" partial credit ensures the hard task never scores a
flat 0.0 when the payload is unmodified — giving the agent a gradient to
climb even before it fixes a single field name.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Type-checking helpers
# ---------------------------------------------------------------------------

def _expected_type(type_str: str) -> type | tuple:
    """Map schema type string to Python type(s)."""
    mapping = {
        "int":      int,
        "float":    (float, int),   # int is acceptable as float
        "str":      str,
        "bool":     bool,
        "list":     list,
        "list[str]": list,
        "list[int]": list,
        "list[float]": list,
        "dict":     dict,
    }
    return mapping.get(type_str, object)


def _type_ok(value: Any, type_str: str) -> bool:
    expected = _expected_type(type_str)
    return isinstance(value, expected)


def _value_ok(agent_val: Any, correct_val: Any) -> bool:
    """
    Flexible value comparison.
    - For lists: elements are compared as strings (order-insensitive for items).
    - For floats: allow ±0.01 tolerance.
    - For ints: exact match (but also accept float like 49.0 for 49).
    - For strings: case-insensitive strip match.
    """
    if isinstance(correct_val, list):
        if not isinstance(agent_val, list):
            return False
        return sorted(str(x) for x in agent_val) == sorted(str(x) for x in correct_val)

    if isinstance(correct_val, float):
        try:
            return abs(float(agent_val) - correct_val) < 0.015
        except (TypeError, ValueError):
            return False

    if isinstance(correct_val, int) and not isinstance(correct_val, bool):
        try:
            return int(agent_val) == correct_val
        except (TypeError, ValueError):
            return False

    if isinstance(correct_val, str):
        return str(agent_val).strip().lower() == correct_val.strip().lower()

    return agent_val == correct_val


# ---------------------------------------------------------------------------
# Fuzzy key matching — detects a field present under wrong name
# ---------------------------------------------------------------------------

def _snake_to_camel(snake: str) -> str:
    """user_id → userId"""
    parts = snake.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])

def _camel_to_snake(camel: str) -> str:
    """userId → user_id"""
    import re
    s = re.sub(r'([A-Z])', r'_\1', camel).lower().lstrip("_")
    return s

def _aliases(field: str) -> List[str]:
    """
    Return all plausible wrong-name variants for a snake_case field.
    e.g. 'user_id' → ['userId', 'UserId', 'userid', 'user-id']
    """
    camel = _snake_to_camel(field)
    pascal = camel[0].upper() + camel[1:]
    flat = field.replace("_", "")
    kebab = field.replace("_", "-")
    return [camel, pascal, flat, kebab]

def _find_fuzzy(field: str, agent_payload: Dict[str, Any]) -> Any:
    """
    Return (alias_key, value) if the field exists under a wrong name,
    or (None, None) if not found at all.
    """
    for alias in _aliases(field):
        if alias in agent_payload:
            return alias, agent_payload[alias]
    # also try camel→snake on all agent keys
    for agent_key in agent_payload:
        if _camel_to_snake(agent_key) == field:
            return agent_key, agent_payload[agent_key]
    return None, None


# ---------------------------------------------------------------------------
# Per-field mismatch detection (used for reward breakdown)
# ---------------------------------------------------------------------------

def _detect_mismatch_category(
    field_key: str,
    agent_val: Any,
    correct_val: Any,
    target_schema: Dict[str, str] | None,
) -> str | None:
    """
    Return which mismatch category this field falls under, or None if correct.
    Used for the reward breakdown counters.
    """
    if field_key not in (target_schema or {}):
        return None
    type_str = (target_schema or {}).get(field_key, "")

    if isinstance(correct_val, list) and isinstance(agent_val, str):
        return "array_scalar"
    if isinstance(correct_val, (int, float)) and isinstance(agent_val, str):
        return "type_coercion"
    if isinstance(correct_val, dict) and isinstance(agent_val, dict):
        return "shape_nesting"
    return None


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def grade(
    agent_payload: Dict[str, Any],
    correct_output: Dict[str, Any],
    target_schema: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Score the agent's payload against correct_output.

    Args:
        agent_payload:  The dict submitted by the agent.
        correct_output: Ground-truth output dict.
        target_schema:  Optional {field: type_str} for type-checking.
                        If None, only value correctness is checked.

    Returns:
        {
          score: float,
          field_scores: dict[str, float],
          correct_fields: int,
          total_fields: int,
          extra_field_penalty: float,
          fixed_field_names: int,
          fixed_type_coercions: int,
          fixed_shapes: int,
          fixed_array_scalars: int,
          fixed_missing_fields: int,
          done: bool,
          success: bool,
          feedback: str,
        }
    """
    field_scores: Dict[str, float] = {}
    feedback_lines: List[str] = []

    # Counters for breakdown
    fixed_names   = 0
    fixed_types   = 0
    fixed_shapes  = 0
    fixed_arrays  = 0
    fixed_missing = 0

    for field, correct_val in correct_output.items():
        if field not in agent_payload:
            # Check if field exists under a wrong name (camelCase etc.)
            alias_key, alias_val = _find_fuzzy(field, agent_payload)
            if alias_key is not None:
                # Field is present but under wrong name — partial credit 0.15
                field_scores[field] = 0.15
                feedback_lines.append(
                    f"  ~ \'{field}\': found as \'{alias_key}\' — rename to snake_case."
                )
            else:
                field_scores[field] = 0.0
                feedback_lines.append(f"  ✗ \'{field}\': MISSING in your payload.")
            fixed_missing -= 1   # not fixed yet
            continue

        agent_val = agent_payload[field]

        # Type check
        if target_schema and field in target_schema:
            type_str = target_schema[field]
            if not _type_ok(agent_val, type_str):
                field_scores[field] = 0.25
                feedback_lines.append(
                    f"  ~ '{field}': wrong type (got {type(agent_val).__name__}, "
                    f"expected {type_str})."
                )
                fixed_types += 0   # not fixed
                continue

        # Value check
        if _value_ok(agent_val, correct_val):
            field_scores[field] = 1.0
            fixed_names   += 1
            fixed_types   += 1
            fixed_shapes  += 1
            fixed_arrays  += 1
            fixed_missing += 1
        else:
            field_scores[field] = 0.5
            feedback_lines.append(
                f"  ~ '{field}': type OK but value wrong "
                f"(got {repr(agent_val)}, expected {repr(correct_val)})."
            )

    # Penalise extra fields (small penalty per field)
    # BUT: camelCase/alias versions of expected fields are NOT truly extra —
    # they are wrong-name versions already penalised by the 0.15 partial credit.
    all_aliases = set()
    for expected_field in correct_output:
        for alias in _aliases(expected_field):
            all_aliases.add(alias)

    truly_extra = [
        k for k in agent_payload
        if k not in correct_output and k not in all_aliases
    ]
    extra_penalty = len(truly_extra) * 0.05
    if truly_extra:
        feedback_lines.append(
            f"  ! Truly extra fields (remove these): {truly_extra}"
        )
    # Inform about wrong-name fields separately (already scored 0.15 above)
    wrong_name_fields = [
        k for k in agent_payload
        if k not in correct_output and k in all_aliases
    ]
    if wrong_name_fields:
        feedback_lines.append(
            f"  ~ Wrong-name fields (rename to snake_case): {wrong_name_fields}"
        )

    total = len(correct_output)
    raw_score = sum(field_scores.values()) / total if total > 0 else 0.0
    score = max(0.0, min(1.0, raw_score - extra_penalty))

    correct = sum(1 for v in field_scores.values() if v == 1.0)
    success = (score >= 0.99)

    if success:
        feedback = "Perfect! All fields are correct."
    elif correct == 0:
        feedback = "No fields were correct. Check field names, types, and values.\n" + "\n".join(feedback_lines)
    else:
        feedback = (
            f"{correct}/{total} fields fully correct (score={score:.2f}).\n"
            + "\n".join(feedback_lines)
        )

    return {
        "score":               round(score, 4),
        "field_scores":        field_scores,
        "correct_fields":      correct,
        "total_fields":        total,
        "extra_field_penalty": extra_penalty,
        "fixed_field_names":   fixed_names,
        "fixed_type_coercions": fixed_types,
        "fixed_shapes":        fixed_shapes,
        "fixed_array_scalars": fixed_arrays,
        "fixed_missing_fields": fixed_missing,
        "done":    success or False,  # environment decides done by max steps
        "success": success,
        "feedback": feedback,
    }