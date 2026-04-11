"""
DataBridgeEnv — Core environment logic.

Manages episodes, calls generator and grader, enforces max_steps.
"""

from __future__ import annotations
import datetime
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from models import (
    TransformAction,
    ContractObservation,
    ContractReward,
    BridgeState,
)
from generator import generate_episode
from grader import grade

MAX_STEPS = {
    "easy":   3,
    "medium": 4,
    "hard":   5,
}

logger = logging.getLogger(__name__)


def _log_event(message: str) -> None:
    logger.warning("[run_environment] %s", message)


def _normalize_key(name: str) -> str:
    if not isinstance(name, str):
        return ""
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    return normalized.strip("_").lower()


def _is_int_like(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _matches_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, str):
        normalized = expected_type.strip().lower()
        if normalized == "int":
            return _is_int_like(value)
        if normalized == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if normalized == "str":
            return isinstance(value, str)
        if normalized == "bool":
            return isinstance(value, bool)
        if normalized == "dict":
            return isinstance(value, dict)
        if normalized == "list":
            return isinstance(value, list)
        list_match = re.match(r"^list\[(.+)\]$", normalized)
        if list_match:
            if not isinstance(value, list):
                return False
            subtype = list_match.group(1).strip()
            return all(_matches_type(item, subtype) for item in value)
        return False

    if expected_type is int:
        return _is_int_like(value)
    if expected_type is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type is bool:
        return isinstance(value, bool)
    if expected_type is Any:
        return True
    if isinstance(expected_type, type):
        return isinstance(value, expected_type)
    return False


def _strict_match(payload: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not isinstance(schema, dict):
        return False
    if not schema:
        return False
    if set(payload.keys()) != set(schema.keys()):
        return False
    for key, expected_type in schema.items():
        if key not in payload:
            return False
        if not _matches_type(payload[key], expected_type):
            return False
    return True


def _coerce_value(value: Any, expected_type: Any) -> Tuple[Any, bool]:
    if isinstance(expected_type, str):
        normalized = expected_type.strip().lower()
        if normalized == "int":
            expected_type = int
        elif normalized == "float":
            expected_type = float
        elif normalized == "str":
            expected_type = str
        elif normalized == "bool":
            expected_type = bool
        elif normalized.startswith("list"):
            expected_type = list
        elif normalized == "dict":
            expected_type = dict

    if expected_type is Any or value is None:
        return value, False

    if expected_type is int:
        if _is_int_like(value):
            return value, False
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.isdigit() or (cleaned.startswith("-") and cleaned[1:].isdigit()):
                return int(cleaned), True
        if isinstance(value, float):
            return int(value), True
        return value, False

    if expected_type is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), not isinstance(value, float)
        if isinstance(value, str):
            cleaned = value.strip()
            try:
                return float(cleaned), True
            except ValueError:
                return value, False
        return value, False

    if expected_type is bool:
        if isinstance(value, bool):
            return value, False
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True, True
            if lowered in {"false", "0", "no"}:
                return False, True
        return value, False

    if expected_type is list:
        if isinstance(value, list):
            return value, False
        if isinstance(value, tuple):
            return list(value), True
        if isinstance(value, str):
            parts = [item.strip() for item in value.split(",")]
            normalized = [item for item in parts if item]
            return normalized, True
        return [value], True

    if expected_type is str:
        if isinstance(value, str):
            return value, False
        return str(value), True

    if expected_type is dict:
        if isinstance(value, dict):
            return value, False
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                return value, False
            if isinstance(parsed, dict):
                return parsed, True
        return value, False

    if isinstance(expected_type, type) and not isinstance(value, expected_type):
        try:
            return expected_type(value), True
        except Exception:
            return value, False

    return value, False


def _parse_iso_to_epoch(value: str) -> Optional[int]:
    try:
        dt = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except ValueError:
        return None


def _infer_missing_from_common_shapes(payload: Dict[str, Any], schema_key: str) -> Tuple[Optional[Any], Optional[str]]:
    user_obj = payload.get("user")
    if isinstance(user_obj, dict):
        if schema_key == "user_id" and "id" in user_obj:
            return user_obj.get("id"), "user.id"
        if schema_key == "user_name" and "name" in user_obj:
            return user_obj.get("name"), "user.name"
        if schema_key == "email" and isinstance(user_obj.get("name"), str):
            parts = [p.lower() for p in user_obj["name"].split() if p.strip()]
            if len(parts) >= 2:
                return f"{parts[0]}.{parts[-1]}@example.com", "derived from user.name"

    address_obj = payload.get("address")
    if isinstance(address_obj, dict):
        if schema_key == "street" and "street" in address_obj:
            return address_obj.get("street"), "address.street"
        if schema_key == "city" and "city" in address_obj:
            return address_obj.get("city"), "address.city"

    return None, None


def _infer_email_from_name(name: Any) -> Optional[str]:
    if not isinstance(name, str):
        return None
    parts = [p.lower() for p in name.split() if p.strip()]
    if len(parts) < 2:
        return None
    return f"{parts[0]}.{parts[-1]}@example.com"


def _infer_email_candidate(payload: Dict[str, Any], partial_payload: Dict[str, Any]) -> Optional[str]:
    candidates = [
        partial_payload.get("user_name"),
        payload.get("user_name"),
        payload.get("full_name"),
    ]

    user_obj = payload.get("user")
    if isinstance(user_obj, dict):
        candidates.append(user_obj.get("name"))

    for candidate in candidates:
        inferred = _infer_email_from_name(candidate)
        if inferred:
            return inferred
    return None


def _heuristic_transform_without_schema(
    payload: Dict[str, Any],
    include_intro_step: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    fixed_payload: Dict[str, Any] = {}
    steps: List[str] = []
    if include_intro_step:
        steps.append("Applied no-schema heuristic fallback.")

    for source_key, raw_value in payload.items():
        normalized_key = _normalize_key(source_key)
        if normalized_key.startswith("debug"):
            steps.append(f"Dropped debug-like field '{source_key}'.")
            continue

        # Flatten common nested shape mismatches like address.city -> city.
        if isinstance(raw_value, dict) and normalized_key in {"address", "location"}:
            for nested_key, nested_value in raw_value.items():
                nk = _normalize_key(nested_key)
                fixed_payload[nk] = nested_value
            steps.append(f"Flattened nested object '{source_key}'.")
            continue

        value = raw_value
        changed = False

        if isinstance(value, str):
            cleaned = value.strip()
            if "," in cleaned:
                value = [item.strip() for item in cleaned.split(",") if item.strip()]
                changed = True
            elif cleaned.isdigit() or (cleaned.startswith("-") and cleaned[1:].isdigit()):
                value = int(cleaned)
                changed = True
            else:
                try:
                    value = float(cleaned)
                    changed = True
                except ValueError:
                    maybe_epoch = _parse_iso_to_epoch(cleaned)
                    if maybe_epoch is not None and normalized_key.endswith("_at"):
                        value = maybe_epoch
                        changed = True

        fixed_payload[normalized_key] = value
        if normalized_key != source_key:
            steps.append(f"Renamed '{source_key}' to '{normalized_key}'.")
        if changed:
            steps.append(f"Normalized value for '{normalized_key}'.")

    return fixed_payload, steps


def _is_string_schema_type(expected_type: Any) -> bool:
    if expected_type is str:
        return True
    if isinstance(expected_type, str):
        return expected_type.strip().lower() == "str"
    return False


def _enrich_semantic_fields(
    fixed_payload: Dict[str, Any],
    source_payload: Dict[str, Any],
    schema: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    enriched = dict(fixed_payload)
    steps: List[str] = []

    if "email" in schema and _is_string_schema_type(schema["email"]):
        email_value = enriched.get("email")
        email_missing = email_value is None or (isinstance(email_value, str) and not email_value.strip())
        if email_missing:
            inferred_email = _infer_email_candidate(source_payload, enriched)
            if inferred_email:
                enriched["email"] = inferred_email
                steps.append("Filled missing 'email' using inferred value from user name.")

    return enriched, steps


def _rule_based_transform(payload: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    if not schema:
        return _heuristic_transform_without_schema(payload)

    fixed_payload: Dict[str, Any] = {}
    steps: List[str] = ["Applied rule-based transformation fallback."]
    normalized_payload_map = {
        _normalize_key(key): key for key in payload.keys() if isinstance(key, str)
    }
    used_source_keys = set()

    for schema_key, expected_type in schema.items():
        source_key = schema_key if schema_key in payload else normalized_payload_map.get(_normalize_key(schema_key))
        source_in_payload = False
        if source_key is None:
            inferred_value, inferred_source = _infer_missing_from_common_shapes(payload, schema_key)
            if inferred_source is not None:
                source_key = inferred_source
                raw_value = inferred_value
                steps.append(f"Inferred '{schema_key}' from {inferred_source}.")
                if inferred_source.startswith("user.") and "user" in payload:
                    used_source_keys.add("user")
                if inferred_source.startswith("address.") and "address" in payload:
                    used_source_keys.add("address")
            else:
                fixed_payload[schema_key] = None
                steps.append(f"Missing field '{schema_key}'; set to null.")
                continue
        else:
            used_source_keys.add(source_key)
            raw_value = payload[source_key]
            source_in_payload = True

        coerced_value, changed = _coerce_value(raw_value, expected_type)
        if (
            schema_key == "email"
            and (
                coerced_value is None
                or (isinstance(coerced_value, str) and not coerced_value.strip())
            )
        ):
            inferred_email = _infer_email_candidate(payload, fixed_payload)
            if inferred_email:
                coerced_value = inferred_email
                changed = True
                steps.append("Inferred missing 'email' from user name.")

        fixed_payload[schema_key] = coerced_value

        if source_key != schema_key and source_in_payload:
            steps.append(f"Mapped '{source_key}' to '{schema_key}'.")
        if changed:
            expected_name = getattr(expected_type, "__name__", str(expected_type))
            steps.append(f"Coerced '{schema_key}' to {expected_name}.")

    dropped_keys = [key for key in payload.keys() if key not in used_source_keys]
    if dropped_keys:
        steps.append(f"Dropped extra fields not in schema: {dropped_keys}.")

    return fixed_payload, steps


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return str(content)


def _parse_llm_json(raw_content: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text = _extract_text_content(raw_content).strip()
    if not text:
        return None, "empty LLM response"

    candidates: List[str] = [text]
    fenced_matches = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(match.strip() for match in fenced_matches if match.strip())

    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1].strip())

    last_error = "unknown parsing failure"
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(parsed, dict):
            return parsed, None
        last_error = "parsed JSON is not an object"

    return None, last_error


def _extract_llm_result(parsed: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str], Optional[str]]:
    fixed_payload = parsed.get("fixed_payload", parsed.get("transformed_payload"))
    explanation = parsed.get("explanation", parsed.get("steps", []))

    if not isinstance(fixed_payload, dict):
        return None, [], "missing or non-object 'fixed_payload'"

    if isinstance(explanation, list):
        steps = [str(item) for item in explanation]
    elif isinstance(explanation, str):
        steps = [explanation]
    else:
        steps = []

    return fixed_payload, steps, None


def run_environment(payload: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    safe_payload = payload if isinstance(payload, dict) else {}
    safe_schema = schema if isinstance(schema, dict) else {}
    steps: List[str] = []

    try:
        if _strict_match(safe_payload, safe_schema):
            steps.append("Payload already strictly matches schema.")
            return dict(safe_payload), steps

        if safe_schema:
            fixed_payload, fallback_steps = _rule_based_transform(safe_payload, safe_schema)
            fixed_payload, semantic_steps = _enrich_semantic_fields(fixed_payload, safe_payload, safe_schema)
            steps.extend(fallback_steps)
            steps.extend(semantic_steps)
            return fixed_payload, steps

        heuristic_payload, heuristic_steps = _heuristic_transform_without_schema(
            safe_payload,
            include_intro_step=True,
        )
        steps.extend(heuristic_steps)
        return heuristic_payload, steps
    except Exception as exc:
        _log_event(f"Unexpected error: {exc!r}")
        return dict(safe_payload), [
            f"Unexpected error in run_environment ({type(exc).__name__}); returned original payload."
        ]

class DataBridgeEnvironment:
    """
    Stateful environment for one active episode.

    Lifecycle:
        reset(task_id) → ContractObservation
        step(action)   → (ContractObservation, ContractReward)
        state()        → BridgeState
    """

    def __init__(self) -> None:
        self._episode_id: str = ""
        self._task_id:    str = ""
        self._step:       int = 0
        self._done:       bool = True   # True until reset() is called
        self._current_score: float = 0.0
        self._episode_data: Dict[str, Any] = {}
        self._last_reward: Optional[ContractReward] = None
        self._seed: int = 42

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy", seed: int = 42) -> ContractObservation:
        """Start a fresh episode for the given task."""
        if task_id not in MAX_STEPS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(MAX_STEPS)}")

        self._seed          = seed
        random.seed(seed)
        self._episode_id    = f"id_{random.randint(1000, 9999)}"
        self._task_id       = task_id
        self._step          = 0
        self._done          = False
        self._current_score = 0.0
        self._episode_data  = generate_episode(task_id, seed=seed)
        self._last_reward   = None

        return self._build_observation()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: TransformAction) -> tuple[ContractObservation, ContractReward, bool, Dict[str, Any]]:
        """Execute one agent action and return the new observation + reward."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step += 1

        # Grade the action
        result = grade(
            agent_payload  = action.transformed_payload,
            correct_output = self._episode_data["correct_output"],
            target_schema  = self._episode_data.get("target_schema"),
        )

        self._current_score = result["score"]

        # Decide if episode ends
        max_steps = MAX_STEPS[self._task_id]
        done = (result["score"] >= 1.0) or (self._step >= max_steps)
        self._done = done

        reward = ContractReward(
            score               = result["score"],
            field_scores        = result["field_scores"],
            fixed_field_names   = result["fixed_field_names"],
            fixed_type_coercions= result["fixed_type_coercions"],
            fixed_shapes        = result["fixed_shapes"],
            fixed_array_scalars = result["fixed_array_scalars"],
            fixed_missing_fields= result["fixed_missing_fields"],
            total_fields        = result["total_fields"],
            correct_fields      = result["correct_fields"],
            done                = done,
            success             = result["success"],
            feedback            = result["feedback"],
        )
        self._last_reward = reward

        obs = self._build_observation(
            last_reward=result["score"],
            last_feedback=result["feedback"],
        )

        info = {
            "score": result["score"],
            "feedback": result["feedback"],
            "step_count": self._step,
        }

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> BridgeState:
        return BridgeState(
            episode_id            = self._episode_id,
            task_id               = self._task_id,
            step_number           = self._step,
            max_steps             = MAX_STEPS.get(self._task_id, 3),
            done                  = self._done,
            current_score         = self._current_score,
            mismatch_types_present= self._episode_data.get("mismatch_types", []),
            total_fields          = len(self._episode_data.get("correct_output", {})),
        )

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        last_reward: Optional[float] = None,
        last_feedback: Optional[str] = None,
    ) -> ContractObservation:
        return ContractObservation(
            task_id         = self._task_id,
            episode_id      = self._episode_id,
            broken_payload  = self._episode_data["broken_payload"],
            target_schema   = self._episode_data.get("target_schema"),
            schema_examples = self._episode_data.get("schema_examples"),
            mismatch_hints  = self._episode_data.get("mismatch_types", []),
            step_number     = self._step,
            max_steps       = MAX_STEPS[self._task_id],
            done            = self._done,
            last_reward     = last_reward,
            last_feedback   = last_feedback,
        )

