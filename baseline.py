"""
DataBridgeEnv — Baseline inference script.

Uses the OpenAI API client (reads OPENAI_API_KEY from env) to run a
language model as a zero-shot agent against all 3 tasks.

Produces reproducible baseline scores.

Usage:
    python baseline.py                     # pretty-print results
    python baseline.py --json              # print JSON on last line
    python baseline.py --episodes 1 --json # one episode per task
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


load_dotenv()


BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MODEL = os.environ.get("BASELINE_MODEL", "gpt-4o-mini")
DEFAULT_EPISODES_PER_TASK = 1
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(obs: Dict[str, Any]) -> str:
    task_id = obs.get("task_id", "unknown")
    broken = json.dumps(obs.get("broken_payload", {}), indent=2)
    schema = obs.get("target_schema")
    hints = obs.get("mismatch_hints", [])
    examples = obs.get("schema_examples")
    feedback = obs.get("last_feedback", "")

    prompt_parts = [
        "You are a data contract debugging agent.",
        f"Task difficulty: {task_id}",
        "",
        "## Broken payload (what the upstream service sent):",
        f"```json\n{broken}\n```",
    ]

    if schema:
        prompt_parts += [
            "",
            "## Target schema (what the downstream expects):",
            f"```json\n{json.dumps(schema, indent=2)}\n```",
        ]
    elif examples:
        prompt_parts += [
            "",
            "## No explicit schema. Infer it from these (broken_input → correct_output) examples:",
        ]
        for i, ex in enumerate(examples, 1):
            prompt_parts.append(f"\n### Example {i}:")
            prompt_parts.append(f"Broken: {json.dumps(ex['broken_input'], indent=2)}")
            prompt_parts.append(f"Correct: {json.dumps(ex['correct_output'], indent=2)}")

    if hints:
        prompt_parts += ["", f"## Mismatch hint categories present: {hints}"]

    if feedback:
        prompt_parts += ["", f"## Feedback from previous attempt: {feedback}"]

    prompt_parts += [
        "",
        "## Your job:",
        "Fix all mismatches and return ONLY a valid JSON object — the corrected payload.",
        "Do NOT include any explanation, markdown fences, or extra keys.",
        "Return only the JSON object.",
    ]

    return "\n".join(prompt_parts)


def _normalize_key(name: str) -> str:
    out = []
    for idx, ch in enumerate(str(name)):
        if ch.isupper() and idx > 0 and str(name)[idx - 1].isalnum():
            out.append("_")
        out.append(ch.lower())
    return "".join(out).replace("-", "_").replace(" ", "_").strip("_")


def _snake_to_camel(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _coerce_for_type(value: Any, type_str: str) -> Any:
    if value is None:
        return None
    t = (type_str or "").strip().lower()
    if t == "int":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            s = value.strip()
            if s:
                try:
                    return int(float(s))
                except ValueError:
                    return value
        return value
    if t == "float":
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if s:
                try:
                    return float(s)
                except ValueError:
                    return value
        return value
    if t == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"true", "1", "yes"}:
                return True
            if s in {"false", "0", "no"}:
                return False
        return bool(value)
    if t.startswith("list"):
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        return [value]
    if t == "str":
        return str(value)
    if t == "dict":
        return value if isinstance(value, dict) else {}
    return value


def _pick_value(broken: Dict[str, Any], target_key: str) -> Any:
    if target_key in broken:
        return broken[target_key]

    candidates = [
        _snake_to_camel(target_key),
        target_key.replace("_", ""),
        target_key.replace("_", "-"),
        target_key.replace("_", " "),
    ]
    for cand in candidates:
        if cand in broken:
            return broken[cand]

    normalized = {_normalize_key(k): k for k in broken.keys() if isinstance(k, str)}
    k_norm = _normalize_key(target_key)
    if k_norm in normalized:
        return broken[normalized[k_norm]]

    return None


def build_rule_based_payload(obs: Dict[str, Any]) -> Dict[str, Any]:
    broken = obs.get("broken_payload") or {}
    if not isinstance(broken, dict):
        return {}

    schema = obs.get("target_schema")
    if not isinstance(schema, dict) or not schema:
        payload: Dict[str, Any] = {}
        for key, value in broken.items():
            new_key = _normalize_key(key)
            if isinstance(value, str):
                s = value.strip()
                if s and (s.isdigit() or (s.startswith("-") and s[1:].isdigit())):
                    payload[new_key] = int(s)
                    continue
                try:
                    payload[new_key] = float(s)
                    continue
                except ValueError:
                    pass
            payload[new_key] = value
        return payload

    fixed: Dict[str, Any] = {}
    for key, type_str in schema.items():
        raw = _pick_value(broken, key)
        fixed[key] = _coerce_for_type(raw, str(type_str))

    return fixed


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------

def run_episode(
    task_id: str,
    openai_client: Any = None,
    verbose: bool = False,
    seed: int = DEFAULT_SEED,
    base_url: str = BASE_URL,
) -> float:
    """Run one episode; return final score."""
    resp = requests.post(
        f"{base_url}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=10,
    )
    resp.raise_for_status()
    obs = resp.json()

    episode_score = 0.0

    while not obs.get("done", False):
        # Decision flow: prefer deterministic rule-based repair unless an OpenAI client is provided.
        if openai_client is None:
            transformed = build_rule_based_payload(obs)
        else:
            prompt = build_prompt(obs)
            llm_failed = False
            try:
                completion = openai_client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    seed=seed,
                    timeout=1.0,
                )
                raw = completion.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                transformed = json.loads(raw)
            except Exception:
                transformed = None
                llm_failed = True
            if llm_failed or not isinstance(transformed, dict):
                transformed = build_rule_based_payload(obs)

        step_resp = requests.post(
            f"{base_url}/step",
            json={"transformed_payload": transformed, "reasoning": ""},
            timeout=10,
        )
        step_resp.raise_for_status()
        data = step_resp.json()
        obs = data["observation"]
        reward = data["reward"]
        episode_score = reward["score"]

        if verbose:
            print(f"  step {obs['step_number']}: score={episode_score:.3f} | {reward['feedback'][:80]}")

    return episode_score


# ---------------------------------------------------------------------------
# Multi-task baseline
# ---------------------------------------------------------------------------

def run_baseline(
    episodes: int = DEFAULT_EPISODES_PER_TASK,
    openai_client: Any = None,
    verbose: bool = False,
    seed_start: int = DEFAULT_SEED,
    base_url: str = BASE_URL,
) -> Dict[str, Dict[str, Any]]:
    task_ids = ["easy", "medium", "hard"]
    results: Dict[str, Dict[str, Any]] = {}

    for task_id in task_ids:
        scores = []
        errors = []
        for ep in range(max(1, int(episodes))):
            try:
                score = run_episode(
                    task_id=task_id,
                    openai_client=openai_client,
                    verbose=verbose,
                    seed=seed_start + ep,
                    base_url=base_url,
                )
                scores.append(score)
            except Exception as e:
                err_msg = str(e)
                errors.append(err_msg)
                scores.append(0.0)

        avg = sum(scores) / len(scores) if scores else 0.0
        successful = len(scores) - len(errors)
        if len(errors) == len(scores):
            status = "error"
        elif errors:
            status = "partial"
        else:
            status = "ok"

        results[task_id] = {
            "scores": scores,
            "mean": round(avg, 4),
            "status": status,
            "successful_episodes": successful,
            "failed_episodes": len(errors),
            "errors": errors[:3],
        }

    return results


def _build_openai_client_from_env() -> Optional[Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print results as JSON on last line.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES_PER_TASK, help="Episodes per task.")
    args = parser.parse_args()

    client = _build_openai_client_from_env()
    if client is None:
        print(
            "WARNING: OPENAI_API_KEY not set (or openai missing). Using deterministic rule-based baseline.",
            file=sys.stderr,
        )

    results = run_baseline(
        episodes=max(1, args.episodes),
        openai_client=client,
        verbose=args.verbose,
        seed_start=DEFAULT_SEED,
        base_url=BASE_URL,
    )

    if args.json:
        print(json.dumps(results))
    else:
        print("\n=== BASELINE SUMMARY ===")
        for tid, r in results.items():
            print(f"  {tid:8s}: mean={r['mean']:.4f}  status={r['status']}  scores={r['scores']}")


if __name__ == "__main__":
    main()
