import json
import os
from typing import Any, Dict, Optional

import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


BASE_URL = os.getenv("BASE_URL", "http://localhost:7860").rstrip("/")
TASK_ID = os.getenv("TASK_ID")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
ENV_NAME = os.getenv("BENCHMARK", "databridge-env")

SYSTEM_PROMPT = (
    "You fix broken JSON payloads. Return ONLY one valid JSON object for transformed_payload. "
    "No markdown. No explanation."
)


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    return str(content or "")


def _parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return None
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _llm_transform(observation: Dict[str, Any], client: Any) -> Dict[str, Any]:
    broken = observation.get("broken_payload")
    if not isinstance(broken, dict):
        return {}
    if client is None or not MODEL_NAME:
        return dict(broken)
    user_payload = {
        "task_id": observation.get("task_id"),
        "broken_payload": broken,
        "target_schema": observation.get("target_schema"),
        "schema_examples": observation.get("schema_examples"),
        "mismatch_hints": observation.get("mismatch_hints"),
        "last_feedback": observation.get("last_feedback"),
    }
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
            temperature=0.0,
            max_tokens=700,
        )
        parsed = _parse_json_object(_extract_text(completion.choices[0].message.content))
        return parsed if isinstance(parsed, dict) else dict(broken)
    except Exception:
        return dict(broken)


def _discover_task_ids() -> list[str]:
    if TASK_ID:
        return [TASK_ID]
    try:
        data = requests.get(f"{BASE_URL}/tasks", timeout=20).json()
        task_ids = [str(t.get("task_id")) for t in data.get("tasks", []) if t.get("task_id")]
        if task_ids:
            return task_ids
    except Exception:
        pass
    return ["easy", "medium", "hard"]


def _run_task(task_id: str, client: Any) -> float:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")
    try:
        observation = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=20).json()
    except Exception:
        print("[END] success=false steps=0 score=0.0000 rewards=")
        return 0.0

    rewards, final_score, done, step = [], 0.0, bool(observation.get("done", False)), 0
    while not done and step < MAX_STEPS:
        step += 1
        transformed = _llm_transform(observation, client)
        action = {"transformed_payload": transformed}
        error = "null"
        try:
            data = requests.post(f"{BASE_URL}/step", json=action, timeout=20).json()
            reward = data.get("reward", {}) or {}
            observation = data.get("observation", {}) or {}
            done = bool(data.get("done", reward.get("done", False)))
            final_score = float(reward.get("score", 0.0))
        except Exception as exc:
            done = True
            final_score = 0.0
            error = str(exc).replace(" ", "_")
        rewards.append(f"{final_score:.4f}")
        action_str = json.dumps(transformed, separators=(",", ":"), ensure_ascii=True)
        print(
            f"[STEP] step={step} action={action_str} reward={final_score:.4f} "
            f"done={str(done).lower()} error={error}"
        )

    success = str(done).lower()
    print(f"[END] success={success} steps={step} score={final_score:.4f} rewards={','.join(rewards)}")
    return final_score


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if (OpenAI and API_KEY and MODEL_NAME) else None
    task_ids = _discover_task_ids()
    for task_id in task_ids:
        _run_task(task_id, client)


if __name__ == "__main__":
    main()
