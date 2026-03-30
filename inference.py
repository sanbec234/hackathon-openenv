import json
import os

import requests
from openai import OpenAI

BASE_URL = os.getenv("BASE_URL", "http://localhost:7860").rstrip("/")
TASK_ID = os.getenv("TASK_ID", "easy")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

try:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
except Exception:
    client = None

SYSTEM_PROMPT = (
    "You are a payload-repair agent. Fix the broken JSON payload to match the expected contract. "
    "Return ONLY a valid JSON object (the corrected payload). No markdown, no explanation."
)


def _extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) and item.get("type") == "text" else str(item)
            for item in content
        )
    return str(content)


def _parse_json_object(raw_text):
    if not raw_text:
        return None
    text = raw_text.strip()
    candidates = [text]
    if "```" in text:
        for block in text.split("```"):
            cleaned = block.replace("json", "", 1).strip()
            if cleaned:
                candidates.append(cleaned)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


def llm_transform(observation):
    broken = observation.get("broken_payload", {})
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
        raw = _extract_text(completion.choices[0].message.content)
        parsed = _parse_json_object(raw)
        return parsed if isinstance(parsed, dict) else dict(broken)
    except Exception:
        return dict(broken)


def main():
    try:
        reset_resp = requests.post(f"{BASE_URL}/reset", json={"task_id": TASK_ID}, timeout=20)
        reset_resp.raise_for_status()
        observation = reset_resp.json()
    except Exception as exc:
        print(f"Failed to call /reset: {exc}")
        return

    final_reward = {}
    steps = 0
    while not observation.get("done", False) and steps < MAX_STEPS:
        steps += 1
        action = {"transformed_payload": llm_transform(observation)}
        try:
            step_resp = requests.post(f"{BASE_URL}/step", json=action, timeout=20)
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except Exception as exc:
            print(f"Failed to call /step: {exc}")
            return
        observation = step_data.get("observation", {})
        final_reward = step_data.get("reward", {})

    print(f"Final reward: {final_reward.get('score')} | done={final_reward.get('done')}")


if __name__ == "__main__":
    main()
