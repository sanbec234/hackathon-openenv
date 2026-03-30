import os

import requests

BASE_URL = os.environ.get("BASE_URL", "http://localhost:7860").rstrip("/")
TASK_ID = os.environ.get("TASK_ID", "easy")


def baseline_transform(observation):
    """Minimal generic baseline: copy broken payload; coerce if schema is available."""
    broken = observation.get("broken_payload", {})
    if not isinstance(broken, dict):
        return {}

    schema = observation.get("target_schema")
    if not isinstance(schema, dict) or not schema:
        return dict(broken)

    transformed = {}
    for key, type_name in schema.items():
        value = broken.get(key)
        if value is None:
            parts = key.split("_")
            camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
            value = broken.get(camel)
        try:
            if type_name == "int" and value is not None:
                value = int(float(value))
            elif type_name == "float" and value is not None:
                value = float(value)
            elif type_name.startswith("list") and isinstance(value, str):
                value = [x.strip() for x in value.split(",") if x.strip()]
            elif type_name == "str" and value is not None:
                value = str(value)
        except Exception:
            pass
        transformed[key] = value
    return transformed


def main():
    try:
        reset_resp = requests.post(f"{BASE_URL}/reset", json={"task_id": TASK_ID}, timeout=15)
        reset_resp.raise_for_status()
        observation = reset_resp.json()
    except requests.RequestException as exc:
        print(f"Failed to reset environment: {exc}")
        return

    final_reward = {}
    while not observation.get("done", False):
        action = {"transformed_payload": baseline_transform(observation), "reasoning": "minimal baseline"}
        try:
            step_resp = requests.post(f"{BASE_URL}/step", json=action, timeout=15)
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except requests.RequestException as exc:
            print(f"Step request failed: {exc}")
            return
        observation = step_data.get("observation", {})
        final_reward = step_data.get("reward", {})

    print(f"Final reward: {final_reward.get('score')} | done={final_reward.get('done')}")


if __name__ == "__main__":
    main()