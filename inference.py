"""Minimal inference runner for OpenEnv submission compliance.

Demonstrates agent interaction via:
1) POST /reset
2) repeated POST /step until done
3) final reward summary
"""

import argparse
import os

import requests


def baseline_transform(observation):
    """Simple generic strategy: map broken payload into target schema when available."""
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
            value = broken.get(parts[0] + "".join(p.capitalize() for p in parts[1:]))
        try:
            if type_name == "int" and value is not None:
                value = int(float(value))
            elif type_name == "float" and value is not None:
                value = float(value)
            elif isinstance(type_name, str) and type_name.startswith("list") and isinstance(value, str):
                value = [x.strip() for x in value.split(",") if x.strip()]
            elif type_name == "str" and value is not None:
                value = str(value)
        except Exception:
            pass
        transformed[key] = value
    return transformed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:7860"))
    parser.add_argument("--task-id", default=os.environ.get("TASK_ID", "easy"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    reset_body = {"task_id": args.task_id}
    if args.seed is not None:
        reset_body["seed"] = args.seed

    try:
        observation = requests.post(f"{base_url}/reset", json=reset_body, timeout=15).json()
        episode_id = observation.get("episode_id")
        print(f"Episode started | task={args.task_id} | episode_id={episode_id} | max_steps={observation.get('max_steps')}")
    except Exception as exc:
        print(f"Failed to call /reset: {exc}")
        return

    final_reward = {}
    while not observation.get("done", False):
        action = {"transformed_payload": baseline_transform(observation), "reasoning": "minimal baseline inference"}
        try:
            step_data = requests.post(f"{base_url}/step", json=action, timeout=15).json()
        except Exception as exc:
            print(f"Failed to call /step: {exc}")
            return
        observation = step_data.get("observation", {})
        final_reward = step_data.get("reward", {})
        print(
            f"step={observation.get('step_number')} "
            f"score={final_reward.get('score')} done={final_reward.get('done')}"
        )

    print("Final summary")
    print(f"  score: {final_reward.get('score')}")
    print(f"  done: {final_reward.get('done')}")
    print(f"  success: {final_reward.get('success')}")


if __name__ == "__main__":
    main()