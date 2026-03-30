"""
DataBridgeEnv — Integration test + pre-submission validator.

Run against the live server before submitting.

Usage:
    python integration_test.py
    python integration_test.py --base-url http://your-space-url

Exit code 0 = all checks pass.
Exit code 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any, Dict

import requests

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
INFO = "\033[94m  INFO\033[0m"

results: list[tuple[str, bool]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    marker = PASS if condition else FAIL
    print(f"{marker}  {name}" + (f" — {detail}" if detail else ""))
    results.append((name, condition))
    return condition


def get(base: str, path: str) -> requests.Response:
    return requests.get(f"{base}{path}", timeout=15)


def post(base: str, path: str, body: Dict[str, Any] | None = None) -> requests.Response:
    return requests.post(f"{base}{path}", json=body or {}, timeout=15)


def is_snake_case_key(key: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*", key))


def run(base: str) -> bool:
    results.clear()

    print(f"\n{'=' * 56}")
    print("  DataBridgeEnv Pre-Submission Validator")
    print(f"  Target: {base}")
    print(f"{'=' * 56}\n")

    # ------------------------------------------------------------------
    # 1. Server responds
    # ------------------------------------------------------------------
    print("── 1. Server health ──────────────────────────────────")
    try:
        r = get(base, "/health")
        check("GET /health returns 200", r.status_code == 200, f"status={r.status_code}")
        check("GET / returns 200", get(base, "/").status_code == 200)
    except Exception as exc:
        check("Server reachable", False, str(exc))
        print("\nServer unreachable — aborting.")
        return False

    # ------------------------------------------------------------------
    # 2. Required endpoints
    # ------------------------------------------------------------------
    print("\n── 2. Required endpoints ─────────────────────────────")
    r = get(base, "/tasks")
    check("GET /tasks returns 200", r.status_code == 200)
    tasks_data = r.json()
    check("GET /tasks has 'tasks' key", "tasks" in tasks_data)
    check("GET /tasks has 'action_schema' key", "action_schema" in tasks_data)
    check("GET /tasks returns 3 tasks", len(tasks_data.get("tasks", [])) == 3)

    task_ids = [task["task_id"] for task in tasks_data.get("tasks", [])]
    check("Tasks include easy/medium/hard", set(task_ids) == {"easy", "medium", "hard"}, str(task_ids))

    difficulties = sorted(task["difficulty"] for task in tasks_data.get("tasks", []))
    check("Difficulty range easy→medium→hard", difficulties == ["easy", "hard", "medium"])

    r = post(base, "/grader")
    check("POST /grader returns 200", r.status_code == 200)
    check("POST /grader has 'current_score'", "current_score" in r.json())

    r = post(base, "/baseline")
    check("POST /baseline returns 200", r.status_code == 200)

    # ------------------------------------------------------------------
    # 3. reset / state
    # ------------------------------------------------------------------
    print("\n── 3. reset() and state() ────────────────────────────")
    seed_by_task = {"easy": 101, "medium": 102, "hard": 103}
    for task_id in ["easy", "medium", "hard"]:
        r = post(base, "/reset", {"task_id": task_id, "seed": seed_by_task[task_id]})
        check(f"POST /reset task_id={task_id} returns 200", r.status_code == 200)
        obs = r.json()
        check(f"  obs.task_id == '{task_id}'", obs.get("task_id") == task_id)
        check("  obs.step_number == 0", obs.get("step_number") == 0)
        check("  obs.done == False", obs.get("done") is False)
        check(
            "  obs.broken_payload is dict",
            isinstance(obs.get("broken_payload"), dict) and len(obs["broken_payload"]) > 0,
        )
        check(
            "  obs.mismatch_hints is list",
            isinstance(obs.get("mismatch_hints"), list) and len(obs["mismatch_hints"]) > 0,
        )

        if task_id in ("easy", "medium"):
            check(
                "  obs.target_schema is dict (not None)",
                isinstance(obs.get("target_schema"), dict),
                f"got {type(obs.get('target_schema'))}",
            )
        else:
            check(
                "  obs.target_schema is None (hard task infers schema)",
                obs.get("target_schema") is None,
            )
            check(
                "  obs.schema_examples has 3 items",
                isinstance(obs.get("schema_examples"), list) and len(obs["schema_examples"]) == 3,
            )

        r2 = get(base, "/state")
        check(f"  GET /state task_id == '{task_id}'", r2.json().get("task_id") == task_id)

    # ------------------------------------------------------------------
    # 4. step() mechanics
    # ------------------------------------------------------------------
    print("\n── 4. step() reward mechanics ────────────────────────")

    post(base, "/reset", {"task_id": "easy", "seed": 110})
    r = post(base, "/step", {"transformed_payload": {}})
    check("POST /step returns 200", r.status_code == 200)
    data = r.json()
    check("Response has 'observation' key", "observation" in data)
    check("Response has 'reward' key", "reward" in data)

    reward = data["reward"]
    check(
        "reward.score is float in [0,1]",
        isinstance(reward.get("score"), (int, float)) and 0.0 <= reward["score"] <= 1.0,
        f"score={reward.get('score')}",
    )
    check("reward.score == 0.0 for empty payload", reward["score"] == 0.0, f"got {reward.get('score')}")
    check("reward.done is bool", isinstance(reward.get("done"), bool))
    check("reward.feedback is str", isinstance(reward.get("feedback"), str))
    check("reward.field_scores is dict", isinstance(reward.get("field_scores"), dict))

    obs = post(base, "/reset", {"task_id": "easy", "seed": 111}).json()
    broken = obs["broken_payload"]
    partial = dict(broken)
    r2 = post(base, "/step", {"transformed_payload": partial})
    score_broken = r2.json()["reward"]["score"]
    check(
        "Reward varies (not always 0 or 1)",
        0.0 <= score_broken < 1.0,
        f"score with broken payload={score_broken}",
    )

    # ------------------------------------------------------------------
    # 5. Full episode — easy task with near-correct answer
    # ------------------------------------------------------------------
    print("\n── 5. Full episode (easy, correct answer) ────────────")

    easy_trial_seeds = [4, 40, 50]
    episode_scores = []
    for idx, seed in enumerate(easy_trial_seeds, start=1):
        obs = post(base, "/reset", {"task_id": "easy", "seed": seed}).json()
        broken = obs["broken_payload"]
        schema = obs["target_schema"]

        fixed = {}
        snake_to_camel = {
            "user_id": "userId",
            "order_id": "orderId",
            "amount": "amount",
            "quantity": "quantity",
            "status": "status",
            "j_unk": "jUnk",
        }
        for field, type_str in schema.items():
            camel_key = snake_to_camel.get(field, field)
            raw_val = broken[camel_key] if camel_key in broken else broken.get(field)
            if raw_val is None:
                fixed[field] = None
                continue
            try:
                if type_str == "int":
                    fixed[field] = int(float(str(raw_val).replace(",", "")))
                elif type_str == "float":
                    fixed[field] = float(str(raw_val).replace(",", "").replace("$", ""))
                else:
                    fixed[field] = str(raw_val)
            except (TypeError, ValueError):
                fixed[field] = raw_val

        r = post(base, "/step", {"transformed_payload": fixed})
        score = r.json()["reward"]["score"]
        episode_scores.append(score)
        print(f"  {INFO}  trial {idx} (seed={seed}): score={score:.3f}")

    check("Perfect fix scores 1.0 at least once", max(episode_scores) == 1.0, f"scores={episode_scores}")
    check(
        "Score varies across episodes (grader not always same)",
        len(set(round(score, 4) for score in episode_scores)) > 1,
        f"scores={episode_scores}",
    )

    # ------------------------------------------------------------------
    # 6. max_steps enforcement
    # ------------------------------------------------------------------
    print("\n── 6. max_steps enforcement ──────────────────────────")
    obs = post(base, "/reset", {"task_id": "easy", "seed": 120}).json()
    max_steps = obs["max_steps"]
    check("max_steps == 3 for easy", max_steps == 3, f"got {max_steps}")

    for step_idx in range(3):
        r = post(base, "/step", {"transformed_payload": {}})
        rd = r.json()["reward"]
        if step_idx == 2:
            check(f"Episode done after {max_steps} steps", rd["done"] is True)

    r = post(base, "/step", {"transformed_payload": {}})
    check("POST /step after done returns 400", r.status_code == 400, f"got {r.status_code}")

    # ------------------------------------------------------------------
    # 7. Episode diversity (grader not always same score)
    # ------------------------------------------------------------------
    print("\n── 7. Episode diversity ──────────────────────────────")
    diversity_seeds = [40, 42, 43, 47, 50]
    scores_easy = []
    for seed in diversity_seeds:
        post(base, "/reset", {"task_id": "easy", "seed": seed})
        r = post(base, "/step", {"transformed_payload": {"junk": 1}})
        scores_easy.append(r.json()["reward"]["score"])

    check(
        "Grader doesn't always return same score (5 episodes)",
        len(set(round(score, 4) for score in scores_easy)) > 1,
        f"scores={scores_easy}",
    )

    # ------------------------------------------------------------------
    # 8. Hard task schema_examples
    # ------------------------------------------------------------------
    print("\n── 8. Hard task schema inference setup ───────────────")
    obs = post(base, "/reset", {"task_id": "hard", "seed": 130}).json()
    examples = obs.get("schema_examples", [])
    check("Hard task provides schema_examples", len(examples) == 3)
    if examples:
        ex = examples[0]
        check("Each example has 'broken_input'", "broken_input" in ex)
        check("Each example has 'correct_output'", "correct_output" in ex)
        check(
            "correct_output has snake_case keys",
            all(isinstance(key, str) and is_snake_case_key(key) for key in ex["correct_output"].keys()),
        )

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    failed = [name for name, ok in results if not ok]

    print(f"\n{'=' * 56}")
    print(f"  Result: {passed}/{total} checks passed")
    if failed:
        print("\n  FAILED checks:")
        for name in failed:
            print(f"    ✗ {name}")
    else:
        print("\n  ALL CHECKS PASSED — safe to submit!")
    print(f"{'=' * 56}\n")

    return len(failed) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://0.0.0.0:7860",
        help="Base URL of the running DataBridgeEnv server.",
    )
    args = parser.parse_args()

    ok = run(args.base_url)
    sys.exit(0 if ok else 1)
