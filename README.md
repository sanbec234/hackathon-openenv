---
title: DataBridgeEnv
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# DataBridgeEnv

DataBridgeEnv is an OpenEnv RL environment for **debugging API contract breakages** between services.

It models a real backend reliability problem: one service emits malformed payloads, another service enforces a strict contract, and an agent must repair payloads through `reset() / step() / state()` interactions.

## Why This Matters

In production systems, contract drift causes outages, silent data corruption, and integration failures. This environment trains and evaluates agents on realistic payload repair behavior:

- rename fields (`camelCase` -> `snake_case`)
- coerce types (`"42"` -> `42`, `"49.99"` -> `49.99`)
- flatten nested shapes
- convert scalar/list forms
- add missing fields and drop extras

## Task Design

Three tasks with increasing difficulty and deterministic generation from `(task_id, seed)`.

| Task | Difficulty | Max Steps | Core Challenge |
|---|---|---:|---|
| `easy` | Easy | 3 | field-name + type-coercion mismatches |
| `medium` | Medium | 4 | shape, array/scalar, missing/extra, and type mismatches |
| `hard` | Hard | 5 | all mismatch classes + schema inference from examples |

### Hard Task: Schema Inference

`hard` does not expose `target_schema`. The agent must infer it from three `(broken_input, correct_output)` examples, then fix a new live broken payload. `correct_output` keys are normalized to snake_case.

## Reward and Episode Mechanics

Grading is deterministic and returns a scalar score in `[0.0, 1.0]` plus field-level details.

Per-field scoring:
- `1.0` correct name, type, value
- `0.5` correct type but wrong value
- `0.25` wrong type
- `0.15` alias/wrong-name field detected
- `0.0` missing field

Overall score is the mean per-field score minus small extra-field penalties.

Episode terminates when:
- score reaches `1.0`, or
- `max_steps` is reached.

## Seed-Driven Determinism and Diversity

- Same `(task_id, seed)` -> same payload and expected output
- Different seeds -> structurally different payloads and score distributions
- Generator is parametric (not fixed templates), with varying numeric/text formats, optional fields, and shape variants

## OpenEnv Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `POST /baseline`

## Quick Start

```bash
./run.sh
```

Server runs on `http://localhost:7860`.

Run validation:

```bash
python integration_test.py --base-url http://localhost:7860
```

Run baseline:

```bash
python baseline.py --episodes 1
```

## Hugging Face Deployment

Use this workflow to avoid PATH and CLI issues on fresh machines.

### 1. Install Hugging Face CLI

```bash
./setup_hf.sh
```

This installs `huggingface_hub[cli]` and verifies `huggingface-cli` is available (including `.venv/bin/huggingface-cli` when using the project venv).
If your installed `huggingface_hub` exposes only `hf`, the script creates a compatibility shim so `huggingface-cli login` and `huggingface-cli whoami` still work.

### 2. Login (Option A: CLI)

```bash
huggingface-cli login
huggingface-cli whoami
```

### 3. Login (Option B: fallback if CLI is not available)

```bash
git config --global credential.helper store
git push
```

When prompted:
- Username: your Hugging Face username
- Password: your Hugging Face access token

### 4. Push to Hugging Face Space

```bash
git remote add space https://huggingface.co/spaces/<username>/<space-name>
git push space main
```

If `space` already exists:

```bash
git remote set-url space https://huggingface.co/spaces/<username>/<space-name>
git push space main
```

## Baseline Results

Two baseline modes are supported in `baseline.py`:

- **Rule-based fallback** (default when `OPENAI_API_KEY` is missing)
- **LLM baseline** (enabled when `OPENAI_API_KEY` is set)

Reference run (`python baseline.py --episodes 1 --json`):

| Baseline Mode | Easy Mean | Medium Mean | Hard Mean | Notes |
|---|---:|---:|---:|---|
| Rule-based fallback | 1.0000 | 1.0000 | 1.0000 | Deterministic, no external API dependency |
| LLM (`gpt-4o-mini`) | Requires API key | Requires API key | Requires API key | Same harness, model-driven action generation |

Example output shape:

```json
{
  "easy": {"scores": [1.0], "mean": 1.0, "status": "ok"},
  "medium": {"scores": [1.0], "mean": 1.0, "status": "ok"},
  "hard": {"scores": [1.0], "mean": 1.0, "status": "ok"}
}
```

## Repository Files

Core submission files:

- `app.py`
- `environment.py`
- `generator.py`
- `grader.py`
- `models.py`
- `baseline.py`
- `integration_test.py`
- `requirements.txt`
- `README.md`
- `run.sh`
- `openenv.yaml`
- `Dockerfile`
