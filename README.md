---
title: Hyperlocal Inventory Curation
emoji: "🛒"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - quick-commerce
  - data-curation
  - catalog-ops
---

# Hyperlocal Inventory Curation

`hyperlocal_inventory_curation_env` is a real-world OpenEnv benchmark for **Automated Dataset Curation for Hyper-Local Inventory**.

The environment simulates a catalog-operations workflow used in quick commerce platforms such as Zepto, Blinkit, and Instamart. An AI agent receives noisy merchant and dark-store inventory records and must clean them into reliable structured catalog data used by downstream systems like search, pricing, substitutions, and fulfillment.

This is not a toy environment. It models a real operational task:
- normalize product titles
- standardize units and quantities
- assign taxonomy
- merge duplicate records
- correct obvious pricing anomalies
- flag ambiguous records for review

## Why This Environment Matters

Hyper-local commerce relies on fragmented, messy inventory feeds from local stores, merchant uploads, POS exports, and internal catalog sync jobs. Poor curation causes:
- broken search results
- bad substitutions
- incorrect pricing
- fulfillment errors
- poor customer experience

This benchmark evaluates whether an agent can perform safe, policy-aware, multi-step inventory curation under realistic operational constraints.

## Environment Overview

Each episode is one inventory curation batch.

The agent sees:
- a batch of noisy inventory records
- policy snippets
- current batch state
- previous actions
- remaining steps
- reward details

The agent must improve the batch over several steps using structured actions.

## Action Space

The environment uses a typed `InventoryCurationAction` model.

Supported actions:
- `normalize_title`
- `normalize_size`
- `assign_category`
- `merge_duplicate_records`
- `correct_price`
- `fill_missing_attribute`
- `flag_for_review`
- `finalize_batch`

Example action payload:

```json
{
  "action_type": "assign_category",
  "record_id": "med_3",
  "field_name": "category",
  "value": "household",
  "reason": "Assign category using known product identity."
}
```

## Observation Space

The environment returns a typed `InventoryCurationObservation`.

It includes:
- `task_id`
- `difficulty`
- `objective`
- `records`
- `policy_snippets`
- `action_history`
- `remaining_steps`
- `last_action_error`
- `reward_details`
- `done`

## State Space

The internal state is represented by `InventoryCurationState`.

It tracks:
- current task metadata
- episode id
- step count
- working records
- merged pairs
- flagged records
- action history
- last reward
- cumulative score
- progress score
- done flag

## Reward Design

The environment provides dense reward across the full trajectory.

The agent receives positive reward for:
- correct title normalization
- correct unit normalization
- correct category assignment
- correct duplicate resolution
- correct price correction
- correct escalation of ambiguous records

The agent is penalized for:
- invalid categories
- invalid actions
- incorrect or repeated low-value actions
- premature finalization
- unsafe curation behavior

This makes the benchmark suitable for both evaluation and learning.

## Tasks

The benchmark includes 3 deterministic tasks with increasing difficulty.

### `easy_title_unit_cleanup`

Difficulty: `easy`

A small single-store batch with obvious title cleanup, unit normalization, and category assignment.

Examples:
- `Coca cola 1000ml`
- `Amul taaza toned milk 500 ML`
- `Banana 6 pc`

### `medium_duplicate_price_fix`

Difficulty: `medium`

A batch with duplicate records, one obvious pricing anomaly, and multiple category assignments.

Examples:
- `Coke 1 ltr`
- `Coca Cola 1000 ml`
- `Surf exel easy wash 1kg`

### `hard_ambiguous_multisource_batch`

Difficulty: `hard`

A multi-source batch with ambiguity, near-duplicates, incorrect prices, and cases that should be flagged instead of guessed.

Examples:
- `Tomato loose 1kg`
- `Tomatoes 1 kg pack`
- `Coke Zero can 300ml`
- `Coca Cola Zero 330 ml can`

This task is intentionally safety-oriented: the best agent is not the one that guesses the most, but the one that knows when to escalate ambiguity.

## Grading

Each task has a deterministic grader that returns a score in `[0.0, 1.0]`.

The grader evaluates:
- title normalization
- size normalization
- category assignment
- duplicate handling
- price handling
- escalation quality

This allows the environment to distinguish weak, partial, and strong agent behavior.

## Baseline Inference

The baseline script is `inference.py` at the project root.

It:
- uses the OpenAI client
- reads `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME`
- runs all 3 tasks
- emits strict `[START]`, `[STEP]`, and `[END]` logs
- includes heuristic fallback for reproducible behavior even when model access is weak or unavailable

Run it with:

```powershell
$env:HF_TOKEN="your_token"
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

## Local Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If your project uses `uv`, you can also install with:

```powershell
uv sync
```

## Running Tests

```powershell
pytest -q
```

Current status:
- tests pass
- environment validation passes
- Docker build works
- API endpoints work locally
- Hugging Face Space is deployed and responding

## OpenEnv Validation

Validate the environment with:

```powershell
openenv validate --verbose
```

## Running the Server Locally

```powershell
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Useful endpoints:
- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`

## Docker

Build the image:

```powershell
docker build -t hyperlocal-inventory-curation .
```

Run the container:

```powershell
docker run -p 8000:8000 hyperlocal-inventory-curation
```

## Hugging Face Space Deployment

This project is deployed as a Docker Space.

Space page:
- [https://huggingface.co/spaces/MadhuraMadhu/hyperlocal-inventory-curation-env](https://huggingface.co/spaces/MadhuraMadhu/hyperlocal-inventory-curation-env)

Live app endpoint:
- [https://madhuramadhu-hyperlocal-inventory-curation-env.hf.space](https://madhuramadhu-hyperlocal-inventory-curation-env.hf.space)

Typical deployment flow:
1. Create a new Hugging Face Space.
2. Choose `Docker` as the SDK.
3. Push this repository to the Space.
4. Add secrets if needed:
   - `HF_TOKEN`
   - `API_BASE_URL`
   - `MODEL_NAME`

You can also use OpenEnv tooling:

```powershell
openenv push
```

## Example API Usage

Local reset example:

```powershell
Invoke-WebRequest -UseBasicParsing `
  -Uri "http://127.0.0.1:8000/reset" `
  -Method POST `
  -ContentType "application/json" `
  -Body "{}" | Select-Object -ExpandProperty Content
```

Local step example:

```powershell
Invoke-WebRequest -UseBasicParsing `
  -Uri "http://127.0.0.1:8000/step" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"action":{"action_type":"normalize_title","record_id":"easy_1","field_name":"normalized_title","value":"Coca Cola 1 L"}}' |
Select-Object -ExpandProperty Content
```

Deployed health check example:

```powershell
Invoke-WebRequest -UseBasicParsing `
  -Uri "https://madhuramadhu-hyperlocal-inventory-curation-env.hf.space/health" |
Select-Object -ExpandProperty Content
```

## Project Structure

```text
hyperlocal_inventory_curation_env/
├── __init__.py
├── client.py
├── grader.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── tasks.py
├── tests/
│   └── test_environment.py
└── server/
    ├── __init__.py
    ├── app.py
    ├── environment.py
    └── requirements.txt
```

## Submission Readiness

This environment currently satisfies the core hackathon requirements:
- real-world task
- full OpenEnv workflow
- typed models
- 3 deterministic tasks
- agent graders with scores in `[0,1]`
- shaped rewards
- root `inference.py`
- Docker build
- OpenEnv validation
- local API verification
- Hugging Face Space deployment

## Summary

This benchmark evaluates whether an AI agent can safely and effectively curate noisy merchant inventory into usable structured catalog data for hyper-local commerce systems.

It is designed to reward operationally useful behavior, not just generic text output.
