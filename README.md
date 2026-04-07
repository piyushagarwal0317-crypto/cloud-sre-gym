---
title: CloudScaleRL
sdk: docker
app_port: 8000
tags:
  - openenv
---

# CloudScaleRL

CloudScaleRL is an OpenEnv benchmark for training and evaluating autonomous SRE agents on real cloud autoscaling workflows.

The full project documentation is in [cloudscalerl/README.md](cloudscalerl/README.md).

## Quick start

1. Build container:
   docker build -t cloudscalerl .
2. Run server:
   docker run --rm -p 8000:8000 cloudscalerl
3. Validate environment:
   openenv validate http://localhost:8000

## Inference entrypoint

The submission script is [inference.py](inference.py) and supports:
- API_BASE_URL (default: http://localhost:8000)
- MODEL_NAME (default: gpt-4o)
- HF_TOKEN (required)

## Hugging Face Space setup

Use a Docker Space for this repo.

1. Create a new Space with SDK set to `Docker`.
2. Push this repository to the Space.
3. In Space `Settings -> Secrets`, add:
   - `HF_TOKEN`: use a **Fine-grained** token with **read-only** permissions.
4. In Space `Settings -> Variables`, optionally add:
   - `MODEL_NAME=hardcoded-controller`
   - `API_BASE_URL=http://127.0.0.1:8000`
   - `BENCHMARK_NAME=cloudscalerl`

Reference template: [hf_space.env.example](hf_space.env.example)
Quick print helper: `./pre_submit_check.sh --print-hf-space-settings`

Notes:
- Keep `HF_TOKEN` in Secrets only.
- `MODEL_NAME` is used as an output label in `[START]` lines.
- This project does not require runtime write permissions for normal inference runs.

## Presubmit check

Run one command before submitting:

HF_TOKEN=<your_token> ./pre_submit_check.sh

This checks:
- root inference.py presence and required env-var contract
- OpenAI client import and initialization markers in inference.py
- missing HF_TOKEN failure behavior
- server health and auto-start (if needed)
- strict [START]/[STEP]/[END] output format for selected tasks
- optional openenv validate and docker build smoke checks

Useful options:
- AUTO_START_SERVER=true|false (default: true)
- RUN_OPENENV_VALIDATE=auto|true|false (default: auto)
- RUN_DOCKER_BUILD=true|false (default: false)
- PYTHON_BIN=<path_to_python>
- --print-hf-space-settings (prints exact Space Secrets/Variables to paste)

Examples:
- HF_TOKEN=<your_token> ./pre_submit_check.sh task1_hpa
- HF_TOKEN=<your_token> RUN_OPENENV_VALIDATE=true ./pre_submit_check.sh
- HF_TOKEN=<your_token> RUN_DOCKER_BUILD=true ./pre_submit_check.sh
- ./pre_submit_check.sh --print-hf-space-settings
