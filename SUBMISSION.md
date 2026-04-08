# CloudScaleRL Hackathon Submission

## Project
- Name: CloudScaleRL
- Space SDK: Docker
- App port: 8000
- Space tag: openenv

## Environment URL
- Replace with your deployed URL: `https://<user>-cloudscalerl.hf.space`

## Quick Validation Commands
```bash
# Local validation
HF_TOKEN=<token> ./pre_submit_check.sh

# Remote validation against your Space
API_BASE_URL=https://<user>-cloudscalerl.hf.space \
ENV_URL=https://<user>-cloudscalerl.hf.space \
HF_TOKEN=<token> \
./pre_submit_check.sh
```

## OpenEnv Compliance Evidence
1. `CloudScaleAction`, `CloudScaleObservation`, and `CloudScaleState` inherit OpenEnv base types in `cloudscalerl/models.py`.
2. Environment client integration uses `EnvClient` in `cloudscalerl/client.py`.
3. Environment server integration uses `EnvServer` in `cloudscalerl/server/cloudscalerl_env.py`.
4. Task definitions and graders are declared in `cloudscalerl/openenv.yaml`.
5. OpenEnv-oriented endpoints exist in `cloudscalerl/server/app.py`: `/metadata`, `/schema`, `/tasks`, `/grade`, `/health`.

## Inference Contract
`inference.py` emits required episode lines:
- `[START] task=<task> env=<benchmark> model=<model>`
- `[STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`

## Reproducibility
```bash
API_BASE_URL=http://127.0.0.1:8000 \
ENV_URL=http://127.0.0.1:8000 \
HF_TOKEN=<token> \
SEED=42 \
INFERENCE_TEMPERATURE=0.0 \
ENABLE_LLM_PROBE=false \
python inference.py task1_hpa
```

The baseline policy is deterministic and hardcoded; OpenAI client initialization remains for submission compatibility.
