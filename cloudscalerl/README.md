# CloudScaleRL

CloudScaleRL is a Kubernetes-inspired RL benchmark for training and evaluating autonomous SRE agents on realistic cloud scaling operations. The agent observes live cluster telemetry and chooses scaling actions every simulated minute.

## Environment Description and Motivation

The environment models real SRE work, not a toy game:

- latency and error-rate SLO management
- horizontal and vertical scaling decisions
- budget-aware infrastructure operation
- multi-region traffic management and incident response
- optional carbon-aware routing tradeoffs

This makes it suitable for training and evaluating LLM-based control policies for production-like cloud operations.

Current baseline controller: deterministic hardcoded cloud policy.
The policy uses production-style heuristics (SLO-first HPA, incident failover,
budget-aware downscaling, VPA right-sizing, node cooldowns, and optional
carbon-aware traffic rebalance).

## Action Space

The top-level action type is `CloudScaleAction` with optional sub-actions:

- `hpa`: adjust replicas, min/max bounds, or CPU target for one service
- `vpa`: adjust CPU and memory requests for one service
- `traffic`: change region traffic weights or perform explicit failover
- `node`: add/remove/change node type in a region
- `no_op`: explicit wait action

Actions are validated by Pydantic models in `models.py`.

## Observation Space

Each step returns a typed `CloudScaleObservation` with:

- `step`
- `services`: per-service metrics (replicas, CPU, memory, p99 latency, error rate)
- `regions`: per-region state (traffic weight, nodes, node type, cost, degradation)
- `total_cost_usd_per_hour`
- `budget_remaining_usd`
- `global_slo_met`
- `pending_events`
- `counterfactual_cost_usd_per_hour`

## Tasks and Expected Difficulty

| Task ID | Difficulty | Objective | Grader (0.0-1.0) |
|---|---|---|---|
| task1_hpa | easy | Single-service reactive autoscaling under diurnal spikes | Fraction of ticks with SLO met |
| task2_cost | medium | Multi-service operation under strict budget and flash-sale surge | 0.6 * SLO compliance + 0.4 * cost compliance |
| task3_incident | hard | Multi-region failover and recovery during simulated AZ degradation | 0.5 * availability + 0.3 * recovery SLO + 0.2 * cost |

Task metadata and grader paths are declared in `openenv.yaml`.

## Reward Design

Dense step reward combines positive and negative signals over the full trajectory:

$$R = 0.4 \cdot \text{SLO} + 0.3 \cdot \text{CostEff} + 0.2 \cdot \text{Stability} + \text{AnticipationBonus} - 0.3 \cdot \text{Penalty}$$

Where penalties cover undesirable behavior such as invalid traffic weights, invalid service targets, and operating over budget.

## Setup and Usage

### Python setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run environment server

```bash
uvicorn cloudscalerl.server.app:app --host 127.0.0.1 --port 8000
```

### Validate OpenEnv endpoint

```bash
openenv validate http://localhost:8000
```

### Run baseline client

```bash
python -m cloudscalerl.client task1_hpa task2_cost task3_incident
```

### Run submission inference script

```bash
export API_BASE_URL="http://localhost:8000"
export ENV_URL="http://localhost:8000"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="<token>"
export SEED=42
export INFERENCE_TEMPERATURE=0.0
export ENABLE_LLM_PROBE=false
python inference.py
```

`inference.py` emits required line formats:

- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`

### Evaluator endpoint checks

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/tasks
curl -fsS http://localhost:8000/metadata
curl -fsS http://localhost:8000/schema
curl -fsS -X POST http://localhost:8000/mcp \
	-H "Content-Type: application/json" \
	-d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'
```

### Docker usage

```bash
docker build -t cloudscalerl .
docker run --rm -p 8000:8000 cloudscalerl
```

## Baseline Scores

Example baseline summary (hardcoded controller) from project experiments:

| Metric | Static Rule Baseline | CloudScaleRL Hardcoded Cloud Policy |
|---|---|---|
| SLO adherence | 84.2% | 97.8% |
| Average cost per hour | $18.40 | $12.10 |
| Carbon intensity | 420 gCO2/kWh | 315 gCO2/kWh |
| Aggregate success score | 0.62 | 0.89 |

## Future Implementation

LLM reasoning controller is planned as a future optional mode.

Planned approach:

- keep current deterministic policy as the reliability and safety baseline
- add LLM planner for long-horizon strategy and incident sequencing
- enforce hard safety guardrails before action execution
- fall back to deterministic policy when LLM output is invalid or unavailable
