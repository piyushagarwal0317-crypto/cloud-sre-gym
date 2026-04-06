import json

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) managing a Kubernetes cluster.
Your job: observe real-time cluster metrics each tick, reason about risks, and
output a scaling action. You are the agent in a CloudScaleRL environment.

=== TIMING (CRITICAL) ===
- Each tick = 60 seconds of simulated time.
- Pod scheduling takes 2 ticks. PRE-SCALE before traffic arrives.
- Node provisioning takes 5 ticks. Plan node changes well ahead.
- Pod startup adds ~45s latency. Factor this into latency decisions.

=== SLO RULES ===
- NEVER let p99 latency exceed 200ms on any service.
- NEVER let error_rate exceed 0.1%.
- If a region is marked DEGRADED, shift ALL traffic away immediately.

=== OUTPUT FORMAT ===
Think step by step:
1. Identify the biggest risk (latency breach? cost overrun? region failure?)
2. Decide what action addresses that risk without causing a new problem.
3. Output ONLY a valid JSON object. No markdown, no preamble.

ACTION SCHEMA:
{
  "hpa": {"service": str, "target_replicas": int} | null,
  "no_op": false
}
"""

def make_example(user_text, assistant_json):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_json}
        ]
    }

dataset = [
    # Scenario 1: High CPU, Latency climbing -> Scale Up
    make_example(
        "=== Tick 10 ===\nBudget remaining: $45.00  |  Cost now: $1.20/hr\nGlobal SLO met: True\n\nSERVICES:\n  api-gateway ✅\n    replicas=3  cpu=85%  mem=40%  p99=180ms  rps=250.0  errors=0.0000%\n\nThink step by step, then output your JSON action.",
        "{\n  \"hpa\": {\"service\": \"api-gateway\", \"target_replicas\": 6},\n  \"no_op\": false\n}"
    ),
    # Scenario 2: Normal Load -> Do Nothing
    make_example(
        "=== Tick 11 ===\nBudget remaining: $44.50  |  Cost now: $1.20/hr\nGlobal SLO met: True\n\nSERVICES:\n  api-gateway ✅\n    replicas=6  cpu=45%  mem=40%  p99=50ms  rps=250.0  errors=0.0000%\n\nThink step by step, then output your JSON action.",
        "{\n  \"no_op\": true\n}"
    ),
    # Scenario 3: Very Low Load -> Scale Down to save cost
    make_example(
        "=== Tick 50 ===\nBudget remaining: $30.00  |  Cost now: $2.40/hr\nGlobal SLO met: True\n\nSERVICES:\n  api-gateway ✅\n    replicas=10  cpu=15%  mem=40%  p99=20ms  rps=50.0  errors=0.0000%\n\nThink step by step, then output your JSON action.",
        "{\n  \"hpa\": {\"service\": \"api-gateway\", \"target_replicas\": 3},\n  \"no_op\": false\n}"
    ),
    # Scenario 4: Impending Traffic Spike -> Pre-scale
    make_example(
        "=== Tick 58 ===\nBudget remaining: $28.00  |  Cost now: $0.60/hr\nGlobal SLO met: True\n\nSERVICES:\n  api-gateway ✅\n    replicas=3  cpu=30%  mem=40%  p99=45ms  rps=100.0  errors=0.0000%\n\n⏰ UPCOMING: traffic_spike_t+2\n\nThink step by step, then output your JSON action.",
        "{\n  \"hpa\": {\"service\": \"api-gateway\", \"target_replicas\": 8},\n  \"no_op\": false\n}"
    ),
    # Scenario 5: SLO Breach Recovery
    make_example(
        "=== Tick 100 ===\nBudget remaining: $20.00  |  Cost now: $0.80/hr\nGlobal SLO met: False\n\nSERVICES:\n  api-gateway ⚠️ SLO BREACH\n    replicas=4  cpu=98%  mem=60%  p99=340ms  rps=600.0  errors=0.0050%\n\nThink step by step, then output your JSON action.",
        "{\n  \"hpa\": {\"service\": \"api-gateway\", \"target_replicas\": 15},\n  \"no_op\": false\n}"
    ),
    # Scenario 6: Replicas pending -> Wait
    make_example(
        "=== Tick 101 ===\nBudget remaining: $19.90  |  Cost now: $0.80/hr\nGlobal SLO met: False\n\nSERVICES:\n  api-gateway ⚠️ SLO BREACH\n    replicas=4 (pending→15)  cpu=95%  mem=60%  p99=310ms  rps=600.0  errors=0.0030%\n\nThink step by step, then output your JSON action.",
        "{\n  \"no_op\": true\n}"
    ),
    # Scenario 7: High cost, mild latency -> Minor scale down
    make_example(
        "=== Tick 200 ===\nBudget remaining: $5.00  |  Cost now: $5.00/hr\nGlobal SLO met: True\n\nSERVICES:\n  api-gateway ✅\n    replicas=20  cpu=20%  mem=30%  p99=30ms  rps=300.0  errors=0.0000%\n\nThink step by step, then output your JSON action.",
        "{\n  \"hpa\": {\"service\": \"api-gateway\", \"target_replicas\": 10},\n  \"no_op\": false\n}"
    ),
    # Scenario 8: Steady state
    make_example(
        "=== Tick 210 ===\nBudget remaining: $4.00  |  Cost now: $2.50/hr\nGlobal SLO met: True\n\nSERVICES:\n  api-gateway ✅\n    replicas=10  cpu=65%  mem=30%  p99=80ms  rps=300.0  errors=0.0000%\n\nThink step by step, then output your JSON action.",
        "{\n  \"no_op\": true\n}"
    ),
    # Scenario 9: High CPU but upcoming drop
    make_example(
        "=== Tick 250 ===\nBudget remaining: $2.00  |  Cost now: $2.50/hr\nGlobal SLO met: True\n\nSERVICES:\n  api-gateway ✅\n    replicas=10  cpu=80%  mem=30%  p99=150ms  rps=380.0  errors=0.0000%\n\n⏰ UPCOMING: traffic_drop_t+1\n\nThink step by step, then output your JSON action.",
        "{\n  \"no_op\": true\n}"
    ),
    # Scenario 10: Spike landed
    make_example(
        "=== Tick 60 ===\nBudget remaining: $26.00  |  Cost now: $1.60/hr\nGlobal SLO met: True\n\nSERVICES:\n  api-gateway ✅\n    replicas=8  cpu=60%  mem=40%  p99=90ms  rps=400.0  errors=0.0000%\n\nThink step by step, then output your JSON action.",
        "{\n  \"no_op\": true\n}"
    )
]

with open("sre_finetuning_data.jsonl", "w") as f:
    for entry in dataset:
        f.write(json.dumps(entry) + "\n")

print("Generated dummy synthetic dataset: sre_finetuning_data.jsonl")