"""
Task 2 — Medium: Cost-Aware Multi-Service
==========================================
4 services: frontend, api, worker, db-proxy
2 regions: us-east-1, eu-west-1
Budget cap: $15/hr
Traffic: weekly pattern + embedded flash-sale spike
Episode: 720 ticks (12 simulated hours)

Objective: SLO >= 99% on all services AND total cost <= $15/hr.
Score: 0.6 * slo_score + 0.4 * cost_score
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from cloudscalerl.models import Observation

TASK_CONFIG: dict[str, Any] = {
    "id": "task2_cost",
    "name": "Cost-Aware Multi-Service",
    "difficulty": "medium",
    "max_steps": 720,
    "trace_pattern": "flash_sale",
    "services": ["frontend", "api", "worker", "db-proxy"],
    "regions": ["us-east-1", "eu-west-1"],
    "budget_usd_per_hr": 15.0,
}

BUDGET_CAP = 15.0


def grade_task2(trajectory: list[Observation]) -> float:
    """
    Composite score: 60% SLO compliance + 40% cost compliance.
    SLO compliance: fraction of ticks where ALL services meet SLO.
    Cost compliance: fraction of ticks where total_cost <= $15/hr.
    """
    if not trajectory:
        return 0.0

    slo_scores: list[float] = []
    cost_scores: list[float] = []

    for obs in trajectory:
        # SLO: all services must meet latency and error rate thresholds
        all_slo_met = all(
            svc.p99_latency_ms < 200 and svc.error_rate < 0.001
            for svc in obs.services.values()
        )
        slo_scores.append(1.0 if all_slo_met else 0.0)

        # Cost: must be at or under budget
        cost_ok = obs.total_cost_usd_per_hour <= BUDGET_CAP
        cost_scores.append(1.0 if cost_ok else 0.0)

    slo_score = mean(slo_scores)
    cost_score = mean(cost_scores)
    return 0.6 * slo_score + 0.4 * cost_score


def describe_task2() -> str:
    return """
Task 2 — Cost-Aware Multi-Service (Medium)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Services: frontend, api, worker, db-proxy (4 services)
Regions:  us-east-1, eu-west-1 (2 regions)
Episode:  720 ticks = 12 simulated hours
Budget:   $15/hr HARD CAP

Traffic pattern: Weekly baseline + 3x flash-sale spike at tick 240

Scoring: 0.6 × (SLO ticks / total) + 0.4 × (on-budget ticks / total)

Key agent behaviours to learn:
  1. Right-size all 4 services independently — they have different load profiles
  2. Use VPA to reduce per-pod resource requests before adding replicas
  3. Shift some traffic to eu-west-1 (cheaper if node_count optimised)
  4. Pre-scale for the flash-sale spike (visible in pending_events)
  5. Scale DOWN aggressively after spike — cost budget is tight
  6. Spot instances can help cost but risk preemption — manage carefully

Flash sale timing: Ticks 240–270. Load 3× normal. Pre-scale by tick 238.
"""