"""
Task 1 — Easy: Reactive HPA
=============================
Single service (api-gateway), single region (us-east-1).
Predictable diurnal traffic with CPU spikes at ticks 60, 240, 420.
Episode length: 480 ticks (8 simulated hours).

Objective: Keep p99 < 200ms and cpu_utilization < 75% throughout.
Score: fraction of ticks where api-gateway SLO is met.
"""

from __future__ import annotations

from typing import Any

from cloudscalerl.models import Observation

TASK_CONFIG: dict[str, Any] = {
    "id": "task1_hpa",
    "name": "Reactive HPA",
    "difficulty": "easy",
    "max_steps": 480,
    "trace_pattern": "diurnal",
    "services": ["api-gateway"],
    "regions": ["us-east-1"],
    "budget_usd_per_hr": 50.0,
}


def grade_task1(trajectory: list[Observation]) -> float:
    """
    Score = fraction of ticks where api-gateway meets SLO.
    Fully deterministic. Range: [0.0, 1.0].
    """
    if not trajectory:
        return 0.0

    slo_ticks = 0
    for obs in trajectory:
        svc = obs.services.get("api-gateway")
        if svc is None:
            continue
        if svc.p99_latency_ms < 200 and svc.error_rate < 0.001:
            slo_ticks += 1

    return slo_ticks / len(trajectory)


def describe_task1() -> str:
    return """
Task 1 — Reactive HPA (Easy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Service:  api-gateway (single service)
Region:   us-east-1 (single region)
Episode:  480 ticks = 8 simulated hours
Budget:   $50/hr (generous — focus on SLO, not cost)

Traffic pattern: Diurnal sine wave
CPU spikes:      Tick 60, 240, 420 → 85% utilization

Scoring: fraction of ticks with p99 < 200ms AND error_rate < 0.1%
Max score: 1.0 (SLO met every tick)

Key agent behaviours to learn:
  1. Pre-scale 2 ticks before the spike arrives (pod scheduling delay)
  2. Scale back down after spike to save cost
  3. Don't thrash replicas (>50% change in 3 ticks penalised)
"""