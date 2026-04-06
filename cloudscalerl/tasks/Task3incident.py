"""
Task 3 — Hard: Multi-Region Incident Response
===============================================
3 regions: us-east-1, eu-west-1, ap-south-1
3 services: api-gateway, worker, db-proxy
AZ degradation in us-east-1 at tick 30 (50% packet loss simulated).
Episode: 300 ticks (5 simulated hours).

Agent must:
  1. Detect the degradation (or anticipate from pending_events)
  2. Shift ALL traffic away from us-east-1 immediately
  3. Scale up VPA/HPA in eu-west-1 and ap-south-1 to handle extra load
  4. Maintain global SLO throughout the incident (ticks 30–90)
  5. Manage cost under $20/hr during incident
  6. Handle recovery cleanly (ticks 91–150) — don't leave stale over-provisioning

Score: 0.5 * availability + 0.3 * recovery_slo + 0.2 * cost_ok
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from cloudscalerl.models import Observation

TASK_CONFIG: dict[str, Any] = {
    "id": "task3_incident",
    "name": "Multi-Region Incident Response",
    "difficulty": "hard",
    "max_steps": 300,
    "trace_pattern": "incident",
    "services": ["api-gateway", "worker", "db-proxy"],
    "regions": ["us-east-1", "eu-west-1", "ap-south-1"],
    "budget_usd_per_hr": 20.0,
    "incident_tick": 30,
    "incident_region": "us-east-1",
}

INCIDENT_START = 30
INCIDENT_END = 90
RECOVERY_END = 150
BUDGET_INCIDENT = 20.0
INCIDENT_REGION = "us-east-1"


def grade_task3(trajectory: list[Observation]) -> float:
    """
    Composite score across incident and recovery windows.

    availability (incident window 30–90):
      - 1.0 if agent shifted traffic away from degraded region
        (non-degraded regions carry > 80% weight)
      - 0.0 if agent left traffic in degraded region

    recovery_slo (recovery window 91–150):
      - Fraction of ticks where global_slo_met is True

    cost_ok (incident window):
      - Fraction of ticks where total_cost < $20/hr
    """
    if not trajectory:
        return 0.0

    incident = trajectory[INCIDENT_START:INCIDENT_END]
    recovery = trajectory[INCIDENT_END:RECOVERY_END]

    # --- Availability score ---
    avail_scores: list[float] = []
    for obs in incident:
        degraded_region = obs.regions.get(INCIDENT_REGION)
        if degraded_region and degraded_region.is_degraded:
            # Good: non-degraded regions carry >80% of traffic
            non_degraded_weight = sum(
                r.traffic_weight
                for rid, r in obs.regions.items()
                if not r.is_degraded
            )
            avail_scores.append(1.0 if non_degraded_weight > 0.8 else 0.0)
        else:
            # Region not yet degraded or already recovered
            avail_scores.append(1.0)

    availability = mean(avail_scores) if avail_scores else 1.0

    # --- Recovery SLO score ---
    recovery_slo = mean(
        1.0 if obs.global_slo_met else 0.0 for obs in recovery
    ) if recovery else 1.0

    # --- Cost score during incident ---
    cost_ok = mean(
        1.0 if obs.total_cost_usd_per_hour < BUDGET_INCIDENT else 0.0
        for obs in incident
    ) if incident else 1.0

    return 0.5 * availability + 0.3 * recovery_slo + 0.2 * cost_ok


def describe_task3() -> str:
    return """
Task 3 — Multi-Region Incident Response (Hard)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Services: api-gateway, worker, db-proxy (3 services)
Regions:  us-east-1 (will degrade), eu-west-1, ap-south-1 (3 regions)
Episode:  300 ticks = 5 simulated hours
Budget:   $20/hr during incident

Timeline:
  Tick 0–29:   Normal operations. Pre-position resources.
  Tick 30:     us-east-1 enters AZ degradation (is_degraded=True).
               pending_events will show this 5 ticks in advance.
  Tick 30–90:  Incident window. INCIDENT SCORING.
  Tick 91–150: Recovery window. Recovery SLO scoring.
  Tick 151+:   us-east-1 recovers. Wind down and rebalance.

Scoring: 0.5 × availability + 0.3 × recovery_SLO + 0.2 × cost_ok
  availability: did you route away from degraded region? (>80% non-degraded)
  recovery_SLO: global SLO met during ticks 91–150
  cost_ok:      total cost < $20/hr during incident

Key agent behaviours to learn:
  1. Watch pending_events for 'az_degradation_us-east-1_t+N' at tick ~25
  2. At tick 25–28: start shifting traffic to eu-west-1 and ap-south-1
  3. At tick 28: pre-scale api-gateway and worker in healthy regions (2 tick delay!)
  4. At tick 30+: set us-east-1 traffic_weight to 0.0 immediately
  5. Monitor cost — ap-south-1 has high carbon AND higher cost; eu-west-1 preferred
  6. At tick 90+: gradually reintroduce us-east-1 as it recovers at tick 150
  7. Scale DOWN redundant replicas in healthy regions to recover budget

Carbon hint: eu-west-1 (210 gCO2/kWh) is greener than ap-south-1 (720 gCO2/kWh).
             Prefer eu-west-1 for the traffic shift.
"""