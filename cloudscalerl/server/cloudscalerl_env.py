"""
CloudScaleRL Core Environment
================================
Simulates a Kubernetes cluster with:
  - Realistic action delays (pod scheduling: 2 ticks, node provisioning: 5 ticks)
  - HPA / VPA / Traffic routing / Node management
  - Workload traces (diurnal, flash-sale, incident)
  - Spot instance preemption
  - Carbon intensity per region
  - Pending events visible to agent
  - Counterfactual oracle cost
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from cloudscalerl.models import (
    Action,
    HPAAction,
    NodeAction,
    Observation,
    RegionState,
    Reward,
    ServiceMetrics,
    TrafficAction,
    VPAAction,
)

TRACES_DIR = Path(__file__).parent / "workload_traces"

# Static carbon intensity fallbacks (gCO2/kWh) when Electricity Maps API is absent
CARBON_DEFAULTS = {
    "us-east-1": 380.0,
    "eu-west-1": 210.0,
    "ap-south-1": 720.0,
}

NODE_COSTS: dict[str, float] = {
    "m5.large": 0.096,
    "m5.xlarge": 0.192,
    "c5.xlarge": 0.170,
    "c5.2xlarge": 0.340,
    "spot.c5.xlarge": 0.051,   # 70% cheaper than on-demand
    "spot.m5.large": 0.029,
}

POD_SCHEDULING_TICKS = 2
NODE_PROVISIONING_TICKS = 5
POD_STARTUP_MEAN_S = 45
POD_STARTUP_STD_S = 10
TICK_DURATION_S = 60


@dataclass
class PendingScaleEvent:
    service: str
    target_replicas: int
    ticks_remaining: int


@dataclass
class PendingNodeEvent:
    region: str
    operation: str
    node_type: str
    count: int
    ticks_remaining: int


@dataclass
class ServiceState:
    replicas: int
    cpu_request_millicores: int
    memory_request_mb: int
    min_replicas: int = 1
    max_replicas: int = 50
    target_cpu_utilization: float = 0.7
    # Replica history for thrash detection
    replica_history: list[int] = field(default_factory=list)


class CloudScaleEnvironment:
    """
    Core RL environment. Instantiated per-episode via reset().
    Not thread-safe — run one episode per instance.
    """

    def __init__(self, task_config: dict[str, Any], seed: int = 42) -> None:
        self.task_config = task_config
        self.seed = seed
        self.rng = random.Random(seed)
        self._load_trace()
        self.step_count = 0
        self.max_steps = task_config["max_steps"]
        self.budget_usd_per_hr = task_config.get("budget_usd_per_hr", 50.0)
        self.budget_remaining = self.budget_usd_per_hr * (self.max_steps * TICK_DURATION_S / 3600)
        self.services: dict[str, ServiceState] = {}
        self.regions: dict[str, RegionState] = {}
        self.pending_scale: list[PendingScaleEvent] = []
        self.pending_nodes: list[PendingNodeEvent] = []
        self._init_cluster()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load_trace(self) -> None:
        pattern = self.task_config.get("trace_pattern", "diurnal")
        trace_file = TRACES_DIR / f"{pattern}.json"
        if trace_file.exists():
            with open(trace_file) as f:
                self.trace: list[float] = json.load(f)
        else:
            # Generate synthetic trace on-the-fly
            self.trace = _generate_trace(self.seed, pattern, self.task_config["max_steps"])

    def _init_cluster(self) -> None:
        for svc_name in self.task_config["services"]:
            self.services[svc_name] = ServiceState(
                replicas=2,
                cpu_request_millicores=500,
                memory_request_mb=512,
            )

        for i, region_id in enumerate(self.task_config["regions"]):
            n = len(self.task_config["regions"])
            self.regions[region_id] = RegionState(
                region_id=region_id,
                traffic_weight=round(1.0 / n, 4),
                node_count=3,
                node_type="m5.large",
                cost_per_hour=3 * NODE_COSTS["m5.large"],
                is_degraded=False,
                carbon_intensity_gco2_kwh=_fetch_carbon(region_id),
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self.__init__(self.task_config, self.seed)
        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        # 1. Validate & apply penalty for constraint violations
        penalty = self._check_constraints(action)

        # 2. Apply action (with delays)
        self._apply_action(action)

        # 3. Advance simulation clock
        self._advance_pending_events()
        self._advance_workload()

        # 4. Inject task-specific events (AZ failure, spot preemption, etc.)
        self._inject_events()

        # 5. Compute reward components
        slo_score = self._compute_slo_score()
        cost_score = self._compute_cost_efficiency()
        stability = self._compute_stability()

        # 6. Shape reward
        raw = (
            0.4 * slo_score
            + 0.3 * cost_score
            + 0.2 * stability
            - 0.3 * penalty
        )
        total_reward = max(-1.0, min(1.0, raw))

        # 7. Update budget
        hourly_cost = self._current_cost_per_hour()
        self.budget_remaining -= hourly_cost * (TICK_DURATION_S / 3600)

        self.step_count += 1

        done = (
            self.step_count >= self.max_steps
            or self.budget_remaining <= -50.0  # allow small overage before terminating
            or self._catastrophic_failure()
        )

        obs = self._build_observation()
        reward = Reward(
            total=total_reward,
            slo_component=slo_score,
            cost_component=cost_score,
            stability_component=stability,
            penalty=penalty,
            breakdown=self._reward_breakdown(),
        )
        return obs, reward, done, {"step": self.step_count}

    # ── Constraint Checking ───────────────────────────────────────────────────

    def _check_constraints(self, action: Action) -> float:
        penalty = 0.0

        if action.hpa:
            svc = self.services.get(action.hpa.service)
            if svc is None:
                penalty += 0.5  # unknown service
            elif action.hpa.target_replicas and action.hpa.target_replicas < svc.min_replicas:
                penalty += 0.3  # below PodDisruptionBudget min

        if action.traffic and action.traffic.region_weights:
            total = sum(action.traffic.region_weights.values())
            if abs(total - 1.0) > 0.02:
                penalty += 0.4  # weights don't sum to 1

        if self.budget_remaining <= 0:
            penalty += 0.2  # already over budget

        return min(penalty, 1.0)

    # ── Action Application ────────────────────────────────────────────────────

    def _apply_action(self, action: Action) -> None:
        if action.no_op:
            return

        if action.hpa:
            self._apply_hpa(action.hpa)
        if action.vpa:
            self._apply_vpa(action.vpa)
        if action.traffic:
            self._apply_traffic(action.traffic)
        if action.node:
            self._apply_node(action.node)

    def _apply_hpa(self, hpa: HPAAction) -> None:
        svc = self.services.get(hpa.service)
        if svc is None:
            return
        if hpa.min_replicas:
            svc.min_replicas = hpa.min_replicas
        if hpa.max_replicas:
            svc.max_replicas = hpa.max_replicas
        if hpa.target_cpu_utilization:
            svc.target_cpu_utilization = hpa.target_cpu_utilization
        if hpa.target_replicas:
            target = max(svc.min_replicas, min(svc.max_replicas, hpa.target_replicas))
            # Schedule with 2-tick delay
            self.pending_scale.append(
                PendingScaleEvent(hpa.service, target, POD_SCHEDULING_TICKS)
            )

    def _apply_vpa(self, vpa: VPAAction) -> None:
        svc = self.services.get(vpa.service)
        if svc is None:
            return
        if vpa.cpu_request_millicores:
            svc.cpu_request_millicores = vpa.cpu_request_millicores
        if vpa.memory_request_mb:
            svc.memory_request_mb = vpa.memory_request_mb

    def _apply_traffic(self, traffic: TrafficAction) -> None:
        if traffic.region_weights:
            for rid, w in traffic.region_weights.items():
                if rid in self.regions:
                    self.regions[rid].traffic_weight = w
        elif traffic.failover_from and traffic.failover_to:
            src = self.regions.get(traffic.failover_from)
            dst = self.regions.get(traffic.failover_to)
            if src and dst:
                dst.traffic_weight += src.traffic_weight
                src.traffic_weight = 0.0

    def _apply_node(self, node: NodeAction) -> None:
        region = self.regions.get(node.region)
        if region is None:
            return
        nt = node.node_type or region.node_type
        self.pending_nodes.append(
            PendingNodeEvent(node.region, node.operation, nt, node.count, NODE_PROVISIONING_TICKS)
        )

    # ── Pending Event Resolution ──────────────────────────────────────────────

    def _advance_pending_events(self) -> None:
        # Pod scaling
        resolved_scale = []
        for ev in self.pending_scale:
            ev.ticks_remaining -= 1
            if ev.ticks_remaining <= 0:
                svc = self.services.get(ev.service)
                if svc:
                    svc.replicas = ev.target_replicas
                    svc.replica_history.append(ev.target_replicas)
            else:
                resolved_scale.append(ev)
        self.pending_scale = resolved_scale

        # Node provisioning
        resolved_nodes = []
        for ev in self.pending_nodes:
            ev.ticks_remaining -= 1
            if ev.ticks_remaining <= 0:
                region = self.regions.get(ev.region)
                if region:
                    if ev.operation == "add":
                        region.node_count += ev.count
                    elif ev.operation == "remove":
                        region.node_count = max(1, region.node_count - ev.count)
                    elif ev.operation == "change_type":
                        region.node_type = ev.node_type
                        region.is_spot = ev.node_type.startswith("spot.")
                    region.cost_per_hour = region.node_count * NODE_COSTS.get(region.node_type, 0.096)
            else:
                resolved_nodes.append(ev)
        self.pending_nodes = resolved_nodes

    # ── Workload Simulation ───────────────────────────────────────────────────

    def _advance_workload(self) -> None:
        """
        Compute simulated metrics for this tick based on the workload trace.
        Uses a simplified queueing model: latency ~ load / (capacity - load).
        """
        tick_idx = self.step_count % len(self.trace)
        load_multiplier = self.trace[tick_idx]

        for svc_name, svc in self.services.items():
            capacity = svc.replicas * (svc.cpu_request_millicores / 500.0)
            demand = load_multiplier * 2.0 * svc.replicas  # base RPS proportional to replicas

            # CPU utilization
            cpu_util = min(0.99, (load_multiplier * 0.6) / max(capacity / svc.replicas, 0.01))
            cpu_util += self.rng.gauss(0, 0.02)
            cpu_util = max(0.0, min(0.99, cpu_util))

            # Memory utilization (slower to change)
            mem_util = 0.4 + (cpu_util * 0.3) + self.rng.gauss(0, 0.01)
            mem_util = max(0.0, min(0.99, mem_util))

            # Latency: M/D/1 queue model approximation
            rho = cpu_util  # server utilization
            if rho < 0.99:
                base_latency = 50.0  # ms service time
                queueing_latency = base_latency * rho / (2 * (1 - rho))
                p99_latency = base_latency + queueing_latency * 2.5
            else:
                p99_latency = 2000.0  # saturated

            p99_latency = max(20.0, p99_latency + self.rng.gauss(0, 5.0))

            # Error rate rises sharply when latency > 500ms
            if p99_latency > 500:
                error_rate = min(0.05, (p99_latency - 500) / 10000)
            else:
                error_rate = max(0.0, self.rng.gauss(0.0001, 0.00005))

            # Update metrics (in-place via dict mutation — ServiceState holds scalars)
            self._service_metrics_cache[svc_name] = ServiceMetrics(
                replicas=svc.replicas,
                cpu_utilization=round(cpu_util, 4),
                memory_utilization=round(mem_util, 4),
                requests_per_second=round(demand, 1),
                p99_latency_ms=round(p99_latency, 1),
                error_rate=round(error_rate, 6),
                cpu_request_millicores=svc.cpu_request_millicores,
                memory_request_mb=svc.memory_request_mb,
                pending_replicas=self._get_pending_replicas(svc_name),
            )

    @property
    def _service_metrics_cache(self) -> dict[str, ServiceMetrics]:
        if not hasattr(self, "_svc_cache"):
            self._svc_cache: dict[str, ServiceMetrics] = {}
        return self._svc_cache

    def _get_pending_replicas(self, service: str) -> int | None:
        for ev in self.pending_scale:
            if ev.service == service:
                return ev.target_replicas
        return None

    # ── Event Injection ───────────────────────────────────────────────────────

    def _inject_events(self) -> None:
        task_id = self.task_config.get("id", "")

        # Task 3: AZ degradation at tick 30
        if task_id == "task3_incident":
            incident_tick = self.task_config.get("incident_tick", 30)
            incident_region = self.task_config.get("incident_region", "us-east-1")
            if self.step_count == incident_tick:
                if incident_region in self.regions:
                    self.regions[incident_region].is_degraded = True
            # Recovery at tick 150
            if self.step_count == 150:
                if incident_region in self.regions:
                    self.regions[incident_region].is_degraded = False

        # Spot preemption (random, ~5% chance per tick per spot region)
        for rid, region in self.regions.items():
            if region.is_spot and self.rng.random() < 0.05:
                region.node_count = max(1, region.node_count - 1)
                region.cost_per_hour = region.node_count * NODE_COSTS.get(region.node_type, 0.029)

    # ── Reward Computation ────────────────────────────────────────────────────

    def _compute_slo_score(self) -> float:
        if not self._service_metrics_cache:
            return 0.5
        compliant = sum(
            1
            for m in self._service_metrics_cache.values()
            if m.p99_latency_ms < 200 and m.error_rate < 0.001
        )
        return compliant / len(self._service_metrics_cache)

    def _compute_cost_efficiency(self) -> float:
        actual = self._current_cost_per_hour()
        optimal = self._oracle_optimal_cost()
        if optimal <= 0:
            return 1.0
        efficiency = 1.0 - (actual / optimal)
        return max(0.0, min(1.0, efficiency))

    def _compute_stability(self) -> float:
        """Penalise thrashing: replica count changed >50% in last 3 ticks."""
        score = 1.0
        for svc in self.services.values():
            hist = svc.replica_history[-3:]
            if len(hist) >= 2:
                for i in range(1, len(hist)):
                    if hist[i - 1] > 0:
                        change = abs(hist[i] - hist[i - 1]) / hist[i - 1]
                        if change > 0.5:
                            score -= 0.2
        return max(0.0, score)

    def _reward_breakdown(self) -> dict[str, float]:
        breakdown = {}
        for name, m in self._service_metrics_cache.items():
            breakdown[f"{name}_p99"] = m.p99_latency_ms
            breakdown[f"{name}_err"] = m.error_rate
        breakdown["cost_per_hr"] = self._current_cost_per_hour()
        return breakdown

    def _current_cost_per_hour(self) -> float:
        return sum(r.cost_per_hour for r in self.regions.values())

    def _oracle_optimal_cost(self) -> float:
        """
        Greedy bin-packing oracle: minimum cost to serve current load.
        Uses smallest node type that can handle the demand.
        """
        # Simplified: 1 m5.large per 2 replicas across all services
        total_replicas = sum(s.replicas for s in self.services.values())
        nodes_needed = max(1, math.ceil(total_replicas / 2))
        return nodes_needed * NODE_COSTS["m5.large"]

    def _catastrophic_failure(self) -> bool:
        """Episode ends early if ALL regions are degraded simultaneously."""
        return all(r.is_degraded for r in self.regions.values())

    # ── Observation Builder ───────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        if not self._service_metrics_cache:
            # First tick — build synthetic initial metrics
            for svc_name in self.services:
                svc = self.services[svc_name]
                self._service_metrics_cache[svc_name] = ServiceMetrics(
                    replicas=svc.replicas,
                    cpu_utilization=0.3,
                    memory_utilization=0.4,
                    requests_per_second=100.0,
                    p99_latency_ms=80.0,
                    error_rate=0.0001,
                    cpu_request_millicores=svc.cpu_request_millicores,
                    memory_request_mb=svc.memory_request_mb,
                )

        global_slo = all(
            m.p99_latency_ms < 200 and m.error_rate < 0.001
            for m in self._service_metrics_cache.values()
        )

        pending_events = self._build_pending_events()

        return Observation(
            step=self.step_count,
            services=dict(self._service_metrics_cache),
            regions=dict(self.regions),
            total_cost_usd_per_hour=self._current_cost_per_hour(),
            budget_remaining_usd=self.budget_remaining,
            global_slo_met=global_slo,
            pending_events=pending_events,
            counterfactual_cost_usd_per_hour=self._oracle_optimal_cost(),
        )

    def _build_pending_events(self) -> list[str]:
        events: list[str] = []

        # Pending scale events
        for ev in self.pending_scale:
            events.append(f"scale_{ev.service}_to_{ev.target_replicas}_t+{ev.ticks_remaining}")

        # Pending node events
        for ev in self.pending_nodes:
            events.append(f"node_{ev.operation}_{ev.region}_t+{ev.ticks_remaining}")

        # Upcoming trace spikes (look-ahead 5 ticks)
        for lookahead in range(1, 6):
            future_idx = (self.step_count + lookahead) % len(self.trace)
            current_idx = self.step_count % len(self.trace)
            if self.trace[future_idx] > self.trace[current_idx] * 1.5:
                events.append(f"traffic_spike_t+{lookahead}")
                break

        # Upcoming task events
        task_id = self.task_config.get("id", "")
        if task_id == "task3_incident":
            incident_tick = self.task_config.get("incident_tick", 30)
            ticks_to_incident = incident_tick - self.step_count
            if 0 < ticks_to_incident <= 5:
                region = self.task_config.get("incident_region", "us-east-1")
                events.append(f"az_degradation_{region}_t+{ticks_to_incident}")

        return events

    def render(self) -> dict[str, Any]:
        """JSON dashboard snapshot for debugging and judging."""
        obs = self._build_observation()
        return {
            "step": self.step_count,
            "slo_status": "MET" if obs.global_slo_met else "BREACHED",
            "cost_usd_per_hr": round(obs.total_cost_usd_per_hour, 4),
            "oracle_cost_usd_per_hr": round(obs.counterfactual_cost_usd_per_hour or 0, 4),
            "budget_remaining": round(obs.budget_remaining_usd, 2),
            "services": {
                name: {
                    "replicas": m.replicas,
                    "cpu": f"{m.cpu_utilization:.0%}",
                    "p99_ms": m.p99_latency_ms,
                    "errors": f"{m.error_rate:.4%}",
                }
                for name, m in self._service_metrics_cache.items()
            },
            "regions": {
                rid: {
                    "weight": f"{r.traffic_weight:.0%}",
                    "nodes": r.node_count,
                    "degraded": r.is_degraded,
                    "carbon": r.carbon_intensity_gco2_kwh,
                }
                for rid, r in self.regions.items()
            },
            "pending_events": obs.pending_events,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _generate_trace(seed: int, pattern: str, length: int) -> list[float]:
    """
    Generate a synthetic workload trace.
    Patterns: diurnal, flash_sale, incident, adversarial
    """
    rng = random.Random(seed)
    trace: list[float] = []

    for t in range(length):
        hour = (t * TICK_DURATION_S / 3600) % 24

        if pattern == "diurnal":
            # Sine wave with business-hours peak
            base = 0.4 + 0.4 * math.sin(math.pi * (hour - 6) / 12)
            lunch_boost = 0.15 * math.exp(-((hour - 12) ** 2) / 2)
            val = base + lunch_boost + rng.gauss(0, 0.02)

        elif pattern == "flash_sale":
            # Weekly baseline + 3x spike at tick 240
            base = 0.3 + 0.2 * math.sin(math.pi * (hour - 6) / 12)
            spike = 3.0 if 240 <= t <= 270 else 1.0
            val = base * spike + rng.gauss(0, 0.02)

        elif pattern == "incident":
            # Steady traffic that continues through AZ degradation
            base = 0.5 + 0.2 * math.sin(math.pi * hour / 24)
            val = base + rng.gauss(0, 0.02)

        elif pattern == "adversarial":
            # Designed to fool threshold policies: spike then immediately drop
            phase = t % 20
            if phase < 2:
                val = 1.8  # trigger over-provisioning
            elif phase < 10:
                val = 0.2  # then drop — wasted resources
            else:
                val = 0.7  # normal load
            val += rng.gauss(0, 0.02)

        else:
            val = 0.5 + rng.gauss(0, 0.05)

        trace.append(max(0.05, min(3.0, val)))

    return trace


def _fetch_carbon(region_id: str) -> float:
    """
    Fetch live carbon intensity from Electricity Maps API.
    Falls back to static defaults if API key is absent.
    """
    api_key = os.environ.get("ELECTRICITY_MAPS_API_KEY")
    if not api_key:
        return CARBON_DEFAULTS.get(region_id, 400.0)

    # Map region IDs to Electricity Maps zone codes
    zone_map = {
        "us-east-1": "US-MIDA-PJM",
        "eu-west-1": "IE",
        "ap-south-1": "IN-SO",
    }
    zone = zone_map.get(region_id, "US-MIDA-PJM")

    try:
        import requests

        resp = requests.get(
            f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}",
            headers={"auth-token": api_key},
            timeout=5,
        )
        if resp.ok:
            return float(resp.json().get("carbonIntensity", CARBON_DEFAULTS[region_id]))
    except Exception:
        pass

    return CARBON_DEFAULTS.get(region_id, 400.0)