"""
CloudScaleRL Environment Server — built on openenv.core.EnvServer.

Follows the same pattern as dm_control_env's server/app.py:
  - Inherits EnvServer (or composes FastAPI app the openenv way)
  - _step() applies the action and returns (observation, reward, done, info)
  - _reset() initialises a fresh episode and returns the first observation
  - _get_state() returns CloudScaleState for GET /state
  - render() returns a JSON dashboard snapshot (like dm_control render=True)

The simulation physics:
  - Pod scheduling delay: 2 ticks
  - Node provisioning delay: 5 ticks
  - Latency model: M/D/1 queue approximation
  - Spot preemption: ~5% chance per tick per spot region
  - Carbon intensity: live from Electricity Maps API (falls back to static defaults)
"""

from __future__ import annotations

import json
import math
import os
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from openenv.core.env_server import EnvServer
    from openenv.core.env_server.types import StepResult
    _HAS_OPENENV = True
except ImportError:
    _HAS_OPENENV = False
    EnvServer = object  # type: ignore[assignment,misc]
    StepResult = dict   # type: ignore[assignment]

from cloudscalerl.models import (
    CloudScaleAction,
    CloudScaleObservation,
    CloudScaleReward,
    CloudScaleState,
    RegionState,
    ServiceMetrics,
)

# ── Constants ─────────────────────────────────────────────────────────────────

TRACES_DIR = Path(__file__).parent / "workload_traces"

TICK_DURATION_S = 60
# Default to realistic delays, but allow curriculum overrides via env vars.
POD_SCHEDULING_TICKS = int(os.environ.get("POD_SCHEDULING_TICKS", "2"))
NODE_PROVISIONING_TICKS = int(os.environ.get("NODE_PROVISIONING_TICKS", "5"))
POD_STARTUP_MEAN_S = 45
POD_STARTUP_STD_S = 10

NODE_COSTS: dict[str, float] = {
    "m5.large":      0.096,
    "m5.xlarge":     0.192,
    "c5.xlarge":     0.170,
    "c5.2xlarge":    0.340,
    "spot.c5.xlarge": 0.051,
    "spot.m5.large":  0.029,
}

CARBON_DEFAULTS: dict[str, float] = {
    "us-east-1":  380.0,
    "eu-west-1":  210.0,
    "ap-south-1": 720.0,
}

ACTION_SPEC: dict[str, Any] = {
    "hpa": {
        "target_replicas": [1, 100],
        "min_replicas": [1, 100],
        "max_replicas": [1, 100],
        "target_cpu_utilization": [0.1, 0.95],
    },
    "vpa": {
        "cpu_request_millicores": [10, 32000],
        "memory_request_mb": [64, 131072],
    },
    "traffic": {
        "region_weights": "Dict[str, float] — must sum to 1.0",
    },
    "node": {
        "operation": ["add", "remove", "change_type"],
        "count": [1, 20],
    },
}

OBSERVATION_SPEC: dict[str, Any] = {
    "services": {
        "replicas": "int ge 0",
        "cpu_utilization": "float 0–1",
        "memory_utilization": "float 0–1",
        "requests_per_second": "float ge 0",
        "p99_latency_ms": "float ge 0",
        "error_rate": "float 0–1",
        "cpu_request_millicores": "int ge 1",
        "memory_request_mb": "int ge 1",
        "pending_replicas": "Optional[int]",
    },
    "regions": {
        "traffic_weight": "float 0–1 (sum across regions = 1.0)",
        "node_count": "int ge 0",
        "node_type": "str",
        "cost_per_hour": "float ge 0",
        "is_degraded": "bool",
        "is_spot": "bool",
        "carbon_intensity_gco2_kwh": "float",
    },
}


# ── Internal simulation state (not exposed to agent) ─────────────────────────


@dataclass
class _ServiceState:
    replicas: int
    cpu_request_millicores: int
    memory_request_mb: int
    min_replicas: int = 1
    max_replicas: int = 50
    target_cpu_utilization: float = 0.70
    replica_history: list[int] = field(default_factory=list)


@dataclass
class _PendingScale:
    service: str
    target_replicas: int
    ticks_remaining: int


@dataclass
class _PendingNode:
    region: str
    operation: str
    node_type: str
    count: int
    ticks_remaining: int


# ── Environment class ─────────────────────────────────────────────────────────


class CloudScaleEnvServer(EnvServer):
    """
    CloudScaleRL environment — openenv EnvServer subclass.

    Instantiate directly or mount onto an existing FastAPI app:

        app = FastAPI()
        env_server = CloudScaleEnvServer()
        app.include_router(env_server.router)

    Or use standalone (openenv handles the FastAPI wiring):

        server = CloudScaleEnvServer()
        server.serve(port=8000)
    """

    def __init__(self) -> None:
        if _HAS_OPENENV:
            super().__init__()

        # Episode state — reset by _reset()
        self._task_config: dict[str, Any] = {}
        self._seed: int = 42
        self._rng: random.Random = random.Random(42)
        self._step_count: int = 0
        self._max_steps: int = 480
        self._budget_usd_per_hr: float = 50.0
        self._budget_remaining: float = 0.0
        self._episode_id: Optional[str] = None

        self._services: dict[str, _ServiceState] = {}
        self._regions: dict[str, RegionState] = {}
        self._pending_scale: list[_PendingScale] = []
        self._pending_nodes: list[_PendingNode] = []
        self._trace: list[float] = []
        self._metrics_cache: dict[str, ServiceMetrics] = {}
        self._trajectory: list[CloudScaleObservation] = []
        self._task_grader: Optional[Callable[[list[CloudScaleObservation]], float]] = None
        self._final_task_score: Optional[float] = None
        self._grader_error: Optional[str] = None

    # ── openenv EnvServer interface ───────────────────────────────────────────

    def _reset(
        self,
        task_id: str = "task1_hpa",
        seed: int = 42,
        render: bool = False,
        **kwargs: Any,
    ) -> CloudScaleObservation:
        """
        Initialise a fresh episode.
        Called by EnvServer when POST /reset is received.
        Mirrors DMControlEnv.reset() signature pattern.
        """
        from cloudscalerl.tasks.Task1hpa import (
            TASK_CONFIG as TASK1_CONFIG,
            grade_task1,
        )
        from cloudscalerl.tasks.Task2cost import (
            TASK_CONFIG as TASK2_CONFIG,
            grade_task2,
        )
        from cloudscalerl.tasks.Task3incident import (
            TASK_CONFIG as TASK3_CONFIG,
            grade_task3,
        )

        task_map: dict[
            str,
            tuple[dict[str, Any], Callable[[list[CloudScaleObservation]], float]],
        ] = {
            "task1_hpa": (TASK1_CONFIG, grade_task1),
            "task2_cost": (TASK2_CONFIG, grade_task2),
            "task3_incident": (TASK3_CONFIG, grade_task3),
        }
        if task_id not in task_map:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {list(task_map.keys())}"
            )

        self._task_config, self._task_grader = task_map[task_id]
        self._seed = seed
        self._rng = random.Random(seed)
        self._step_count = 0
        self._max_steps = self._task_config["max_steps"]
        self._budget_usd_per_hr = self._task_config.get("budget_usd_per_hr", 50.0)
        self._budget_remaining = self._budget_usd_per_hr * (
            self._max_steps * TICK_DURATION_S / 3600
        )
        self._episode_id = str(uuid.uuid4())
        self._pending_scale = []
        self._pending_nodes = []
        self._metrics_cache = {}
        self._trajectory = []
        self._final_task_score = None
        self._grader_error = None

        self._load_trace()
        self._init_cluster()

        initial_observation = self._build_observation(render=render)
        self._trajectory.append(self._snapshot_observation(initial_observation))
        return initial_observation

    def _step(
        self,
        action: CloudScaleAction,
        render: bool = False,
        **kwargs: Any,
    ) -> tuple[CloudScaleObservation, float, bool, dict[str, Any]]:
        """
        Advance one tick.
        Called by EnvServer when POST /step is received.
        Returns (observation, reward_scalar, done, info).
        """
        # 1. Constraint checking → penalty
        penalty = self._check_constraints(action)

        # 2. Apply action (with realistic delays)
        self._apply_action(action)

        # 3. Advance pending events (pod scheduling, node provisioning)
        self._advance_pending_events()

        # 4. Advance workload trace → update metrics cache
        self._advance_workload()

        # 5. Inject task-specific events (AZ failure, spot preemption, etc.)
        self._inject_events()

        # Reward Shaping: Anticipation bonus
        anticipation_bonus = 0.0
        if action.hpa and action.hpa.service in self._services:
            m = self._metrics_cache.get(action.hpa.service)
            if m and m.cpu_utilization > 0.80 and action.hpa.target_replicas:
                current_replicas = self._services[action.hpa.service].replicas
                if action.hpa.target_replicas > current_replicas:
                    anticipation_bonus = 0.2

        # 6. Compute reward components
        slo_score   = self._compute_slo_score()
        cost_score  = self._compute_cost_efficiency()
        stability   = self._compute_stability()

        raw_reward = (
            0.4 * slo_score
            + 0.3 * cost_score
            + 0.2 * stability
            + anticipation_bonus
            - 0.3 * penalty
        )
        total_reward = max(-1.0, min(1.0, raw_reward))

        # 7. Drain budget
        hourly_cost = self._current_cost_per_hour()
        self._budget_remaining -= hourly_cost * (TICK_DURATION_S / 3600)

        self._step_count += 1

        done = (
            self._step_count >= self._max_steps
            or self._budget_remaining <= -50.0
            or self._catastrophic_failure()
        )

        obs = self._build_observation(render=render)
        self._trajectory.append(self._snapshot_observation(obs))

        final_task_score: Optional[float] = None
        if done:
            final_task_score = self._compute_task_score()

        reward_detail = CloudScaleReward(
            total=total_reward,
            slo_component=slo_score,
            cost_component=cost_score,
            stability_component=stability,
            penalty=penalty,
            breakdown=self._reward_breakdown(),
        )

        info: dict[str, Any] = {
            "step": self._step_count,
            "reward_detail": reward_detail.model_dump(),
            "done": done,
        }
        if final_task_score is not None:
            info["final_task_score"] = final_task_score
        if self._grader_error is not None:
            info["grader_error"] = self._grader_error
        return obs, total_reward, done, info

    def grade_current_episode(self) -> dict[str, Any]:
        """
        Return a deterministic task score for the currently collected trajectory.

        Score is clipped to [0.0, 1.0]. This can be called before or after done.
        """
        score = self._compute_task_score()
        report: dict[str, Any] = {
            "task_id": self._task_config.get("id", ""),
            "score": score,
            "steps": self._step_count,
            "trajectory_length": len(self._trajectory),
        }
        if self._grader_error is not None:
            report["grader_error"] = self._grader_error
        return report

    def _get_state(self) -> CloudScaleState:
        """
        Return current environment metadata.
        Called by EnvServer when GET /state is received.
        Mirrors DMControlState structure.
        """
        return CloudScaleState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_config.get("id", ""),
            max_steps=self._max_steps,
            budget_usd_per_hr=self._budget_usd_per_hr,
            services=list(self._services.keys()),
            regions=list(self._regions.keys()),
            trace_pattern=self._task_config.get("trace_pattern", "diurnal"),
            action_spec=ACTION_SPEC,
            observation_spec=OBSERVATION_SPEC,
        )

    def render(self) -> dict[str, Any]:
        """
        JSON dashboard snapshot — human-readable cluster state.
        Analogous to dm_control render=True returning pixels.
        Called by GET /render.
        """
        obs = self._build_observation()
        return {
            "step": self._step_count,
            "episode_id": self._episode_id,
            "slo_status": "MET" if obs.global_slo_met else "BREACHED",
            "cost_usd_per_hr": round(obs.total_cost_usd_per_hour, 4),
            "oracle_cost_usd_per_hr": round(
                obs.counterfactual_cost_usd_per_hour or 0, 4
            ),
            "budget_remaining": round(obs.budget_remaining_usd, 2),
            "services": {
                name: {
                    "replicas": m.replicas,
                    "pending_replicas": m.pending_replicas,
                    "cpu": f"{m.cpu_utilization:.0%}",
                    "mem": f"{m.memory_utilization:.0%}",
                    "p99_ms": round(m.p99_latency_ms, 1),
                    "errors": f"{m.error_rate:.4%}",
                    "cpu_req": f"{m.cpu_request_millicores}m",
                    "mem_req": f"{m.memory_request_mb}MB",
                }
                for name, m in self._metrics_cache.items()
            },
            "regions": {
                rid: {
                    "weight": f"{r.traffic_weight:.0%}",
                    "nodes": r.node_count,
                    "type": r.node_type,
                    "cost_hr": round(r.cost_per_hour, 3),
                    "degraded": r.is_degraded,
                    "spot": r.is_spot,
                    "carbon_gco2_kwh": r.carbon_intensity_gco2_kwh,
                }
                for rid, r in self._regions.items()
            },
            "pending_events": self._build_pending_events(),
        }

    # ── Cluster initialisation ────────────────────────────────────────────────

    def _load_trace(self) -> None:
        pattern = self._task_config.get("trace_pattern", "diurnal")
        trace_file = TRACES_DIR / f"{pattern}.json"
        if trace_file.exists():
            with open(trace_file) as f:
                self._trace = json.load(f)
        else:
            self._trace = _generate_trace(
                self._seed, pattern, self._max_steps
            )

    def _init_cluster(self) -> None:
        self._services = {
            name: _ServiceState(
                replicas=2,
                cpu_request_millicores=500,
                memory_request_mb=512,
            )
            for name in self._task_config["services"]
        }

        n_regions = len(self._task_config["regions"])
        self._regions = {
            rid: RegionState(
                region_id=rid,
                traffic_weight=round(1.0 / n_regions, 4),
                node_count=3,
                node_type="m5.large",
                cost_per_hour=3 * NODE_COSTS["m5.large"],
                carbon_intensity_gco2_kwh=_fetch_carbon(rid),
            )
            for rid in self._task_config["regions"]
        }

    # ── Constraint checking ───────────────────────────────────────────────────

    def _check_constraints(self, action: CloudScaleAction) -> float:
        penalty = 0.0

        if action.hpa:
            svc = self._services.get(action.hpa.service)
            if svc is None:
                penalty += 0.5
            elif (
                action.hpa.target_replicas
                and action.hpa.target_replicas < svc.min_replicas
            ):
                penalty += 0.3  # PodDisruptionBudget violation

        if action.traffic and action.traffic.region_weights:
            total = sum(action.traffic.region_weights.values())
            if abs(total - 1.0) > 0.02:
                penalty += 0.4

        if self._budget_remaining <= 0:
            penalty += 0.2

        return min(penalty, 1.0)

    # ── Action application ────────────────────────────────────────────────────

    def _apply_action(self, action: CloudScaleAction) -> None:
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

    def _apply_hpa(self, hpa: Any) -> None:
        svc = self._services.get(hpa.service)
        if svc is None:
            return
        if hpa.min_replicas:
            svc.min_replicas = hpa.min_replicas
        if hpa.max_replicas:
            svc.max_replicas = hpa.max_replicas
        if hpa.target_cpu_utilization:
            svc.target_cpu_utilization = hpa.target_cpu_utilization
        if hpa.target_replicas:
            target = max(
                svc.min_replicas, min(svc.max_replicas, hpa.target_replicas)
            )
            self._pending_scale.append(
                _PendingScale(hpa.service, target, POD_SCHEDULING_TICKS)
            )

    def _apply_vpa(self, vpa: Any) -> None:
        svc = self._services.get(vpa.service)
        if svc is None:
            return
        if vpa.cpu_request_millicores:
            svc.cpu_request_millicores = vpa.cpu_request_millicores
        if vpa.memory_request_mb:
            svc.memory_request_mb = vpa.memory_request_mb

    def _apply_traffic(self, traffic: Any) -> None:
        if traffic.region_weights:
            for rid, w in traffic.region_weights.items():
                if rid in self._regions:
                    self._regions[rid].traffic_weight = w
        elif traffic.failover_from and traffic.failover_to:
            src = self._regions.get(traffic.failover_from)
            dst = self._regions.get(traffic.failover_to)
            if src and dst:
                dst.traffic_weight += src.traffic_weight
                src.traffic_weight = 0.0

    def _apply_node(self, node: Any) -> None:
        if node.region not in self._regions:
            return
        nt = node.node_type or self._regions[node.region].node_type
        self._pending_nodes.append(
            _PendingNode(
                node.region, node.operation, nt, node.count, NODE_PROVISIONING_TICKS
            )
        )

    # ── Pending event resolution ──────────────────────────────────────────────

    def _advance_pending_events(self) -> None:
        # Pod scheduling
        still_pending: list[_PendingScale] = []
        for ev in self._pending_scale:
            ev.ticks_remaining -= 1
            if ev.ticks_remaining <= 0:
                svc = self._services.get(ev.service)
                if svc:
                    svc.replicas = ev.target_replicas
                    svc.replica_history.append(ev.target_replicas)
            else:
                still_pending.append(ev)
        self._pending_scale = still_pending

        # Node provisioning
        still_nodes: list[_PendingNode] = []
        for ev in self._pending_nodes:
            ev.ticks_remaining -= 1
            if ev.ticks_remaining <= 0:
                region = self._regions.get(ev.region)
                if region:
                    if ev.operation == "add":
                        region.node_count += ev.count
                    elif ev.operation == "remove":
                        region.node_count = max(1, region.node_count - ev.count)
                    elif ev.operation == "change_type":
                        region.node_type = ev.node_type
                        region.is_spot = ev.node_type.startswith("spot.")
                    region.cost_per_hour = region.node_count * NODE_COSTS.get(
                        region.node_type, 0.096
                    )
            else:
                still_nodes.append(ev)
        self._pending_nodes = still_nodes

    # ── Workload simulation ───────────────────────────────────────────────────

    def _advance_workload(self) -> None:
        """
        Compute per-service metrics for this tick.
        Uses an M/D/1 queue approximation: p99 latency rises sharply as
        CPU utilisation approaches 1.0 — forces the agent to maintain headroom.
        """
        tick_idx = self._step_count % len(self._trace)
        load_mult = self._trace[tick_idx]

        for name, svc in self._services.items():
            # Effective capacity scales with replica count and per-pod CPU request.
            # Service demand is traffic-driven and should not rise with replica count.
            capacity_factor = svc.replicas * (svc.cpu_request_millicores / 500.0)
            demand_rps = load_mult * 200.0
            cpu_demand_units = load_mult * 4.0

            # CPU utilisation
            cpu_util = min(
                0.99,
                cpu_demand_units / max(capacity_factor * 4.0, 0.01),
            )
            cpu_util = max(0.0, min(0.99, cpu_util + self._rng.gauss(0, 0.02)))

            mem_util = max(
                0.0, min(0.99, 0.4 + cpu_util * 0.3 + self._rng.gauss(0, 0.01))
            )

            # M/D/1 queue: p99 ≈ base_service_time + queueing_delay * 2.5
            rho = cpu_util
            if rho < 0.99:
                base_ms = 50.0
                queue_ms = base_ms * rho / (2 * (1 - rho))
                p99 = base_ms + queue_ms * 2.5
            else:
                p99 = 2000.0

            p99 = max(20.0, p99 + self._rng.gauss(0, 5.0))

            # Error rate spikes sharply above 500ms
            error_rate = (
                min(0.05, (p99 - 500) / 10000) if p99 > 500
                else max(0.0, self._rng.gauss(0.0001, 0.00005))
            )

            self._metrics_cache[name] = ServiceMetrics(
                replicas=svc.replicas,
                cpu_utilization=round(cpu_util, 4),
                memory_utilization=round(mem_util, 4),
                requests_per_second=round(demand_rps, 1),
                p99_latency_ms=round(p99, 1),
                error_rate=round(error_rate, 6),
                cpu_request_millicores=svc.cpu_request_millicores,
                memory_request_mb=svc.memory_request_mb,
                pending_replicas=self._get_pending_replicas(name),
            )

    def _get_pending_replicas(self, service: str) -> Optional[int]:
        for ev in self._pending_scale:
            if ev.service == service:
                return ev.target_replicas
        return None

    # ── Task event injection ──────────────────────────────────────────────────

    def _inject_events(self) -> None:
        task_id = self._task_config.get("id", "")

        # Task 3: AZ degradation
        if task_id == "task3_incident":
            incident_tick = self._task_config.get("incident_tick", 30)
            incident_region = self._task_config.get("incident_region", "us-east-1")
            if self._step_count == incident_tick and incident_region in self._regions:
                self._regions[incident_region].is_degraded = True
            if self._step_count == 150 and incident_region in self._regions:
                self._regions[incident_region].is_degraded = False

        # Spot preemption: ~5% chance per tick per spot region
        for region in self._regions.values():
            if region.is_spot and self._rng.random() < 0.05:
                region.node_count = max(1, region.node_count - 1)
                region.cost_per_hour = region.node_count * NODE_COSTS.get(
                    region.node_type, 0.029
                )

    # ── Reward computation ────────────────────────────────────────────────────

    def _compute_slo_score(self) -> float:
        if not self._metrics_cache:
            return 0.5

        ok = 0
        breaches = 0
        for m in self._metrics_cache.values():
            if m.p99_latency_ms >= 200 or m.error_rate >= 0.001:
                breaches += 1
            else:
                ok += 1

        total = len(self._metrics_cache)
        # Normalized range [-1, 1]: +1 all services healthy, -1 all breached.
        return max(-1.0, min(1.0, (ok - breaches) / total))

    def _compute_cost_efficiency(self) -> float:
        actual = self._current_cost_per_hour()
        optimal = self._oracle_optimal_cost()
        if optimal <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - actual / optimal))

    def _compute_stability(self) -> float:
        score = 1.0
        for svc in self._services.values():
            hist = svc.replica_history[-3:]
            for i in range(1, len(hist)):
                if hist[i - 1] > 0:
                    change = abs(hist[i] - hist[i - 1]) / hist[i - 1]
                    if change > 0.5:
                        score -= 0.2
        return max(0.0, score)

    def _reward_breakdown(self) -> dict[str, float]:
        bd: dict[str, float] = {"cost_per_hr": self._current_cost_per_hour()}
        for name, m in self._metrics_cache.items():
            bd[f"{name}_p99"] = m.p99_latency_ms
            bd[f"{name}_err"] = m.error_rate
        return bd

    def _snapshot_observation(self, obs: CloudScaleObservation) -> CloudScaleObservation:
        # Keep trajectory compact and deterministic for task graders.
        return obs.model_copy(update={"dashboard_json": None}, deep=True)

    def _compute_task_score(self) -> Optional[float]:
        if self._task_grader is None:
            return None

        try:
            raw_score = float(self._task_grader(self._trajectory))
            clipped = max(0.0, min(1.0, raw_score))
            self._final_task_score = clipped
            self._grader_error = None
            return clipped
        except Exception as exc:
            self._grader_error = str(exc)
            self._final_task_score = None
            return None

    def _current_cost_per_hour(self) -> float:
        return sum(r.cost_per_hour for r in self._regions.values())

    def _oracle_optimal_cost(self) -> float:
        total_replicas = sum(s.replicas for s in self._services.values())
        nodes_needed = max(1, math.ceil(total_replicas / 2))
        return nodes_needed * NODE_COSTS["m5.large"]

    def _catastrophic_failure(self) -> bool:
        return bool(self._regions) and all(
            r.is_degraded for r in self._regions.values()
        )

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_observation(self, render: bool = False) -> CloudScaleObservation:
        if not self._metrics_cache:
            for name, svc in self._services.items():
                self._metrics_cache[name] = ServiceMetrics(
                    replicas=svc.replicas,
                    cpu_utilization=0.30,
                    memory_utilization=0.40,
                    requests_per_second=100.0,
                    p99_latency_ms=80.0,
                    error_rate=0.0001,
                    cpu_request_millicores=svc.cpu_request_millicores,
                    memory_request_mb=svc.memory_request_mb,
                )

        global_slo = all(
            m.p99_latency_ms < 200 and m.error_rate < 0.001
            for m in self._metrics_cache.values()
        )

        return CloudScaleObservation(
            step=self._step_count,
            services=dict(self._metrics_cache),
            regions=dict(self._regions),
            total_cost_usd_per_hour=self._current_cost_per_hour(),
            budget_remaining_usd=self._budget_remaining,
            global_slo_met=global_slo,
            pending_events=self._build_pending_events(),
            counterfactual_cost_usd_per_hour=self._oracle_optimal_cost(),
            dashboard_json=self.render() if render else None,
        )

    def _build_pending_events(self) -> list[str]:
        events: list[str] = []

        for ev in self._pending_scale:
            events.append(
                f"scale_{ev.service}_to_{ev.target_replicas}_t+{ev.ticks_remaining}"
            )
        for ev in self._pending_nodes:
            events.append(
                f"node_{ev.operation}_{ev.region}_t+{ev.ticks_remaining}"
            )

        # Traffic spike look-ahead (5 ticks)
        for ahead in range(1, 6):
            future_idx = (self._step_count + ahead) % len(self._trace)
            now_idx = self._step_count % len(self._trace)
            if self._trace[future_idx] > self._trace[now_idx] * 1.5:
                events.append(f"traffic_spike_t+{ahead}")
                break

        # AZ degradation look-ahead (task 3)
        if self._task_config.get("id") == "task3_incident":
            incident_tick = self._task_config.get("incident_tick", 30)
            ticks_to = incident_tick - self._step_count
            if 0 < ticks_to <= 5:
                region = self._task_config.get("incident_region", "us-east-1")
                events.append(f"az_degradation_{region}_t+{ticks_to}")

        return events


# ── Helpers ───────────────────────────────────────────────────────────────────


def _generate_trace(seed: int, pattern: str, length: int) -> list[float]:
    """Synthetic workload trace generator — used when JSON file is absent."""
    rng = random.Random(seed)
    trace: list[float] = []

    for t in range(length):
        hour = (t * TICK_DURATION_S / 3600) % 24

        if pattern == "diurnal":
            base = 0.4 + 0.4 * math.sin(math.pi * (hour - 6) / 12)
            boost = 0.15 * math.exp(-((hour - 12) ** 2) / 2)
            val = base + boost + rng.gauss(0, 0.02)
            if t in (60, 61, 240, 241, 420, 421):
                val = 1.8  # CPU spikes for task 1

        elif pattern == "flash_sale":
            weekend = 0.7 if (t * TICK_DURATION_S / 3600 / 24) % 7 >= 5 else 1.0
            base = weekend * (0.3 + 0.2 * math.sin(math.pi * (hour - 6) / 12))
            val = base * (3.0 if 240 <= t <= 270 else 1.0) + rng.gauss(0, 0.02)

        elif pattern == "incident":
            base = 0.5 + 0.2 * math.sin(math.pi * hour / 24)
            if 30 <= t <= 45:
                base *= 1.3
            val = base + rng.gauss(0, 0.02)

        else:
            val = 0.5 + rng.gauss(0, 0.05)

        trace.append(round(max(0.05, min(3.0, val)), 4))

    return trace


def _fetch_carbon(region_id: str) -> float:
    """Live carbon intensity from Electricity Maps API. Falls back to static defaults."""
    api_key = os.environ.get("ELECTRICITY_MAPS_API_KEY")
    if not api_key:
        return CARBON_DEFAULTS.get(region_id, 400.0)

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