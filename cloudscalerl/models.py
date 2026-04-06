"""
CloudScaleRL Pydantic models — built on openenv.core base types.

Follows the same pattern as dm_control_env:
  - CloudScaleAction    inherits openenv Action
  - CloudScaleObservation  inherits openenv Observation
  - CloudScaleState     inherits openenv State

This makes the environment plug-and-play with EnvClient and EnvServer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Fallback: plain Pydantic BaseModel so the file works standalone
    from pydantic import BaseModel as _Base

    Action = _Base        # type: ignore[assignment,misc]
    Observation = _Base   # type: ignore[assignment,misc]
    State = _Base         # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models (not top-level openenv types — used as nested fields)
# ─────────────────────────────────────────────────────────────────────────────


class ServiceMetrics(Action):
    """
    Per-service metrics snapshot at one tick.
    Nested inside CloudScaleObservation.observations["<service_name>"].
    """

    replicas: int = Field(..., ge=0)
    cpu_utilization: float = Field(..., ge=0.0, le=1.0, description="Fraction 0–1")
    memory_utilization: float = Field(..., ge=0.0, le=1.0)
    requests_per_second: float = Field(..., ge=0.0)
    p99_latency_ms: float = Field(..., ge=0.0)
    error_rate: float = Field(..., ge=0.0, le=1.0)
    cpu_request_millicores: int = Field(..., ge=1, description="VPA-managed CPU request")
    memory_request_mb: int = Field(..., ge=1, description="VPA-managed memory request")
    pending_replicas: Optional[int] = Field(
        None,
        description="Replicas scheduled but not yet active (2-tick scheduling delay)",
    )


class RegionState(Action):
    """
    Per-region cluster state snapshot.
    Nested inside CloudScaleObservation.regions["<region_id>"].
    """

    region_id: str = Field(..., description='e.g. "us-east-1", "eu-west-1"')
    traffic_weight: float = Field(..., ge=0.0, le=1.0, description="Fraction of total traffic")
    node_count: int = Field(..., ge=0)
    node_type: str = Field(..., description='e.g. "m5.large", "spot.c5.xlarge"')
    cost_per_hour: float = Field(..., ge=0.0)
    is_degraded: bool = Field(False, description="True during simulated AZ failure")
    is_spot: bool = Field(False, description="True if running on spot instances")
    carbon_intensity_gco2_kwh: float = Field(
        200.0,
        description="gCO2/kWh from Electricity Maps API — lower is greener",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CloudScaleObservation  (openenv Observation)
# ─────────────────────────────────────────────────────────────────────────────


class CloudScaleObservation(Observation):
    """
    Full cluster observation returned each tick.

    Analogous to DMControlObservation.observations — the agent reads this
    dict-of-metrics and decides what scaling action to take.

    Example:
        >>> obs.services["api-gateway"].p99_latency_ms
        142.3
        >>> obs.regions["us-east-1"].is_degraded
        True
        >>> "traffic_spike_t+3" in obs.pending_events
        True
    """

    step: int = Field(..., ge=0)

    # Per-service metrics keyed by service name (e.g. "api-gateway", "worker")
    services: Dict[str, ServiceMetrics] = Field(
        ...,
        description="Keyed by service name",
    )

    # Per-region state keyed by region_id
    regions: Dict[str, RegionState] = Field(
        ...,
        description="Keyed by region_id",
    )

    total_cost_usd_per_hour: float = Field(..., ge=0.0)
    budget_remaining_usd: float = Field(
        ..., description="Decreases each tick. Can go negative (budget breach)."
    )
    global_slo_met: bool = Field(
        ..., description="True only if ALL services meet SLO this tick"
    )

    # Look-ahead events the agent can act on proactively
    pending_events: List[str] = Field(
        default_factory=list,
        description=(
            "Upcoming events visible to agent, e.g. "
            '["traffic_spike_t+3", "az_degradation_us-east-1_t+2", '
            '"node_preemption_eu-west-1_t+1"]'
        ),
    )

    # Oracle cost for learning signal (like dm_control's info dict)
    counterfactual_cost_usd_per_hour: Optional[float] = Field(
        None,
        description="What a greedy oracle would spend — agent can learn from the delta",
    )

    # Optional rendered dashboard (analogous to DMControlObservation.pixels)
    dashboard_json: Optional[Dict[str, Any]] = Field(
        None,
        description="JSON dashboard snapshot when render=True is passed to reset/step",
    )

    @model_validator(mode="after")
    def check_traffic_weights_sum(self) -> "CloudScaleObservation":
        if self.regions:
            total = sum(r.traffic_weight for r in self.regions.values())
            if abs(total - 1.0) > 0.05:
                raise ValueError(
                    f"Region traffic weights sum to {total:.3f}, expected ~1.0"
                )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Action sub-types
# ─────────────────────────────────────────────────────────────────────────────


class HPAAction(Action):
    """
    Horizontal Pod Autoscaler directive for one service.

    Example:
        >>> HPAAction(service="api-gateway", target_replicas=6)
        >>> HPAAction(service="worker", target_cpu_utilization=0.7)
    """

    service: str
    target_replicas: Optional[int] = Field(None, ge=1, le=100)
    min_replicas: Optional[int] = Field(None, ge=1)
    max_replicas: Optional[int] = Field(None, ge=1, le=100)
    target_cpu_utilization: Optional[float] = Field(
        None, ge=0.1, le=0.95,
        description="Target CPU fraction for the HPA autoscaler",
    )

    @model_validator(mode="after")
    def check_min_max(self) -> "HPAAction":
        if self.min_replicas and self.max_replicas:
            if self.min_replicas > self.max_replicas:
                raise ValueError("min_replicas must be <= max_replicas")
        if self.target_replicas and self.min_replicas:
            if self.target_replicas < self.min_replicas:
                raise ValueError("target_replicas must be >= min_replicas")
        return self


class VPAAction(Action):
    """
    Vertical Pod Autoscaler directive — adjusts per-pod resource requests.

    Example:
        >>> VPAAction(service="api-gateway", cpu_request_millicores=1000, memory_request_mb=1024)
    """

    service: str
    cpu_request_millicores: Optional[int] = Field(None, ge=10, le=32000)
    memory_request_mb: Optional[int] = Field(None, ge=64, le=131072)


class TrafficAction(Action):
    """
    Traffic routing directive across regions.

    region_weights must sum to 1.0 if provided.

    Example (shift all traffic to eu-west-1):
        >>> TrafficAction(region_weights={"us-east-1": 0.0, "eu-west-1": 1.0})

    Example (quick failover):
        >>> TrafficAction(failover_from="us-east-1", failover_to="eu-west-1")
    """

    region_weights: Optional[Dict[str, float]] = Field(
        None,
        description="New traffic weight per region. Must sum to 1.0.",
    )
    failover_from: Optional[str] = Field(None, description="Region to drain traffic from")
    failover_to: Optional[str] = Field(None, description="Region to receive drained traffic")
    canary_service: Optional[str] = None
    canary_percent: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("region_weights")
    @classmethod
    def weights_must_sum_to_one(
        cls, v: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        if v is not None:
            total = sum(v.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"region_weights must sum to 1.0, got {total:.3f}")
        return v


class NodeAction(Action):
    """
    Node provisioning / decommissioning directive.
    Node provisioning takes 5 ticks to take effect.

    Example:
        >>> NodeAction(region="eu-west-1", operation="add", node_type="spot.c5.xlarge", count=2)
    """

    region: str
    operation: Literal["add", "remove", "change_type"]
    node_type: Optional[str] = Field(
        None, description='e.g. "m5.large", "c5.xlarge", "spot.c5.xlarge"'
    )
    count: int = Field(1, ge=1, le=20)


# ─────────────────────────────────────────────────────────────────────────────
# CloudScaleAction  (openenv Action)
# ─────────────────────────────────────────────────────────────────────────────


class CloudScaleAction(Action):
    """
    Top-level agent action for one tick.
    Passed directly to EnvClient.step() — analogous to DMControlAction.

    All sub-actions are optional. Set only what you want to change this tick.
    Set no_op=True to explicitly skip (environment still advances one tick).

    Example — scale up and reroute simultaneously:
        >>> CloudScaleAction(
        ...     hpa=HPAAction(service="api-gateway", target_replicas=8),
        ...     traffic=TrafficAction(region_weights={"us-east-1": 0.0, "eu-west-1": 1.0}),
        ... )

    Example — do nothing:
        >>> CloudScaleAction(no_op=True)
    """

    hpa: Optional[HPAAction] = None
    vpa: Optional[VPAAction] = None
    traffic: Optional[TrafficAction] = None
    node: Optional[NodeAction] = None
    no_op: bool = Field(
        False,
        description="Explicit no-op — agent consciously chose to wait this tick",
    )

    @model_validator(mode="after")
    def check_not_empty(self) -> "CloudScaleAction":
        has_action = any([self.hpa, self.vpa, self.traffic, self.node])
        if not has_action and not self.no_op:
            raise ValueError(
                "CloudScaleAction must specify at least one of: "
                "hpa, vpa, traffic, node — or set no_op=True"
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# CloudScaleState  (openenv State)
# ─────────────────────────────────────────────────────────────────────────────


class CloudScaleState(State):
    """
    Extended environment state returned by GET /state.
    Analogous to DMControlState — describes the currently loaded task
    and action/observation specifications.

    Example:
        >>> state = client.get_state()
        >>> state.task_id
        'task3_incident'
        >>> state.action_spec
        {'hpa': {'target_replicas': [1, 100]}, ...}
    """

    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: str = ""
    max_steps: int = 480
    budget_usd_per_hr: float = 50.0
    services: List[str] = Field(default_factory=list)
    regions: List[str] = Field(default_factory=list)
    trace_pattern: str = "diurnal"

    # Analogous to DMControlState.action_spec / observation_spec
    action_spec: Dict[str, Any] = Field(
        default_factory=dict,
        description="Describes valid action space bounds for agent introspection",
    )
    observation_spec: Dict[str, Any] = Field(
        default_factory=dict,
        description="Describes observation field names, shapes, and value ranges",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CloudScaleReward (not an openenv type — returned in step info dict)
# ─────────────────────────────────────────────────────────────────────────────


class CloudScaleReward(Action):  # plain BaseModel stand-in
    """
    Detailed reward breakdown returned alongside each step.
    total is the scalar clipped to [-1.0, 1.0] that RL algorithms consume.
    breakdown is for logging / debugging / agent self-reflection in prompts.
    """

    total: float = Field(..., description="Scalar reward clipped to [-1.0, 1.0]")
    slo_component: float = Field(..., description="SLO score 0–1")
    cost_component: float = Field(..., description="Cost efficiency score 0–1")
    stability_component: float = Field(..., description="Anti-thrash score 0–1")
    penalty: float = Field(..., description="Constraint violation penalty 0+")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-service and per-region detail for logging",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task registry — analogous to AVAILABLE_ENVIRONMENTS in dm_control_env
# ─────────────────────────────────────────────────────────────────────────────

AVAILABLE_TASKS: List[tuple[str, str, int, str]] = [
    # (task_id, difficulty, max_steps, description)
    (
        "task1_hpa",
        "easy",
        480,
        "Single service (api-gateway), single region. Reactive HPA. Diurnal traffic.",
    ),
    (
        "task2_cost",
        "medium",
        720,
        "4 services, 2 regions. Budget $15/hr. Flash-sale spike at tick 240.",
    ),
    (
        "task3_incident",
        "hard",
        300,
        "3 regions. AZ degradation in us-east-1 at tick 30. Failover + VPA scale-up.",
    ),
]