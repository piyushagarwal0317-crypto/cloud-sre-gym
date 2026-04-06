"""
CloudScaleRL Pydantic models.

Three core model groups:
  - Observation  : what the agent sees each tick
  - Action       : what the agent can do
  - Reward       : scalar + breakdown sent back after each step
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Observation ───────────────────────────────────────────────────────────────


class ServiceMetrics(BaseModel):
    """Per-service metrics snapshot at one tick."""

    replicas: int = Field(..., ge=0, description="Current running replica count")
    cpu_utilization: float = Field(..., ge=0.0, le=1.0, description="Fraction 0.0–1.0")
    memory_utilization: float = Field(..., ge=0.0, le=1.0, description="Fraction 0.0–1.0")
    requests_per_second: float = Field(..., ge=0.0)
    p99_latency_ms: float = Field(..., ge=0.0, description="p99 latency in milliseconds")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction 0.0–1.0")
    cpu_request_millicores: int = Field(..., ge=1, description="VPA-managed CPU request")
    memory_request_mb: int = Field(..., ge=1, description="VPA-managed memory request")

    # Pending scaling state (set by environment, read-only for agent)
    pending_replicas: Optional[int] = Field(
        None, description="Replicas that will become active after scheduling delay"
    )


class RegionState(BaseModel):
    """Per-region cluster state at one tick."""

    region_id: str = Field(..., description='e.g. "us-east-1", "eu-west-1", "ap-south-1"')
    traffic_weight: float = Field(..., ge=0.0, le=1.0, description="Must sum to 1.0 across regions")
    node_count: int = Field(..., ge=0)
    node_type: str = Field(..., description='e.g. "m5.large", "c5.xlarge", "spot.c5.xlarge"')
    cost_per_hour: float = Field(..., ge=0.0)
    is_degraded: bool = Field(False, description="True during simulated AZ failure")
    is_spot: bool = Field(False, description="True if node_type is a spot instance")
    carbon_intensity_gco2_kwh: float = Field(
        200.0, description="gCO2/kWh — lower is greener. Live from Electricity Maps API."
    )


class Observation(BaseModel):
    """Full cluster observation at one tick — what the agent receives."""

    step: int = Field(..., ge=0)
    services: Dict[str, ServiceMetrics] = Field(..., description="Keyed by service name")
    regions: Dict[str, RegionState] = Field(..., description="Keyed by region_id")
    total_cost_usd_per_hour: float = Field(..., ge=0.0)
    budget_remaining_usd: float = Field(..., description="Can go negative (budget breach)")
    global_slo_met: bool = Field(..., description="True if ALL services meet SLO this tick")
    pending_events: List[str] = Field(
        default_factory=list,
        description='Upcoming events visible to agent, e.g. ["traffic_spike_t+3", "node_preemption_t+1"]',
    )
    counterfactual_cost_usd_per_hour: Optional[float] = Field(
        None, description="What the oracle would have spent. For learning signal."
    )

    @model_validator(mode="after")
    def check_traffic_weights_sum(self) -> "Observation":
        total = sum(r.traffic_weight for r in self.regions.values())
        if self.regions and abs(total - 1.0) > 0.05:
            raise ValueError(f"Region traffic weights sum to {total:.3f}, expected ~1.0")
        return self


# ── Action ────────────────────────────────────────────────────────────────────


class HPAAction(BaseModel):
    """Horizontal Pod Autoscaler action for one service."""

    service: str
    target_replicas: Optional[int] = Field(None, ge=1, le=100)
    min_replicas: Optional[int] = Field(None, ge=1)
    max_replicas: Optional[int] = Field(None, ge=1, le=100)
    target_cpu_utilization: Optional[float] = Field(None, ge=0.1, le=0.95)

    @model_validator(mode="after")
    def check_min_max(self) -> "HPAAction":
        if self.min_replicas and self.max_replicas:
            if self.min_replicas > self.max_replicas:
                raise ValueError("min_replicas must be <= max_replicas")
        if self.target_replicas and self.min_replicas:
            if self.target_replicas < self.min_replicas:
                raise ValueError("target_replicas must be >= min_replicas")
        return self


class VPAAction(BaseModel):
    """Vertical Pod Autoscaler action for one service."""

    service: str
    cpu_request_millicores: Optional[int] = Field(None, ge=10, le=32000)
    memory_request_mb: Optional[int] = Field(None, ge=64, le=131072)


class TrafficAction(BaseModel):
    """Traffic routing action across regions."""

    region_weights: Optional[Dict[str, float]] = Field(
        None, description="New traffic weights per region. Must sum to 1.0."
    )
    failover_from: Optional[str] = Field(None, description="Region to drain traffic from")
    failover_to: Optional[str] = Field(None, description="Region to route traffic to")
    canary_service: Optional[str] = Field(None, description="Service for canary deployment")
    canary_percent: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("region_weights")
    @classmethod
    def weights_must_sum_to_one(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if v is not None:
            total = sum(v.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"region_weights must sum to 1.0, got {total:.3f}")
        return v


class NodeAction(BaseModel):
    """Node provisioning / decommissioning action."""

    region: str
    operation: Literal["add", "remove", "change_type"]
    node_type: Optional[str] = Field(None, description='e.g. "m5.large", "spot.c5.xlarge"')
    count: int = Field(1, ge=1, le=20)


class Action(BaseModel):
    """
    Full agent action for one tick.
    All sub-actions are optional — set only what you want to change.
    Set no_op=True to explicitly do nothing (skip action).
    """

    hpa: Optional[HPAAction] = None
    vpa: Optional[VPAAction] = None
    traffic: Optional[TrafficAction] = None
    node: Optional[NodeAction] = None
    no_op: bool = Field(False, description="Explicit no-op — agent consciously chose to wait")

    @model_validator(mode="after")
    def check_not_empty(self) -> "Action":
        has_action = any([self.hpa, self.vpa, self.traffic, self.node])
        if not has_action and not self.no_op:
            raise ValueError(
                "Action must specify at least one of: hpa, vpa, traffic, node — or set no_op=True"
            )
        return self


# ── Reward ────────────────────────────────────────────────────────────────────


class Reward(BaseModel):
    """Reward signal returned after each step."""

    total: float = Field(..., description="Final scalar sent to agent, clipped to [-1.0, 1.0]")
    slo_component: float = Field(..., description="SLO satisfaction score 0.0–1.0")
    cost_component: float = Field(..., description="Cost efficiency score 0.0–1.0")
    stability_component: float = Field(..., description="Anti-thrash score 0.0–1.0")
    penalty: float = Field(..., description="Constraint violation penalty 0.0+")
    breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Per-service and per-region breakdown for debugging"
    )