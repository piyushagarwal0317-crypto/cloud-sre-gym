from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class SREAction(BaseModel):
    action_type: str = Field(..., description="One of: 'scale_up', 'scale_down', 'change_instance_type', 'do_nothing'")
    amount: int = Field(0, description="Number of instances to scale by")
    instance_type: Optional[str] = Field(None, description="Target instance type (e.g., 'standard', 'compute_optimized')")

class SREObservation(BaseModel):
    tool_outputs: Dict[str, Any] = Field(default_factory=dict, description="Outputs from recently called tools")
    time_step: int = Field(..., description="Current simulation minute")

class SREReward(BaseModel):
    reward: float = Field(..., description="Dense reward scalar")
    cost_penalty: float
    sla_penalty: float
    stability_penalty: float