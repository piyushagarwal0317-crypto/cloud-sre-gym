"""
CloudScaleRL FastAPI Server
=============================
Exposes the RL environment over HTTP so the LLM agent (client.py)
can interact with it via POST /reset, POST /step, GET /state, GET /render.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cloudscalerl.models import CloudScaleAction, CloudScaleObservation, CloudScaleReward
from cloudscalerl.server.cloudscalerl_env import CloudScaleEnvServer

# Load task configs from openenv.yaml
_OPENENV_PATH = Path(__file__).parent.parent / "openenv.yaml"
with open(_OPENENV_PATH) as f:
    _OPENENV = yaml.safe_load(f)

TASK_CONFIGS: dict[str, dict[str, Any]] = {
    t["id"]: t for t in _OPENENV["tasks"]
}

app = FastAPI(
    title="CloudScaleRL",
    description="Kubernetes-inspired cloud autoscaling RL environment",
    version="0.1.0",
)

# One active environment instance per server process (single-agent use)
_env: Optional[CloudScaleEnvServer] = None


# ── Request / Response Models ─────────────────────────────────────────────────


class ResetRequest(BaseModel):
    task_id: str
    seed: int = 42


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/")
def root() -> dict[str, Any]:
    """Root route for Space health visibility and quick endpoint discovery."""
    return {
        "name": "CloudScaleRL",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/reset", "/step", "/state", "/render", "/tasks", "/grade"],
    }


@app.post("/reset", response_model=dict[str, Any])
def reset(req: ResetRequest) -> dict[str, Any]:
    """Reset the environment for a new episode."""
    global _env

    if req.task_id not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Valid: {list(TASK_CONFIGS.keys())}",
        )

    config = TASK_CONFIGS[req.task_id]
    _env = CloudScaleEnvServer()
    obs = _env._reset(task_id=req.task_id, seed=req.seed)
    return obs.model_dump()

@app.post("/step", response_model=StepResponse)
def step(action: CloudScaleAction) -> StepResponse:
    """Advance the environment one tick with the given action."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")

    obs, reward, done, info = _env._step(action)

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=dict[str, Any])
def get_state() -> dict[str, Any]:
    """Get current environment state without advancing."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    state = _env._get_state()
    return state.model_dump()


@app.get("/render", response_model=dict[str, Any])
def render() -> dict[str, Any]:
    """JSON dashboard snapshot — human-readable cluster state for debugging."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return _env.render()


@app.get("/tasks", response_model=list[dict[str, Any]])
def list_tasks() -> list[dict[str, Any]]:
    """List all available tasks with metadata."""
    return list(TASK_CONFIGS.values())


@app.get("/grade", response_model=dict[str, Any])
def grade() -> dict[str, Any]:
    """Return current task grader score for the collected trajectory."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return _env.grade_current_episode()


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint. Returns healthy status."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    """Return environment metadata per OpenEnv spec."""
    return {
        "name": "cloudscalerl",
        "version": "0.1.0",
        "description": _OPENENV.get("description", "Kubernetes-inspired autoscaling RL environment"),
        "tasks": list(TASK_CONFIGS.keys()),
        "reward_range": _OPENENV.get("reward_range", "[-1.0, 1.0]"),
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    """Return action, observation, and state schemas per OpenEnv spec."""
    return {
        "action": {
            "type": "object",
            "properties": {
                "hpa": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},
                        "target_replicas": {"type": "integer", "minimum": 1, "maximum": 100},
                        "min_replicas": {"type": "integer", "minimum": 1},
                        "max_replicas": {"type": "integer", "minimum": 1, "maximum": 100},
                        "target_cpu_utilization": {"type": "number", "minimum": 0.1, "maximum": 0.95},
                    },
                },
                "vpa": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},
                        "cpu_request_millicores": {"type": "integer", "minimum": 10, "maximum": 32000},
                        "memory_request_mb": {"type": "integer", "minimum": 64, "maximum": 131072},
                    },
                },
                "traffic": {
                    "type": "object",
                    "properties": {
                        "region_weights": {"type": "object", "additionalProperties": {"type": "number"}},
                        "failover_from": {"type": "string"},
                        "failover_to": {"type": "string"},
                    },
                },
                "node": {
                    "type": "object",
                    "properties": {
                        "region": {"type": "string"},
                        "operation": {"type": "string", "enum": ["add", "remove", "change_type"]},
                        "node_type": {"type": "string"},
                        "count": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                },
                "no_op": {"type": "boolean"},
            },
        },
        "observation": {
            "type": "object",
            "properties": {
                "step": {"type": "integer"},
                "services": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "replicas": {"type": "integer"},
                            "cpu_utilization": {"type": "number"},
                            "memory_utilization": {"type": "number"},
                            "requests_per_second": {"type": "number"},
                            "p99_latency_ms": {"type": "number"},
                            "error_rate": {"type": "number"},
                        },
                    },
                },
                "regions": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "traffic_weight": {"type": "number"},
                            "node_count": {"type": "integer"},
                            "is_degraded": {"type": "boolean"},
                            "cost_per_hour": {"type": "number"},
                        },
                    },
                },
                "total_cost_usd_per_hour": {"type": "number"},
                "budget_remaining_usd": {"type": "number"},
                "global_slo_met": {"type": "boolean"},
                "pending_events": {"type": "array", "items": {"type": "string"}},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "task_id": {"type": "string"},
                "max_steps": {"type": "integer"},
                "budget_usd_per_hr": {"type": "number"},
                "services": {"type": "array", "items": {"type": "string"}},
                "regions": {"type": "array", "items": {"type": "string"}},
            },
        },
    }


# ── MCP (Model Context Protocol) Support ────────────────────────────────────────

class MCPRequest(BaseModel):
    """JSON-RPC 2.0 request format for MCP."""
    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] = {}
    id: Optional[int] = None


@app.post("/mcp")
def mcp(request: MCPRequest) -> dict[str, Any]:
    """
    MCP (Model Context Protocol) endpoint for agent integration.

    Supports methods:
    - initialize: Return server capabilities
    - list_tools: Return available tools (reset, step, state, render, grade)
    - call_tool: Execute a tool with parameters
    """
    if request.jsonrpc != "2.0":
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32600, "message": "Invalid JSON-RPC version"},
            "id": request.id,
        }

    method = request.method
    params = request.params

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": {
                    "name": "cloudscalerl",
                    "version": "0.1.0",
                },
            },
            "id": request.id,
        }

    elif method == "list_tools":
        return {
            "jsonrpc": "2.0",
            "result": {
                "tools": [
                    {
                        "name": "reset",
                        "description": "Reset the environment for a new episode",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string", "description": "Task ID to run"},
                                "seed": {"type": "integer", "description": "Random seed", "default": 42},
                            },
                            "required": ["task_id"],
                        },
                    },
                    {
                        "name": "step",
                        "description": "Advance the environment one tick with an action",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "hpa": {"type": "object", "description": "Horizontal Pod Autoscaler action"},
                                "vpa": {"type": "object", "description": "Vertical Pod Autoscaler action"},
                                "traffic": {"type": "object", "description": "Traffic routing action"},
                                "node": {"type": "object", "description": "Node provisioning action"},
                                "no_op": {"type": "boolean", "description": "Explicit no-op action"},
                            },
                        },
                    },
                    {
                        "name": "state",
                        "description": "Get current environment state",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "render",
                        "description": "Get JSON dashboard snapshot",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "grade",
                        "description": "Get current task score",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ],
            },
            "id": request.id,
        }

    elif method == "call_tool":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        try:
            result = _execute_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "result": {"content": [{"type": "text", "text": json.dumps(result)}]},
                "id": request.id,
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(e)},
                "id": request.id,
            }

    else:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": f"Method not found: {method}"},
            "id": request.id,
        }


def _execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute an MCP tool by name."""
    global _env

    if name == "reset":
        task_id = args.get("task_id", "task1_hpa")
        seed = args.get("seed", 42)
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {task_id}")
        _env = CloudScaleEnvServer()
        obs = _env._reset(task_id=task_id, seed=seed)
        return {"observation": obs.model_dump()}

    elif name == "step":
        if _env is None:
            raise ValueError("Environment not initialized. Call reset first.")
        action = CloudScaleAction(**args)
        obs, reward, done, info = _env._step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }

    elif name == "state":
        if _env is None:
            raise ValueError("Environment not initialized. Call reset first.")
        return _env._get_state().model_dump()

    elif name == "render":
        if _env is None:
            raise ValueError("Environment not initialized. Call reset first.")
        return _env.render()

    elif name == "grade":
        if _env is None:
            raise ValueError("Environment not initialized. Call reset first.")
        return _env.grade_current_episode()

    else:
        raise ValueError(f"Unknown tool: {name}")