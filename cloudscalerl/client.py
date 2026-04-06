"""
CloudScaleRL — LLM Reasoning Agent Client
==========================================
Mirrors the DMControlEnv client pattern:
  - CloudScaleEnv extends EnvClient[CloudScaleAction, CloudScaleObservation, CloudScaleState]
  - _step_payload()  converts CloudScaleAction → JSON dict for the server
  - _parse_result()  converts server JSON → StepResult[CloudScaleObservation]
  - _parse_state()   converts server JSON → CloudScaleState

On top of the openenv plumbing, this module adds the LLM reasoning layer:
  - SYSTEM_PROMPT      : SRE rules and action schema
  - ContextManager     : rolling 5-tick window + episode compression
  - obs_to_prompt()    : formats CloudScaleObservation for the LLM
  - parse_action_*()   : LLM output → validated CloudScaleAction
  - run_episode()      : main agent loop using the EnvClient

Usage:
    python -m cloudscalerl.client task1_hpa
    USE_TOOLS=true python -m cloudscalerl.client task3_incident
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Optional

import httpx
from openai import OpenAI
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
except ImportError:
    # Minimal shims so the file is importable without openenv installed
    from dataclasses import dataclass

    @dataclass
    class StepResult:           # type: ignore[no-redef]
        observation: Any
        reward: Optional[float]
        done: bool

    class EnvClient:            # type: ignore[no-redef]
        def __init__(self, base_url: str, **kwargs: Any) -> None:
            self.base_url = base_url

        def reset(self, **kwargs: Any) -> Any: ...
        def step(self, action: Any, **kwargs: Any) -> Any: ...
        def get_state(self) -> Any: ...
        def close(self) -> None: ...
        def __enter__(self) -> "EnvClient": return self
        def __exit__(self, *_: Any) -> None: self.close()

from cloudscalerl.models import (
    AVAILABLE_TASKS,
    CloudScaleAction,
    CloudScaleObservation,
    CloudScaleState,
    HPAAction,
    NodeAction,
    TrafficAction,
    VPAAction,
)

# ── Setup ─────────────────────────────────────────────────────────────────────

console = Console()


def _resolve_base_url() -> Optional[str]:
    explicit = os.environ.get("OPENAI_BASE_URL")
    if explicit:
        return explicit

    if os.environ.get("USE_OLLAMA", "false").lower() == "true":
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ollama_url.rstrip("/") + "/v1"

    return None


_resolved_base_url = _resolve_base_url()
_resolved_api_key = os.environ.get("OPENAI_API_KEY")
if not _resolved_api_key and _resolved_base_url and "11434" in _resolved_base_url:
    _resolved_api_key = "ollama"


def _resolve_default_model() -> str:
    explicit = os.environ.get("AGENT_MODEL")
    if explicit:
        return explicit

    if os.environ.get("USE_OLLAMA", "false").lower() == "true":
        return os.environ.get("OLLAMA_MODEL", "sre-agent")

    return "gpt-4o"

openai_client = OpenAI(
    api_key=_resolved_api_key or "dummy_key_for_local_testing",
    base_url=_resolved_base_url,  # Allows pointing to vLLM, Ollama, HuggingFace, etc.
)
USE_TOOLS = os.environ.get("USE_TOOLS", "false").lower() == "true"
FORCE_JSON_RESPONSE_FORMAT = (
    os.environ.get("FORCE_JSON_RESPONSE_FORMAT", "true").lower() == "true"
)
DEFAULT_MODEL = _resolve_default_model()
COMPRESS_MODEL = "gpt-4o-mini"
ENABLE_INTERNAL_FEEDBACK = os.environ.get("ENABLE_INTERNAL_FEEDBACK", "false").lower() == "true"
FEEDBACK_MODEL = os.environ.get("FEEDBACK_MODEL", "tinyllama-sre-critic")
FEEDBACK_TEMPERATURE = float(os.environ.get("FEEDBACK_TEMPERATURE", "0.0"))
FEEDBACK_MAX_TOKENS = int(os.environ.get("FEEDBACK_MAX_TOKENS", "300"))

FEEDBACK_SYSTEM_PROMPT = """
You are an internal SRE action critic for CloudScaleRL.
You will receive:
- current cluster observation summary
- the agent's proposed action
- recent reward trend

Your job is to either approve the action or provide a safer/better revised action.

Return ONLY valid JSON with this schema:
{
    "accept": true|false,
    "reason": "short reason",
    "revised_action": {CloudScaleAction JSON} | null
}

Rules:
- If action risks SLO breach (p99>=200ms or error_rate>=0.001), reject and revise.
- If a region is DEGRADED and traffic remains there, reject and revise.
- Avoid aggressive replica thrashing (>50% change repeatedly).
- Prefer no_op when pending_replicas already indicates scaling is in flight.
- revised_action must be valid for the CloudScaleAction schema.
""".strip()


# ═══════════════════════════════════════════════════════════════════════════════
# CloudScaleEnv — openenv EnvClient subclass
# ═══════════════════════════════════════════════════════════════════════════════


class CloudScaleEnv(EnvClient):
    """
    Client for CloudScaleRL environments.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Follows the same pattern as DMControlEnv.

    Supported tasks:
        - task1_hpa      (easy)   : Reactive HPA, single service
        - task2_cost     (medium) : Cost-aware multi-service
        - task3_incident (hard)   : Multi-region incident response

    Example:
        >>> with CloudScaleEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset(task_id="task1_hpa")
        ...     print(result.observation.global_slo_met)
        ...
        ...     action = CloudScaleAction(
        ...         hpa=HPAAction(service="api-gateway", target_replicas=5)
        ...     )
        ...     result = env.step(action)
        ...     print(f"Reward: {result.reward:.3f}")

    Example switching tasks (like dm_control switching domains):
        >>> client = CloudScaleEnv(base_url="http://localhost:8000")
        >>> result = client.reset(task_id="task1_hpa")
        >>> # ... train on task 1 ...
        >>> result = client.reset(task_id="task3_incident")
        >>> # ... train on task 3 ...
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        provider: Optional[Any] = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            provider=provider,
        )

    # ── openenv EnvClient interface ───────────────────────────────────────────

    def _step_payload(self, action: CloudScaleAction) -> dict[str, Any]:
        """
        Convert CloudScaleAction → JSON payload for POST /step.
        Mirrors DMControlEnv._step_payload().
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: dict[str, Any]
    ) -> StepResult[CloudScaleObservation]:
        """
        Parse server JSON response → StepResult[CloudScaleObservation].
        Mirrors DMControlEnv._parse_result().
        """
        obs_data = payload.get("observation", {})
        observation = CloudScaleObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> CloudScaleState:
        """
        Parse GET /state response → CloudScaleState.
        Mirrors DMControlEnv._parse_state().
        """
        return CloudScaleState(**payload)

    # ── Public API (mirrors DMControlEnv.reset / step signatures) ─────────────

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        render: bool = False,
        **kwargs: Any,
    ) -> StepResult[CloudScaleObservation]:
        """
        Reset the environment for a new episode.

        Args:
            task_id: Optionally switch to a different task
                     (like switching domain_name in dm_control).
            seed: Random seed for reproducibility.
            render: If True, include JSON dashboard in observation.

        Returns:
            StepResult with initial CloudScaleObservation.
        """
        reset_kwargs = dict(kwargs)
        if task_id is not None:
            reset_kwargs["task_id"] = task_id
        if seed is not None:
            reset_kwargs["seed"] = seed
        reset_kwargs["render"] = render
        return super().reset(**reset_kwargs)

    def step(
        self,
        action: CloudScaleAction,
        render: bool = False,
        **kwargs: Any,
    ) -> StepResult[CloudScaleObservation]:
        """
        Execute one tick in the environment.

        Args:
            action: CloudScaleAction (hpa / vpa / traffic / node / no_op).
            render: If True, include JSON dashboard in returned observation.

        Returns:
            StepResult with new observation, reward scalar, and done flag.
        """
        return super().step(action, render=render, **kwargs)

    @staticmethod
    def available_tasks() -> list[tuple[str, str, int, str]]:
        """
        List available CloudScaleRL tasks.
        Mirrors DMControlEnv.available_environments().

        Returns:
            List of (task_id, difficulty, max_steps, description) tuples.
        """
        return AVAILABLE_TASKS

    @classmethod
    def from_direct(
        cls,
        task_id: str = "task1_hpa",
        port: int = 8000,
    ) -> "CloudScaleEnv":
        """
        Create a CloudScaleEnv client with an embedded local server.
        Mirrors DMControlEnv.from_direct().

        Starts a local uvicorn server in a subprocess and returns a client
        connected to it. Useful for quick testing without Docker.

        Args:
            task_id: Default task to load.
            port: Port for the local server.

        Returns:
            CloudScaleEnv client connected to the local server.

        Example:
            >>> client = CloudScaleEnv.from_direct(task_id="task3_incident")
            >>> try:
            ...     result = client.reset()
            ...     for _ in range(100):
            ...         result = client.step(CloudScaleAction(no_op=True))
            ... finally:
            ...     client.close()
        """
        import subprocess
        import time

        import requests as req_lib

        cmd = [
            sys.executable, "-m", "uvicorn",
            "cloudscalerl.server.app:app",
            "--host", "127.0.0.1",
            "--port", str(port),
        ]

        env = {
            **os.environ,
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
        }

        server_process = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        base_url = f"http://127.0.0.1:{port}"
        for _ in range(30):
            try:
                r = req_lib.get(
                    f"{base_url}/health", timeout=2,
                    proxies={"http": None, "https": None},
                )
                if r.status_code == 200:
                    break
            except req_lib.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            server_process.kill()
            raise RuntimeError(
                f"Failed to start local CloudScaleRL server on port {port}."
            )

        class _DirectProvider:
            def __init__(self, proc: subprocess.Popen) -> None:
                self._proc = proc

            def stop(self) -> None:
                if self._proc:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()

        return cls(base_url=base_url, provider=_DirectProvider(server_process))


class CloudScaleHTTPEndpoint:
    """
    Thin HTTP client for the current FastAPI app contract.

    The local app exposes REST endpoints (/reset, /step, /state) rather than
    openenv websocket endpoints, so run_episode uses this adapter.
    """

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)

    def __enter__(self) -> "CloudScaleHTTPEndpoint":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        render: bool = False,
        **kwargs: Any,
    ) -> StepResult[CloudScaleObservation]:
        payload: dict[str, Any] = {
            "task_id": task_id or "task1_hpa",
            "seed": seed or 42,
        }
        if render:
            payload["render"] = True
        payload.update(kwargs)

        response = self._client.post("/reset", json=payload)
        response.raise_for_status()
        observation = CloudScaleObservation(**response.json())
        return StepResult(observation=observation, reward=None, done=False)

    def step(
        self,
        action: CloudScaleAction,
        render: bool = False,
        **kwargs: Any,
    ) -> StepResult[CloudScaleObservation]:
        payload = action.model_dump(exclude_none=True)
        if render:
            payload["render"] = True
        payload.update(kwargs)

        response = self._client.post("/step", json=payload)
        response.raise_for_status()
        data = response.json()
        observation = CloudScaleObservation(**data.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=data.get("reward"),
            done=bool(data.get("done", False)),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Reasoning Layer
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are an expert Site Reliability Engineer (SRE) managing a Kubernetes cluster.
Your job: observe real-time cluster metrics each tick, reason about risks, and
output a scaling action. You are the agent in a CloudScaleRL environment.

=== TIMING (CRITICAL) ===
- Each tick = 60 seconds of simulated time.
- Pod scheduling takes 2 ticks. PRE-SCALE before traffic arrives.
- Node provisioning takes 5 ticks. Plan node changes well ahead.
- Pod startup adds ~45s latency. Factor this into latency decisions.

=== SLO RULES ===
- NEVER let p99 latency exceed 200ms on any service.
- NEVER let error_rate exceed 0.1% (0.001 fraction).
- If a region is marked DEGRADED, shift ALL traffic away immediately.

=== COST RULES ===
- Never exceed budget_remaining. Over-provisioning wastes money.
- Spot instances are 70% cheaper but can be preempted at any tick.
- Use VPA to right-size pods before adding nodes.
- Prefer greener regions (lower carbon_intensity_gco2_kwh) when cost/SLO allow.

=== STABILITY RULES ===
- Avoid thrashing: do not scale a service by >50% in 3 consecutive ticks.
- Watch RECENT REWARDS — if trending down, your last action was wrong.
- UPCOMING EVENTS show what is coming — act early (2 ticks ahead for pods).

=== OUTPUT FORMAT ===
Think step by step:
1. Identify the biggest risk (latency breach? cost overrun? region failure?)
2. Decide what action addresses that risk without causing a new problem.
3. Output ONLY a valid JSON object. No markdown, no preamble.

ACTION SCHEMA:
{
  "hpa": {"service": str, "target_replicas": int|null, "min_replicas": int|null,
          "max_replicas": int|null, "target_cpu_utilization": float|null} | null,
  "vpa": {"service": str, "cpu_request_millicores": int|null,
          "memory_request_mb": int|null} | null,
  "traffic": {"region_weights": {region_id: float}|null,
              "failover_from": str|null, "failover_to": str|null} | null,
  "node": {"region": str, "operation": "add"|"remove"|"change_type",
           "node_type": str|null, "count": int} | null,
  "no_op": false
}
region_weights must sum to 1.0. Set no_op: true to explicitly wait.
"""

# OpenAI tool definitions (USE_TOOLS=true mode)
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "scale_hpa",
            "description": "Scale a service horizontally. Use when CPU > 75% or p99 is rising.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "target_replicas": {"type": "integer", "minimum": 1, "maximum": 100},
                    "min_replicas": {"type": "integer", "minimum": 1},
                    "max_replicas": {"type": "integer", "minimum": 1, "maximum": 100},
                    "target_cpu_utilization": {"type": "number", "minimum": 0.3, "maximum": 0.9},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_vpa",
            "description": "Adjust per-pod resource requests. Use when memory OOM or CPU throttled.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "cpu_request_millicores": {"type": "integer", "minimum": 10, "maximum": 32000},
                    "memory_request_mb": {"type": "integer", "minimum": 64, "maximum": 131072},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reroute_traffic",
            "description": "Shift traffic weights between regions. Weights must sum to 1.0.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_weights": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    "failover_from": {"type": "string"},
                    "failover_to": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_nodes",
            "description": "Add/remove/change nodes. Node provisioning takes 5 ticks — act early.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string"},
                    "operation": {"type": "string", "enum": ["add", "remove", "change_type"]},
                    "node_type": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["region", "operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "no_op",
            "description": "Do nothing this tick. Use when the cluster is stable.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ── Context Manager ───────────────────────────────────────────────────────────


class ContextManager:
    """
    Manages LLM context across ticks.
    Keeps a rolling 5-tick window of obs/action/reward history.
    Compresses older history every 20 ticks to stay within token budget.
    """

    def __init__(self, window_size: int = 5, compress_after: int = 20) -> None:
        self.window_size = window_size
        self.compress_after = compress_after
        self.history: list[dict[str, Any]] = []
        self.episode_summary: str = ""

    def add(self, obs_text: str, action: CloudScaleAction, reward: float) -> None:
        self.history.append({
            "obs": obs_text,
            "action": action.model_dump_json(exclude_none=True),
            "reward": round(reward, 4),
        })
        if len(self.history) > self.compress_after:
            self._compress()

    def build_messages(self, current_obs_text: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        if self.episode_summary:
            messages += [
                {"role": "user", "content": f"EPISODE CONTEXT:\n{self.episode_summary}"},
                {"role": "assistant", "content": "Understood. Using this context."},
            ]
        for entry in self.history[-self.window_size:]:
            messages.append({"role": "user", "content": entry["obs"]})
            messages.append({
                "role": "assistant",
                "content": f"Action: {entry['action']}\nReward: {entry['reward']}",
            })
        messages.append({"role": "user", "content": current_obs_text})
        return messages

    def _compress(self) -> None:
        old = self.history[: -self.window_size]
        prompt = (
            "Summarise these Kubernetes cluster management decisions in 5 bullet points. "
            "Focus on: actions taken, what worked, what didn't, reward trend, cluster state.\n\n"
            + "\n---\n".join(
                f"obs: {e['obs'][:250]}...\naction: {e['action']}\nreward: {e['reward']}"
                for e in old
            )
        )
        resp = openai_client.chat.completions.create(
            model=COMPRESS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        new_chunk = resp.choices[0].message.content or ""
        self.episode_summary = (
            f"{self.episode_summary}\n\n[LATER]\n{new_chunk}"
            if self.episode_summary
            else new_chunk
        )
        self.history = self.history[-self.window_size:]
        console.print("[dim]🗜  Context compressed[/dim]")


# ── Observation → Prompt ──────────────────────────────────────────────────────


def obs_to_prompt(obs: CloudScaleObservation, reward_history: list[float]) -> str:
    lines = [
        f"=== Tick {obs.step} ===",
        f"Budget remaining: ${obs.budget_remaining_usd:.2f}  |  "
        f"Cost now: ${obs.total_cost_usd_per_hour:.2f}/hr",
        f"Global SLO met: {obs.global_slo_met}",
    ]

    if obs.counterfactual_cost_usd_per_hour is not None:
        delta = obs.total_cost_usd_per_hour - obs.counterfactual_cost_usd_per_hour
        lines.append(
            f"Oracle cost: ${obs.counterfactual_cost_usd_per_hour:.2f}/hr  "
            f"({'over' if delta > 0 else 'under'} by ${abs(delta):.2f}/hr)"
        )

    lines += ["", "SERVICES:"]
    for name, svc in obs.services.items():
        slo_ok = svc.p99_latency_ms < 200 and svc.error_rate < 0.001
        flag = "✅" if slo_ok else "⚠️ SLO BREACH"
        pending = f" (pending→{svc.pending_replicas})" if svc.pending_replicas else ""
        lines.append(
            f"  {name} {flag}\n"
            f"    replicas={svc.replicas}{pending}  cpu={svc.cpu_utilization:.0%}  "
            f"mem={svc.memory_utilization:.0%}  p99={svc.p99_latency_ms:.0f}ms  "
            f"rps={svc.requests_per_second:.1f}  errors={svc.error_rate:.4%}\n"
            f"    cpu_req={svc.cpu_request_millicores}m  mem_req={svc.memory_request_mb}MB"
        )

    lines += ["", "REGIONS:"]
    for rid, r in obs.regions.items():
        status = "⚠️  DEGRADED" if r.is_degraded else "✅ healthy"
        spot = "  [SPOT]" if r.is_spot else ""
        lines.append(
            f"  {rid} {status}{spot}\n"
            f"    weight={r.traffic_weight:.0%}  nodes={r.node_count}  "
            f"type={r.node_type}  cost=${r.cost_per_hour:.2f}/hr  "
            f"carbon={r.carbon_intensity_gco2_kwh:.0f} gCO₂/kWh"
        )

    if obs.pending_events:
        lines += ["", f"⏰ UPCOMING: {', '.join(obs.pending_events)}"]

    if reward_history:
        recent = reward_history[-5:]
        trend = "↑" if len(recent) > 1 and recent[-1] > recent[0] else "↓"
        lines += ["", f"📈 RECENT REWARDS: {[round(r, 3) for r in recent]}  ({trend})"]

    lines += ["", "Think step by step, then output your JSON action."]
    return "\n".join(lines)


# ── Action Parsers ────────────────────────────────────────────────────────────


def parse_action_from_json(raw: str) -> CloudScaleAction:
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return CloudScaleAction(**json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[yellow][WARN] Parse failed: {e}. Using no_op.[/yellow]")
        return CloudScaleAction(no_op=True)


def parse_action_from_tools(tool_calls: list[Any]) -> CloudScaleAction:
    """Merge multiple tool calls into one CloudScaleAction (compound actions)."""
    hpa: Optional[HPAAction] = None
    vpa: Optional[VPAAction] = None
    traffic: Optional[TrafficAction] = None
    node: Optional[NodeAction] = None
    no_op = False

    for tc in tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments)
        if name == "scale_hpa":
            hpa = HPAAction(**args)
        elif name == "adjust_vpa":
            vpa = VPAAction(**args)
        elif name == "reroute_traffic":
            traffic = TrafficAction(**args)
        elif name == "manage_nodes":
            node = NodeAction(**args)
        elif name == "no_op":
            no_op = True

    if not any([hpa, vpa, traffic, node]):
        no_op = True
    return CloudScaleAction(hpa=hpa, vpa=vpa, traffic=traffic, node=node, no_op=no_op)


def _strip_markdown_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def maybe_apply_internal_feedback(
    obs: CloudScaleObservation,
    proposed_action: CloudScaleAction,
    reward_history: list[float],
    model: str = FEEDBACK_MODEL,
) -> tuple[CloudScaleAction, dict[str, Any]]:
    """
    Run a second-pass critic model over the proposed action.
    If critic rejects and returns a valid revised_action, use it.
    Otherwise keep original action.
    """
    prompt = (
        f"OBSERVATION:\n{obs_to_prompt(obs, reward_history)}\n\n"
        f"PROPOSED_ACTION:\n{proposed_action.model_dump_json(exclude_none=True)}\n\n"
        "Respond with JSON only."
    )

    try:
        completion = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=FEEDBACK_TEMPERATURE,
            max_tokens=FEEDBACK_MAX_TOKENS,
        )

        raw = completion.choices[0].message.content or "{}"
        data = json.loads(_strip_markdown_fence(raw))

        accepted = bool(data.get("accept", True))
        reason = str(data.get("reason", "approved"))

        if accepted:
            return proposed_action, {
                "enabled": True,
                "accepted": True,
                "reason": reason,
                "model": model,
            }

        revised = data.get("revised_action")
        if revised is None:
            return proposed_action, {
                "enabled": True,
                "accepted": True,
                "reason": f"critic_rejected_without_revision:{reason}",
                "model": model,
            }

        revised_action = CloudScaleAction(**revised)
        return revised_action, {
            "enabled": True,
            "accepted": False,
            "reason": reason,
            "model": model,
        }

    except Exception as exc:
        console.print(
            f"[yellow][WARN] Internal feedback unavailable: {exc}. Using original action.[/yellow]"
        )
        return proposed_action, {
            "enabled": True,
            "accepted": True,
            "reason": "feedback_unavailable",
            "model": model,
        }


# ── Main Agent Loop ───────────────────────────────────────────────────────────


def run_episode(
    env_url: str,
    task_id: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Run one full episode using the LLM reasoning agent + CloudScaleEnv client."""
    ctx = ContextManager()
    reward_history: list[float] = []

    console.print(Panel(
        f"[bold cyan]CloudScaleRL Agent[/bold cyan]\n"
        f"Task: [yellow]{task_id}[/yellow]  Model: {model}  Tools: {USE_TOOLS}"
    ))

    with CloudScaleHTTPEndpoint(base_url=env_url) as env:
        result = env.reset(task_id=task_id)
        obs: CloudScaleObservation = result.observation

        total_reward = 0.0
        done = False
        step_times: list[float] = []

        while not done:
            t0 = time.time()
            obs_text = obs_to_prompt(obs, reward_history)
            messages = ctx.build_messages(obs_text)
            feedback_meta: Optional[dict[str, Any]] = None

            if USE_TOOLS:
                completion = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=512,
                    temperature=0.15,
                )
                msg = completion.choices[0].message
                action = (
                    parse_action_from_tools(msg.tool_calls)
                    if msg.tool_calls
                    else CloudScaleAction(no_op=True)
                )
            else:
                completion_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": 512,
                    "temperature": 0.15,
                }
                if FORCE_JSON_RESPONSE_FORMAT:
                    completion_kwargs["response_format"] = {"type": "json_object"}

                completion = openai_client.chat.completions.create(
                    **completion_kwargs,
                )
                raw = completion.choices[0].message.content or "{}"
                action = parse_action_from_json(raw)

            if ENABLE_INTERNAL_FEEDBACK:
                action, feedback_meta = maybe_apply_internal_feedback(
                    obs=obs,
                    proposed_action=action,
                    reward_history=reward_history,
                )

            # Step via EnvClient
            result = env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            ctx.add(obs_text, action, reward)
            reward_history.append(reward)
            total_reward += reward

            elapsed = time.time() - t0
            step_times.append(elapsed)
            _print_tick(
                obs,
                action,
                reward,
                elapsed,
                getattr(completion, "usage", None),
                feedback_meta,
            )

    mean_reward = total_reward / max(len(reward_history), 1)
    console.print(Panel(
        f"[bold green]Episode complete[/bold green]\n"
        f"Ticks: {len(reward_history)}  "
        f"Total: {total_reward:.3f}  Mean: {mean_reward:.4f}\n"
        f"Avg step: {sum(step_times)/len(step_times):.2f}s"
    ))
    return {
        "task_id": task_id,
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "ticks": len(reward_history),
    }


def _print_tick(
    obs: CloudScaleObservation,
    action: CloudScaleAction,
    reward: float,
    elapsed: float,
    usage: Any,
    feedback_meta: Optional[dict[str, Any]] = None,
) -> None:
    color = "green" if reward > 0 else "red"
    slo = "✅" if obs.global_slo_met else "⚠️ "
    action_str = action.model_dump_json(exclude_none=True)
    tok = f"{usage.total_tokens}tok" if usage else ""
    feedback_str = ""
    if feedback_meta and feedback_meta.get("enabled"):
        if feedback_meta.get("accepted"):
            feedback_str = "critic=approved"
        else:
            feedback_str = "critic=rewrote"
    console.print(
        f"[dim]tick {obs.step:4d}[/dim]  "
        f"slo={slo}  cost=${obs.total_cost_usd_per_hour:.2f}/hr  "
        f"[{color}]reward={reward:+.3f}[/{color}]  "
        f"[dim]{action_str[:80]}  {elapsed:.1f}s {tok} {feedback_str}[/dim]"
    )


def main() -> None:
    env_url = os.environ.get("ENV_URL", "http://localhost:8000")
    tasks = sys.argv[1:] if len(sys.argv) > 1 else ["task1_hpa"]
    model = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)

    all_scores: list[dict[str, Any]] = []
    for task in tasks:
        all_scores.append(run_episode(env_url, task, model=model))

    if len(all_scores) > 1:
        table = Table(title="CloudScaleRL Results")
        table.add_column("Task", style="cyan")
        table.add_column("Ticks", justify="right")
        table.add_column("Mean Reward", justify="right")
        table.add_column("Total Reward", justify="right")
        for s in all_scores:
            c = "green" if s["mean_reward"] > 0 else "red"
            table.add_row(
                s["task_id"], str(s["ticks"]),
                f"[{c}]{s['mean_reward']:+.4f}[/{c}]",
                f"{s['total_reward']:.3f}",
            )
        console.print(table)
    else:
        console.print(f"\n[bold]Final score:[/bold] {all_scores[0]}")


if __name__ == "__main__":
    main()