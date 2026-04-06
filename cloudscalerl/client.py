"""
CloudScaleRL — LLM Reasoning Agent
====================================
Replaces hardcoded rule-based policies with a GPT-4o agent that:
  1. Reads the current cluster observation (formatted as human-readable text)
  2. Reasons via chain-of-thought about what scaling actions to take
  3. Outputs a structured JSON Action
  4. Maintains a rolling memory window + episode compression for long episodes

Usage:
    python -m cloudscalerl.client task1_hpa
    USE_TOOLS=true python -m cloudscalerl.client task3_incident
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import httpx
from openai import OpenAI
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cloudscalerl.models import (
    Action,
    HPAAction,
    NodeAction,
    Observation,
    TrafficAction,
    VPAAction,
)

# ── Setup ─────────────────────────────────────────────────────────────────────

console = Console()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
USE_TOOLS = os.environ.get("USE_TOOLS", "false").lower() == "true"
DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "gpt-4o")
COMPRESS_MODEL = "gpt-4o-mini"  # cheaper model for context compression


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert Site Reliability Engineer (SRE) managing a Kubernetes cluster.
Your job is to observe real-time cluster metrics each tick and decide scaling actions.

=== TIMING RULES (CRITICAL) ===
- Each tick = 60 seconds of simulated time.
- Pod scheduling takes 2 ticks to take effect. You must PRE-SCALE before traffic arrives.
- Node provisioning takes 5 ticks. Plan node changes well in advance.
- Pod startup adds ~45s latency (±10s). Factor this into latency decisions.

=== SLO RULES ===
- NEVER let p99 latency exceed 200ms on any service.
- NEVER let error_rate exceed 0.1% on any service.
- If a region is marked DEGRADED, shift ALL traffic away from it immediately.

=== COST RULES ===
- Never exceed budget_remaining. Over-provisioning wastes money.
- Spot instances (spot.*) are 70% cheaper but can be preempted at any tick.
- Use VPA to right-size pods before adding nodes.
- Prefer greener regions (lower carbon_intensity_gco2_kwh) when SLO and cost allow.

=== STABILITY RULES ===
- Avoid thrashing: do not scale a service by more than 50% in 3 consecutive ticks.
- Observe RECENT REWARDS: if rewards are trending down, your last action was wrong.
- UPCOMING EVENTS in the observation tell you what is coming — act early.

=== OUTPUT FORMAT ===
Think step by step:
1. Identify the biggest risk right now (latency breach? cost overrun? region failure?)
2. Decide what action addresses that risk without causing a new problem.
3. Output a valid JSON object matching the Action schema. Raw JSON only — no markdown.

ACTION SCHEMA:
{
  "hpa": {
    "service": "<service_name>",
    "target_replicas": <int|null>,
    "min_replicas": <int|null>,
    "max_replicas": <int|null>,
    "target_cpu_utilization": <float 0.3–0.9|null>
  } | null,
  "vpa": {
    "service": "<service_name>",
    "cpu_request_millicores": <int|null>,
    "memory_request_mb": <int|null>
  } | null,
  "traffic": {
    "region_weights": {"<region_id>": <float>, ...} | null,
    "failover_from": "<region_id>" | null,
    "failover_to": "<region_id>" | null,
    "canary_service": "<service>" | null,
    "canary_percent": <float 0–1|null>
  } | null,
  "node": {
    "region": "<region_id>",
    "operation": "add" | "remove" | "change_type",
    "node_type": "<type>" | null,
    "count": <int>
  } | null,
  "no_op": false
}

All fields are optional. Set unused sub-actions to null. If waiting is correct, set no_op: true.
region_weights must sum to 1.0 if provided.
"""

# ── OpenAI Tool Definitions (used when USE_TOOLS=true) ───────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "scale_hpa",
            "description": (
                "Scale a service horizontally. Use when CPU > 75%, p99 is rising, "
                "or a traffic spike is predicted within 2 ticks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name to scale"},
                    "target_replicas": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Absolute target replica count",
                    },
                    "min_replicas": {"type": "integer", "minimum": 1},
                    "max_replicas": {"type": "integer", "minimum": 1, "maximum": 100},
                    "target_cpu_utilization": {
                        "type": "number",
                        "minimum": 0.3,
                        "maximum": 0.9,
                        "description": "Target CPU fraction for HPA autoscaler",
                    },
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_vpa",
            "description": (
                "Adjust vertical resource requests for a service. "
                "Use when memory OOM kills occur or CPU throttling is observed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "cpu_request_millicores": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 32000,
                    },
                    "memory_request_mb": {
                        "type": "integer",
                        "minimum": 64,
                        "maximum": 131072,
                    },
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reroute_traffic",
            "description": (
                "Shift traffic weights between regions. Weights must sum to 1.0. "
                "Use when a region is degraded or for carbon-aware routing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region_weights": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Map of region_id -> traffic fraction. Must sum to 1.0.",
                    },
                    "failover_from": {"type": "string", "description": "Region to drain"},
                    "failover_to": {"type": "string", "description": "Region to receive traffic"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_nodes",
            "description": (
                "Add, remove, or change node types in a region. "
                "Node provisioning takes 5 ticks — act early."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string"},
                    "operation": {
                        "type": "string",
                        "enum": ["add", "remove", "change_type"],
                    },
                    "node_type": {
                        "type": "string",
                        "description": 'e.g. "m5.large", "c5.xlarge", "spot.c5.xlarge"',
                    },
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
            "description": "Explicitly do nothing this tick. Use when the cluster is stable.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ── Context Manager ───────────────────────────────────────────────────────────


class ContextManager:
    """
    Manages what goes into the LLM context each tick.
    Keeps a rolling window of recent observations + a compressed episode summary.
    Prevents context window overflow on long episodes (720 ticks for Task 2).
    """

    def __init__(self, window_size: int = 5, compress_after: int = 20) -> None:
        self.window_size = window_size
        self.compress_after = compress_after
        self.history: list[dict[str, Any]] = []  # {obs_text, action_json, reward}
        self.episode_summary: str = ""

    def add(self, obs_text: str, action: Action, reward: float) -> None:
        self.history.append(
            {
                "obs": obs_text,
                "action": action.model_dump_json(exclude_none=True),
                "reward": round(reward, 4),
            }
        )
        if len(self.history) > self.compress_after:
            self._compress()

    def build_messages(self, current_obs_text: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if self.episode_summary:
            messages.append(
                {
                    "role": "user",
                    "content": f"EPISODE CONTEXT SO FAR:\n{self.episode_summary}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "Understood. I'll use this context to inform my decisions.",
                }
            )

        # Rolling window as alternating user/assistant turns
        window = self.history[-self.window_size :]
        for entry in window:
            messages.append({"role": "user", "content": entry["obs"]})
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Action taken: {entry['action']}\n(Reward received: {entry['reward']})",
                }
            )

        messages.append({"role": "user", "content": current_obs_text})
        return messages

    def _compress(self) -> None:
        """
        Ask a cheap model to summarise old history into a short narrative.
        Called automatically when history grows beyond compress_after ticks.
        """
        old = self.history[: -self.window_size]
        summary_prompt = (
            "Summarise the following Kubernetes cluster management decisions in 5 bullet points. "
            "Focus on: what scaling actions were taken, what worked, what didn't, "
            "reward trend, and current cluster state.\n\n"
            + "\n---\n".join(
                f"Tick obs: {e['obs'][:300]}...\nAction: {e['action']}\nReward: {e['reward']}"
                for e in old
            )
        )
        resp = client.chat.completions.create(
            model=COMPRESS_MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=400,
        )
        new_summary = resp.choices[0].message.content or ""
        if self.episode_summary:
            self.episode_summary = (
                f"{self.episode_summary}\n\n[LATER SUMMARY]\n{new_summary}"
            )
        else:
            self.episode_summary = new_summary
        self.history = self.history[-self.window_size :]
        console.print("[dim]🗜  Context compressed[/dim]")


# ── Observation → Prompt ──────────────────────────────────────────────────────


def obs_to_prompt(obs: Observation, reward_history: list[float]) -> str:
    lines = [
        f"=== Tick {obs.step} ===",
        f"Budget remaining: ${obs.budget_remaining_usd:.2f}  |  Cost now: ${obs.total_cost_usd_per_hour:.2f}/hr",
        f"Global SLO met: {obs.global_slo_met}",
    ]

    if obs.counterfactual_cost_usd_per_hour is not None:
        delta = obs.total_cost_usd_per_hour - obs.counterfactual_cost_usd_per_hour
        lines.append(
            f"Oracle optimal cost: ${obs.counterfactual_cost_usd_per_hour:.2f}/hr  "
            f"(you're {'over' if delta > 0 else 'under'} by ${abs(delta):.2f}/hr)"
        )

    lines += ["", "SERVICES:"]
    for name, svc in obs.services.items():
        slo_ok = svc.p99_latency_ms < 200 and svc.error_rate < 0.001
        slo_flag = "✅" if slo_ok else "⚠️ SLO BREACH"
        lines.append(
            f"  {name} {slo_flag}"
            f"\n    replicas={svc.replicas}"
            + (f" (pending→{svc.pending_replicas})" if svc.pending_replicas else "")
            + f"  cpu={svc.cpu_utilization:.0%}"
            f"  mem={svc.memory_utilization:.0%}"
            f"  p99={svc.p99_latency_ms:.0f}ms"
            f"  rps={svc.requests_per_second:.1f}"
            f"  errors={svc.error_rate:.4%}"
            f"\n    cpu_req={svc.cpu_request_millicores}m  mem_req={svc.memory_request_mb}MB"
        )

    lines += ["", "REGIONS:"]
    for rid, r in obs.regions.items():
        status = "⚠️  DEGRADED" if r.is_degraded else "✅ healthy"
        spot = "  [SPOT]" if r.is_spot else ""
        lines.append(
            f"  {rid} {status}{spot}"
            f"\n    weight={r.traffic_weight:.0%}  nodes={r.node_count}"
            f"  type={r.node_type}  cost=${r.cost_per_hour:.2f}/hr"
            f"  carbon={r.carbon_intensity_gco2_kwh:.0f} gCO₂/kWh"
        )

    if obs.pending_events:
        lines += ["", f"⏰ UPCOMING EVENTS: {', '.join(obs.pending_events)}"]

    if reward_history:
        recent = reward_history[-5:]
        trend = "↑" if len(recent) > 1 and recent[-1] > recent[0] else "↓"
        lines += [
            "",
            f"📈 RECENT REWARDS: {[round(r, 3) for r in recent]}  (trend: {trend})",
        ]

    lines += ["", "Think step by step, then output your JSON action."]
    return "\n".join(lines)


# ── Action Parser ─────────────────────────────────────────────────────────────


def parse_action_from_json(raw: str, obs: Observation) -> Action:
    """Parse LLM JSON output → validated Action. Falls back to no_op on error."""
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
        return Action(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[yellow][WARN] Action parse failed: {e}. Using no_op.[/yellow]")
        return Action(no_op=True)


def parse_action_from_tools(tool_calls: list[Any]) -> Action:
    """
    Merge multiple tool calls (scale_hpa + reroute_traffic in same tick) → one Action.
    This lets the agent do compound actions when using function calling mode.
    """
    hpa: HPAAction | None = None
    vpa: VPAAction | None = None
    traffic: TrafficAction | None = None
    node: NodeAction | None = None
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

    return Action(hpa=hpa, vpa=vpa, traffic=traffic, node=node, no_op=no_op)


# ── Main Agent Loop ───────────────────────────────────────────────────────────


def run_episode(
    env_url: str,
    task_id: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Run one full episode using the LLM reasoning agent."""
    ctx = ContextManager(window_size=5, compress_after=20)
    reward_history: list[float] = []

    console.print(Panel(f"[bold cyan]CloudScaleRL Agent[/bold cyan]\nTask: [yellow]{task_id}[/yellow]  Model: {model}  Tools: {USE_TOOLS}"))

    # Reset environment
    with httpx.Client(timeout=30.0) as http:
        resp = http.post(f"{env_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        obs = Observation(**resp.json())

        total_reward = 0.0
        done = False
        step_times: list[float] = []

        while not done:
            t0 = time.time()

            # Build prompt
            obs_text = obs_to_prompt(obs, reward_history)
            messages = ctx.build_messages(obs_text)

            # LLM inference
            if USE_TOOLS:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=512,
                    temperature=0.15,
                )
                msg = completion.choices[0].message
                if msg.tool_calls:
                    action = parse_action_from_tools(msg.tool_calls)
                else:
                    # Model chose not to call a tool — force no_op
                    action = Action(no_op=True)
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.15,
                    response_format={"type": "json_object"},
                )
                raw = completion.choices[0].message.content or "{}"
                action = parse_action_from_json(raw, obs)

            # Step environment
            step_resp = http.post(
                f"{env_url}/step",
                json=action.model_dump(exclude_none=True),
            )
            step_resp.raise_for_status()
            result = step_resp.json()

            obs = Observation(**result["observation"])
            reward = result["reward"]["total"]
            done = result["done"]

            # Update memory
            ctx.add(obs_text, action, reward)
            reward_history.append(reward)
            total_reward += reward

            elapsed = time.time() - t0
            step_times.append(elapsed)

            # Live display
            _print_tick(obs, action, reward, elapsed, completion.usage)  # type: ignore[arg-type]

    mean_reward = total_reward / max(len(reward_history), 1)
    console.print(
        Panel(
            f"[bold green]Episode complete[/bold green]\n"
            f"Ticks: {len(reward_history)}  "
            f"Total reward: {total_reward:.3f}  "
            f"Mean reward: {mean_reward:.3f}\n"
            f"Avg step time: {sum(step_times)/len(step_times):.2f}s"
        )
    )
    return {
        "task_id": task_id,
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "ticks": len(reward_history),
    }


def _print_tick(
    obs: Observation,
    action: Action,
    reward: float,
    elapsed: float,
    usage: Any,
) -> None:
    """Pretty-print one tick summary to the console."""
    color = "green" if reward > 0 else "red"
    slo = "✅" if obs.global_slo_met else "⚠️ "
    action_summary = action.model_dump_json(exclude_none=True, exclude={"no_op": False})

    tokens = f"{usage.total_tokens}tok" if usage else ""
    console.print(
        f"[dim]tick {obs.step:4d}[/dim]  "
        f"slo={slo}  "
        f"cost=${obs.total_cost_usd_per_hour:.2f}/hr  "
        f"[{color}]reward={reward:+.3f}[/{color}]  "
        f"[dim]{action_summary[:80]}  {elapsed:.1f}s {tokens}[/dim]"
    )


def print_leaderboard(scores: list[dict[str, Any]]) -> None:
    table = Table(title="CloudScaleRL Results")
    table.add_column("Task", style="cyan")
    table.add_column("Ticks", justify="right")
    table.add_column("Mean Reward", justify="right")
    table.add_column("Total Reward", justify="right")
    for s in scores:
        color = "green" if s["mean_reward"] > 0 else "red"
        table.add_row(
            s["task_id"],
            str(s["ticks"]),
            f"[{color}]{s['mean_reward']:+.4f}[/{color}]",
            f"{s['total_reward']:.3f}",
        )
    console.print(table)


def main() -> None:
    env_url = os.environ.get("ENV_URL", "http://localhost:8000")
    tasks = sys.argv[1:] if len(sys.argv) > 1 else ["task1_hpa"]
    model = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)

    all_scores: list[dict[str, Any]] = []
    for task in tasks:
        scores = run_episode(env_url, task, model=model)
        all_scores.append(scores)

    if len(all_scores) > 1:
        print_leaderboard(all_scores)
    else:
        console.print(f"\n[bold]Final score:[/bold] {all_scores[0]}")


if __name__ == "__main__":
    main()