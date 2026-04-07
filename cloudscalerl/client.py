"""
CloudScaleRL hardcoded controller client.

This module provides:
    - openenv-compatible client wrappers
    - an HTTP endpoint adapter for the FastAPI server
    - a deterministic, cloud-like control policy used by run_episode()

Usage:
        python -m cloudscalerl.client task1_hpa
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable, Optional

import httpx
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
DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "hardcoded-controller")


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


# ── Main Agent Loop ───────────────────────────────────────────────────────────


def _parse_t_plus(event: str) -> Optional[int]:
    if "_t+" not in event:
        return None
    try:
        return int(event.rsplit("_t+", 1)[1])
    except ValueError:
        return None


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        n = max(len(clean), 1)
        equal = 1.0 / n
        return {k: equal for k in clean}
    normalized = {k: v / total for k, v in clean.items()}
    anchor = max(normalized, key=normalized.get)
    normalized[anchor] += 1.0 - sum(normalized.values())
    return normalized


def _budget_pressure(obs: CloudScaleObservation) -> bool:
    return obs.budget_remaining_usd <= max(2.0 * obs.total_cost_usd_per_hour, 10.0)


def _budget_critical(obs: CloudScaleObservation) -> bool:
    return obs.budget_remaining_usd <= max(obs.total_cost_usd_per_hour, 5.0)


def _decrement_policy_state(policy_state: dict[str, Any]) -> None:
    for key in ("hpa_cooldowns", "vpa_cooldowns"):
        cooldowns: dict[str, int] = policy_state[key]
        for name in list(cooldowns.keys()):
            cooldowns[name] -= 1
            if cooldowns[name] <= 0:
                cooldowns.pop(name, None)
    policy_state["traffic_cooldown"] = max(0, int(policy_state["traffic_cooldown"]) - 1)
    policy_state["node_cooldown"] = max(0, int(policy_state["node_cooldown"]) - 1)


def _choose_hardcoded_action(
    obs: CloudScaleObservation,
    task_id: str,
    policy_state: dict[str, Any],
) -> tuple[CloudScaleAction, Optional[str], str]:
    """
    Deterministic cloud-like control policy.

    Priority order:
      1) traffic failover for degraded or imminent incident regions
      2) SLO-protecting HPA scale-up (with cooldown)
      3) budget-aware HPA scale-down
      4) VPA right-sizing for sustained pressure or budget protection
      5) node add/remove with conservative cooldown
      6) optional carbon-aware traffic rebalance when stable
      7) no-op
    """
    _decrement_policy_state(policy_state)
    hpa_cooldowns: dict[str, int] = policy_state["hpa_cooldowns"]
    vpa_cooldowns: dict[str, int] = policy_state["vpa_cooldowns"]

    try:
        spike_soon = any(
            event.startswith("traffic_spike_t+") and (_parse_t_plus(event) or 99) <= 2
            for event in obs.pending_events
        )

        degraded_regions = [rid for rid, r in obs.regions.items() if r.is_degraded]
        upcoming_avoid: set[str] = set(degraded_regions)
        for event in obs.pending_events:
            if not event.startswith("az_degradation_"):
                continue
            ticks = _parse_t_plus(event)
            if ticks is None or ticks > 2:
                continue
            parts = event.split("_")
            if len(parts) >= 4:
                region_id = parts[2]
                if region_id in obs.regions and obs.regions[region_id].traffic_weight > 0.15:
                    upcoming_avoid.add(region_id)

        if upcoming_avoid and policy_state["traffic_cooldown"] == 0:
            healthy = {
                rid: r for rid, r in obs.regions.items() if rid not in upcoming_avoid
            }
            if healthy:
                score = {
                    rid: ((r.node_count + 1.0) / max(r.cost_per_hour, 0.1))
                    + (180.0 / max(r.carbon_intensity_gco2_kwh, 80.0))
                    for rid, r in healthy.items()
                }
                weights = {rid: 0.0 for rid in obs.regions}
                for rid, val in _normalize_weights(score).items():
                    weights[rid] = val
                policy_state["traffic_cooldown"] = 3
                return (
                    CloudScaleAction(
                        traffic=TrafficAction(region_weights=_normalize_weights(weights))
                    ),
                    None,
                    "traffic_failover",
                )

        hottest_name: Optional[str] = None
        hottest_score = 0.0
        hottest_metrics = None
        for svc_name, svc in obs.services.items():
            if svc.pending_replicas is not None:
                continue
            if hpa_cooldowns.get(svc_name, 0) > 0:
                continue
            risk = max(
                svc.cpu_utilization / 0.75,
                svc.p99_latency_ms / 200.0,
                svc.error_rate / 0.001,
            )
            if risk > hottest_score:
                hottest_name = svc_name
                hottest_score = risk
                hottest_metrics = svc

        if hottest_name is not None and hottest_metrics is not None:
            urgent = hottest_score >= 1.0
            proactive = spike_soon and hottest_score >= 0.85
            if urgent or proactive:
                demand_factor = max(
                    hottest_metrics.cpu_utilization / 0.65,
                    hottest_metrics.p99_latency_ms / 170.0,
                    1.0 + min(1.0, hottest_metrics.error_rate / 0.001) * 0.25,
                )
                target = max(
                    hottest_metrics.replicas + 1,
                    int(hottest_metrics.replicas * demand_factor),
                )
                if proactive:
                    target += 1
                target = min(100, min(target, hottest_metrics.replicas + 4))
                hpa_cooldowns[hottest_name] = 2
                return (
                    CloudScaleAction(
                        hpa=HPAAction(
                            service=hottest_name,
                            target_replicas=target,
                            target_cpu_utilization=0.65,
                        )
                    ),
                    None,
                    "hpa_scale_up",
                )

        if _budget_pressure(obs):
            cool_candidates = []
            for svc_name, svc in obs.services.items():
                if svc.replicas <= 1 or svc.pending_replicas is not None:
                    continue
                if hpa_cooldowns.get(svc_name, 0) > 0:
                    continue
                if (
                    svc.p99_latency_ms < 130
                    and svc.error_rate < 0.0007
                    and svc.cpu_utilization < 0.45
                ):
                    cool_candidates.append((svc_name, svc))
            if cool_candidates:
                cool_candidates.sort(
                    key=lambda item: (item[1].cpu_utilization, item[1].p99_latency_ms)
                )
                svc_name, svc = cool_candidates[0]
                step_down = 2 if _budget_critical(obs) and svc.replicas > 3 else 1
                target = max(1, svc.replicas - step_down)
                hpa_cooldowns[svc_name] = 3
                return (
                    CloudScaleAction(hpa=HPAAction(service=svc_name, target_replicas=target)),
                    None,
                    "budget_scale_down",
                )

        for svc_name, svc in sorted(
            obs.services.items(),
            key=lambda item: item[1].memory_utilization,
            reverse=True,
        ):
            if svc.pending_replicas is not None:
                continue
            if vpa_cooldowns.get(svc_name, 0) > 0:
                continue

            if svc.memory_utilization > 0.85 or svc.cpu_utilization > 0.85:
                new_cpu = (
                    int(svc.cpu_request_millicores * 1.15)
                    if svc.cpu_utilization > 0.85
                    else svc.cpu_request_millicores
                )
                new_mem = (
                    int(svc.memory_request_mb * 1.20)
                    if svc.memory_utilization > 0.85
                    else svc.memory_request_mb
                )
                vpa_cooldowns[svc_name] = 8
                return (
                    CloudScaleAction(
                        vpa=VPAAction(
                            service=svc_name,
                            cpu_request_millicores=new_cpu,
                            memory_request_mb=new_mem,
                        )
                    ),
                    None,
                    "vpa_tune_up",
                )

            if _budget_pressure(obs) and svc.cpu_utilization < 0.35 and svc.memory_utilization < 0.45:
                new_cpu = max(250, int(svc.cpu_request_millicores * 0.90))
                new_mem = max(256, int(svc.memory_request_mb * 0.90))
                if new_cpu != svc.cpu_request_millicores or new_mem != svc.memory_request_mb:
                    vpa_cooldowns[svc_name] = 8
                    return (
                        CloudScaleAction(
                            vpa=VPAAction(
                                service=svc_name,
                                cpu_request_millicores=new_cpu,
                                memory_request_mb=new_mem,
                            )
                        ),
                        None,
                        "vpa_tune_down",
                    )

        if policy_state["node_cooldown"] == 0 and obs.services:
            avg_cpu = sum(s.cpu_utilization for s in obs.services.values()) / len(obs.services)
            healthy_regions = [rid for rid, r in obs.regions.items() if not r.is_degraded]

            if avg_cpu > 0.86 and not _budget_critical(obs) and healthy_regions:
                target_region = max(healthy_regions, key=lambda rid: obs.regions[rid].traffic_weight)
                node_type = obs.regions[target_region].node_type
                if _budget_pressure(obs) and not node_type.startswith("spot."):
                    node_type = "spot.c5.xlarge"
                policy_state["node_cooldown"] = 5
                return (
                    CloudScaleAction(
                        node=NodeAction(
                            region=target_region,
                            operation="add",
                            node_type=node_type,
                            count=1,
                        )
                    ),
                    None,
                    "node_scale_up",
                )

            if _budget_critical(obs) and avg_cpu < 0.55:
                candidates = [
                    rid
                    for rid, r in obs.regions.items()
                    if not r.is_degraded and r.node_count > 1 and r.traffic_weight < 0.70
                ]
                if candidates:
                    target_region = max(candidates, key=lambda rid: obs.regions[rid].cost_per_hour)
                    policy_state["node_cooldown"] = 5
                    return (
                        CloudScaleAction(
                            node=NodeAction(region=target_region, operation="remove", count=1)
                        ),
                        None,
                        "node_scale_down",
                    )

        if (
            policy_state["traffic_cooldown"] == 0
            and obs.global_slo_met
            and len(obs.regions) >= 2
            and not any(r.is_degraded for r in obs.regions.values())
        ):
            source = max(
                obs.regions,
                key=lambda rid: (
                    obs.regions[rid].carbon_intensity_gco2_kwh
                    * obs.regions[rid].traffic_weight
                ),
            )
            target = min(
                obs.regions,
                key=lambda rid: obs.regions[rid].carbon_intensity_gco2_kwh,
            )
            carbon_gap = (
                obs.regions[source].carbon_intensity_gco2_kwh
                - obs.regions[target].carbon_intensity_gco2_kwh
            )
            if source != target and obs.regions[source].traffic_weight > 0.30 and carbon_gap > 150:
                shift = min(0.15, obs.regions[source].traffic_weight - 0.20)
                if shift > 0:
                    weights = {rid: r.traffic_weight for rid, r in obs.regions.items()}
                    weights[source] -= shift
                    weights[target] += shift
                    policy_state["traffic_cooldown"] = 4
                    return (
                        CloudScaleAction(
                            traffic=TrafficAction(region_weights=_normalize_weights(weights))
                        ),
                        None,
                        "carbon_rebalance",
                    )

        return CloudScaleAction(no_op=True), None, "steady_noop"
    except (ValidationError, ValueError, TypeError) as exc:
        return CloudScaleAction(no_op=True), str(exc), "policy_fallback_noop"


def run_episode(
    env_url: str,
    task_id: str,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    emit_console: bool = True,
    temperature: float = 0.15,
    on_step: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Run one full episode using deterministic hardcoded cloud controller."""
    _ = model, temperature
    reward_history: list[float] = []
    policy_state: dict[str, Any] = {
        "hpa_cooldowns": {},
        "vpa_cooldowns": {},
        "traffic_cooldown": 0,
        "node_cooldown": 0,
    }

    if emit_console:
        console.print(Panel(
            f"[bold cyan]CloudScaleRL Hardcoded Controller[/bold cyan]\n"
            f"Task: [yellow]{task_id}[/yellow]  Policy: deterministic-real-cloud"
        ))

    with CloudScaleHTTPEndpoint(base_url=env_url) as env:
        result = env.reset(task_id=task_id, seed=seed)
        obs: CloudScaleObservation = result.observation

        total_reward = 0.0
        done = False
        step_times: list[float] = []

        while not done:
            t0 = time.time()
            action, action_error, policy_reason = _choose_hardcoded_action(
                obs,
                task_id,
                policy_state,
            )

            # Step via EnvClient
            result = env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            reward_history.append(reward)
            total_reward += reward

            elapsed = time.time() - t0
            step_times.append(elapsed)
            if emit_console:
                _print_tick(
                    obs,
                    action,
                    reward,
                    elapsed,
                    policy_reason=policy_reason,
                )

            if on_step is not None:
                on_step(
                    {
                        "step": len(reward_history),
                        "action": action.model_dump_json(exclude_none=True),
                        "reward": reward,
                        "done": done,
                        "error": action_error,
                    }
                )

    mean_reward = total_reward / max(len(reward_history), 1)
    if emit_console:
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
        "rewards": reward_history,
        "success": bool(done and len(reward_history) > 0),
    }


def _print_tick(
    obs: CloudScaleObservation,
    action: CloudScaleAction,
    reward: float,
    elapsed: float,
    policy_reason: Optional[str] = None,
) -> None:
    color = "green" if reward > 0 else "red"
    slo = "✅" if obs.global_slo_met else "⚠️ "
    action_str = action.model_dump_json(exclude_none=True)
    policy_str = f"policy={policy_reason}" if policy_reason else ""
    console.print(
        f"[dim]tick {obs.step:4d}[/dim]  "
        f"slo={slo}  cost=${obs.total_cost_usd_per_hour:.2f}/hr  "
        f"[{color}]reward={reward:+.3f}[/{color}]  "
        f"[dim]{action_str[:80]}  {elapsed:.1f}s {policy_str}[/dim]"
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