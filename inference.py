"""
CloudScaleRL inference entrypoint.

This script follows the OpenEnv RL challenge output contract:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Required env vars:
  API_BASE_URL  (default provided)
  MODEL_NAME    (default provided)
  HF_TOKEN      (required)
"""

from __future__ import annotations

import os
import sys
from typing import Any

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Reserved for optional LLM-backed policies.
# Current baseline uses deterministic hardcoded control logic.
LLM_CLIENT = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Keep compatibility with existing client env names.
os.environ.setdefault("OPENAI_API_KEY", HF_TOKEN)
os.environ.setdefault("ENV_URL", API_BASE_URL)
os.environ.setdefault("AGENT_MODEL", MODEL_NAME)

from cloudscalerl.client import run_episode

DEFAULT_TASKS = ["task1_hpa", "task2_cost", "task3_incident"]
BENCHMARK_NAME = os.getenv("BENCHMARK_NAME", "cloudscalerl")
SEED = int(os.getenv("SEED", "42"))
INFERENCE_TEMPERATURE = float(os.getenv("INFERENCE_TEMPERATURE", "0.0"))


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _sanitize_single_line(text: str) -> str:
    return " ".join(text.splitlines()).strip()


def _format_error(err: Any) -> str:
    if err is None:
        return "null"
    return _sanitize_single_line(str(err)) or "null"


def _run_task(task: str) -> None:
    rewards: list[float] = []
    steps = 0
    success = False

    print(f"[START] task={task} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)

    def _on_step(event: dict[str, Any]) -> None:
        nonlocal steps
        step_num = int(event.get("step", 0))
        action_str = _sanitize_single_line(str(event.get("action", "{}")))
        reward = float(event.get("reward", 0.0))
        done = bool(event.get("done", False))
        error = _format_error(event.get("error"))

        steps = step_num
        rewards.append(reward)

        print(
            f"[STEP] step={step_num} action={action_str} reward={reward:.2f} "
            f"done={_bool_str(done)} error={error}",
            flush=True,
        )

    try:
        result = run_episode(
            env_url=API_BASE_URL,
            task_id=task,
            model=MODEL_NAME,
            seed=SEED,
            emit_console=False,
            temperature=INFERENCE_TEMPERATURE,
            on_step=_on_step,
        )
        success = bool(result.get("success", False))
    except Exception:
        success = False
    finally:
        rewards_csv = ",".join(f"{reward:.2f}" for reward in rewards)
        print(
            f"[END] success={_bool_str(success)} steps={steps} rewards={rewards_csv}",
            flush=True,
        )


def main() -> None:
    tasks = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TASKS
    for task in tasks:
        _run_task(task)


if __name__ == "__main__":
    main()
