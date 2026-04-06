"""
CloudScaleRL inference entrypoint.

This script is a thin wrapper around cloudscalerl.client.run_episode and
works with any OpenAI-compatible endpoint, including Ollama.

Example with Ollama:
  export ENV_URL='http://127.0.0.1:8000'
  export USE_OLLAMA='true'
  export OLLAMA_BASE_URL='http://127.0.0.1:11434'
  export AGENT_MODEL='cloudscalerl-sre:latest'
  export USE_TOOLS='false'
  /workspaces/cloud-sre-gym/.venv/bin/python inference.py task1_hpa
"""

from __future__ import annotations

import os
import sys

from cloudscalerl.client import DEFAULT_MODEL, run_episode


def main() -> None:
    env_url = os.environ.get("ENV_URL", "http://localhost:8000")
    model = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
    tasks = sys.argv[1:] if len(sys.argv) > 1 else [os.environ.get("TASK_ID", "task1_hpa")]

    for task in tasks:
        result = run_episode(env_url=env_url, task_id=task, model=model)
        print(
            f"[RESULT] task={result['task_id']} ticks={result['ticks']} "
            f"mean_reward={result['mean_reward']:.4f} total_reward={result['total_reward']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
