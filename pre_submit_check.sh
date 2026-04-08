#!/usr/bin/env bash
set -euo pipefail

print_help() {
  cat <<'EOF'
Run CloudScaleRL pre-submit checks in one command.

Usage:
  ./pre_submit_check.sh [task1_hpa task2_cost task3_incident ...]
  ./pre_submit_check.sh --print-hf-space-settings

Required:
  HF_TOKEN                    Hugging Face token for inference.py

Defaults:
  API_BASE_URL               http://127.0.0.1:8000
  MODEL_NAME                 hardcoded-controller
  AUTO_START_SERVER          true
  CHECK_API_ENDPOINTS        true   (validate /tasks, /metadata, /schema)
  CHECK_MCP                  auto   (probe /mcp initialize; fail only when true)
  RUN_OPENENV_VALIDATE       auto   (run only if openenv CLI is available)
  RUN_DOCKER_BUILD           false
  PYTHON_BIN                 ./.venv/bin/python

Examples:
  HF_TOKEN=abc ./pre_submit_check.sh
  HF_TOKEN=abc ./pre_submit_check.sh task1_hpa
  HF_TOKEN=abc CHECK_MCP=true ./pre_submit_check.sh task1_hpa
  HF_TOKEN=abc RUN_OPENENV_VALIDATE=true RUN_DOCKER_BUILD=true ./pre_submit_check.sh
  ./pre_submit_check.sh --print-hf-space-settings
EOF
}

print_hf_space_settings() {
  cat <<'EOF'
Hugging Face Space settings for CloudScaleRL

Secrets (required):
  HF_TOKEN=<fine-grained-read-token>

Variables (optional):
  MODEL_NAME=hardcoded-controller
  API_BASE_URL=http://127.0.0.1:8000
  BENCHMARK_NAME=cloudscalerl
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

if [[ "${1:-}" == "--print-hf-space-settings" ]]; then
  print_hf_space_settings
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_FILE="${ROOT_DIR}/inference.py"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
MODEL_NAME="${MODEL_NAME:-hardcoded-controller}"
AUTO_START_SERVER="${AUTO_START_SERVER:-true}"
RUN_OPENENV_VALIDATE="${RUN_OPENENV_VALIDATE:-auto}"
RUN_DOCKER_BUILD="${RUN_DOCKER_BUILD:-false}"
CHECK_API_ENDPOINTS="${CHECK_API_ENDPOINTS:-true}"
CHECK_MCP="${CHECK_MCP:-auto}"
HEALTH_RETRIES="${HEALTH_RETRIES:-40}"
HEALTH_DELAY_S="${HEALTH_DELAY_S:-1}"

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  TASKS=("task1_hpa" "task2_cost" "task3_incident")
fi

if [[ -z "${HF_TOKEN+x}" ]]; then
  echo "[FAIL] HF_TOKEN is required."
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FAIL] PYTHON_BIN is not executable: ${PYTHON_BIN}"
  exit 1
fi

server_started="false"
server_pid=""
server_log=""

cleanup() {
  if [[ "${server_started}" == "true" && -n "${server_pid}" ]]; then
    echo "[INFO] Stopping temporary server (PID ${server_pid})"
    kill "${server_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${server_log}" && -f "${server_log}" ]]; then
    rm -f "${server_log}"
  fi
}
trap cleanup EXIT

require_pattern() {
  local pattern="$1"
  local description="$2"
  if grep -qE "${pattern}" "${INFERENCE_FILE}"; then
    echo "[PASS] ${description}"
  else
    echo "[FAIL] ${description}"
    exit 1
  fi
}

probe_mcp_initialize() {
  local mcp_url="${API_BASE_URL%/}/mcp"
  local mcp_payload='{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'
  local mcp_body

  if ! mcp_body="$(curl -fsS -X POST "${mcp_url}" -H "Content-Type: application/json" -d "${mcp_payload}")"; then
    return 1
  fi

  "${PYTHON_BIN}" - "${mcp_body}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
if payload.get("jsonrpc") != "2.0":
    raise SystemExit(1)

result = payload.get("result")
if not isinstance(result, dict):
    raise SystemExit(1)

server_info = result.get("serverInfo")
if not isinstance(server_info, dict) or not server_info.get("name"):
    raise SystemExit(1)

print("[PASS] MCP initialize endpoint is valid")
PY
}

echo "[INFO] Checking static submission contract in inference.py"
if [[ ! -f "${INFERENCE_FILE}" ]]; then
  echo "[FAIL] Missing root inference.py"
  exit 1
fi
echo "[PASS] Root inference.py exists"

require_pattern '^API_BASE_URL = os.getenv\("API_BASE_URL", "[^"]+"\)' "API_BASE_URL has default"
require_pattern '^MODEL_NAME = os.getenv\("MODEL_NAME", "[^"]+"\)' "MODEL_NAME has default"
require_pattern '^if HF_TOKEN is None:' "HF_TOKEN required guard exists"
require_pattern '^from openai import OpenAI$' "OpenAI import exists"
require_pattern 'OpenAI\(base_url=API_BASE_URL, api_key=HF_TOKEN\)' "OpenAI client initialization exists"

echo "[INFO] Verifying missing HF_TOKEN fails fast"
hf_missing_log="$(mktemp)"
set +e
env -u HF_TOKEN API_BASE_URL="${API_BASE_URL}" MODEL_NAME="${MODEL_NAME}" \
  "${PYTHON_BIN}" "${INFERENCE_FILE}" task1_hpa >"${hf_missing_log}" 2>&1
hf_missing_rc=$?
set -e
if [[ ${hf_missing_rc} -eq 0 ]]; then
  echo "[FAIL] inference.py should fail when HF_TOKEN is missing"
  cat "${hf_missing_log}"
  rm -f "${hf_missing_log}"
  exit 1
fi
if ! grep -q 'HF_TOKEN environment variable is required' "${hf_missing_log}"; then
  echo "[FAIL] Missing HF_TOKEN error message was not found"
  cat "${hf_missing_log}"
  rm -f "${hf_missing_log}"
  exit 1
fi
rm -f "${hf_missing_log}"
echo "[PASS] Missing HF_TOKEN behavior is correct"

health_url="${API_BASE_URL%/}/health"
if curl -fsS "${health_url}" >/dev/null 2>&1; then
  echo "[PASS] Using existing server at ${API_BASE_URL}"
elif [[ "${AUTO_START_SERVER}" == "true" ]]; then
  stripped="${API_BASE_URL#http://}"
  stripped="${stripped#https://}"
  hostport="${stripped%%/*}"
  server_host="${hostport%%:*}"
  server_port="8000"
  if [[ "${hostport}" == *:* ]]; then
    server_port="${hostport##*:}"
  fi

  server_log="$(mktemp)"
  echo "[INFO] Starting temporary server on ${server_host}:${server_port}"
  "${PYTHON_BIN}" -m uvicorn cloudscalerl.server.app:app \
    --host "${server_host}" --port "${server_port}" >"${server_log}" 2>&1 &
  server_pid="$!"
  server_started="true"

  if ! curl --silent --fail --retry "${HEALTH_RETRIES}" --retry-connrefused \
    --retry-delay "${HEALTH_DELAY_S}" "${health_url}" >/dev/null; then
    echo "[FAIL] Server failed to become healthy at ${API_BASE_URL}"
    echo "[INFO] Last server log lines:"
    tail -n 60 "${server_log}" || true
    exit 1
  fi
  echo "[PASS] Temporary server is healthy"
else
  echo "[FAIL] Server not reachable at ${API_BASE_URL} and AUTO_START_SERVER=false"
  exit 1
fi

if [[ "${CHECK_API_ENDPOINTS}" == "true" ]]; then
  echo "[INFO] Validating API discovery endpoints"
  tasks_payload="$(curl -fsS "${API_BASE_URL%/}/tasks")"
  metadata_payload="$(curl -fsS "${API_BASE_URL%/}/metadata")"
  schema_payload="$(curl -fsS "${API_BASE_URL%/}/schema")"

  if ! "${PYTHON_BIN}" - "${tasks_payload}" "${metadata_payload}" "${schema_payload}" <<'PY'
import json
import sys

tasks = json.loads(sys.argv[1])
metadata = json.loads(sys.argv[2])
schema = json.loads(sys.argv[3])

errors = []

if not isinstance(tasks, list) or not tasks:
    errors.append("/tasks must return a non-empty list")
else:
    required_task_keys = {"id", "name", "difficulty", "max_steps"}
    missing = required_task_keys.difference(tasks[0].keys())
    if missing:
        errors.append(f"/tasks first entry missing keys: {sorted(missing)}")

if not isinstance(metadata, dict):
    errors.append("/metadata must return a JSON object")
else:
    for key in ("name", "version", "tasks"):
        if key not in metadata:
            errors.append(f"/metadata missing key: {key}")

if not isinstance(schema, dict):
    errors.append("/schema must return a JSON object")
else:
    for key in ("action", "observation", "state"):
        if key not in schema:
            errors.append(f"/schema missing key: {key}")

if errors:
    print("[FAIL] API discovery validation failed:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    raise SystemExit(1)

print("[PASS] API discovery endpoints are valid")
PY
  then
    exit 1
  fi
else
  echo "[INFO] Skipping API discovery endpoint checks"
fi

case "${CHECK_MCP}" in
  true)
    echo "[INFO] Probing MCP initialize endpoint"
    if ! probe_mcp_initialize; then
      echo "[FAIL] MCP initialize probe failed"
      exit 1
    fi
    ;;
  auto)
    echo "[INFO] Probing MCP initialize endpoint (auto)"
    if ! probe_mcp_initialize; then
      echo "[INFO] Skipping MCP probe result in auto mode"
    fi
    ;;
  false)
    echo "[INFO] Skipping MCP probe"
    ;;
  *)
    echo "[FAIL] Invalid CHECK_MCP value: ${CHECK_MCP}"
    exit 1
    ;;
esac

echo "[INFO] Running inference for tasks: ${TASKS[*]}"
inference_log="$(mktemp)"
set +e
HF_TOKEN="${HF_TOKEN}" API_BASE_URL="${API_BASE_URL}" MODEL_NAME="${MODEL_NAME}" \
  "${PYTHON_BIN}" "${INFERENCE_FILE}" "${TASKS[@]}" >"${inference_log}" 2>&1
inference_rc=$?
set -e
if [[ ${inference_rc} -ne 0 ]]; then
  echo "[FAIL] inference.py exited with non-zero status (${inference_rc})"
  tail -n 80 "${inference_log}" || true
  rm -f "${inference_log}"
  exit 1
fi

echo "[INFO] Validating [START]/[STEP]/[END] output contract"
if ! "${PYTHON_BIN}" - "${inference_log}" "${TASKS[@]}" <<'PY'
import re
import sys

log_path = sys.argv[1]
expected_tasks = sys.argv[2:]

start_re = re.compile(r"^\[START\] task=([^\s]+) env=([^\s]+) model=([^\s]+)$")
step_re = re.compile(r"^\[STEP\] step=(\d+) action=(.*) reward=(-?\d+\.\d{2}) done=(true|false) error=(.*)$")
end_re = re.compile(r"^\[END\] success=(true|false) steps=(\d+) rewards=(.*)$")
reward_re = re.compile(r"^-?\d+\.\d{2}$")

with open(log_path, "r", encoding="utf-8") as fh:
    lines = [line.rstrip("\n") for line in fh]

errors = []
episodes = []
current = None

for lineno, line in enumerate(lines, start=1):
    m_start = start_re.match(line)
    if m_start:
        if current is not None:
            errors.append(f"line {lineno}: encountered START before previous END")
        current = {
            "task": m_start.group(1),
            "step_count": 0,
            "last_step": 0,
            "success": None,
        }
        episodes.append(current)
        continue

    m_step = step_re.match(line)
    if m_step:
        if current is None:
            errors.append(f"line {lineno}: STEP appeared before START")
            continue
        step_no = int(m_step.group(1))
        if step_no != current["last_step"] + 1:
            errors.append(
                f"line {lineno}: non-sequential step number {step_no}, expected {current['last_step'] + 1}"
            )
        current["last_step"] = step_no
        current["step_count"] += 1
        continue

    m_end = end_re.match(line)
    if m_end:
        if current is None:
            errors.append(f"line {lineno}: END appeared before START")
            continue

        success = m_end.group(1) == "true"
        steps = int(m_end.group(2))
        rewards_raw = m_end.group(3)

        if steps != current["step_count"]:
            errors.append(
                f"line {lineno}: END steps={steps} but observed {current['step_count']} STEP lines"
            )

        rewards = [] if rewards_raw == "" else rewards_raw.split(",")
        if steps != len(rewards):
            errors.append(
                f"line {lineno}: END steps={steps} but rewards list has {len(rewards)} items"
            )
        for idx, reward in enumerate(rewards, start=1):
            if not reward_re.match(reward):
                errors.append(
                    f"line {lineno}: reward #{idx} has invalid format '{reward}'"
                )

        if not success:
            errors.append(f"line {lineno}: END success=false")

        current["success"] = success
        current = None
        continue

    errors.append(f"line {lineno}: unexpected line type '{line[:120]}'")

if current is not None:
    errors.append("file ended before closing END line")

if len(episodes) != len(expected_tasks):
    errors.append(
        f"expected {len(expected_tasks)} episodes but found {len(episodes)}"
    )

for idx, task_name in enumerate(expected_tasks):
    if idx >= len(episodes):
        break
    found_task = episodes[idx]["task"]
    if found_task != task_name:
        errors.append(
            f"episode order mismatch at position {idx + 1}: expected {task_name}, got {found_task}"
        )

if errors:
    print("[FAIL] Output contract validation failed:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    sys.exit(1)

print(f"[PASS] Output contract valid for {len(episodes)} episode(s)")
PY
then
  tail -n 80 "${inference_log}" || true
  rm -f "${inference_log}"
  exit 1
fi
rm -f "${inference_log}"

case "${RUN_OPENENV_VALIDATE}" in
  true)
    if ! command -v openenv >/dev/null 2>&1; then
      echo "[FAIL] RUN_OPENENV_VALIDATE=true but openenv CLI is not installed"
      exit 1
    fi
    echo "[INFO] Running openenv validate"
    openenv validate "${API_BASE_URL}"
    echo "[PASS] openenv validate"
    ;;
  auto)
    if command -v openenv >/dev/null 2>&1; then
      echo "[INFO] Running openenv validate"
      openenv validate "${API_BASE_URL}"
      echo "[PASS] openenv validate"
    else
      echo "[INFO] Skipping openenv validate (CLI not found)"
    fi
    ;;
  false)
    echo "[INFO] Skipping openenv validate"
    ;;
  *)
    echo "[FAIL] Invalid RUN_OPENENV_VALIDATE value: ${RUN_OPENENV_VALIDATE}"
    exit 1
    ;;
esac

if [[ "${RUN_DOCKER_BUILD}" == "true" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "[FAIL] RUN_DOCKER_BUILD=true but docker is not installed"
    exit 1
  fi
  echo "[INFO] Running docker build smoke check"
  docker build -q -t cloudscalerl-presubmit "${ROOT_DIR}" >/dev/null
  echo "[PASS] docker build"
else
  echo "[INFO] Skipping docker build"
fi

echo "[PASS] All presubmit checks passed"
