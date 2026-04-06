#!/usr/bin/env bash
set -euo pipefail

print_help() {
  cat <<'EOF'
Run all CloudScaleRL tasks with a local Ollama model.

Usage:
  ./run_all_tasks_ollama.sh [task1_hpa task2_cost task3_incident ...]

Defaults:
  tasks: task1_hpa task2_cost task3_incident
  ENV_URL: http://127.0.0.1:8000
  OLLAMA_BASE_URL: http://127.0.0.1:11434
  AGENT_MODEL: sre-agent
  USE_TOOLS: false
  FORCE_JSON_RESPONSE_FORMAT: false
  AUTO_START_SERVER: true

Environment overrides:
  PYTHON_BIN                 Python executable (default: ./.venv/bin/python)
  ENV_URL                    CloudScaleRL API base URL
  OLLAMA_BASE_URL            Ollama base URL without /v1
  AGENT_MODEL                Ollama model name
  USE_TOOLS                  true|false
  FORCE_JSON_RESPONSE_FORMAT true|false
  AUTO_START_SERVER          true|false

Examples:
  ./run_all_tasks_ollama.sh
  AGENT_MODEL=sre-agent ./run_all_tasks_ollama.sh task1_hpa
  AUTO_START_SERVER=false ENV_URL=http://127.0.0.1:8000 ./run_all_tasks_ollama.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
ENV_URL="${ENV_URL:-http://127.0.0.1:8000}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
AGENT_MODEL="${AGENT_MODEL:-sre-agent}"
USE_TOOLS="${USE_TOOLS:-false}"
FORCE_JSON_RESPONSE_FORMAT="${FORCE_JSON_RESPONSE_FORMAT:-false}"
AUTO_START_SERVER="${AUTO_START_SERVER:-true}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python executable not found at ${PYTHON_BIN}"
  echo "        Set PYTHON_BIN, or create/activate your virtual environment first."
  exit 1
fi

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  TASKS=("task1_hpa" "task2_cost" "task3_incident")
fi

if ! curl -fsS "${OLLAMA_BASE_URL%/}/api/tags" >/dev/null 2>&1; then
  echo "[WARN] Ollama is not reachable at ${OLLAMA_BASE_URL}."
  echo "       Start Ollama first (for example: ollama serve)."
fi

server_started="false"
server_pid=""

cleanup() {
  if [[ "${server_started}" == "true" && -n "${server_pid}" ]]; then
    echo "[INFO] Stopping env server (PID ${server_pid})"
    kill "${server_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if curl -fsS "${ENV_URL%/}/health" >/dev/null 2>&1; then
  echo "[INFO] Using existing env server at ${ENV_URL}"
elif [[ "${AUTO_START_SERVER}" == "true" ]]; then
  stripped="${ENV_URL#http://}"
  stripped="${stripped#https://}"
  hostport="${stripped%%/*}"
  server_host="${hostport%%:*}"
  server_port="8000"
  if [[ "${hostport}" == *:* ]]; then
    server_port="${hostport##*:}"
  fi

  server_log="${ROOT_DIR}/.cloudscalerl_server.log"
  echo "[INFO] Starting env server on ${server_host}:${server_port}"
  "${PYTHON_BIN}" -m uvicorn cloudscalerl.server.app:app --host "${server_host}" --port "${server_port}" >"${server_log}" 2>&1 &
  server_pid="$!"
  server_started="true"

  if ! curl --silent --fail --retry 40 --retry-connrefused --retry-delay 1 "${ENV_URL%/}/health" >/dev/null; then
    echo "[ERROR] Env server failed to become healthy at ${ENV_URL}"
    echo "[INFO] Last server log lines:"
    tail -n 50 "${server_log}" || true
    exit 1
  fi
else
  echo "[ERROR] Env server not reachable at ${ENV_URL} and AUTO_START_SERVER=false"
  exit 1
fi

echo "[INFO] Running tasks: ${TASKS[*]}"
tmp_log="$(mktemp)"

USE_OLLAMA=true \
OLLAMA_BASE_URL="${OLLAMA_BASE_URL}" \
AGENT_MODEL="${AGENT_MODEL}" \
USE_TOOLS="${USE_TOOLS}" \
FORCE_JSON_RESPONSE_FORMAT="${FORCE_JSON_RESPONSE_FORMAT}" \
ENV_URL="${ENV_URL}" \
"${PYTHON_BIN}" "${ROOT_DIR}/inference.py" "${TASKS[@]}" | tee "${tmp_log}"

echo
echo "=== Summary ==="
if ! grep -E "^\[RESULT\]" "${tmp_log}"; then
  echo "[WARN] No [RESULT] lines found in output."
fi

rm -f "${tmp_log}"
