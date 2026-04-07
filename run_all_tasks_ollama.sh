#!/usr/bin/env bash
set -euo pipefail

print_help() {
  cat <<'EOF'
Run all CloudScaleRL tasks with a local Ollama model.

Usage:
  ./run_all_tasks_ollama.sh [task1_hpa task2_cost task3_incident ...]

Defaults:
  tasks: task1_hpa task2_cost task3_incident
  API_BASE_URL: http://127.0.0.1:8000
  OLLAMA_BASE_URL: http://127.0.0.1:11434
  MODEL_NAME: sre-agent
  HF_TOKEN: ollama
  OLLAMA_AUTO_START: true
  USE_TOOLS: false
  FORCE_JSON_RESPONSE_FORMAT: false
  AUTO_START_SERVER: true

Environment overrides:
  PYTHON_BIN                 Python executable (default: ./.venv/bin/python)
  API_BASE_URL               CloudScaleRL API base URL
  ENV_URL                    Legacy alias for API_BASE_URL
  OLLAMA_BASE_URL            Ollama base URL without /v1
  MODEL_NAME                 Inference model name
  AGENT_MODEL                Ollama model name
  HF_TOKEN                   Required by inference.py (defaults to OPENAI_API_KEY or 'ollama')
  OLLAMA_AUTO_START          true|false (start ollama serve if reachable locally)
  USE_TOOLS                  true|false
  FORCE_JSON_RESPONSE_FORMAT true|false
  AUTO_START_SERVER          true|false

Examples:
  ./run_all_tasks_ollama.sh
  AGENT_MODEL=sre-agent ./run_all_tasks_ollama.sh task1_hpa
  AUTO_START_SERVER=false API_BASE_URL=http://127.0.0.1:8000 ./run_all_tasks_ollama.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
API_BASE_URL="${API_BASE_URL:-${ENV_URL:-http://127.0.0.1:8000}}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
MODEL_NAME="${MODEL_NAME:-${AGENT_MODEL:-sre-agent}}"
AGENT_MODEL="${AGENT_MODEL:-${MODEL_NAME}}"
HF_TOKEN="${HF_TOKEN:-${OPENAI_API_KEY:-ollama}}"
OLLAMA_AUTO_START="${OLLAMA_AUTO_START:-true}"
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

ollama_started="false"
ollama_pid=""
server_started="false"
server_pid=""

ollama_healthcheck() {
  curl -fsS "${OLLAMA_BASE_URL%/}/api/tags" >/dev/null 2>&1
}

ensure_ollama() {
  if ollama_healthcheck; then
    echo "[INFO] Using Ollama endpoint at ${OLLAMA_BASE_URL}"
  else
    echo "[WARN] Ollama is not reachable at ${OLLAMA_BASE_URL}."

    if [[ "${OLLAMA_AUTO_START}" == "true" ]] && command -v ollama >/dev/null 2>&1; then
      ollama_log="${ROOT_DIR}/.ollama_serve.log"
      echo "[INFO] Starting Ollama service"
      ollama serve >"${ollama_log}" 2>&1 &
      ollama_pid="$!"
      ollama_started="true"

      if ! curl --silent --fail --retry 30 --retry-connrefused --retry-delay 1 "${OLLAMA_BASE_URL%/}/api/tags" >/dev/null; then
        echo "[ERROR] Failed to start or reach Ollama at ${OLLAMA_BASE_URL}."
        echo "[INFO] Last Ollama log lines:"
        tail -n 50 "${ollama_log}" || true
        exit 1
      fi
    else
      echo "[ERROR] Ollama is required but unreachable at ${OLLAMA_BASE_URL}."
      if ! command -v ollama >/dev/null 2>&1; then
        echo "[HINT] Ollama CLI is not installed in this environment."
      fi
      echo "[HINT] Start Ollama first (for example: ollama serve)."
      echo "[HINT] If this is a remote container, localhost points to the container."
      echo "       Set OLLAMA_BASE_URL to a reachable Ollama host."
      exit 1
    fi
  fi

  if ! curl -fsS "${OLLAMA_BASE_URL%/}/api/show" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"${AGENT_MODEL}\"}" >/dev/null 2>&1; then
    echo "[ERROR] Model '${AGENT_MODEL}' not found at ${OLLAMA_BASE_URL}."
    echo "[HINT] Create it: ollama create ${AGENT_MODEL} -f Modelfile"
    echo "[HINT] Or set AGENT_MODEL to an existing model name."
    exit 1
  fi
}

cleanup() {
  if [[ "${server_started}" == "true" && -n "${server_pid}" ]]; then
    echo "[INFO] Stopping env server (PID ${server_pid})"
    kill "${server_pid}" >/dev/null 2>&1 || true
  fi
  if [[ "${ollama_started}" == "true" && -n "${ollama_pid}" ]]; then
    echo "[INFO] Stopping ollama service (PID ${ollama_pid})"
    kill "${ollama_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

ensure_ollama

if curl -fsS "${API_BASE_URL%/}/health" >/dev/null 2>&1; then
  echo "[INFO] Using existing env server at ${API_BASE_URL}"
elif [[ "${AUTO_START_SERVER}" == "true" ]]; then
  stripped="${API_BASE_URL#http://}"
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

  if ! curl --silent --fail --retry 40 --retry-connrefused --retry-delay 1 "${API_BASE_URL%/}/health" >/dev/null; then
    echo "[ERROR] Env server failed to become healthy at ${API_BASE_URL}"
    echo "[INFO] Last server log lines:"
    tail -n 50 "${server_log}" || true
    exit 1
  fi
else
  echo "[ERROR] Env server not reachable at ${API_BASE_URL} and AUTO_START_SERVER=false"
  exit 1
fi

echo "[INFO] Running tasks: ${TASKS[*]}"
tmp_log="$(mktemp)"

USE_OLLAMA=true \
OLLAMA_BASE_URL="${OLLAMA_BASE_URL}" \
AGENT_MODEL="${AGENT_MODEL}" \
MODEL_NAME="${MODEL_NAME}" \
HF_TOKEN="${HF_TOKEN}" \
USE_TOOLS="${USE_TOOLS}" \
FORCE_JSON_RESPONSE_FORMAT="${FORCE_JSON_RESPONSE_FORMAT}" \
API_BASE_URL="${API_BASE_URL}" \
ENV_URL="${API_BASE_URL}" \
"${PYTHON_BIN}" "${ROOT_DIR}/inference.py" "${TASKS[@]}" | tee "${tmp_log}"

echo
echo "=== Summary ==="
if ! grep -E "^\[END\]" "${tmp_log}"; then
  echo "[WARN] No [END] lines found in output."
fi

rm -f "${tmp_log}"
