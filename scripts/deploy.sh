#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
FORCE_DOWN_UP="${FORCE_DOWN_UP:-0}"
PRUNE_IMAGES="${PRUNE_IMAGES:-0}"

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "Error: neither 'docker compose' nor 'docker-compose' is available on this host." >&2
  exit 1
fi

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "Error: compose file '${COMPOSE_FILE}' was not found in ${REPO_DIR}." >&2
  exit 1
fi

echo "Deploying with ${COMPOSE_CMD[*]} -f ${COMPOSE_FILE}"

if [[ "${FORCE_DOWN_UP}" == "1" ]]; then
  "${COMPOSE_CMD[@]}" -f "${COMPOSE_FILE}" down --remove-orphans
fi

"${COMPOSE_CMD[@]}" -f "${COMPOSE_FILE}" pull || true
"${COMPOSE_CMD[@]}" -f "${COMPOSE_FILE}" up -d --build --remove-orphans
"${COMPOSE_CMD[@]}" -f "${COMPOSE_FILE}" ps

if [[ "${PRUNE_IMAGES}" == "1" ]]; then
  docker image prune -f
fi

echo "Deploy completed."
