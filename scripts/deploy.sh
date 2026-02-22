#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

CURRENT_BRANCH="$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
FORCE_DOWN_UP="${FORCE_DOWN_UP:-0}"
PRUNE_IMAGES="${PRUNE_IMAGES:-0}"
DEPLOY_PROJECT_NAME="${DEPLOY_PROJECT_NAME:-$(basename "${COMPOSE_FILE}" .yml)}"
TRAEFIK_PROJECT_NAME="${TRAEFIK_PROJECT_NAME:-traefik}"

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "Error: neither 'docker compose' nor 'docker-compose' is available on this host." >&2
  exit 1
fi

TRAEFIK_COMPOSE_CMD=("${COMPOSE_CMD[@]}" -p "${TRAEFIK_PROJECT_NAME}")
DEPLOY_COMPOSE_CMD=("${COMPOSE_CMD[@]}" -p "${DEPLOY_PROJECT_NAME}")

if [[ "${CURRENT_BRANCH}" == "main" ]]; then
  if ! "${TRAEFIK_COMPOSE_CMD[@]}" -f docker-compose.traefik.yml ps --services --filter "status=running" 2>/dev/null | grep -q traefik; then
    echo "Traefik is not running. Starting Traefik..."
    "${TRAEFIK_COMPOSE_CMD[@]}" -f docker-compose.traefik.yml up -d
  else
    echo "Traefik is already running."
  fi
fi

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "Error: compose file '${COMPOSE_FILE}' was not found in ${REPO_DIR}." >&2
  exit 1
fi

echo "Deploying with ${DEPLOY_COMPOSE_CMD[*]} -f ${COMPOSE_FILE}"

if [[ "${FORCE_DOWN_UP}" == "1" ]]; then
  "${DEPLOY_COMPOSE_CMD[@]}" -f "${COMPOSE_FILE}" down --remove-orphans
fi

"${DEPLOY_COMPOSE_CMD[@]}" -f "${COMPOSE_FILE}" pull || true
"${DEPLOY_COMPOSE_CMD[@]}" -f "${COMPOSE_FILE}" up -d --build --remove-orphans
"${DEPLOY_COMPOSE_CMD[@]}" -f "${COMPOSE_FILE}" ps

if [[ "${PRUNE_IMAGES}" == "1" ]]; then
  docker image prune -f
fi

echo "Deploy completed."
