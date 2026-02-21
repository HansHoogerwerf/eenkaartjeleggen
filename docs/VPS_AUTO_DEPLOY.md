# VPS Auto Deploy (GitHub Actions + SSH)

This repo includes:

- `scripts/deploy.sh`: Runs your Docker Compose deployment on the VPS.
- `.github/workflows/ci.yml` deploy job: Runs after tests pass on pushes to `main`.
- `.github/workflows/deploy-acceptance.yml`: Deploys the acceptance build on pushes to `develop` (or manual run), using acceptance-specific compose defaults.

The acceptance compose file (`docker-compose.acceptance.yml`) routes traffic to `acceptance.eenkaartjeleggen.nl`.

## 1) One-time VPS setup

1. Install Docker + Docker Compose plugin on the VPS.
2. Clone this repo on the VPS (for example into `/opt/claude_code_test_project`).
3. Add and maintain the correct `.env` for each environment on the VPS (production/acceptance).
4. Ensure your VPS user can run Docker commands.
5. Test manually once:

```bash
cd /opt/claude_code_test_project
git checkout main
bash scripts/deploy.sh
```

## 2) GitHub repository secrets

### Production (`ci.yml` deploy job)

Set these in `Settings -> Secrets and variables -> Actions`:

- `VPS_HOST` (required): VPS hostname or IP.
- `VPS_USER` (required): SSH user for deployments.
- `VPS_SSH_KEY` (required): Private key content used by GitHub Actions.
- `VPS_APP_DIR` (required): Absolute path to repo on VPS (for example `/opt/claude_code_test_project`).

Optional:

- `VPS_PORT`: SSH port (default `22`).
- `VPS_COMPOSE_FILE`: Compose file path relative to repo root (default `docker-compose.prod.yml`).
- `VPS_BRANCH`: Branch to deploy (default `main`).
- `VPS_KNOWN_HOSTS`: Full known_hosts entry; if omitted, workflow runs `ssh-keyscan`.

### Acceptance (`deploy-acceptance.yml`)

Set these in `Settings -> Secrets and variables -> Actions`:

- `ACCEPTANCE_VPS_HOST` (required): Acceptance VPS hostname or IP.
- `ACCEPTANCE_VPS_USER` (required): SSH user for acceptance deployments.
- `ACCEPTANCE_VPS_SSH_KEY` (required): Private key content used by GitHub Actions.
- `ACCEPTANCE_VPS_APP_DIR` (required): Absolute path to repo on acceptance VPS.

Optional:

- `ACCEPTANCE_VPS_PORT`: SSH port (default `22`).
- `ACCEPTANCE_VPS_COMPOSE_FILE`: Compose file path relative to repo root (default `docker-compose.acceptance.yml`).
- `ACCEPTANCE_VPS_BRANCH`: Branch to deploy (default `develop`).
- `ACCEPTANCE_VPS_KNOWN_HOSTS`: Full known_hosts entry; if omitted, workflow runs `ssh-keyscan`.

## 3) Deploy behavior

### Production

On each push to `main`, CI runs tests. If they pass, deploy runs:

1. SSH to VPS
2. `git fetch` / `git pull --ff-only`
3. `bash scripts/deploy.sh`

By default `scripts/deploy.sh` runs:

- `docker compose pull` (best effort)
- `docker compose up -d --build --remove-orphans`

You can force your old `down && up` behavior by setting `FORCE_DOWN_UP=1` when running the script.

### Acceptance

On each push to `develop` (or when manually triggered via `workflow_dispatch`), the acceptance deploy workflow:

1. Verifies the required `ACCEPTANCE_*` secrets are configured.
2. SSHes to the acceptance VPS.
3. Checks out/pulls the configured branch (default `develop`).
4. Runs `scripts/deploy.sh` with `COMPOSE_FILE=docker-compose.acceptance.yml` (unless overridden).
