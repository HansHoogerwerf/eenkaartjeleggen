# VPS Auto Deploy (GitHub Actions + SSH)

This repo includes:

- `scripts/deploy.sh`: Runs your Docker Compose deployment on the VPS.
- `.github/workflows/ci.yml` deploy job: Runs after tests pass on pushes to `main`.

## 1) One-time VPS setup

1. Install Docker + Docker Compose plugin on the VPS.
2. Clone this repo on the VPS (for example into `/opt/claude_code_test_project`).
3. Add and maintain your production `.env` on the VPS.
4. Ensure your VPS user can run Docker commands.
5. Test manually once:

```bash
cd /opt/claude_code_test_project
git checkout main
bash scripts/deploy.sh
```

## 2) GitHub repository secrets

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

## 3) Deploy behavior

On each push to `main`, CI runs tests. If they pass, deploy runs:

1. SSH to VPS
2. `git fetch` / `git pull --ff-only`
3. `bash scripts/deploy.sh`

By default `scripts/deploy.sh` runs:

- `docker compose pull` (best effort)
- `docker compose up -d --build --remove-orphans`

You can force your old `down && up` behavior by setting `FORCE_DOWN_UP=1` when running the script.
