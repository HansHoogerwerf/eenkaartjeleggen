# eenkaartjeleggen

A real-time, browser-based multiplayer **Klaverjassen** card game. Players create or join rooms, take a seat, and play with other humans or AI opponents. Empty seats fill with AI when the game starts.

Live at [eenkaartjeleggen.nl](https://eenkaartjeleggen.nl).

## Features

- **Four-seat Klaverjassen** with two teams (seats 0/2 vs seats 1/3).
- **Real-time multiplayer** over Socket.IO, with reconnect/snapshot support.
- **AI opponents** at three difficulty levels — Beginner, Advanced, Expert (neural).
- **Rules variants:** Rotterdam (default) and Amsterdam.
- **Game modes:** score limit, boom (16-round fixed), and free play.
- **Languages:** Dutch and English.
- **Progressive web app** — installable, offline-cached static assets, generated icons.

## Tech Stack

| Layer | Stack |
|---|---|
| Backend | Python 3.12, Flask 3.1, Flask-SocketIO, Gevent + gevent-websocket, Gunicorn |
| Frontend | Vanilla JavaScript, CSS, Socket.IO client (served locally) |
| AI | Heuristic engine + optional PyTorch neural card-play model |
| Infra | Docker / Docker Compose, Traefik reverse proxy, Let's Encrypt TLS |
| CI/CD | GitHub Actions (test + benchmark gate, SSH deploy) |

## Repository Layout

```
app.py                # Flask + Socket.IO server
main.py               # Game engine, AI, replay
config.py             # Frozen runtime config
klaverjas/            # Card/rules primitives (Card, Deck, Trick, roem)
server/               # Room state and game-flow bridge
neural/               # Neural feature encoding, model, inference wrapper
models/               # Tracked production models (neural_best.pt, neural_gpu.pt)
static/ templates/    # Frontend SPA + PWA assets
tools/                # Benchmarks, replay CLI, training pipelines
tests/                # Unit + Playwright + integration tests
docs/                 # Detailed architecture, AI strategy, deploy runbooks
```

For a full architecture walkthrough, see [docs/claude-analysis.md](docs/claude-analysis.md).

## Quick Start

### Local (Python)

```bash
pip install -r requirements.txt
cp .env.example .env        # set SECRET_KEY, CORS_ORIGINS
python app.py
```

Visit `http://localhost:5000`.

### Local (Docker Compose)

```bash
docker compose up --build
```

### Tests

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The CI quality gate runs `tools/ci_quality_gate.py` to detect AI regressions between the public `expert` and `advanced` profiles.

## Configuration

All runtime configuration is in [config.py](config.py); secrets and per-environment overrides come from environment variables.

| Variable | Purpose | Default |
|---|---|---|
| `SECRET_KEY` | Flask session secret — **set this in production** | `klaverjas-secret` (insecure fallback) |
| `CORS_ORIGINS` | Socket.IO CORS allow-list | `*` |
| `FLASK_DEBUG` | Enable Flask debug when `1` | `0` |
| `SEAT_RECONNECT_TIMEOUT_SECONDS` | Per-seat reconnect grace period | `60` |
| `ROOM_LOBBY_TTL_SECONDS` | Inactive lobby expiry | `3600` |
| `ROOM_STARTED_TTL_SECONDS` | Inactive started-room expiry | `21600` |
| `ACME_EMAIL` | Traefik / Let's Encrypt contact email | — |

See [.env.example](.env.example) for the minimum env file.

## AI

The public lobby exposes three difficulty levels. They map to the strongest profile generation, with neural play on Expert when a model is available.

| Public level | Behavior |
|---|---|
| Beginner | Heuristic expert: card inference, Monte Carlo trick-win probability, endgame minimax |
| Advanced | Beginner profile + 3-trick sampled lookahead |
| Expert | Neural card play (PyTorch `.pt` model), with heuristic fallback if `torch` or model is unavailable |

`torch` is **not** in `requirements.txt`, but the [Dockerfile](Dockerfile) installs the CPU build on top of it — so the production and acceptance Docker deploys run neural Expert. For a plain `pip install -r requirements.txt` (no Docker), Expert falls back to the heuristic profile unless you also install torch:

```bash
pip install torch
```

Details and tuning notes live in [docs/ai-strategy.md](docs/ai-strategy.md). The neural model design and feature encoding are described in [docs/neural-ai-plan.md](docs/neural-ai-plan.md).

## Deployment

Production runs behind Traefik with Let's Encrypt TLS, on a single Gunicorn worker (room state is in-process memory).

- [docs/VPS_AUTO_DEPLOY.md](docs/VPS_AUTO_DEPLOY.md) — GitHub Actions deploy workflow and required secrets.
- [docs/vps-setup-runbook.md](docs/vps-setup-runbook.md) — VPS bootstrap runbook.

Production deploys on push to `main` after CI passes; acceptance deploys on push to `develop`.

## Known Limitations

- **Single-worker only.** Room state is in-process; multi-worker deployments would split state. Use Redis or persistent storage to scale out.
- **Name-based reconnect.** A disconnected seat can be reclaimed by anyone with the room code and matching player name. Reconnect tokens are tracked work.
- **No rate limiting** on Socket.IO events.

See [docs/claude-analysis.md §11](docs/claude-analysis.md#11-security-and-reliability) for the full hardening backlog.

## License

TBD.
