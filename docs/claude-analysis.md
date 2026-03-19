# Codebase Analysis: Klaverjassen Web Game (eenkaartjeleggen)

**Repository:** https://github.com/HansHoogerwerf/eenkaartjeleggen
**Analysis Date:** 2026-02-23
**Current Branch:** develop
**Project Size:** ~3.0 MB

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Technology Stack](#3-technology-stack)
4. [Architecture](#4-architecture)
5. [Configuration](#5-configuration)
6. [Deployment & Scripts](#6-deployment--scripts)
7. [Game Engine Deep Dive](#7-game-engine-deep-dive)
8. [Docker & Containerization](#8-docker--containerization)
9. [CI/CD Pipelines](#9-cicd-pipelines)
10. [Dependencies](#10-dependencies)
11. [Testing & Code Quality](#11-testing--code-quality)
12. [Security](#12-security)
13. [Internationalization](#13-internationalization)
14. [Code Patterns & Conventions](#14-code-patterns--conventions)
15. [Deployment Configuration Summary](#15-deployment-configuration-summary)
16. [Issues & Areas for Improvement](#16-issues--areas-for-improvement)
17. [Summary & Recommendations](#17-summary--recommendations)

---

## 1. Project Overview

A multiplayer **Klaverjassen** card game (known locally as "eenkaartjeleggen") built as a real-time web application. Players can create/join game rooms, fill empty seats with AI opponents, and play against each other in-browser.

| Attribute | Value |
|-----------|-------|
| Type | Multiplayer card game web app |
| Primary Backend Language | Python 3.12 |
| Primary Frontend Language | Vanilla JavaScript |
| Real-time Protocol | WebSocket (Socket.IO) |
| Deployment Target | VPS with Docker + Traefik |
| Players per Room | Up to 4 (human or AI) |
| Supported Languages | Dutch (default), English |

---

## 2. Project Structure

```
/
├── app.py                          # Flask + Socket.IO server (~416 lines)
├── main.py                         # Game engine core logic (~1,660 lines)
├── config.py                       # Centralized config via frozen dataclasses (~74 lines)
├── requirements.txt                # Python runtime dependencies
├── requirements-dev.txt            # Dev dependencies (Playwright)
│
├── klaverjas/                      # Game rules & card logic module
│   ├── __init__.py
│   ├── constants.py                # Card ranks, suits, scoring rules (23 lines)
│   └── core.py                     # Card, Deck, Trick classes (98 lines)
│
├── server/                         # Web server helpers
│   ├── __init__.py
│   ├── room_state.py               # Room management, player seats (153 lines)
│   └── game_flow.py                # Game lifecycle, event handling (235 lines)
│
├── static/                         # Frontend assets
│   ├── game_app.js                 # Main app logic (1,004 lines)
│   ├── game_render.js              # Game board rendering (178 lines)
│   ├── game_state.js               # Client-side state (116 lines)
│   ├── i18n.js                     # Internationalization (378 lines)
│   └── style.css                   # Styling (~26 KB)
│
├── templates/
│   └── index.html                  # Main HTML page (~18 KB)
│
├── tests/                          # 15 test files
│   ├── test_app_integration.py     # Flask/Socket.IO integration tests
│   ├── test_ai_decisions.py        # AI logic tests
│   ├── test_ai_strength_levels.py  # AI difficulty levels
│   ├── test_ai_advanced_tactics.py # Advanced AI tactics
│   ├── test_ai_seat_logic.py       # Seat-specific AI logic
│   ├── test_ai_signaling.py        # Partner signal tests
│   ├── test_game_flow_unit.py      # Game flow tests
│   ├── test_room_state_unit.py     # Room management tests
│   ├── test_core_unit.py           # Card/Deck/Trick unit tests
│   ├── test_browser_integration.py # Playwright E2E browser tests
│   ├── test_benchmark_smoke.py     # Quick benchmark sanity check
│   ├── test_ci_quality_gate.py     # CI regression gate tests
│   ├── test_replay_determinism.py  # Seed-based replay determinism
│   ├── test_frontend_split.py      # Checks JS files are split correctly and key element IDs exist (not i18n)
│   └── test_refactor_structure.py  # Architecture/import tests
│
├── tools/                          # Dev & CI tools
│   ├── ai_benchmark.py             # AI performance benchmarking (181 lines)
│   ├── ci_quality_gate.py          # CI quality gate checks (105 lines)
│   ├── replay_cli.py               # Game replay debugging (135 lines)
│   └── benchmark_baseline.json     # Baseline metrics for regression
│
├── scripts/
│   └── deploy.sh                   # Docker Compose deployment script (~54 lines)
│
├── docs/
│   └── VPS_AUTO_DEPLOY.md          # Deployment documentation
│
├── .github/workflows/
│   ├── ci.yml                      # Main CI pipeline + production deploy
│   └── deploy-acceptance.yml       # Acceptance environment deploy
│
├── docker-compose.yml              # Local dev environment
├── docker-compose.prod.yml         # Production environment
├── docker-compose.acceptance.yml   # Acceptance/staging environment
├── docker-compose.traefik.yml      # Traefik reverse proxy config
├── Dockerfile                      # Container image definition
├── .dockerignore                   # Docker build exclusions
├── .gitignore                      # Git exclusions
├── .env.example                    # Environment variable template
└── .claude/settings.local.json     # Claude Code CLI permissions
```

---

## 3. Technology Stack

### Backend

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.12 |
| Web Framework | Flask | 3.1.2 |
| Real-time | Flask-SocketIO | Latest |
| WSGI Server | Gunicorn | Latest |
| Async Runtime | Gevent + gevent-websocket | Latest |
| Threading | Python threading | stdlib |

### Frontend

| Component | Technology |
|-----------|-----------|
| Language | Vanilla JavaScript (no framework) |
| Real-time Client | Socket.IO 4.7.5 client (loaded from cdnjs.cloudflare.com; no SRI) |
| Styling | CSS3 with CSS variables |
| Internationalization | Custom i18n implementation |
| E2E Testing | Playwright 1.52.0 |

### Infrastructure

| Component | Technology |
|-----------|-----------|
| Containers | Docker (python:3.12-slim base) |
| Orchestration | Docker Compose |
| Reverse Proxy | Traefik (latest) |
| SSL/TLS | Let's Encrypt (ACME) |
| CI/CD | GitHub Actions |

### Domains

| Environment | Domain |
|-------------|--------|
| Production | eenkaartjeleggen.nl |
| Acceptance | acceptance.eenkaartjeleggen.nl |

---

## 4. Architecture

### Backend: Layered Architecture with Real-time Event Streaming

```
┌──────────────────────────────────────────┐
│           Flask Web Layer (app.py)        │  HTTP + Socket.IO events
├──────────────────────────────────────────┤
│       Server Helpers (server/)            │  Room state, game flow bridge
├──────────────────────────────────────────┤
│       Game Engine (main.py)               │  Pure game logic in daemon thread
├──────────────────────────────────────────┤
│       Game Rules (klaverjas/)             │  Cards, Tricks, Constants
├──────────────────────────────────────────┤
│       Configuration (config.py)           │  Frozen dataclasses + env vars
└──────────────────────────────────────────┘
```

**1. Flask Web Layer (`app.py`)**
- HTTP route `/` serves `index.html`
- Socket.IO event handlers for all multiplayer actions
- Request validation and routing to server helpers

**2. Server Helpers Layer (`server/`)**
- `room_state.py`: Manages game rooms; lobby, active games, seat assignments
- `game_flow.py`: Bridges the game engine thread and Socket.IO event layer
- Handles reconnection logic and broadcasting state to clients

**3. Game Engine Layer (`main.py`)**
- Pure game logic (~1,660 lines), no web dependencies
- `Player` base class with `HumanPlayer` and `AIPlayer` subclasses
- `KlaverjasGame` orchestrates rounds, bidding, and trick play
- Runs in a separate daemon thread per room

**4. Configuration Layer (`config.py`)**
- Centralized frozen dataclasses (`ServerConfig`, `RoomConfig`)
- Environment variable overrides at instantiation time
- Immutable once created

**5. Game Rules Module (`klaverjas/`)**
- `Card`, `Deck`, `Trick` implementations
- Trick winner calculation
- Roem (honour card combination) scoring
- Rule variant definitions

### Frontend: Event-Driven State Machine

```
┌────────────────────────────────────────────┐
│   game_app.js (1,004 lines)                 │  Socket.IO I/O, user input, notifications
├────────────────────────────────────────────┤
│   game_state.js (116 lines)                 │  Client-side state cache
├────────────────────────────────────────────┤
│   game_render.js (178 lines)                │  Card table visualization
├────────────────────────────────────────────┤
│   i18n.js (378 lines)                       │  Translation key system
└────────────────────────────────────────────┘
```

### Multiplayer Architecture

- **Room-based Lobbies:** Up to 4 players per room (identified by 4-char alphanumeric code)
- **AI Fill:** Empty seats can be filled with AI players at configurable strength
- **Host Migration:** If the room creator disconnects, host role passes to another player
- **Disconnect/Reconnect:** Players can temporarily disconnect; game pauses and resumes on reconnect (120s timeout)
- **Room Lifecycle:**
  - Lobby phase TTL: 3600s
  - Game phase TTL: 21600s
  - Automatic expiration and cleanup

### Thread Model

```
Main thread       → Flask HTTP + Socket.IO event handling (Gevent greenlets)
Game thread       → One daemon thread per active room (blocks on human input via threading.Event)
```

This hybrid model (gevent for the web layer, OS threads for game execution) is pragmatic and works, but it increases concurrency complexity. Shared mutable state (`sid_to_room`, `rooms`, room snapshots) is accessed from both gevent handlers and game threads without explicit synchronisation. Under normal load this is safe; under reconnect/disconnect churn race conditions can surface.

**Note:** Production Gunicorn is configured with `-w 1` (single worker). This is correct and intentional — with in-memory room state and process-local WebSocket connections, multiple workers would break state consistency.

---

## 5. Configuration

### `config.py` — Frozen Dataclasses

```python
ServerConfig:
  secret_key        # Flask session encryption (default: "klaverjas-secret"; override via SECRET_KEY)
  cors_origins      # CORS allowed origins (env: CORS_ORIGINS)
  ping_timeout      # 30s
  ping_interval     # 10s
  host              # 0.0.0.0
  port              # 5000
  debug             # Controlled by FLASK_DEBUG env var

RoomConfig:
  default_player_name       # "Player"
  max_player_name_len       # 16 characters
  max_chat_message_len      # 200 characters
  seat_count                # 4
  room_code_length          # 4 chars
  allowed_game_modes        # ["score_limit", "boom", "free_play"]
  default_game_mode         # "score_limit"
  allowed_ai_strengths      # ["beginner", "advanced", "expert"]
  default_ai_strength       # "expert"
  allowed_rules_variants    # ["rotterdam", "amsterdam"]
  default_rules_variant     # "rotterdam"
  score_limit_range         # 50–5000 (default: 500)
  max_team_name_len         # 16 characters
```

### Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `SECRET_KEY` | Flask session encryption | Yes (production) |
| `CORS_ORIGINS` | Allowed CORS origins | Optional |
| `FLASK_DEBUG` | Enable debug mode | Dev only |
| `ACME_EMAIL` | Let's Encrypt contact email | VPS deployment |

---

## 6. Deployment & Scripts

### `scripts/deploy.sh`

A 54-line bash script orchestrating Docker Compose deployments:

- Detects Docker Compose plugin vs legacy `docker-compose` command
- Isolates Traefik and app as separate Compose projects
- Ensures Traefik is running before starting the app
- Pulls latest images (best effort, non-fatal)
- Builds and starts services with `--remove-orphans`
- Optional flags via environment variables:

| Env Var | Default | Effect |
|---------|---------|--------|
| `COMPOSE_FILE` | `docker-compose.prod.yml` | Override compose file |
| `FORCE_DOWN_UP` | `0` | Force `down` then `up` |
| `PRUNE_IMAGES` | `0` | Prune unused Docker images after deploy |
| `TRAEFIK_PROJECT_NAME` | `traefik` | Compose project name for Traefik |

---

## 7. Game Engine Deep Dive

### Core Classes (`main.py`)

#### `Player` (Abstract Base Class)

| Property | Description |
|----------|-------------|
| `name` | Player's display name |
| `team` | 0 or 1 (two teams: seats 0+2 vs 1+3) |
| `seat_idx` | 0–3 (South, West, North, East) |
| `hand` | Current cards in hand |
| `played_cards` | Cards seen played this round |
| `rules_variant` | `"rotterdam"` or `"amsterdam"` |

Key methods:
- `legal_moves(trick, trump)` — Returns playable cards per rules variant
- `choose_card(trick, trump)` — Abstract; implemented by subclasses
- `choose_trump(suit, forced)` — Abstract; for bidding phase
- `receive_hand(cards)` — Deals cards to player
- `observe_card(card)` / `observe_trick_play(...)` — State tracking for inference

#### Rules Variants

| Rule | Rotterdam (Default) | Amsterdam |
|------|-------------------|-----------|
| Trump requirement | Must trump if partner is losing | Can follow any card if partner winning |
| Overtrump | Must overtrump if possible | Softer requirement |
| Nat (pit) | 0 points if < 81 trick points | Same |

#### `HumanPlayer(Player)`

- Uses `threading.Event` to pause the game thread waiting for a human's card/bid choice
- Supports disconnect/reconnect:
  - `set_disconnected()` — Pauses game
  - `set_reconnected()` — Resumes and re-fires pending request
  - Disconnect timeout: 120s (configurable)
- Callbacks injected from Flask layer:
  - `_on_move_request(seat_idx, legal_cards)`
  - `_on_bid_request(seat_idx, suit, forced)`
  - `_on_disconnect_pause(seat_idx)`

#### `AIPlayer(Player)`

Three strength profiles with tuned parameters:

| Parameter | Beginner | Advanced | Expert |
|-----------|----------|----------|--------|
| Inference | No | Yes | Yes |
| Trick Probability | No | Yes | Yes |
| Endgame Solver | No | No | Yes |
| Tie Break Delta | 0.95 | 0.55 | 0.35 |
| Mistake Rate | 10% | 3% | 0% |
| Declaration Bias | 0.65 | 0.25 | 0.0 |
| Trick Sim Samples | 4 | 12 | 20 |

Decision making:
- `_declaration_score(suit)` — Evaluates hand strength for bidding (weights: Trump J=1.85, Trump 9=1.30, voids=0.35, etc.)
- `_strategy(legal, trick, trump)` — Card play heuristics
- `_score_pressure()` — Context-aware aggression/caution modifier
- Partner signaling — Infers information from partner's card choices
- Void tracking — Infers which suits opponents cannot play

#### `KlaverjasGame`

Orchestrates the full game loop:

1. **Deal** — Distribute 8 cards × 4 players
2. **Bidding** — Each player is offered the current suit; can pass or declare
3. **Trick Play** — 8 tricks, 4 cards each
4. **Scoring** — Calculate trick points + roem (honour combinations) + nat check

**Game Modes:**

| Mode | Description |
|------|-------------|
| `score_limit` | First team to reach score_limit (default 500) wins |
| `boom` | Play fixed 16 rounds |
| `free_play` | Continuous play, no win condition |

---

## 8. Docker & Containerization

### Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

Optimizations:
- Slim base image for reduced footprint
- Dependencies installed before app code (maximizes layer cache reuse)
- `--no-cache-dir` reduces image size

### `.dockerignore`

Excludes: `__pycache__`, `*.pyc`, `.git`, `.env`, `.env.*` (but not `.env.example`), Docker config files

### Docker Compose Files

| File | Environment | Key Difference |
|------|-------------|----------------|
| `docker-compose.yml` | Local dev | Source volume mount, `FLASK_DEBUG=1`, port 5000 mapped |
| `docker-compose.prod.yml` | Production | Gunicorn+Gevent, Traefik integration, `restart: unless-stopped` |
| `docker-compose.acceptance.yml` | Acceptance | Same as prod, port 5050, different domain |
| `docker-compose.traefik.yml` | Shared VPS | Traefik with ACME, HTTP→HTTPS redirect |

---

## 9. CI/CD Pipelines

### `.github/workflows/ci.yml` (Main Pipeline)

**Triggers:** Push to `main`, pull requests

**Test Job:**
1. Set up Python 3.11
2. Install dependencies
3. Run unit tests: `python -m unittest discover`
4. Run AI benchmark quality gate (CI invokes tool with 256 rounds and seed 42; the tool's own default is 10,000 rounds)

**Quality Gate Thresholds (Expert vs Advanced AI):**

| Metric | Max Allowed Regression |
|--------|----------------------|
| Win rate drop | 3% |
| Avg point diff drop | 20 points |
| Declare success drop | 6% |

**Deploy Job (main branch only, after tests pass):**
1. Set up SSH with key from GitHub Secrets
2. SSH to VPS
3. `git pull --ff-only origin/main`
4. Run `scripts/deploy.sh`

**Required Secrets:**

| Secret | Required? |
|--------|----------|
| `VPS_HOST` | Yes |
| `VPS_USER` | Yes |
| `VPS_SSH_KEY` | Yes |
| `VPS_APP_DIR` | Yes |
| `VPS_PORT` | Optional (default 22) |
| `VPS_COMPOSE_FILE` | Optional |
| `VPS_KNOWN_HOSTS` | Optional |

### `.github/workflows/deploy-acceptance.yml`

**Triggers:** Push to `develop`, manual `workflow_dispatch`

- Same pipeline structure; uses `ACCEPTANCE_*` secret prefix
- Defaults to `docker-compose.acceptance.yml`
- Deploys to `acceptance.eenkaartjeleggen.nl`

> **Gap:** This workflow is deploy-only — it does not run the unit test suite or AI benchmark quality gate before deploying. A failing or regressed `develop` push can reach the acceptance environment without being caught. The `ci.yml` test job does not cover `develop` pushes; only PRs and `main` pushes are tested automatically.

---

## 10. Dependencies

### Python (`requirements.txt`)

| Package | Purpose |
|---------|---------|
| `flask==3.1.2` | Web framework |
| `flask-socketio` | Real-time Socket.IO events |
| `gevent` | Async greenlet runtime |
| `gevent-websocket` | WebSocket support for Gevent |
| `gunicorn` | Production WSGI server |

### Python Dev (`requirements-dev.txt`)

| Package | Purpose |
|---------|---------|
| `playwright==1.52.0` | Browser automation for E2E tests |

### JavaScript

- Socket.IO 4.7.5 client loaded from cdnjs.cloudflare.com (`index.html:301`); no SRI integrity attribute
- No JavaScript package manager used

### Docker Images

| Image | Use |
|-------|-----|
| `python:3.12-slim` | Application container |
| `traefik:latest` | Reverse proxy |

---

## 11. Testing & Code Quality

### Test Suite (15 files)

| Test File | Focus |
|-----------|-------|
| `test_app_integration.py` | Room creation, joining, chat, game start via Socket.IO |
| `test_game_flow_unit.py` | Game phase transitions and state |
| `test_room_state_unit.py` | Seats, host migration, TTL expiry |
| `test_ai_decisions.py` | Bidding, card choices, strategy |
| `test_ai_strength_levels.py` | Strength profile parameter effects |
| `test_ai_advanced_tactics.py` | Signaling, inference, endgame solver |
| `test_ai_seat_logic.py` | Position-aware AI logic |
| `test_ai_signaling.py` | Partner signal encoding/decoding |
| `test_core_unit.py` | Card, Deck, Trick winner calculation |
| `test_browser_integration.py` | Full game flow with Playwright |
| `test_benchmark_smoke.py` | Quick benchmark sanity check |
| `test_ci_quality_gate.py` | CI regression baseline loading |
| `test_replay_determinism.py` | Seed-based reproducibility |
| `test_frontend_split.py` | Checks that the frontend JS is split into the three expected files and that key element IDs (e.g. `lobby-ai-picker`) are present; does not test i18n/translations |
| `test_refactor_structure.py` | Module imports and architecture |

### Quality Gate Tools (`tools/`)

**`ai_benchmark.py`**
- Runs configurable rounds (CLI default: 10,000; CI workflow invocation uses 256 rounds with seed 42) of AI vs AI
- Tracks: wins, total points, declarations, nats
- Compares expert vs advanced AI

**`ci_quality_gate.py`**
- Loads baseline from `tools/benchmark_baseline.json`
- Runs benchmark and compares vs baseline
- Fails CI (exit code 1) if regression exceeds any threshold

**`replay_cli.py`**
- CLI tool to replay a game with the same seed for determinism debugging

---

## 12. Security

### Environment Variables & Secrets

| Secret | Storage |
|--------|---------|
| `SECRET_KEY` | `.env` (git-ignored) + GitHub Secrets |
| `ACME_EMAIL` | `.env` + VPS env file |
| `VPS_SSH_KEY` | GitHub Secrets (encrypted) |

### Input Validation

**Server-side (partial and inconsistent):**
The following fields do have server-side guards: player names (trimmed, max 16 chars), chat messages (trimmed, max 200 chars), room codes (uppercase, 4 chars), game mode (allow-list), AI strength (allow-list), score limit (range 50–5000). However, many Socket.IO handlers use raw direct indexing (`data["card"]`) and unguarded type casts (`int(requested_seat)`) without catching exceptions. Input validation should be considered mixed quality, not a broadly satisfied property.

**Client-side:**
- HTML `maxlength` attributes
- JavaScript cleanup before Socket.IO emit

### Network Security

- HTTPS enforced via Traefik (HTTP → HTTPS redirect)
- Let's Encrypt certificates (auto-renewed)
- WebSocket upgrades to WSS in production
- App exposed only via Traefik on internal Docker network

### Security Gaps

| Issue | Risk | Suggested Fix |
|-------|------|--------------|
| Partial XSS risk for player names in lobby | Chat messages and sender names ARE escaped via `escapeHtml()` (`game_app.js:788`, `game_app.js:839`). However, player names rendered into the seat picker and lobby seat lists use `innerHTML` without escaping (`game_app.js:105`, `game_app.js:243`) | Wrap `seat.name` in `escapeHtml()` at those two locations |
| No rate limiting on Socket.IO events | DoS via rapid event emission | Add per-socket rate limiter middleware |
| No player authentication | Trolling, room hijacking | Accept for public game; optionally add optional nicknames with abuse reporting |
| Room code collision (36^4 = 1,679,616 combinations; codes are A-Z0-9) | `generate_code()` retries for uniqueness within a process, so in-process collisions are prevented. Risk is guessability/enumeration by outsiders and eventual search-space saturation at high concurrent room counts | Increase `room_code_length` from 4 to 6 if load grows; consider rate-limiting room join attempts |
| **Reconnect identity based on display name** | A third party with the room code can hijack a disconnected seat by joining with the same display name | Issue a per-seat secret token on join; require it for reconnect |
| **Default `SECRET_KEY` is a static placeholder** (`config.py:48`) | Sessions are predictable if the env var is not set in production | Fail fast on startup when `SECRET_KEY` equals the default in non-debug mode |
| **Default CORS origin is `*`** (`config.py:49`) | Any origin can make credentialed requests if the env var is not overridden | Use a restrictive default CORS origin in non-debug mode |
| **Socket.IO CDN loaded without Subresource Integrity (SRI)** | CDN compromise could inject malicious JavaScript | Add `integrity` and `crossorigin` attributes to the `<script>` tag |

---

## 13. Internationalization

**Supported Languages:** Dutch (nl, default), English (en)

**Implementation (`static/i18n.js`):**
- Key-based translation dictionary embedded in JavaScript
- Language switcher UI (flag buttons)
- Persists selected language to `localStorage`
- Translates: text content and placeholder attributes (`applyStaticTranslations()`)

**Coverage:** 200+ translation keys covering lobby UI, game UI, bidding, scoring, errors, and notifications. `applyStaticTranslations()` updates `textContent` and `placeholder` attributes, but there is no generic pass for `aria-label` attributes — many `aria-label` values remain hardcoded in `templates/index.html` and are not translated.

---

## 14. Code Patterns & Conventions

### Naming Conventions

| Context | Convention |
|---------|-----------|
| Python variables/functions | `snake_case` |
| Python constants | `UPPER_CASE` |
| JavaScript functions | `camelCase` |
| CSS variables | `--kebab-case` |

### Python Style

- Modern Python 3.11+ features: `from __future__ import annotations`, `str | None` union types
- Frozen `@dataclass` for immutable config objects
- Type hints on function signatures throughout
- Module-level docstrings for rules explanations
- Method-level docstrings sparse in AI logic (noted as improvement area)

### Git Workflow

- Branch naming: `codex/*`, `feat-*`, `chore-*`
- PR-based development with squash/merge
- Two environments: `develop` → acceptance, `main` → production
- Recent commits show Traefik isolation refactor and deploy script simplification

---

## 15. Deployment Configuration Summary

| Aspect | Dev | Acceptance | Production |
|--------|-----|-----------|-----------|
| Compose File | `docker-compose.yml` | `docker-compose.acceptance.yml` | `docker-compose.prod.yml` |
| Port | 5000 (host mapped) | 5050 (exposed) | 5000 (internal only) |
| Domain | `localhost` | `acceptance.eenkaartjeleggen.nl` | `eenkaartjeleggen.nl` |
| WSGI Server | Flask dev server | Gunicorn + GeventWebSocketWorker | Gunicorn + GeventWebSocketWorker |
| Debug Mode | Yes (`FLASK_DEBUG=1`) | No | No |
| Source Volume | Yes | No | No |
| Restart Policy | None | `unless-stopped` | `unless-stopped` |
| Reverse Proxy | None | Traefik | Traefik |
| SSL | None | Let's Encrypt | Let's Encrypt |
| Docker Network | Default bridge | `traefik-public` (external) | `traefik-public` (external) |

---

## 16. Issues & Areas for Improvement

### Specific Bugs (from code-level analysis)

1. **Stale SID mappings accumulate on reconnect** (`server/game_flow.py:28-33`, `app.py:340-355`) — `reconnect_player()` adds `sid_to_room[new_sid]` but does not remove the old SID mapping. For started games, `handle_disconnect()` intentionally retains the old SID entry. Reconnect cycles therefore accumulate stale entries in the global `sid_to_room` dict, which can cause unexpected lookup behaviour and memory growth over time. Fix: remove `old_sid` from `sid_to_room` on successful reconnect.

2. **`handle_start_game()` has a server crash path** (`app.py:143-174`) — `team_names` is normalised from `names` inside the `isinstance(names, list)` guard, but a second reference to `enumerate(names)` exists outside that guard (inside the `reconnect_timeout_seconds` validation block). If a client sends `reconnect_timeout_seconds` but omits or invalidates `team_names`, `enumerate(names)` raises an exception. The standard browser client always sends valid `team_names`, so this is invisible in normal use but is exploitable with a custom client.

3. **Reconnect identity is based on player name matching** (`app.py:75-96`, `static/game_state.js:42`) — The server matches a reconnecting client to a disconnected seat using exact string equality (`info["name"] == name`) on the server-normalised display name (trimmed and truncated before comparison, no case normalisation). The client stores only `{code, name}` in localStorage. Anyone who knows the room code can claim a disconnected player's seat by joining with the same display name. Fix: issue a per-seat reconnect token on join and store `{code, name, reconnect_token}` client-side.

4. **Socket.IO payload validation is inconsistent** (`app.py:105`, `app.py:201`, `app.py:219`) — Some handlers perform raw index access (`data["card"]`) or unsafe coercions (`int(requested_seat)`) without guarding against missing keys or wrong types. Malformed payloads raise unhandled exceptions. Fix: validate all incoming event payloads defensively; use `.get(...)` and type checks; return socket error events instead of raising.

### Security

5. **Partial XSS in lobby player names** — Chat messages and sender names are correctly escaped via `escapeHtml()` before insertion. However, player names rendered into the seat picker (`game_app.js:105`) and lobby seat list (`game_app.js:243`) are interpolated into `innerHTML` without escaping. A player who joins with a name containing `<script>` or `<img onerror=...>` can trigger execution in other clients' browsers. Fix: wrap `seat.name` in `escapeHtml()` at those two call sites.
6. **No rate limiting** — Rapid Socket.IO event emission could abuse the server. Add per-socket throttling.
7. **Room code guessability** — Codes are 4 characters from A-Z0-9 (36^4 = 1,679,616 combinations). `generate_code()` retries until a unique code is found within the process, so in-process collisions are not possible. The real risk is external enumeration: an attacker can probe room codes to discover active games. Consider increasing `room_code_length` to 6 and rate-limiting `peek_room`/`join_room` events.

### Reliability

4. **In-memory state only** — VPS restart loses all active games. Acceptable for current scale; would need a Redis/database layer for production resilience.
5. **Single VPS** — Single point of failure with no load balancing. Acceptable for hobby project; consider failover for production.

### Code Quality

8. **Undocumented AI inference heuristics** — The AI's card inference and endgame solver logic is complex but lacks explaining comments/docstrings. Makes future maintenance harder.
9. **Sparse method docstrings** — Particularly in `main.py` AI logic and `server/game_flow.py`.
10. **`main.py` is a multi-concern monolith** — Domain model, AI heuristics, endgame solver, game loop, replay generation, and UI pacing all live in one 1,659-line file. Suggested decomposition: `engine/game.py`, `engine/players.py`, `ai/strategy.py`, `ai/inference.py`, `replay/model.py`.
11. **UI pacing delays are global mutable constants patched by tooling** (`klaverjas/constants.py:17-19`) — `tools/ai_benchmark.py` and `tools/replay_cli.py` zero out these constants at module level to disable delays during automation. This couples tool behaviour to global mutable state. Fix: pass pacing configuration into the `KlaverjasGame` constructor so each instance controls its own delays.
12. **Import-time background task startup** (`app.py:41`) — The room cleanup task is started at module import time, creating a hidden side effect. This complicates test isolation and could spawn duplicate loops in alternative deployment patterns. Fix: move task startup into an explicit initialisation hook.
13. **Minor unused code** — `CURRENT_BRANCH` computed but unused (`scripts/deploy.sh:8`); `total_points` computed but unused (`main.py:1591`).
14. **Inline `onclick` handlers in `index.html`** — The template uses inline `onclick` attributes throughout, preventing a Content Security Policy that disallows `unsafe-inline`. Refactoring to `addEventListener` calls in `game_app.js` would enable a stricter CSP.

### Testing

8. **Partial E2E test coverage** — A reconnect-on-reload flow is already covered by `test_reload_auto_reconnect_restores_active_game` in `test_browser_integration.py`. Remaining gaps: full multiplayer game flow (4 humans), host migration E2E, and disconnect-timeout expiry scenarios.
9. **No load testing** — Concurrent room capacity is unknown and untested. Do not make concurrency claims without a benchmark; add load testing before drawing conclusions about scale limits.

### Scalability

10. **Game thread blocks on human input** — Acceptable for 4 players; scales poorly to hundreds of concurrent games. Would need async game engine for high concurrency.
11. **Socket.IO transport configuration not explicitly set** — `game_state.js` initialises Socket.IO without specifying `transports`. This means Socket.IO uses its default behaviour (WebSocket first, falls back to long-polling). This is generally fine and makes the app work on restricted networks. If a WebSocket-only policy is desired for performance reasons, explicitly set `transports: ['websocket']`; otherwise this is acceptable as-is.

### Dependencies

12. **Loose version pinning** — `flask-socketio`, `gevent`, `gunicorn` have no pinned versions in `requirements.txt`. Risk of breaking changes on fresh installs.
13. **No lock file** — No `pip-tools` or `Poetry` lock file. Recommend adding `requirements.lock` for reproducible builds.

### Documentation

14. **No Socket.IO event schema** — Events are hardcoded in `app.py` and `game_app.js` with no central documentation. A simple event catalog (event name, payload shape, direction) would help maintainability.

---

## 17. Summary & Recommendations

### Strengths

- Clean layer separation: game engine, web layer, frontend are independently testable
- Comprehensive test suite: unit, integration, E2E, and AI regression benchmarks
- Robust CI/CD: automated deployment with quality gates and staging environment
- Solid multiplayer reliability: disconnect/reconnect, host migration, AI fill
- Flexible AI: three difficulty levels with tunable parameters and regression protection
- Internationalization: Dutch + English with 200+ translation keys
- Modern Python: type hints, frozen dataclasses, slim dependencies

### Phased Improvement Roadmap

**Phase 1 — High value, low risk (fix now)**

| Task | Notes |
|------|-------|
| Fix reconnect SID cleanup bug | Remove old SID from `sid_to_room` on reconnect |
| Fix `handle_start_game()` crash path | Normalise `team_names` once; guard before all references |
| Harden Socket.IO payload validation | Use `.get()`, type guards, error events in all handlers |
| Fix unescaped player names in lobby seat picker and seat list (XSS) | Wrap `seat.name` in `escapeHtml()` at `game_app.js:105` and `game_app.js:243`; chat content is already escaped |
| Add regression tests for malformed payloads | Prevents future regressions on validation fixes |
| Pin all package versions in `requirements.txt` | ~30 min; prevents breaking-change surprises |
| Remove unused `CURRENT_BRANCH` and `total_points` | Minor noise reduction |

**Phase 2 — Security / reliability**

| Task | Notes |
|------|-------|
| Replace name-based reconnect with server-issued tokens | Store `reconnect_token` in localStorage |
| Add per-socket rate limiting / abuse guards | Covers room probing, chat, rapid card play |
| Fail-fast for default `SECRET_KEY` and `*` CORS in production | Add startup assertion in `app.py` |
| Add SRI hash to Socket.IO CDN `<script>` tag | Prevents CDN-level supply chain attack |
| Add tests to `ci.yml` for `develop` push / before acceptance deploy | Prevents unverified code reaching acceptance |

**Phase 3 — Maintainability / scale**

| Task | Notes |
|------|-------|
| Decompose `main.py` into `engine/`, `ai/`, `replay/` modules | Reduces merge conflict blast radius |
| Move pacing delays into `KlaverjasGame` constructor options | Decouples tooling from global state |
| Move background task startup into explicit init hook | Cleans up import-time side effects |
| Introduce per-room concurrency locks | Guards `sid_to_room` and seat mutations |
| Expand E2E test suite (host migration, 4-player flow, disconnect timeout) | 2–3 days; reconnect-on-reload already covered |
| Add `requirements.lock` via pip-tools | Reproducible builds |
| Add Redis for game state persistence (future) | Only needed if multi-instance or restart resilience required |

### Key Files Reference

Line counts are approximate snapshots from initial analysis and may drift as the codebase evolves.

| File | Lines (approx.) | Role |
|------|-----------------|------|
| [main.py](main.py) | ~1,660 | Game engine — core game logic |
| [app.py](app.py) | ~416 | Flask/Socket.IO server |
| [static/game_app.js](static/game_app.js) | ~1,004 | Frontend game logic |
| [config.py](config.py) | ~74 | Centralized configuration |
| [server/game_flow.py](server/game_flow.py) | ~235 | Game lifecycle bridge |
| [server/room_state.py](server/room_state.py) | ~153 | Room management |
| [scripts/deploy.sh](scripts/deploy.sh) | ~54 | Deployment script |
| [tools/ai_benchmark.py](tools/ai_benchmark.py) | ~181 | AI benchmarking |
| [static/i18n.js](static/i18n.js) | ~378 | Internationalization |

---

*Analysis performed by Claude Sonnet 4.6 on 2026-02-23.*
