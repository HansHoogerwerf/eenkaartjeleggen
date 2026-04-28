# Codebase Analysis: Klaverjassen Web Game (eenkaartjeleggen)

**Repository:** https://github.com/HansHoogerwerf/eenkaartjeleggen  
**Snapshot Date:** 2026-04-22  
**Current Branch:** develop  
**Tracked Project Size:** ~7.9 MB, including the two tracked production neural model artifacts

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Technology Stack](#3-technology-stack)
4. [Architecture](#4-architecture)
5. [Configuration](#5-configuration)
6. [Game Engine Deep Dive](#6-game-engine-deep-dive)
7. [Frontend and PWA](#7-frontend-and-pwa)
8. [Neural AI and Training Tooling](#8-neural-ai-and-training-tooling)
9. [Deployment and CI/CD](#9-deployment-and-cicd)
10. [Testing and Verification](#10-testing-and-verification)
11. [Security and Reliability](#11-security-and-reliability)
12. [Current Issues and Recommendations](#12-current-issues-and-recommendations)
13. [Key Files Reference](#13-key-files-reference)

---

## 1. Project Overview

This repository contains a real-time multiplayer **Klaverjassen** card game, branded as "eenkaartjeleggen". Players can create rooms, join seats, play with other humans, and fill empty seats with AI players. The app is browser-based, uses Socket.IO for real-time state, and keeps room/game state in memory.

| Attribute | Current Value |
|---|---|
| Type | Multiplayer Klaverjassen web app |
| Backend | Python + Flask + Flask-SocketIO |
| Frontend | Vanilla JavaScript, HTML, CSS |
| Real-time protocol | Socket.IO |
| Player seats | 4 total seats: South, West, North, East |
| Teams | Team 0 = seats 0 and 2; Team 1 = seats 1 and 3 |
| AI public options | Beginner, Advanced, Expert |
| AI public mapping | Beginner = previous Expert; Advanced = previous Expert v2; Expert = Neural-backed profile |
| Rules variants | Rotterdam, Amsterdam |
| Game modes | Score limit, Boom, Free play |
| Supported languages | Dutch and English |
| Deployment target | VPS with Docker Compose, Gunicorn, Gevent, Traefik |
| PWA support | Manifest, service worker, generated icons |

The core game is still a four-player Klaverjassen engine. The recent AI change is about **AI strength option labels**, not seat count.

---

## 2. Project Structure

Current tracked structure, grouped by purpose:

```text
/
├── app.py                          # Flask + Socket.IO server (~604 lines)
├── main.py                         # Game engine, AI logic, replay support (~2,255 lines)
├── config.py                       # Frozen runtime config dataclasses (~80 lines)
├── requirements.txt                # Runtime dependencies
├── requirements-dev.txt            # Dev/test dependencies
├── AGENTS.md                       # Local agent/project instructions
│
├── klaverjas/                      # Pure card/rules primitives
│   ├── __init__.py
│   ├── constants.py                # Suits, ranks, scores, seat/team maps
│   └── core.py                     # Card, Deck, Trick, roem helpers
│
├── server/                         # Room and game-flow bridge
│   ├── __init__.py
│   ├── room_state.py               # Room seats, host migration, TTL cleanup (~275 lines)
│   └── game_flow.py                # Game startup, reconnect snapshots, event routing (~441 lines)
│
├── neural/                         # Neural-card-play feature/model/inference code
│   ├── __init__.py
│   ├── bid_features.py
│   ├── features.py                 # State encoder for card-play model
│   ├── model.py                    # PyTorch model definitions
│   └── player.py                   # Neural inference wrapper with heuristic fallback
│
├── models/
│   ├── .gitignore                  # Ignores training data/intermediate models
│   ├── neural_best.pt              # Tracked production neural model
│   └── neural_gpu.pt               # Tracked GPU-trained model artifact
│
├── static/
│   ├── game_app.js                 # Socket.IO I/O, lobby, interactions (~1,346 lines)
│   ├── game_render.js              # Board/card rendering (~225 lines)
│   ├── game_state.js               # Client state and seat rotation (~211 lines)
│   ├── i18n.js                     # Dutch/English translations (~402 lines)
│   ├── socket.io.min.js            # Local Socket.IO client asset
│   ├── style.css                   # Main responsive styling
│   ├── manifest.webmanifest        # PWA manifest
│   ├── sw.js                       # Service worker
│   └── icons/                      # Generated app icons and generator script
│
├── templates/
│   └── index.html                  # Single-page app template (~334 lines)
│
├── tests/                          # 15 unittest/Playwright-oriented test files
│   ├── test_ai_advanced_tactics.py
│   ├── test_ai_decisions.py
│   ├── test_ai_seat_logic.py
│   ├── test_ai_signaling.py
│   ├── test_ai_strength_levels.py
│   ├── test_app_integration.py
│   ├── test_benchmark_smoke.py
│   ├── test_browser_integration.py
│   ├── test_ci_quality_gate.py
│   ├── test_core_unit.py
│   ├── test_frontend_split.py
│   ├── test_game_flow_unit.py
│   ├── test_refactor_structure.py
│   ├── test_replay_determinism.py
│   └── test_room_state_unit.py
│
├── tools/                          # Benchmarks, replay, data generation, neural/RL training
│   ├── ai_benchmark.py
│   ├── benchmark_lookahead.py
│   ├── ci_quality_gate.py
│   ├── generate_bid_data.py
│   ├── generate_training_data.py
│   ├── gpu_engine.py
│   ├── replay_cli.py
│   ├── reward_rules.py
│   ├── run_full_pipeline.py
│   ├── train_bid_neural.py
│   ├── train_bid_pipeline.py
│   ├── train_bid_rl.py
│   ├── train_gpu_selfplay.py
│   ├── train_neural.py
│   ├── train_ppo.py
│   ├── train_rl.py
│   ├── train_selfplay.py
│   ├── train_shaped.py
│   └── tune_bid_threshold.py
│
├── docs/
│   ├── ai-strategy.md
│   ├── claude-analysis.md
│   ├── neural-ai-plan.md
│   ├── VPS_AUTO_DEPLOY.md
│   └── vps-setup-runbook.md
│
├── plans/
│   ├── reconnect-ui-fix.md
│   └── reconnect-ui-fix-implementation.md
│
├── scripts/
│   └── deploy.sh
│
├── .github/workflows/
│   ├── ci.yml
│   └── deploy-acceptance.yml
│
├── Dockerfile
├── docker-compose.yml
├── docker-compose.prod.yml
├── docker-compose.acceptance.yml
├── docker-compose.traefik.yml
├── .dockerignore
├── .env.example
└── .gitignore
```

Notes:

- `analytics/` exists locally but has no tracked files at this snapshot.
- `.gitignore` and `models/.gitignore` ignore `*.npz` and most `*.pt` files, while allowing `neural_best.pt` and `neural_gpu.pt`.
- The Socket.IO client is now served locally from `static/socket.io.min.js`, not from a CDN.

---

## 3. Technology Stack

### Backend

| Component | Technology |
|---|---|
| Language | Python; Docker image uses `python:3.12-slim`; GitHub Actions uses Python 3.11 |
| Web framework | Flask 3.1.2 |
| Real-time | Flask-SocketIO |
| Async runtime | Gevent + gevent-websocket |
| Production WSGI | Gunicorn |
| Numeric dependency | NumPy |
| Neural inference | PyTorch-compatible `.pt` models; `torch` is optional at runtime and not listed in `requirements.txt` |

`neural/player.py` gracefully returns `None` if `torch` or the model file is unavailable. The AI then falls back to heuristic play.

### Frontend

| Component | Technology |
|---|---|
| Language | Vanilla JavaScript |
| Real-time client | Local Socket.IO client at `/static/socket.io.min.js` |
| Styling | CSS variables and responsive CSS |
| Internationalization | Custom key/value dictionary in `static/i18n.js` |
| PWA | `manifest.webmanifest`, `sw.js`, generated PNG icons |
| Browser tests | Playwright 1.52.0 via `requirements-dev.txt` |

### Infrastructure

| Component | Technology |
|---|---|
| Container | Docker |
| Local orchestration | Docker Compose |
| Production process | Gunicorn with `geventwebsocket.gunicorn.workers.GeventWebSocketWorker` |
| Reverse proxy | Traefik |
| TLS | Let's Encrypt via Traefik ACME |
| CI/CD | GitHub Actions |

---

## 4. Architecture

```text
Browser SPA
  ├─ static/game_state.js       local state and seat rotation
  ├─ static/game_render.js      board/card rendering
  ├─ static/game_app.js         Socket.IO events, lobby, controls
  └─ static/i18n.js             translations

Flask / Socket.IO app.py
  ├─ HTTP routes
  ├─ Socket.IO event handlers
  ├─ background room cleanup task
  └─ native threadpool offload hook for CPU-heavy AI decisions

server/
  ├─ room_state.py              rooms, seats, host migration, sid lookup
  └─ game_flow.py               game startup, reconnect snapshots, room events

main.py
  ├─ Player / HumanPlayer / AIPlayer
  ├─ bidding, trick play, scoring, replay data
  ├─ heuristic AI, inference, lookahead, endgame solver
  └─ optional neural card play

klaverjas/
  └─ reusable card, deck, trick, constants, roem logic
```

### Runtime Model

- The web layer is Gevent-based.
- Each active room's game loop runs in its own thread/greenlet path through `start_room_game()`.
- CPU-heavy AI work is offloaded through `main._thread_offload`, set by `app.py` to Gevent's native threadpool.
- Room state is in memory. This is why production uses one process/worker.

### Room Identity and Reconnect

- Seats are indexed `0..3`.
- Socket IDs are transient and tracked in `sid_to_seat`.
- Reconnect identity is still based on `(room code, player name)`.
- Active connected seats reject duplicate reconnect attempts from another socket.
- Disconnected seats can be reclaimed by matching name, which is convenient but not cryptographically secure.

---

## 5. Configuration

Configuration lives in `config.py` as frozen dataclasses:

```text
ServerConfig
  secret_key        default "klaverjas-secret"; override with SECRET_KEY
  cors_origins      default "*"; override with CORS_ORIGINS
  ping_timeout      30
  ping_interval     10
  host              0.0.0.0
  port              5000
  debug             FLASK_DEBUG == "1"

RoomConfig
  default_player_name              "Player"
  max_player_name_len              16
  max_chat_message_len             200
  seat_count                       4
  room_code_length                 4
  allowed_game_modes               score_limit, boom, free_play
  default_game_mode                score_limit
  allowed_ai_strengths             beginner, advanced, expert
  default_ai_strength              expert
  allowed_rules_variants           rotterdam, amsterdam
  default_rules_variant            rotterdam
  score_limit range                50..5000, default 500
  default_team_names               Team A, Team N
  max_team_name_len                16
  seat_reconnect_timeout_seconds   env SEAT_RECONNECT_TIMEOUT_SECONDS, default 60
  lobby_ttl_seconds                env ROOM_LOBBY_TTL_SECONDS, default 3600
  started_ttl_seconds              env ROOM_STARTED_TTL_SECONDS, default 21600
```

Important environment variables:

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Flask session secret; should be set in production |
| `CORS_ORIGINS` | Socket.IO CORS allow-list |
| `FLASK_DEBUG` | Enables Flask debug mode when `1` |
| `SEAT_RECONNECT_TIMEOUT_SECONDS` | Per-seat reconnect grace period |
| `ROOM_LOBBY_TTL_SECONDS` | Expiry for inactive lobby rooms |
| `ROOM_STARTED_TTL_SECONDS` | Expiry for inactive started rooms |
| `ACME_EMAIL` | Traefik/Let's Encrypt contact email |

---

## 6. Game Engine Deep Dive

### Seat and Team Model

`main.KlaverjasGame` is a four-seat game:

| Seat | Direction | Team |
|---|---|---|
| 0 | South | 0 |
| 1 | West | 1 |
| 2 | North | 0 |
| 3 | East | 1 |

Human seats are passed as `human_seats={seat_idx: name}`. Seats not present become `AIPlayer`s.

### Game Modes

| Mode | Meaning |
|---|---|
| `score_limit` | First team to reach configured score limit wins |
| `boom` | Fixed 16-round game |
| `free_play` | Continues without a win condition |

### Rules Variants

| Variant | Notes |
|---|---|
| Rotterdam | Default; stricter trump/overtrump behavior |
| Amsterdam | Softer partner-winning behavior |

### Player Classes

`Player`

- Base class for hands, legal moves, observed cards, void tracking, and rule-aware move constraints.

`HumanPlayer`

- Blocks the game loop using events while waiting for client input.
- Supports pending move, bid, and forced-suit requests.
- Handles disconnect/reconnect pause behavior and timeout interruption.

`AIPlayer`

- Contains bidding heuristics, card-play heuristics, inference, Monte Carlo trick-win probability, partner signaling, lookahead, endgame exact minimax, and optional neural card play.

### Current Public AI Strength Mapping

The public labels were intentionally remapped:

| Public Option | Previous Equivalent | Main Behavior |
|---|---|---|
| Beginner | Previous `expert` | Heuristic expert: inference, trick probability, endgame solver |
| Advanced | Previous `expert_v2` | Expert heuristic plus enhanced 3-trick sampled lookahead |
| Expert | Previous `neural` | Neural card play when available, with heuristic fallback |

The app only accepts `beginner`, `advanced`, and `expert` through `CONFIG.room.allowed_ai_strengths`.

Internal legacy profile keys such as `expert_v2`, `expert_v2_base`, and `neural` still exist in `AI_STRENGTH_PROFILES` for benchmark/training scripts, but they are no longer public lobby options.

### AI Profile Flags

| Parameter | Beginner | Advanced | Expert |
|---|---|---|---|
| `use_inference` | True | True | True |
| `use_trick_prob` | True | True | True |
| `use_endgame_solver` | True | True | True |
| `use_lookahead` | False | True | False |
| `lookahead_enhanced` | False | True | False |
| `lookahead_depth` | 0 | 3 | 0 |
| `lookahead_samples` | 0 | 8 | 0 |
| `use_neural_play` | False | False | True |
| `tie_break_delta` | 0.35 | 0.35 | 0.35 |
| `random_mistake_rate` | 0% | 0% | 0% |
| `declaration_bias` | 0.0 | 0.0 | 0.0 |
| `trick_win_sim_samples` | 20 | 20 | 20 |

---

## 7. Frontend and PWA

### Main Files

| File | Role |
|---|---|
| `templates/index.html` | Single page shell, lobby, board, modals, scripts |
| `static/game_state.js` | Socket init, global state, session persistence, seat rotation |
| `static/game_render.js` | Cards, labels, trick slots, trump badges |
| `static/game_app.js` | Lobby flow, Socket.IO event handlers, bid/play controls, chat, history |
| `static/i18n.js` | Dutch/English translations |
| `static/style.css` | Responsive game UI styling |
| `static/socket.io.min.js` | Locally served Socket.IO client |
| `static/manifest.webmanifest` | PWA metadata |
| `static/sw.js` | Service worker with static asset cache |

### User-Facing Lobby Flow

1. Player enters a name.
2. Host creates a room or a player joins by room code.
3. Joining player chooses an open seat.
4. Host selects game mode, rules variant, score limit, team names, and public AI strength.
5. Empty seats become AI players when the game starts.

### Session Persistence

The browser stores `{code, name}` in localStorage under `klaverjas_session`. On reconnect/page reload, the client emits `rejoin_game`.

### PWA Behavior

- Service worker caches static assets.
- Static assets use network-first refresh with cache fallback.
- WebSocket/Socket.IO requests bypass service-worker interception.

---

## 8. Neural AI and Training Tooling

### Neural Runtime

`neural/player.py` provides `neural_choose_card()`.

Flow:

1. Try to import `torch`.
2. Try to load the configured `.pt` model from `models/neural_best.pt` by default.
3. Encode the current game state through `neural/features.py`.
4. Predict one of 32 card logits and mask to legal moves.
5. Return `None` if unavailable so `AIPlayer._strategy()` falls back to heuristic play.

Because `torch` is not in `requirements.txt`, a plain production install can still run: Expert AI will degrade to non-neural heuristic behavior if PyTorch is absent.

### Tracked Model Artifacts

| File | Purpose |
|---|---|
| `models/neural_best.pt` | Default neural inference model |
| `models/neural_gpu.pt` | GPU-trained model artifact |

Training `.npz` files and intermediate `.pt` files are ignored.

### Tooling

| Tool | Purpose |
|---|---|
| `tools/ai_benchmark.py` | Candidate vs baseline AI benchmark |
| `tools/benchmark_lookahead.py` | Compare lookahead profiles |
| `tools/ci_quality_gate.py` | CI regression check around benchmark metrics |
| `tools/replay_cli.py` | Capture/verify deterministic replay traces |
| `tools/generate_training_data.py` | Generate card-play imitation data |
| `tools/generate_bid_data.py` | Generate bidding imitation data |
| `tools/train_neural.py` | Supervised card-play training |
| `tools/train_bid_neural.py` | Supervised bidding model training |
| `tools/train_ppo.py` / `train_rl.py` / `train_selfplay.py` / `train_shaped.py` | RL/self-play experiments |
| `tools/train_gpu_selfplay.py` / `gpu_engine.py` | GPU self-play tooling |
| `tools/train_bid_rl.py` / `train_bid_pipeline.py` | Bidding RL and pipeline helpers |
| `tools/run_full_pipeline.py` | End-to-end data/train/benchmark script |
| `tools/reward_rules.py` | Reward shaping helpers |
| `tools/tune_bid_threshold.py` | Bid-threshold tuning helper |

---

## 9. Deployment and CI/CD

### Docker

`Dockerfile`

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

Compose files:

| File | Purpose |
|---|---|
| `docker-compose.yml` | Local development, exposes port 5000 |
| `docker-compose.prod.yml` | Production app service behind Traefik |
| `docker-compose.acceptance.yml` | Acceptance/staging deployment |
| `docker-compose.traefik.yml` | Shared Traefik reverse proxy |

`scripts/deploy.sh`

- Detects Docker Compose plugin vs legacy binary.
- Ensures Traefik is running.
- Pulls images best-effort.
- Builds and starts app with `--remove-orphans`.
- Supports `COMPOSE_FILE`, `FORCE_DOWN_UP`, `PRUNE_IMAGES`, and `TRAEFIK_PROJECT_NAME`.

### GitHub Actions

`.github/workflows/ci.yml`

- Runs on pull requests and pushes to `main`.
- PRs targeting `main` must come from `develop`.
- Installs `requirements.txt` and `requirements-dev.txt`.
- Runs `python -m unittest discover -s tests -v`.
- Runs `tools/ci_quality_gate.py` with candidate `expert` vs baseline `advanced`.
- Deploys to VPS on `main` push if required secrets are present.
- Deployment branch defaults to `main` but can be overridden by `VPS_BRANCH`.

`.github/workflows/deploy-acceptance.yml`

- Runs on pushes to `develop` and manual dispatch.
- Deploys acceptance if required secrets are present.
- Deployment branch defaults to `develop` but can be overridden by `ACCEPTANCE_VPS_BRANCH`.
- Still does not run tests before acceptance deploy; this remains a process gap.

### Domains

| Environment | Domain |
|---|---|
| Production | `eenkaartjeleggen.nl` |
| Acceptance | `acceptance.eenkaartjeleggen.nl` |

---

## 10. Testing and Verification

### Test Suite

| File | Focus |
|---|---|
| `test_core_unit.py` | Card, deck, trick winner, roem |
| `test_room_state_unit.py` | Room seats, host migration, reconnect helpers |
| `test_game_flow_unit.py` | Snapshots, reconnect, event routing, start game |
| `test_app_integration.py` | Flask/Socket.IO room, lobby, start, chat, reconnect errors |
| `test_browser_integration.py` | Playwright browser flows |
| `test_frontend_split.py` | Static split-file/template checks |
| `test_ai_decisions.py` | AI bidding/card heuristics |
| `test_ai_advanced_tactics.py` | Endgame solver, inference, weighted discard |
| `test_ai_seat_logic.py` | Seat-index-aware partner/opponent logic |
| `test_ai_signaling.py` | Partner signal encoding/decay |
| `test_ai_strength_levels.py` | Current public AI strength remap |
| `test_benchmark_smoke.py` | Benchmark smoke test |
| `test_ci_quality_gate.py` | Regression gate metric checks |
| `test_replay_determinism.py` | Replay capture and deterministic verification |
| `test_refactor_structure.py` | Module import/refactor boundaries |

### Latest Local Verification

Command run:

```powershell
python -m unittest discover -s tests -v
```

Result:

- 68 tests passed.
- 1 browser integration setup skipped because port `5000` was already in use.

The AI quality gate command attempted in the sandbox failed on Windows multiprocessing pipe permissions. It needs to be rerun outside the sandbox or in CI after the AI strength remap, because the public `expert` vs `advanced` relationship has changed.

---

## 11. Security and Reliability

### Security Improvements Since the Older Snapshot

| Area | Current State |
|---|---|
| Socket.IO client | Served locally from `static/socket.io.min.js`; no CDN/SRI dependency |
| Active-seat hijack guard | A live connected seat rejects another socket trying to rejoin with the same name |
| Room lookup | `sid_to_seat` stores `(room_code, seat_idx)` instead of only room code |

### Remaining Security and Reliability Gaps

| Issue | Current Risk | Suggested Fix |
|---|---|---|
| Name-based reconnect for disconnected seats | Anyone with room code and matching name can reclaim a disconnected seat | Add server-issued per-seat reconnect tokens stored client-side |
| Lobby player-name XSS | Seat picker and lobby seat list interpolate `seat.name` through `innerHTML` without escaping | Use DOM text nodes or `escapeHtml()` for lobby seat names |
| Partial Socket.IO payload validation | Some handlers still use raw direct indexing such as `data["card"]` | Validate `.get()` payloads and emit structured errors |
| Default `SECRET_KEY` | Static fallback if env var is missing | Fail fast in non-debug production when default is used |
| Default `CORS_ORIGINS="*"` | Broad CORS in production if not overridden | Require explicit CORS origins outside debug |
| No rate limiting | Room probing, chat spam, rapid event spam | Add per-socket/per-event rate limits |
| In-memory room state | Restart loses games; multi-worker deployment would split state | Keep one worker or introduce Redis/persistent state |
| Import-time background cleanup task | `socketio.start_background_task()` runs at import time | Move startup into explicit app initialization |
| Acceptance deploy lacks tests | `develop` can deploy to acceptance without the unittest/benchmark job | Add a test job dependency to acceptance workflow |
| AI quality gate baseline | Public AI labels were remapped, so current baseline may be stale | Recalibrate `tools/benchmark_baseline.json` after benchmark verification |

---

## 12. Current Issues and Recommendations

### High Priority

| Task | Why |
|---|---|
| Re-run and recalibrate the AI quality gate | `expert` now means neural-backed profile and `advanced` means lookahead profile |
| Fix lobby player-name escaping | Prevents a straightforward XSS path in lobby views |
| Add reconnect tokens | Name-only reconnect remains the most important identity weakness |
| Harden Socket.IO payload validation | Prevent malformed clients from raising handler exceptions |

### Medium Priority

| Task | Why |
|---|---|
| Add tests to acceptance deploy workflow | Prevents broken `develop` pushes reaching acceptance |
| Require production `SECRET_KEY` and explicit CORS | Avoids insecure defaults in live deployment |
| Add rate limiting | Reduces abuse risk for public rooms |
| Pin loose runtime dependencies or add a lock file | Makes Docker/CI builds reproducible |

### Longer-Term Maintainability

| Task | Why |
|---|---|
| Decompose `main.py` | It now contains game flow, AI heuristics, lookahead, neural hooks, and replay logic |
| Move pacing delays into per-game config | Tools currently patch mutable global delay constants |
| Add a Socket.IO event schema document | Server and client event contracts are scattered across `app.py` and `game_app.js` |
| Add load/concurrency testing | Current capacity is unknown; room state is process-local |

---

## 13. Key Files Reference

Line counts are current approximate counts from this checkout.

| File | Lines | Role |
|---|---:|---|
| `app.py` | 604 | Flask/Socket.IO server |
| `main.py` | 2,255 | Game engine, AI, replay |
| `config.py` | 80 | Runtime config |
| `server/room_state.py` | 275 | Room, seat, host state |
| `server/game_flow.py` | 441 | Game startup/reconnect/event bridge |
| `static/game_app.js` | 1,346 | Client event/control logic |
| `static/game_render.js` | 225 | Board/card rendering |
| `static/game_state.js` | 211 | Client state/session/seat rotation |
| `static/i18n.js` | 402 | Translations |
| `templates/index.html` | 334 | Main page template |
| `neural/model.py` | 178 | Neural model definitions |
| `neural/features.py` | 263 | Neural feature encoding |
| `neural/player.py` | 123 | Neural inference wrapper |
| `tools/ai_benchmark.py` | 242 | AI benchmark runner |
| `tools/ci_quality_gate.py` | 105 | AI regression quality gate |

---

*Updated by Codex on 2026-04-22.*
