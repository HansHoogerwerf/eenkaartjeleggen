# AGENTS.md

Read this first when working in this repository.

This file is the short operational guide for coding agents. For the fuller current repo map, read [`docs/claude-analysis.md`](docs/claude-analysis.md). For deeper AI behavior notes, read [`docs/ai-strategy.md`](docs/ai-strategy.md).

## Project Snapshot

- This is a real-time multiplayer Klaverjassen web game.
- Backend: Flask + Flask-SocketIO + Gevent.
- Frontend: vanilla JavaScript split across `static/game_state.js`, `static/game_render.js`, `static/game_app.js`, plus `static/i18n.js`.
- Game state is in memory. Production must stay single-process/single-worker unless state is externalized.
- The game is four seats only: South, West, North, East.
- Empty seats become AI players when a room starts.

## Current Public AI Strengths

The public lobby options are intentionally remapped:

| Public option | Current behavior |
|---|---|
| `beginner` | Previous `expert` profile |
| `advanced` | Previous `expert_v2` lookahead profile |
| `expert` | Previous `neural` profile |

Only `beginner`, `advanced`, and `expert` should be exposed by the app or accepted through `CONFIG.room.allowed_ai_strengths`.

Internal legacy profile keys such as `expert_v2`, `expert_v2_base`, and `neural` may still exist for benchmark/training tools. Do not re-add them to the lobby unless the user asks.

`expert` uses neural card play when PyTorch and the model are available. If neural inference is unavailable, it falls back to heuristic play.

## Important Runtime Patterns

- `app.py` sets `main._thread_offload` so CPU-heavy AI decisions run in native threads under Gevent.
- `Room` state lives in `server/room_state.py`; game lifecycle and reconnect snapshots live in `server/game_flow.py`.
- Reconnect identity is still based on room code + player name. This is convenient, but not secure against someone reclaiming a disconnected seat by name.
- The Socket.IO client is local at `static/socket.io.min.js`.
- The app has PWA assets: `static/manifest.webmanifest`, `static/sw.js`, and generated icons.

## Testing

Run focused tests:

```bash
python -m unittest tests.test_ai_strength_levels tests.test_app_integration tests.test_frontend_split -v
```

Run all tests:

```bash
python -m unittest discover -s tests -v
```

Browser E2E tests require Playwright and a free port 5000:

```bash
playwright install
python -m unittest tests.test_browser_integration -v
```

The AI quality gate uses multiprocessing and may need to run outside restricted sandboxes on Windows:

```bash
python tools/ci_quality_gate.py --rounds 256 --seed 42
```

## Automation and Benchmark Notes

Tools that run automated games should disable UI pacing delays by patching the imported `main` delay constants:

```python
main.AI_BID_DELAY = 0.0
main.AI_PLAY_DELAY = 0.0
main.TRICK_CLEAR_DELAY = 0.0
```

The benchmark baseline may need recalibration after AI strength changes. Public `expert` now means neural-backed profile and public `advanced` now means lookahead profile.

## Deployment Notes

- Docker image uses `python:3.12-slim`.
- GitHub Actions uses Python 3.11.
- Production deploy goes through `scripts/deploy.sh`.
- Gunicorn must use one worker because WebSocket connections and rooms are process-local.
- Acceptance deploy currently does not run the unittest/benchmark job before deploying from `develop`.

## High-Priority Known Gaps

- Lobby seat names are still interpolated into `innerHTML`; escape or render with text nodes before treating player names as safe.
- Reconnect should use server-issued per-seat tokens instead of name-only identity.
- Socket.IO payload validation is inconsistent in some handlers.
- Production should fail fast if `SECRET_KEY` remains the default or CORS remains `*`.
- Add rate limiting for public Socket.IO events.

## File Map

- `main.py`: game engine, AI profiles, bidding, trick play, scoring, replay.
- `app.py`: Flask/Socket.IO routes and event handlers.
- `config.py`: runtime defaults and allowed public options.
- `server/room_state.py`: room, seats, host migration, SID lookup.
- `server/game_flow.py`: game startup, reconnect snapshots, room events.
- `neural/`: neural feature/model/player helpers.
- `models/neural_best.pt`: default tracked neural model.
- `tools/`: benchmarks, replay, training, RL, and quality gate scripts.
- `docs/claude-analysis.md`: full current repository analysis.
- `docs/ai-strategy.md`: AI strategy and profile details.
