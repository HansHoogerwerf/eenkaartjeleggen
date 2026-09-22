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

## Current Public AI Opponents

The public lobby options are drop-in model players, registered in
`main.AI_PLAYER_FACTORIES` by `model_players/registry.py` (imported at startup
by `app.py`):

| Public option | Implementation |
|---|---|
| `opus` | `model_players/opus_player.py` — PIMC double-dummy card play + its own MC nat-aware bidder |
| `mythos` | `model_players/mythos_player.py` — determinized alpha-beta card play + MC nat-aware bidder |
| `neural` (default) | `model_players/neural_mythos_player.py` — neural-net card play for the first tricks, **Mythos's** exact nat-aware search from 5 cards down, Mythos's MC bidder |

Only `opus`, `mythos`, and `neural` should be exposed by the app or accepted
through `CONFIG.room.allowed_ai_strengths`.

Important distinctions:

- The public `neural` option is NOT the engine-internal `neural` profile. The
  internal `AIPlayer` strength profiles (`beginner`, `advanced`, `expert`,
  `expert_v2`, `neural`) still exist for tests, benchmarks, and training tools;
  the internal `expert`/`neural` profiles still use the old heuristic bidder.
  Do not expose internal profiles in the lobby.
- No public opponent uses a *trained* bidder yet — the public `neural`/`mythos`
  bidding is a hand-written Monte-Carlo round simulation. Training a neural
  bidder is planned future work.
- `AI_CARD_BUDGET` (env, seconds per card decision, default 1.0) bounds the
  Opus/Mythos search; lower it on weak hardware. The `neural` opponent is not
  search-bound. Neural card play needs PyTorch + `models/neural_best.pt`; it
  falls back to heuristic play if unavailable. `NEURAL_MODEL_PATH` (env)
  points the neural opponents at another checkpoint.
- Two neural feature layouts exist (`neural/features.py`): v1 = 267 inputs
  (all older checkpoints) and v2 = 300 inputs (adds trick-roem features). The
  width is read from the checkpoint, so both load. New training data and
  models should use v2; see `docs/neural-roem-retrain.md`.
- Trick roem (stuk / sequences / four of a kind, awarded per trick to the
  trick winner) is scored by the heuristic card play, the endgame solver, the
  lookahead and the CUDA engine (`tools/gpu_engine.py`). The neural path also
  runs `AIPlayer._roem_guard` after the net picks a card. Keep it that way:
  `tests/test_ai_roem.py` and `tests/test_gpu_engine_roem.py` guard it.
- The hybrid's endgame is a nat-aware search to the end of the round: from 5
  cards in hand Mythos's determinized alpha-beta finishes the round
  (`NEURAL_ENDGAME_ENGINE=mythos`, default; exact from 4 cards, 3-reply
  inner pruning at 5 unless `NEURAL_ENDGAME_EXACT=1`); the Python solver
  (`AIPlayer._endgame_minimax`, sampled deals + alpha-beta + nat/pit
  terminal) serves the internal profiles. This is worth ~+155 points per
  game vs Mythos compared with the old points-only 3-card solver; see
  `docs/neural-v3-campaign.md` before changing it.

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

The benchmark baseline may need recalibration after AI strength changes. Note
that `tools/ci_quality_gate.py` benchmarks the engine-internal profiles
(`expert` vs `advanced`), which are no longer what the lobby exposes — the
public opponents (`opus`/`mythos`/`neural`) are the drop-in model players and
are not covered by the quality gate. `tools/ai_benchmark.py` does accept the
drop-ins (`mythos`, `opus`, `neural_mythosbid`) as candidate or baseline, e.g.
`--candidate-strength neural_mythosbid --baseline-strength mythos --model <ckpt>`
compares card play with the same bidder on both sides.

Training tooling: `tools/generate_training_data.py --teacher mythos|pimc|…`
records imitation data (v2 features), `tools/train_neural.py` trains from one
or more `.npz` files (`--init` warm start, `@k` oversampling,
`--save-every-epoch`), `tools/train_gpu_selfplay.py` runs PPO self-play on
the CUDA engine, `tools/screen_checkpoints.py` benchmarks checkpoints one at a
time, and `tools/run_full_pipeline.py` chains the stages. PyTorch is only
needed from the training step onwards. Benchmarks vs Mythos are
load-sensitive: compare candidates only in back-to-back runs under equal
load (see `docs/neural-v4-campaign.md`). `model_players/pimc_player.py`
(net-rollout search on the GPU) is a stronger player than the lobby hybrid
but is not deployable without a GPU.

## Deployment Notes

- Docker image uses `python:3.12-slim`. Runtime dependencies in
  `requirements.txt` are pinned on purpose: the VPS rebuilds the image on
  every deploy, and an unpinned gunicorn upgrade (26.x moved `packaging` into
  its `gevent` extra) once broke the acceptance deploy. Bump pins
  deliberately and keep `gunicorn[gevent]`.
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
