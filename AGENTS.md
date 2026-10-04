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
| `neural` (default) | `model_players/pimc_player.py` — the neural net with **net-rollout search** for the first tricks (`neural/pimc.py`: the net proposes, playouts by the same net over 64 sampled deals judge, 128 on CUDA), **Mythos's** exact nat-aware search from 5 cards down, Mythos's MC bidder. `NEURAL_SEARCH=0` gives the plain net (`model_players/neural_mythos_player.py`) |

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
  Opus/Mythos search and, as the default of `NEURAL_PIMC_BUDGET`, the
  `neural` opponent's net-rollout search (which halves its deals to 32 at
  most and then pauses when a decision runs over: fewer than 32 deals is
  worse than no search). Lower it on weak hardware. Neural card play needs
  PyTorch + `models/neural_best.pt`; it falls back to heuristic play if
  unavailable. `NEURAL_MODEL_PATH` (env) points the neural opponents at
  another checkpoint.
- The search does not live in the weights: every attempt to distil it back
  into the net failed (hard labels, early-trick-only, soft targets from the
  search's values); the search agrees with itself on only ~67 % of decisions,
  so its edge is decision-time averaging. See `docs/neural-v4-campaign.md`
  before trying again.
- Two neural feature layouts exist (`neural/features.py`): v1 = 267 inputs
  (all older checkpoints) and v2 = 300 inputs (adds trick-roem features). The
  width is read from the checkpoint, so both load. New training data and
  models should use v2; see `docs/neural-roem-retrain.md`.
- Trick roem (stuk / sequences / four of a kind, awarded per trick to the
  trick winner) is scored by the heuristic card play, the endgame solver, the
  lookahead and the CUDA engine (`tools/gpu_engine.py`). The neural path also
  runs `AIPlayer._roem_guard` after the net picks a card. Keep it that way:
  `tests/test_ai_roem.py` and `tests/test_gpu_engine_roem.py` guard it.
- The lobby offers two rules variants (Rotterdam, Amsterdam; they differ in
  when a void player must trump). Everything that models play has to follow
  the variant of the game: the shared move generation
  (`AIPlayer._legal_moves_for_cards`, give it the seat to move and
  `rules_variant`), Mythos's `_legal_int`, the batched engine
  (`KlaverjasGPUEngine(rules_variant=...)`, so the net-rollout search keeps
  one evaluator per variant) and the card inference
  (`_apply_inference_from_play`). With Rotterdam inferences in an Amsterdam
  game an opponent's true hand was ruled out in 1 of 6 decisions.
  `tests/test_rules_variant_search.py` checks each of them against
  `Player.legal_moves`.
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
- Reconnect identity is a server-issued per-seat token (`Room.seats[i]["token"]`,
  handed to the client in `room_created` / `room_joined` / `game_starting` /
  `game_state_snapshot` and stored in `localStorage`). A matching token may
  take a seat over even while an old socket still looks connected (the old
  socket gets `session_superseded` and is closed). The player name is only a
  fallback that may reclaim a seat that is currently disconnected, and names
  are unique per room. A `rejoin_game` from the socket that already holds the
  seat is a silent resync (fresh snapshot, no broadcast).
- Intentional leaves are recognised by the sid no longer being in
  `sid_to_seat`; there is no separate "leaving" set (one used to leak across
  rooms and silently swallow later disconnects).
- `Room.phase` (`lobby`, `bidding`, `playing`, `between_rounds`, `game_over`)
  is updated from the engine's state events in `server/game_flow.py` and is
  what the reconnect snapshot uses to rebuild the right screen (bid dialog,
  table with turn arrow, next-round banner, game-over overlay). Keep it in
  sync when adding engine events.
- When the host drops mid-game the role is lent to a connected player
  (`Room.migrate_host_away_from`) so someone can end the game or wait
  longer, and handed back when the original host reconnects in time
  (`Room.restore_host`, emits `host_migrated` again). A reload therefore does
  not demote the host. In the lobby the host keeps the role through the
  grace period.
- Disconnected seats keep a deadline in `Room.reconnect_deadline`; the
  auto-close greenlet sleeps until it and re-checks, so the host can push it
  back with `extend_wait`. Mid-game the timeout is
  `SEAT_RECONNECT_TIMEOUT_SECONDS` (abort the game when it expires), in the
  lobby `LOBBY_RECONNECT_TIMEOUT_SECONDS` (free the seat).
- A countdown only runs while at least one *other* human is connected and
  waiting. A single-player game (one human, three AIs) or a game where
  everyone dropped simply pauses: the game thread blocks on the next human
  input, `Room.pause_countdowns` clears all deadlines, and
  `game_flow.resume_countdowns` starts the absent seats' countdowns when
  somebody returns. Such a paused room only goes away through the idle TTL
  (`ROOM_STARTED_TTL_SECONDS`, 6 h). Do not reintroduce an unconditional
  abort here: it is what closed single-player games on a locked phone.
- The Socket.IO client is local at `static/socket.io.min.js`.
- The app has PWA assets: `static/manifest.webmanifest`, `static/sw.js`, and generated icons.

## Testing

Run focused tests:

```bash
python -m unittest tests.test_ai_strength_levels tests.test_app_integration tests.test_reconnect_scenarios tests.test_frontend_split -v
```

`tests/test_reconnect_scenarios.py` drives the real Socket.IO handlers with
a stand-in game object (no AI) through the disconnect, takeover, phase and
grace-period paths; `tests/test_game_flow_unit.py` covers the snapshot and
the auto-close greenlet.

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
load; for card-play changes prefer the head-to-head protocol
(`--candidate pimc --baseline neural_mythosbid`, or a candidate checkpoint
via `--model` against the default net), which keeps bidder and endgame
equal on both sides and is far less noisy (see `docs/neural-v4-campaign.md`).

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
- The name-only reconnect fallback (no stored token) still lets anyone with
  the room code and a matching name reclaim a *disconnected* seat; consider
  rate limiting it or requiring the token once every client has one.
- A player who drops for longer than the grace period still ends the game
  for everyone; replacing the seat with an AI mid-round is not implemented.
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
