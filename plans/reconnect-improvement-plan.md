# Reconnection: analysis and improvement plan

Date: 2026-09-27. Status: implemented on `feature/reconnect-hardening`
(phases 0 to 6 in one PR). Of phase 5 the "wait longer" button and the
raised default grace period shipped; replacing an absent player with an AI
mid-round (option 3) did not, see the known gaps in `AGENTS.md`.

This document analyses the current disconnect/reconnect implementation and
lays out a phased plan to make it robust. Earlier reconnect work is recorded
in `plans/reconnect-ui-fix.md` (trick-card window) and PR #47 (the
`ConnectionFSM` / snapshot rewrite).

## 1. How reconnect works today

Server (`app.py`, `server/room_state.py`, `server/game_flow.py`):

- A seat's identity is `(room code, player name)`; the Socket.IO `sid` is
  transient and tracked in `sid_to_seat`.
- `disconnect` mid-game (`app.py:527`): the seat's sid is detached, the
  `HumanPlayer` is flagged disconnected, the host migrates if needed, a
  gevent greenlet is scheduled that aborts the whole game after
  `SEAT_RECONNECT_TIMEOUT_SECONDS` (60), and `seat_disconnected` is
  broadcast so everyone else sees a paused overlay with a countdown.
- `disconnect` in the lobby: the host keeps their seat for 60 s, every other
  seat is closed immediately (`app.py:573-581`).
- `rejoin_game` (`app.py:230`): looks the seat up by name, rejects if the
  seat is still marked connected under another sid (`app.py:255`), otherwise
  `apply_reconnect` (`game_flow.py:81`) rebinds the sid, sends a
  `game_state_snapshot` (`game_flow.py:20`) or `room_joined`, and the game
  thread, which never stopped blocking on the human's `threading.Event`,
  simply continues when the next `play_card` / `bid_response` arrives.
- The snapshot carries hand, card counts, current trick cards, trump,
  declarer, round scores, the log, and the pending input request.

Client (`static/game_state.js`, `static/game_app.js`):

- `{code, name}` is stored in `localStorage`; on every `connect` the
  `ConnectionFSM` emits `rejoin_game` if a session exists.
- `rejoin_error` and most `join_error`s clear the stored session and drop the
  user on the lobby form.

The snapshot-on-rejoin design is sound (python-socketio has no connection
state recovery, so a server-authoritative resync is the right tool). The
problems are in the identity rules around it, the client's handling of
failures, and the snapshot not covering every game phase.

## 2. Findings, ranked by user impact

### F1. A returning player is rejected and loses their session (race)

`rejoin_game` refuses a seat whose old sid the server still believes is
connected (`app.py:255`). The server only learns a socket is dead when the
Engine.IO ping cycle times out (`ping_interval` 10 s + `ping_timeout` 30 s,
up to ~40 s), but the client often reconnects much sooner: iOS Safari kills
the WebSocket when the app goes to the background and reconnects on return,
Wi-Fi to cellular hand-over, laptop lid close/open. The result is
`error.seat_already_connected`, and the client's `rejoin_error` handler
(`game_app.js:394`) calls `cancelAutoReconnect()` which **clears the
session**. The player is now on the lobby form with a bogus error, the game
is paused for everyone else, and their only way back is to type name and
code by hand within the 60 s abort window. A second tab from the same
browser triggers the same path and wipes the shared session for the first
tab too.

### F2. After leaving any room, later disconnects are silently ignored (reproduced)

`_leaving` (`app.py:60`) is meant to mark sids that left intentionally so
their disconnect is quiet. It is only ever cleared inside the disconnect
handler, but the socket stays open after `leave_room` / `leave_game`
(`app.py:333`, `app.py:444`, `app.py:466` marks every seat in the room) and
the same sid then creates or joins the next room. Reproduced with the
Flask-SocketIO test client: cancel a waiting room, create a new room, start,
drop the connection. The seat stays `connected=True`, nobody receives
`seat_disconnected`, no auto-abort is scheduled, the game thread blocks on
the absent player forever, and the player's rejoin is refused with
`seat_already_connected`. Every user who has ever pressed Cancel or Leave in
their session is affected. `sid_to_seat` removal already makes such
disconnects no-ops, so `_leaving` is redundant and can be deleted outright.

### F3. The snapshot has no notion of game phase

`build_game_state_snapshot` describes a trick in progress well, but not the
other phases:

- **Between rounds** (`waiting_for_host`, game thread blocked in
  `_next_round_event.wait()` at `main.py:2123`): the snapshot carries no
  "waiting for host" flag and no round result, so a reloaded client sees an
  empty table with no next-round banner. If the host is the one who
  reloaded, nobody can press "Next round": the game is stuck until the 6 h
  room TTL. Only `leave_game` (which tears down the room) gets them out.
- **Game over**: no winner/scores in the snapshot, so the game-over overlay
  and the host's "New game" button never appear after a reload. Worse, the
  client clears the session on `game_over` (`game_app.js:632`), so the
  auto-rejoin does not even fire.
- **Bidding**: `bid_history` is sent but never rendered; the offered suit
  indicator and pass/declare badges are missing, and the re-opened bid
  dialog has no leader name (`game_app.js:918`).
- **Turn indicator**: the snapshot has no current-turn field, so the arrow
  is hidden until the next `trick_played`.
- **Other disconnected seats**: `seats_status` is sent but unused. A player
  who reconnects while a third player is disconnected sees no paused overlay
  or countdown, then gets an unexplained `game_aborted`.

### F4. Host migration between rounds deadlocks the game

When the host drops during the next-round wait, the host role migrates, but
`host_migrated` (`game_app.js:472`) only toggles the game-over buttons. The
new host's next-round banner was rendered without the button (they were not
host when `waiting_for_host` arrived, `game_app.js:617`). When the old host
returns they are no longer host, so their button emits `next_round` which
the server ignores. Nobody can advance.

### F5. A failed rejoin from inside the game leaves a dead table

`cancelAutoReconnect()` (`game_app.js:162`) resets lobby sub-sections but
never re-activates `#lobby-overlay`. When `rejoin_error` arrives while the
game table is showing (room expired, aborted, server restarted, or F1),
the user sees the frozen table with no overlay and the error text hidden
inside the inactive lobby overlay.

### F6. Lobby guests have no grace period

In the lobby a non-host seat is closed on any disconnect (`app.py:581`). A
reload or brief blip removes the guest, their auto-rejoin gets
`name_not_in_room`, their session is cleared, and they must re-enter the
code and pick a seat again (possibly finding it taken). The host gets 60 s.

### F7. Duplicate names break name-based identity

The fresh-join path in `join_room` (`app.py:195`) does not check that the
name is unique among connected seats. Two "Hans" in one room means
`seat_for_name` always returns the first, so the second can never
reconnect, and a stranger who guesses a name can claim a dropped seat
(already listed as a known gap in `AGENTS.md`).

### F8. Client socket settings slow recovery down

- `timeout: 120000` (`game_state.js:8`) is the connect-attempt timeout. One
  hung attempt through a bad proxy blocks reconnection for two minutes,
  longer than the 60 s abort window.
- The `disconnect` handler ignores the reason. After an
  `io server disconnect` (which the takeover in Phase 1 will use) the client
  library does not auto-reconnect unless `socket.connect()` is called.
- No `visibilitychange` / `online` / `pageshow` hooks: a foregrounded mobile
  tab waits for the ping cycle to notice the dead socket instead of
  resyncing at once. The paused-for-everyone clock is ticking meanwhile.
- `room_expired` is emitted by the server but has no client handler.

### F10. Single-player games were closed after a locked phone

The abort countdown ran even when nobody else was in the room. One human
against three AIs who locked their phone lost the socket, the 60 s grace
expired, the abort removed the only seat and deleted the empty room, and
the rejoin on unlock got `room_not_found`. Fixed in the same PR: a
countdown only runs while at least one other human is connected and
waiting; otherwise the game pauses (the game thread blocks on the next
human input) and the absent seats' countdowns start when somebody
returns. A paused room only expires through `ROOM_STARTED_TTL_SECONDS`.

### F9. One dropped player ends the game for four (product)

After 60 s the whole game aborts and disconnected seats are removed. This
was a deliberate choice (PR #35) but it is harsh, and combined with F1 and
F8 it is reached far more often than necessary. Options are listed in
Phase 5; this is a product decision, not a bug.

## 3. Plan

Phases are ordered by impact and each is independently shippable. Phase 0
writes failing tests first so every later phase has a regression guard.

### Phase 0: characterization tests (server side, fast)

Add to `tests/test_app_integration.py`, using the same fake-game pattern as
`tests/test_game_flow_unit.py` (room with `started=True` and a
`SimpleNamespace` game holding `HumanPlayer`s) so no AI runs:

1. Mid-game disconnect then rejoin on a new socket restores the seat, emits
   `seat_disconnected` then `seat_reconnected`, and the snapshot contains the
   pending request.
2. Leave a room, create another, start, disconnect: the seat must be marked
   disconnected and `seat_disconnected` broadcast (fails today, F2).
3. Rejoin while the old sid is still connected with a valid token succeeds
   and the old socket is dropped (fails today, F1). Without a token it is
   still refused.
4. Rejoin during the between-rounds wait returns `phase=between_rounds` and
   the last round result (fails today, F3).
5. Rejoin after game over returns `phase=game_over` with winner and scores.
6. Lobby guest disconnect keeps the seat for the grace period and a rejoin
   lands in the waiting room (fails today, F6).
7. Joining with a name already connected in the room is refused (F7).

### Phase 1: identity and takeover (server)

Files: `server/room_state.py`, `app.py`, `server/game_flow.py`, `config.py`.

- Give each seat a `token` (`secrets.token_urlsafe(16)`) in `add_seat` and
  the `Room` constructor. Return it in `room_created`, `room_joined`,
  `game_starting` and `game_state_snapshot`. Never broadcast it (keep it out
  of `lobby_state()` and `player_names()`).
- `rejoin_game` accepts an optional `token`. Rules:
  - token matches: always accept. If the seat is still bound to another
    live sid, unregister that sid, call `socketio.server.disconnect(old_sid)`
    so the zombie is closed, and bind the new one. The old sid's eventual
    `disconnect` is a no-op because it is no longer in `sid_to_seat`.
  - no token (manual name entry, new device): accept only for a seat that
    is currently disconnected, exactly as today. Rate-limit or leave for the
    security work already tracked in `AGENTS.md`.
- Delete `_leaving` and every reference (F2). Leaving already removes the
  sid from `sid_to_seat`, which is what makes the later disconnect quiet.
- Enforce unique names per room in `join_room`: if the name belongs to a
  connected seat, return a new `error.name_taken` (F7). Keep the existing
  "same name and disconnected" lobby rejoin.
- Make the same-sid `rejoin_game` (the resync case in Phase 2) not
  broadcast `seat_reconnected`, so resyncs do not spam "X reconnected".

Acceptance: Phase 0 tests 2, 3 and 7 pass; existing suites stay green.

### Phase 2: client resilience

Files: `static/game_state.js`, `static/game_app.js`, `static/i18n.js`.

- Store `{code, name, seat, token}` in the session; send the token with
  every `rejoin_game`.
- `rejoin_error` handling:
  - `seat_already_connected`: do not clear the session. Retry `rejoin_game`
    with backoff (for example 1 s, 3 s, 8 s) because the server will reap the
    old socket within its ping timeout; only after the retries show a
    "your seat is open in another tab or device" message with a
    "Take over" button (sends the token, which Phase 1 honours).
  - any other error while the game table is visible: re-activate
    `#lobby-overlay`, hide the table, show the error there (F5). Split
    `cancelAutoReconnect()` into "clear session" and "show lobby form" so
    each caller does the right thing.
- Stop clearing the session on `game_over` (F3). Clear it only on explicit
  leave, `game_left`, `room_expired` (add the handler) and `room_not_found`.
- `io()` options: `timeout` 10 s, keep infinite attempts and the 1 to 5 s
  delay. Pass the disconnect `reason` to the FSM; on `io server disconnect`
  call `socket.connect()` unless the FSM is in `LEFT`.
- Add `visibilitychange` (to visible), `online` and `pageshow` listeners
  that call a `resync()` helper: if the socket is disconnected, `connect()`;
  if connected and a session exists, emit `rejoin_game` (idempotent on the
  server, replaces the whole UI from the snapshot).
- `host_migrated`: also refresh the next-round banner button and the paused
  overlay's host actions (F4).

Acceptance: manual test matrix in section 4 passes on desktop Chrome and
iOS Safari; the Playwright reload test still passes.

### Phase 3: a complete snapshot (phase machine)

Files: `server/room_state.py`, `server/game_flow.py`, `static/game_app.js`.

- Add `room.phase` with values `lobby`, `bidding`, `playing`,
  `between_rounds`, `game_over`, updated in `room_state()`: `deal_done` sets
  `bidding`, `trump_set` sets `playing`, `waiting_for_host` sets
  `between_rounds`, `game_over` sets `game_over`. Reset in
  `start_room_game` and on abort.
- Track what each phase needs on the room:
  - `cur_offered_suit` (from `trump_offered`) and `bid_history` (already
    there) for bidding.
  - `cur_trick_leader` (from `trump_set.leader_idx` and
    `trick_cleared.next_leader`); current turn is
    `(leader + len(cur_trick_cards)) % 4`.
  - `last_round_result` (`t0`, `t1`, `round_num`, `total_rounds`) from
    `round_done`, and `game_result` (`winner`, `scores`) from `game_over`.
- Snapshot additions: `phase`, `offered_suit`, `trick_leader`,
  `current_turn`, `last_round_result`, `game_result`, and
  `disconnected_seats` as `{seat: seconds_remaining}` computed from
  `room.disconnected_at` and the timeout.
- `pending_request` of type `bid` gains `leader_idx` / `leader_name` so the
  reopened dialog matches the live one.
- Client `game_state_snapshot`: render by phase. Bidding: offered-suit
  indicator plus badges replayed from `bid_history`. Playing: turn arrow
  from `current_turn`. Between rounds: next-round banner with the button for
  the host and the stored round result. Game over: overlay with winner and
  the host's "New game". Any phase: paused overlay with the countdown for
  any entry in `disconnected_seats`.
- `handle_next_round` and `handle_new_game` remain host-only; the phase data
  just makes the buttons reachable again.

Acceptance: Phase 0 tests 4 and 5 pass; a reloaded host can advance the
round and start a new game; a reloaded non-host sees the waiting banner.

### Phase 4: symmetric lobby grace

Files: `app.py`, `config.py`, `static/game_app.js`.

- Treat every lobby seat like the host today: detach, mark disconnected,
  schedule the auto-close greenlet, broadcast `lobby_update`. Add
  `LOBBY_RECONNECT_TIMEOUT_SECONDS` (default 30) so a departed guest frees
  the seat reasonably fast.
- `start_game` while a seat is in its grace period: close those seats so
  they become AI players (today a guest is already closed on disconnect, so
  this preserves behaviour) and say so in the lobby UI, which already shows
  a "disconnected" status per seat.
- A guest reload now lands back in the waiting room through the existing
  `rejoin_game` lobby branch of `apply_reconnect`.

Acceptance: Phase 0 test 6 passes.

### Phase 5: product options for the 60 s abort (decide separately)

Not required for correctness. Candidates, cheapest first:

1. Let the host extend the wait from the paused overlay (re-arm the
   greenlet) instead of only "End game".
2. Raise the default `SEAT_RECONNECT_TIMEOUT_SECONDS` once F1, F2 and F8 are
   fixed, because most drops will then recover in seconds.
3. Replace the absent player with an AI for the rest of the game. The engine
   already mixes `HumanPlayer` and `AIPlayer`, but swapping the object
   mid-`choose_card` needs a hand-over inside `HumanPlayer._wait_for_input`
   (raise a `SeatReplaced` exception the round loop handles by asking the AI
   instead). Bigger change, do after everything above.

### Phase 6: tests, docs, deploy notes

- Playwright: add a network-drop test using `context.set_offline(True)` /
  `set_offline(False)` on one page mid-game (not just `reload`), asserting
  the paused overlay appears for the other page and clears, the session
  survives, and the pending move dialog is restored. Add a between-rounds
  reload test for the host.
- Update `AGENTS.md` ("Important Runtime Patterns" and "High-Priority Known
  Gaps"), `docs/claude-analysis.md` sections 4 and 7, and `README.md` config
  table for the new env var and the token model.
- Consider the Engine.IO timings after the client fixes: with the takeover
  in place, `ping_timeout` can drop from 30 s to 20 s so zombie sockets are
  reaped faster without affecting healthy clients.

## 4. Manual test matrix (Phase 2 and 3 sign-off)

| Scenario | Expected |
|---|---|
| Reload page mid-trick | Back at the table in under 2 s, hand and legal cards restored, no error |
| Toggle airplane mode for 10 s on a phone | Others see the paused overlay; on return the player is back without touching the lobby form |
| Background the app on iOS for 20 s, return | Same as above, no "seat already taken" error |
| Second tab with the same session | First tab keeps playing; second tab offers "Take over" |
| Host reloads between rounds | Next-round banner with the button reappears |
| Host drops between rounds for 20 s | New host gets the button; when the old host returns they see the waiting banner |
| Reload after game over | Game-over overlay restored; host can start a new game |
| Server restarted while a session exists | Lobby form with a clear "room no longer exists" message, no dead table |
| Guest reloads in the waiting room | Back in the waiting room in the same seat |
| Cancel a waiting room, create a new one, drop mid-game | Others see the paused overlay; rejoin succeeds |

## 5. Rough sizing

| Phase | Server | Client | Tests | Notes |
|---|---|---|---|---|
| 0 | | | half a day | Uses existing test client patterns |
| 1 | half a day | | | Small, well contained |
| 2 | | 1 day | | Mostly `game_app.js`; needs device testing |
| 3 | half a day | 1 day | half a day | Largest UI change |
| 4 | quarter day | quarter day | | |
| 6 | | | half a day | Playwright offline test |

Phases 1 and 2 together remove the most common failure (F1, F2, F5, F8) and
can be one PR. Phase 3 is a second PR. Phase 4 is small and can ride along
with either.
