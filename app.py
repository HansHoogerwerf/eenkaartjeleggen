"""
Klaverjassen web server (Flask + Socket.IO).
"""

from gevent import monkey

monkey.patch_all()

import gevent
from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room

from config import CONFIG
import main
from main import HumanPlayer
from server.game_flow import (
    abort_game,
    apply_reconnect,
    extend_seat_grace,
    resume_countdowns,
    schedule_seat_auto_close,
    start_room_game,
)
from server.room_state import (
    Room,
    cleanup_expired_rooms,
    generate_code,
    rooms,
    sid_to_seat,
)

# Offload CPU-intensive AI computation to real native OS threads so that
# multiple games can run in parallel without blocking the gevent event loop.
main._thread_offload = lambda fn: gevent.get_hub().threadpool.apply(fn)

# Eagerly load the neural model so it's cached before any gevent thread needs it
try:
    from neural.player import _get_model, DEFAULT_MODEL_PATH
    _get_model(DEFAULT_MODEL_PATH)
except Exception:
    pass

# Register the drop-in model players so "opus" / "mythos" / "neural" are
# selectable AI opponents (populates main.AI_PLAYER_FACTORIES).
import model_players.registry  # noqa: F401,E402

app = Flask(__name__)
app.config["SECRET_KEY"] = CONFIG.server.secret_key
socketio = SocketIO(
    app,
    async_mode="gevent",
    ping_timeout=CONFIG.server.ping_timeout,
    ping_interval=CONFIG.server.ping_interval,
    cors_allowed_origins=CONFIG.server.cors_origins,
)


# Intentional vs. unintentional disconnects are told apart by `sid_to_seat`
# alone: leaving a room (or being superseded by a token takeover) removes the
# sid from the map, so its later `disconnect` finds nothing to do. There is
# deliberately no separate "leaving" set — a socket that leaves one room and
# then joins another must be tracked normally again.


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sw.js")
def service_worker():
    return send_from_directory("static", "sw.js", mimetype="application/javascript")


def _cleanup_rooms_task():
    while True:
        socketio.sleep(30)
        expired = cleanup_expired_rooms()
        for code in expired:
            socketio.emit("room_expired", {"code": code}, room=code)


socketio.start_background_task(_cleanup_rooms_task)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _resolve_room_seat(sid: str):
    """Return (room, seat) for a connected sid, or (None, None)."""
    entry = sid_to_seat.get(sid)
    if not entry:
        return None, None
    code, seat = entry
    room = rooms.get(code)
    if room is None:
        sid_to_seat.pop(sid, None)
        return None, None
    return room, seat


def _unregister_sid(sid: str) -> None:
    sid_to_seat.pop(sid, None)


def _seat_payload(room: Room, seat: int) -> dict:
    """The private join/rejoin acknowledgement for one seat (carries the
    seat's reconnect token, so it must only ever go to that seat's socket)."""
    return {
        "code": room.code,
        "seat": seat,
        "token": room.seat_token(seat),
        "lobby": room.lobby_state(),
        "is_host": room.host_seat == seat,
    }


def _emit_host_migrated(room: Room) -> None:
    new_host_info = room.seats.get(room.host_seat)
    if new_host_info is not None:
        socketio.emit("host_migrated", {
            "seat": room.host_seat,
            "name": new_host_info.get("name"),
        }, room=room.code)


def _detach_from_lobby(socketio_, room: Room, sid: str, seat: int) -> None:
    """Clean up a lobby leave (game not started): remove the seat and
    either delete the room, migrate the host, or just broadcast an update."""
    room.close_seat(seat)
    _unregister_sid(sid)
    try:
        leave_room(room.code)
    except Exception:
        pass
    if not room.seats:
        rooms.pop(room.code, None)
        return
    # close_seat() already migrated the host if needed; broadcast if changed
    socketio_.emit("lobby_update", room.lobby_state(), room=room.code)
    _emit_host_migrated(room)


def _detach_prior_room(sid: str, keep_code: str | None) -> None:
    """If this sid is still registered in another room, leave it cleanly."""
    prior = sid_to_seat.get(sid)
    if prior is None:
        return
    prior_code, prior_seat = prior
    if prior_code == keep_code:
        return
    prior_room = rooms.get(prior_code)
    if prior_room is not None and not prior_room.started:
        _detach_from_lobby(socketio, prior_room, sid, prior_seat)
    else:
        _unregister_sid(sid)


def _clean_name(raw) -> str:
    return (raw or "").strip()[:CONFIG.room.max_player_name_len]


# ─── Lobby ───────────────────────────────────────────────────────────────


@socketio.on("create_room")
def handle_create_room(data):
    sid = request.sid
    data = data or {}
    name = _clean_name(data.get("name")) or CONFIG.room.default_player_name

    # If this sid was already in another room, detach cleanly first
    _detach_prior_room(sid, keep_code=None)

    code = generate_code()
    room = Room(code, sid, name)
    rooms[code] = room
    sid_to_seat[sid] = (code, 0)

    join_room(code)
    emit("room_created", _seat_payload(room, 0))


@socketio.on("join_room")
def handle_join_room(data):
    """First-time lobby join only. Mid-game reconnect uses `rejoin_game`."""
    sid = request.sid
    data = data or {}
    code = (data.get("code") or "").strip().upper()
    name = _clean_name(data.get("name")) or CONFIG.room.default_player_name

    if code not in rooms:
        emit("join_error", {"key": "error.room_not_found"})
        return

    room = rooms[code]
    room.touch()

    if room.started:
        # Game has already started — must use rejoin_game with matching name
        emit("join_error", {"key": "error.game_in_progress"})
        return

    existing_seat = room.seat_for_name(name)
    if existing_seat is not None:
        existing = room.seats[existing_seat]
        if existing["connected"] and existing["sid"] == sid:
            # Already seated on this very socket — just acknowledge again.
            emit("room_joined", _seat_payload(room, existing_seat))
            return
        if existing["connected"]:
            # Someone else is live under that name. Names are the fallback
            # reconnect identity, so they must be unique per room.
            emit("join_error", {"key": "error.name_taken"})
            return
        # The name belongs to a disconnected seat: treat this as a
        # lobby-level reconnect (same person coming back without a token).
        _detach_prior_room(sid, keep_code=code)
        room.cancel_close_greenlet(existing_seat)
        room.attach_sid(existing_seat, sid)
        room.mark_reconnected(existing_seat)
        resume_countdowns(socketio, room, except_seat=existing_seat)
        sid_to_seat[sid] = (code, existing_seat)
        join_room(code)
        emit("room_joined", _seat_payload(room, existing_seat))
        socketio.emit("lobby_update", room.lobby_state(), room=code)
        return

    # Fresh join — pick a seat
    requested_seat = data.get("seat")
    if requested_seat is not None:
        try:
            requested_seat = int(requested_seat)
        except (TypeError, ValueError):
            emit("join_error", {"key": "error.invalid_seat"})
            return
        if requested_seat < 0 or requested_seat >= CONFIG.room.seat_count:
            emit("join_error", {"key": "error.invalid_seat"})
            return
        if requested_seat in room.seats:
            emit("join_error", {"key": "error.seat_taken"})
            return
        seat = requested_seat
    else:
        seat = room.next_free_seat()
        if seat is None:
            emit("join_error", {"key": "error.room_full"})
            return

    # Detach from any previous room before joining this one
    _detach_prior_room(sid, keep_code=code)

    room.add_seat(seat, sid, name, connected=True)
    sid_to_seat[sid] = (code, seat)
    # Somebody is now here to wait: paused seats (e.g. a lone host who
    # dropped) get their grace countdown.
    resume_countdowns(socketio, room, except_seat=seat)
    join_room(code)
    emit("room_joined", _seat_payload(room, seat))
    socketio.emit("lobby_update", room.lobby_state(), room=code)


@socketio.on("rejoin_game")
def handle_rejoin_game(data):
    """Reconnect to an in-progress (or lobby) room.

    Identity, in order of preference:
      1. `token` — the per-seat secret handed out on join. A matching token
         always wins the seat, even if an old socket still looks connected
         (a phone that went to the background, a second tab): the old socket
         is told it was superseded and closed.
      2. `name` — the fallback for a client without a stored session. It may
         only reclaim a seat that is currently disconnected.
    A rejoin on the socket that already holds the seat is a plain resync:
    the snapshot is re-sent and nothing is broadcast.
    """
    sid = request.sid
    data = data or {}
    code = (data.get("code") or "").strip().upper()
    name = _clean_name(data.get("name"))
    token = data.get("token")
    if not isinstance(token, str) or not token:
        token = None

    if not code or (not name and token is None):
        emit("rejoin_error", {"key": "error.missing_credentials"})
        return
    if code not in rooms:
        emit("rejoin_error", {"key": "error.room_not_found"})
        return

    room = rooms[code]
    room.touch()

    seat = room.seat_for_token(token)
    token_ok = seat is not None
    if seat is None and name:
        seat = room.seat_for_name(name)
    if seat is None:
        emit("rejoin_error", {"key": "error.name_not_in_room"})
        return

    info = room.seats[seat]
    old_sid = info.get("sid")
    announce = True
    if info.get("connected") and old_sid == sid:
        # Same socket asking again (visibility change, manual resync).
        announce = False
    elif info.get("connected"):
        if not token_ok:
            # A live socket holds the seat and the caller cannot prove it is
            # the same player: refuse, to prevent hijacking by name.
            emit("rejoin_error", {"key": "error.seat_already_connected"})
            return
        # Token takeover: retire the old socket. Dropping it from
        # sid_to_seat first makes its disconnect handler a no-op.
        sid_to_seat.pop(old_sid, None)
        socketio.emit("session_superseded", {"code": code, "seat": seat}, to=old_sid)
        try:
            socketio.server.disconnect(old_sid)
        except Exception:
            pass
        announce = False
    else:
        # Reclaiming a disconnected seat (the normal reconnect).
        if old_sid is not None and old_sid != sid:
            sid_to_seat.pop(old_sid, None)

    # Detach any other room this sid was in
    _detach_prior_room(sid, keep_code=code)

    sid_to_seat[sid] = (code, seat)
    apply_reconnect(socketio, emit, join_room, room, seat, sid, announce=announce)

    if announce:
        # Broadcast updated lobby so everyone sees the seat as connected
        socketio.emit("lobby_update", room.lobby_state(), room=code)


@socketio.on("peek_room")
def handle_peek_room(data):
    data = data or {}
    code = (data.get("code") or "").strip().upper()
    if code not in rooms:
        emit("join_error", {"key": "error.room_not_found"})
        return
    room = rooms[code]
    room.touch()
    if room.started:
        emit("join_error", {"key": "error.game_in_progress"})
        return
    emit("room_peeked", {"code": code, "lobby": room.lobby_state()})


@socketio.on("start_game")
def handle_start_game(data=None):
    sid = request.sid
    room, _seat = _resolve_room_seat(sid)
    if room is None:
        return
    room.touch()
    if not room.is_host(sid) or room.started:
        if not room.is_host(sid):
            emit("error", {"key": "error.only_host"})
        return

    if data:
        mode = data.get("mode", CONFIG.room.default_game_mode)
        if mode in CONFIG.room.allowed_game_modes:
            room.game_mode = mode
        strength = data.get("ai_strength", CONFIG.room.default_ai_strength)
        if strength in CONFIG.room.allowed_ai_strengths:
            room.ai_strength = strength
        rules_variant = data.get("rules_variant", CONFIG.room.default_rules_variant)
        if rules_variant in CONFIG.room.allowed_rules_variants:
            room.rules_variant = rules_variant
        limit = data.get("score_limit")
        if isinstance(limit, int) and CONFIG.room.min_score_limit <= limit <= CONFIG.room.max_score_limit:
            room.score_limit = limit
        names = data.get("team_names")
        if isinstance(names, list) and len(names) == 2:
            room.team_names = [
                str(n).strip()[:CONFIG.room.max_team_name_len] or CONFIG.room.default_team_names[i]
                for i, n in enumerate(names)
            ]

    # Seats still waiting out their lobby grace period have nobody behind
    # them right now; free them so those seats are dealt to AI players
    # instead of blocking the first trick on an absent human.
    for s in list(room.seats.keys()):
        if not room.seats[s].get("connected"):
            room.close_seat(s)

    start_room_game(socketio, room)


@socketio.on("leave_room")
def handle_leave_room(_data=None):
    """Intentional leave from the lobby. Does NOT trigger reconnect flow."""
    sid = request.sid
    room, seat = _resolve_room_seat(sid)
    if room is None:
        return
    if room.started:
        # Leaving an in-progress game while in the lobby UI — treat as leave_game
        handle_leave_game()
        return
    _detach_from_lobby(socketio, room, sid, seat)


# ─── In-game actions ─────────────────────────────────────────────────────


@socketio.on("play_card")
def handle_play_card(data):
    sid = request.sid
    room, seat = _resolve_room_seat(sid)
    if room is None or not room.game:
        return
    room.touch()
    card = (data or {}).get("card")
    if not isinstance(card, str):
        return
    player = room.game.players[seat]
    if isinstance(player, HumanPlayer):
        player.supply_card(card)


@socketio.on("bid_response")
def handle_bid_response(data):
    sid = request.sid
    room, seat = _resolve_room_seat(sid)
    if room is None or not room.game:
        return
    room.touch()
    declare = (data or {}).get("declare")
    if not isinstance(declare, bool):
        return
    player = room.game.players[seat]
    if isinstance(player, HumanPlayer):
        player.supply_bid(declare)


@socketio.on("forced_suit_response")
def handle_forced_suit_response(data):
    sid = request.sid
    room, seat = _resolve_room_seat(sid)
    if room is None or not room.game:
        return
    room.touch()
    suit = (data or {}).get("suit")
    if not isinstance(suit, str):
        return
    player = room.game.players[seat]
    if isinstance(player, HumanPlayer):
        player.supply_forced_suit(suit)


@socketio.on("new_game")
def handle_new_game(_data=None):
    sid = request.sid
    room, _seat = _resolve_room_seat(sid)
    if room is None or not room.is_host(sid):
        return
    room.touch()

    if room.game:
        room._game_abort_handled = True
        room.game.signal_next_round()
        for p in room.game.players:
            if isinstance(p, HumanPlayer):
                p.interrupt()
    if room.game_thread and room.game_thread.is_alive():
        room.game_thread.join(timeout=3)

    room.started = False
    room.reset_game_state()
    start_room_game(socketio, room)


@socketio.on("next_round")
def handle_next_round(_data=None):
    sid = request.sid
    room, _seat = _resolve_room_seat(sid)
    if room is None or not room.is_host(sid):
        return
    room.touch()
    if room.game:
        room.game.signal_next_round()


@socketio.on("chat_message")
def handle_chat_message(data):
    sid = request.sid
    room, seat = _resolve_room_seat(sid)
    if room is None:
        return
    room.touch()
    text = str((data or {}).get("text", "")).strip()[:CONFIG.room.max_chat_message_len]
    if not text:
        return
    socketio.emit("chat_message", {
        "name": room.seats[seat]["name"],
        "text": text,
        "seat": seat,
    }, room=room.code)


@socketio.on("leave_game")
def handle_leave_game(_data=None):
    """Intentional leave from an in-progress game. Tears down the whole room."""
    sid = request.sid
    room, seat = _resolve_room_seat(sid)
    if room is None:
        return
    room.touch()
    if not room.started:
        _detach_from_lobby(socketio, room, sid, seat)
        return

    name = room.seats.get(seat, {}).get("name", "?") if seat is not None else "?"

    if room.game:
        room._game_abort_handled = True
        room.game.signal_next_round()
        for p in room.game.players:
            if isinstance(p, HumanPlayer):
                p.interrupt()

    socketio.emit("game_left", {"name": name}, room=room.code)
    # Every seat's sid leaves the map, so their later disconnects are quiet.
    for info in room.seats.values():
        info_sid = info.get("sid")
        if info_sid is not None:
            sid_to_seat.pop(info_sid, None)
        g = info.get("close_greenlet")
        if g is not None:
            try:
                g.kill()
            except Exception:
                pass
    rooms.pop(room.code, None)


@socketio.on("host_abort_game")
def handle_host_abort_game(_data=None):
    """Host-only: end the in-progress game immediately without waiting for
    the auto-abort timeout. The game thread is interrupted and its run()
    handler emits `game_aborted` + `lobby_update` so all remaining clients
    return to the lobby. `abort_game()` is a no-op when no game is running,
    so it is safe to call unconditionally."""
    sid = request.sid
    room, _seat = _resolve_room_seat(sid)
    if room is None:
        return
    if not room.is_host(sid):
        emit("error", {"key": "error.only_host"})
        return
    abort_game(socketio, room)


@socketio.on("extend_wait")
def handle_extend_wait(_data=None):
    """Host-only: give every seat that is waiting to reconnect more time
    before the game is aborted. Broadcasts the new remaining time so all
    countdowns stay in sync."""
    sid = request.sid
    room, _seat = _resolve_room_seat(sid)
    if room is None:
        return
    if not room.is_host(sid):
        emit("error", {"key": "error.only_host"})
        return
    room.touch()
    extra = CONFIG.room.reconnect_extend_seconds
    for seat, info in room.disconnected_seats().items():
        remaining = extend_seat_grace(room, seat, extra)
        if remaining is None:
            continue
        socketio.emit("seat_wait_extended", {
            "seat": seat,
            "name": info.get("name"),
            "seconds_remaining": int(remaining),
        }, room=room.code)


@socketio.on("get_hands")
def handle_get_hands(_data=None):
    sid = request.sid
    room, _seat = _resolve_room_seat(sid)
    if room is None:
        return
    room.touch()
    emit("hands_data", {"tricks": room.cur_round_tricks, "players": room.player_names()})


@socketio.on("get_history")
def handle_get_history(_data=None):
    sid = request.sid
    room, _seat = _resolve_room_seat(sid)
    if room is None:
        return
    room.touch()
    emit("history_data", {"rounds": room.round_history})


@socketio.on("connect")
def handle_connect():
    # Nothing to do: the client drives reconnection explicitly by emitting
    # rejoin_game (or join_room for first-time joins) after connecting.
    pass


@socketio.on("disconnect")
def handle_disconnect(_reason=None):
    sid = request.sid

    # A sid that left, never joined, or was superseded by a token takeover
    # is not in the map: nothing to do.
    entry = sid_to_seat.pop(sid, None)
    if not entry:
        return
    code, seat = entry
    room = rooms.get(code)
    if room is None:
        return
    room.touch()
    info = room.seats.get(seat)
    if info is None or info.get("sid") != sid:
        # The seat moved on to another socket already.
        return

    if room.started:
        # Unintentional disconnect mid-game → start reconnect flow
        room.detach_sid(seat)
        room.mark_disconnected(seat)

        player = room.game.players[seat] if room.game else None
        if isinstance(player, HumanPlayer):
            player.set_disconnected()

        # Host migration: somebody connected must be able to end the game
        # or wait longer; the role returns to this seat when it reconnects.
        if room.migrate_host_away_from(seat):
            _emit_host_migrated(room)

        if room.connected_human_count() == 0:
            # Nobody is left to wait for this player (a single-player game,
            # or everyone dropped, e.g. a locked phone): pause instead of
            # counting down to an abort. The game thread simply blocks on
            # the next human input; countdowns start when somebody returns
            # and the room idles out via ROOM_STARTED_TTL_SECONDS.
            room.pause_countdowns()
            return

        timeout = CONFIG.room.seat_reconnect_timeout_seconds
        schedule_seat_auto_close(socketio, room, seat, timeout)

        socketio.emit("seat_disconnected", {
            "seat": seat,
            "name": info["name"],
            "reconnect_timeout_seconds": timeout,
            "seconds_remaining": timeout,
        }, room=code)
        return

    # Lobby disconnect: every seat (host or guest) keeps its place for the
    # lobby grace period so a reload or a short blip does not lose the seat.
    # The host role is not migrated during the grace period; if the host
    # never returns the auto-close greenlet frees the seat and migrates.
    # With nobody else connected there is no countdown either; the room
    # idles out via ROOM_LOBBY_TTL_SECONDS.
    room.detach_sid(seat)
    room.mark_disconnected(seat)
    if room.connected_human_count() == 0:
        room.pause_countdowns()
    else:
        schedule_seat_auto_close(socketio, room, seat, CONFIG.room.lobby_reconnect_timeout_seconds)
    socketio.emit("lobby_update", room.lobby_state(), room=code)


if __name__ == "__main__":
    socketio.run(
        app,
        host=CONFIG.server.host,
        port=CONFIG.server.port,
        debug=CONFIG.server.debug,
        use_reloader=False,
    )
