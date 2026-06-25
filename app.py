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


# Set of SIDs that have explicitly left (via leave_room/leave_game). When
# their socket subsequently disconnects, the disconnect handler will clean
# up quietly without triggering reconnect logic. This is how we distinguish
# "intentional leave" from "unintentional disconnect".
_leaving: set[str] = set()


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


def _detach_from_lobby(socketio_, room: Room, sid: str, seat: int) -> None:
    """Clean up a lobby disconnect (game not started): remove the seat and
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
    new_host_info = room.seats.get(room.host_seat)
    if new_host_info is not None:
        socketio_.emit("host_migrated", {
            "seat": room.host_seat,
            "name": new_host_info.get("name"),
        }, room=room.code)


# ─── Lobby ───────────────────────────────────────────────────────────────


@socketio.on("create_room")
def handle_create_room(data):
    sid = request.sid
    name = (data.get("name") or CONFIG.room.default_player_name).strip()[:CONFIG.room.max_player_name_len] or CONFIG.room.default_player_name

    # If this sid was already in another room, detach cleanly first
    existing_room, existing_seat = _resolve_room_seat(sid)
    if existing_room is not None and not existing_room.started:
        _detach_from_lobby(socketio, existing_room, sid, existing_seat)

    code = generate_code()
    room = Room(code, sid, name)
    rooms[code] = room
    sid_to_seat[sid] = (code, 0)

    join_room(code)
    emit("room_created", {
        "code": code,
        "seat": 0,
        "lobby": room.lobby_state(),
        "is_host": True,
    })


@socketio.on("join_room")
def handle_join_room(data):
    """First-time lobby join only. Mid-game reconnect uses `rejoin_game`."""
    sid = request.sid
    code = (data.get("code") or "").strip().upper()
    name = (data.get("name") or CONFIG.room.default_player_name).strip()[:CONFIG.room.max_player_name_len] or CONFIG.room.default_player_name

    if code not in rooms:
        emit("join_error", {"key": "error.room_not_found"})
        return

    room = rooms[code]
    room.touch()

    if room.started:
        # Game has already started — must use rejoin_game with matching name
        emit("join_error", {"key": "error.game_in_progress"})
        return

    # If the name already exists in the lobby and is currently disconnected,
    # treat this as a lobby-level reconnect (same person coming back).
    existing_seat = room.seat_for_name(name)
    if existing_seat is not None and not room.seats[existing_seat]["connected"]:
        # Detach any prior mapping this sid had
        prior_room, prior_seat = _resolve_room_seat(sid)
        if prior_room is not None and prior_room is not room:
            _detach_from_lobby(socketio, prior_room, sid, prior_seat)

        room.cancel_close_greenlet(existing_seat)
        room.attach_sid(existing_seat, sid)
        room.mark_reconnected(existing_seat)
        sid_to_seat[sid] = (code, existing_seat)
        join_room(code)
        emit("room_joined", {
            "code": code,
            "seat": existing_seat,
            "lobby": room.lobby_state(),
            "is_host": room.host_seat == existing_seat,
        })
        socketio.emit("lobby_update", room.lobby_state(), room=code)
        return

    # Fresh join — pick a seat
    requested_seat = data.get("seat")
    if requested_seat is not None:
        requested_seat = int(requested_seat)
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
    prior_room, prior_seat = _resolve_room_seat(sid)
    if prior_room is not None and prior_room is not room:
        _detach_from_lobby(socketio, prior_room, sid, prior_seat)

    room.add_seat(seat, sid, name, connected=True)
    sid_to_seat[sid] = (code, seat)
    join_room(code)
    emit("room_joined", {
        "code": code,
        "seat": seat,
        "lobby": room.lobby_state(),
        "is_host": room.host_seat == seat,
    })
    socketio.emit("lobby_update", room.lobby_state(), room=code)


@socketio.on("rejoin_game")
def handle_rejoin_game(data):
    """Reconnect to an in-progress (or lobby) game by (code, name)."""
    sid = request.sid
    code = (data.get("code") or "").strip().upper()
    name = (data.get("name") or "").strip()[:CONFIG.room.max_player_name_len]

    if not code or not name:
        emit("rejoin_error", {"key": "error.missing_credentials"})
        return
    if code not in rooms:
        emit("rejoin_error", {"key": "error.room_not_found"})
        return

    room = rooms[code]
    room.touch()
    seat = room.seat_for_name(name)
    if seat is None:
        emit("rejoin_error", {"key": "error.name_not_in_room"})
        return

    if room.seats[seat].get("connected"):
        # The seat is already held by a live socket. If it's THIS sid, just
        # re-send the snapshot (idempotent). Otherwise reject to prevent
        # hijacking of an actively-connected player.
        if room.seats[seat].get("sid") != sid:
            emit("rejoin_error", {"key": "error.seat_already_connected"})
            return

    # Detach any prior room this sid was in
    prior = sid_to_seat.get(sid)
    if prior is not None:
        prior_code, prior_seat = prior
        if prior_code != code:
            prior_room = rooms.get(prior_code)
            if prior_room is not None and not prior_room.started:
                _detach_from_lobby(socketio, prior_room, sid, prior_seat)
            else:
                _unregister_sid(sid)

    # Unregister any old sid that was bound to this seat
    old_sid = room.seats[seat].get("sid")
    if old_sid is not None and old_sid != sid:
        sid_to_seat.pop(old_sid, None)

    sid_to_seat[sid] = (code, seat)
    apply_reconnect(socketio, emit, join_room, room, seat, sid)

    # Broadcast updated lobby so everyone sees the seat as connected
    socketio.emit("lobby_update", room.lobby_state(), room=code)


@socketio.on("peek_room")
def handle_peek_room(data):
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
    start_room_game(socketio, room)


@socketio.on("leave_room")
def handle_leave_room(_data=None):
    """Intentional leave from the lobby. Does NOT trigger reconnect flow."""
    sid = request.sid
    _leaving.add(sid)
    room, seat = _resolve_room_seat(sid)
    if room is None:
        return
    if room.started:
        # Leaving an in-progress game while in the lobby UI — treat as leave_game
        _leaving.discard(sid)
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
    player = room.game.players[seat]
    if isinstance(player, HumanPlayer):
        player.supply_card(data["card"])


@socketio.on("bid_response")
def handle_bid_response(data):
    sid = request.sid
    room, seat = _resolve_room_seat(sid)
    if room is None or not room.game:
        return
    room.touch()
    player = room.game.players[seat]
    if isinstance(player, HumanPlayer):
        player.supply_bid(data["declare"])


@socketio.on("forced_suit_response")
def handle_forced_suit_response(data):
    sid = request.sid
    room, seat = _resolve_room_seat(sid)
    if room is None or not room.game:
        return
    room.touch()
    player = room.game.players[seat]
    if isinstance(player, HumanPlayer):
        player.supply_forced_suit(data["suit"])


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
    room.round_history.clear()
    room.cur_round_tricks.clear()
    room.cur_trick_cards.clear()
    room.cur_bid_history.clear()
    room.cur_log_messages.clear()
    room.cur_roem[:] = [0, 0]
    room.cur_tricks[:] = [0, 0]
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
    text = str(data.get("text", "")).strip()[:CONFIG.room.max_chat_message_len]
    if not text:
        return
    socketio.emit("chat_message", {
        "name": room.seats[seat]["name"],
        "text": text,
        "seat": seat,
    }, room=room.code)


@socketio.on("leave_game")
def handle_leave_game():
    """Intentional leave from an in-progress game. Tears down the whole room."""
    sid = request.sid
    _leaving.add(sid)
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

    # Mark every seat's sid as "leaving" so their disconnects are silent
    for info in room.seats.values():
        info_sid = info.get("sid")
        if info_sid is not None:
            _leaving.add(info_sid)

    socketio.emit("game_left", {"name": name}, room=room.code)
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
def handle_disconnect():
    sid = request.sid

    # Intentional leave → quiet cleanup, no reconnect flow
    if sid in _leaving:
        _leaving.discard(sid)
        _unregister_sid(sid)
        return

    entry = sid_to_seat.pop(sid, None)
    if not entry:
        return
    code, seat = entry
    room = rooms.get(code)
    if room is None:
        return
    room.touch()

    if room.started:
        # Unintentional disconnect mid-game → start reconnect flow
        room.detach_sid(seat)
        room.mark_disconnected(seat)

        player = room.game.players[seat] if room.game else None
        if isinstance(player, HumanPlayer):
            player.set_disconnected()

        # Host migration
        migrated = room._migrate_host()
        if migrated:
            new_host_info = room.seats.get(room.host_seat)
            if new_host_info is not None:
                socketio.emit("host_migrated", {
                    "seat": room.host_seat,
                    "name": new_host_info.get("name"),
                }, room=code)

        schedule_seat_auto_close(socketio, room, seat)

        socketio.emit("seat_disconnected", {
            "seat": seat,
            "name": room.seats[seat]["name"],
            "reconnect_timeout_seconds": CONFIG.room.seat_reconnect_timeout_seconds,
        }, room=code)
        return

    # Lobby disconnect: keep the host's seat alive for reconnect, remove others
    if seat == room.host_seat:
        room.detach_sid(seat)
        room.mark_disconnected(seat)
        schedule_seat_auto_close(socketio, room, seat)
        socketio.emit("lobby_update", room.lobby_state(), room=code)
        return

    room.close_seat(seat)
    try:
        leave_room(code)
    except Exception:
        pass
    if not room.seats:
        rooms.pop(code, None)
        return

    migrated = room._migrate_host()
    if migrated:
        new_host_info = room.seats.get(room.host_seat)
        if new_host_info is not None:
            socketio.emit("host_migrated", {
                "seat": room.host_seat,
                "name": new_host_info.get("name"),
            }, room=code)
    socketio.emit("lobby_update", room.lobby_state(), room=code)


if __name__ == "__main__":
    socketio.run(
        app,
        host=CONFIG.server.host,
        port=CONFIG.server.port,
        debug=CONFIG.server.debug,
        use_reloader=False,
    )
