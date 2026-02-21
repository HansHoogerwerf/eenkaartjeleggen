"""
Klaverjassen web server (Flask + Socket.IO).
"""

from gevent import monkey

monkey.patch_all()

import os

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room

from main import HumanPlayer
from server.game_flow import leave_current_room, reconnect_player, start_room_game
from server.room_state import Room, cleanup_expired_rooms, generate_code, rooms, sid_to_room

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "klaverjas-secret")
_cors_origins = os.environ.get("CORS_ORIGINS", "*")
socketio = SocketIO(
    app,
    async_mode="gevent",
    ping_timeout=30,
    ping_interval=10,
    cors_allowed_origins=_cors_origins,
)


@app.route("/")
def index():
    return render_template("index.html")


def _cleanup_rooms_task():
    while True:
        socketio.sleep(30)
        expired = cleanup_expired_rooms()
        for code in expired:
            socketio.emit("room_expired", {"code": code}, room=code)


socketio.start_background_task(_cleanup_rooms_task)


@socketio.on("create_room")
def handle_create_room(data):
    sid = request.sid
    name = (data.get("name") or "Player").strip()[:16] or "Player"

    leave_current_room(socketio, leave_room, sid)

    code = generate_code()
    room = Room(code, sid, name)
    timeout = data.get("reconnect_timeout_seconds")
    if isinstance(timeout, int) and 15 <= timeout <= 600:
        room.reconnect_timeout_seconds = timeout
    rooms[code] = room
    sid_to_room[sid] = code

    join_room(code)
    emit("room_created", {"code": code, "seat": 0, "lobby": room.lobby_state()})


@socketio.on("join_room")
def handle_join_room(data):
    sid = request.sid
    code = (data.get("code") or "").strip().upper()
    name = (data.get("name") or "Player").strip()[:16] or "Player"

    if code not in rooms:
        emit("join_error", {"key": "error.room_not_found"})
        return

    room = rooms[code]
    room.touch()
    if room.started:
        reconnect_seat = None
        for seat, info in room.seats.items():
            if not info["connected"] and info["name"] == name:
                reconnect_seat = seat
                break
        if reconnect_seat is not None:
            reconnect_player(socketio, emit, join_room, room, reconnect_seat, sid)
            return
        emit("join_error", {"key": "error.game_in_progress"})
        return

    requested_seat = data.get("seat")
    if requested_seat is not None:
        requested_seat = int(requested_seat)
        if requested_seat < 0 or requested_seat > 3:
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

    leave_current_room(socketio, leave_room, sid)

    room.add_seat(seat, sid, name, connected=True)
    sid_to_room[sid] = code
    join_room(code)
    emit("room_joined", {"code": code, "seat": seat, "lobby": room.lobby_state()})
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
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        return
    room = rooms[code]
    room.touch()
    if room.creator_sid != sid or room.started:
        if room.creator_sid != sid:
            emit("error", {"key": "error.only_creator"})
        return

    if data:
        mode = data.get("mode", "score_limit")
        if mode in ("score_limit", "boom", "free_play"):
            room.game_mode = mode
        strength = data.get("ai_strength", "expert")
        if strength in ("beginner", "advanced", "expert"):
            room.ai_strength = strength
        limit = data.get("score_limit")
        if isinstance(limit, int) and 50 <= limit <= 5000:
            room.score_limit = limit
        names = data.get("team_names")
        if isinstance(names, list) and len(names) == 2:
            room.team_names = [str(n).strip()[:16] or f"Team {i}" for i, n in enumerate(names)]
        timeout = data.get("reconnect_timeout_seconds")
        if isinstance(timeout, int) and 15 <= timeout <= 600:
            room.reconnect_timeout_seconds = timeout

    start_room_game(socketio, room)


@socketio.on("leave_room")
def handle_leave_room(_data=None):
    leave_current_room(socketio, leave_room, request.sid)


@socketio.on("play_card")
def handle_play_card(data):
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        return
    room = rooms[code]
    room.touch()
    if not room.game:
        return
    seat = room.seat_for_sid(sid)
    if seat is None:
        return
    player = room.game.players[seat]
    if isinstance(player, HumanPlayer):
        player.supply_card(data["card"])


@socketio.on("bid_response")
def handle_bid_response(data):
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        return
    room = rooms[code]
    room.touch()
    if not room.game:
        return
    seat = room.seat_for_sid(sid)
    if seat is None:
        return
    player = room.game.players[seat]
    if isinstance(player, HumanPlayer):
        player.supply_bid(data["declare"])


@socketio.on("new_game")
def handle_new_game(_data=None):
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        return
    room = rooms[code]
    room.touch()
    if room.creator_sid != sid:
        return

    if room.game:
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
    room.cur_roem[:] = [0, 0]
    room.cur_tricks[:] = [0, 0]
    start_room_game(socketio, room)


@socketio.on("next_round")
def handle_next_round(_data=None):
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        return
    room = rooms[code]
    room.touch()
    if room.creator_sid != sid:
        return
    if room.game:
        room.game.signal_next_round()


@socketio.on("chat_message")
def handle_chat_message(data):
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        return
    room = rooms[code]
    room.touch()
    seat = room.seat_for_sid(sid)
    if seat is None:
        return
    text = str(data.get("text", "")).strip()[:200]
    if not text:
        return
    emit("chat_message", {
        "name": room.seats[seat]["name"],
        "text": text,
        "seat": seat,
    }, room=code)


@socketio.on("leave_game")
def handle_leave_game():
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        return
    room = rooms[code]
    room.touch()
    if not room.started:
        return

    seat = room.seat_for_sid(sid)
    name = room.seats.get(seat, {}).get("name", "?") if seat is not None else "?"

    room.touch()
    if room.game:
        room.game.signal_next_round()
        for p in room.game.players:
            if isinstance(p, HumanPlayer):
                p.interrupt()

    socketio.emit("game_left", {"name": name}, room=code)
    for info in room.seats.values():
        sid_to_room.pop(info.get("sid"), None)
    rooms.pop(code, None)


@socketio.on("get_hands")
def handle_get_hands(_data=None):
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        return
    room = rooms[code]
    room.touch()
    emit("hands_data", {"tricks": room.cur_round_tricks, "players": room.player_names()})


@socketio.on("get_history")
def handle_get_history(_data=None):
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        return
    room = rooms[code]
    room.touch()
    emit("history_data", {"rounds": room.round_history})


@socketio.on("connect")
def handle_connect():
    pass


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        sid_to_room.pop(sid, None)
        return

    room = rooms[code]
    room.touch()
    seat = room.seat_for_sid(sid)
    if seat is None:
        sid_to_room.pop(sid, None)
        return

    if room.started:
        room.seats[seat]["connected"] = False
        room.mark_disconnected(seat)
        player = room.game.players[seat] if room.game else None
        if isinstance(player, HumanPlayer):
            player.set_disconnected()
        if room.creator_sid == sid:
            host_seat = room.host_migration_target()
            if host_seat is not None:
                room.creator_sid = room.seats[host_seat]["sid"]
                socketio.emit("host_migrated", {"seat": host_seat, "name": room.seats[host_seat]["name"]}, room=code)
        socketio.emit("player_disconnected", {
            "seat": seat,
            "name": room.seats[seat]["name"],
            "reconnect_timeout_seconds": room.reconnect_timeout_seconds,
        }, room=code)
        return

    room.remove_seat(seat)
    sid_to_room.pop(sid, None)
    leave_room(code)
    if not room.seats:
        del rooms[code]
    else:
        if room.creator_sid == sid:
            host_seat = room.host_migration_target()
            if host_seat is not None:
                room.creator_sid = room.seats[host_seat]["sid"]
                socketio.emit("host_migrated", {"seat": host_seat, "name": room.seats[host_seat]["name"]}, room=code)
        socketio.emit("lobby_update", room.lobby_state(), room=code)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    socketio.run(app, host="0.0.0.0", port=5000, debug=debug, use_reloader=False)
