"""
Klaverjassen — Flask + Socket.IO web server (multiplayer lobby).

Supports 1–4 human players per game room.  One player creates a room
(gets a 4-character join code), others join with that code, and
remaining empty seats are filled by AI.
"""

from gevent import monkey
monkey.patch_all()

import os
import random
import string
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room

from main import (
    KlaverjasGame, HumanPlayer, AIPlayer, GameInterrupt,
    SUIT_NAMES, RED_SUITS, SEAT_DEFAULTS, SEAT_TEAMS,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "klaverjas-secret")
socketio = SocketIO(app, async_mode="gevent", ping_timeout=120, ping_interval=25)

PLAYER_NAMES_DEFAULT = {0: "South", 1: "West", 2: "North", 3: "East"}


# ─── Room ─────────────────────────────────────────────────────────────────────

class Room:
    """One game room with up to 4 human players."""

    def __init__(self, code: str, creator_sid: str, creator_name: str):
        self.code = code
        self.creator_sid = creator_sid

        # seat_idx → {sid, name, connected}
        self.seats: dict[int, dict] = {}
        self.seats[0] = {"sid": creator_sid, "name": creator_name, "connected": True}

        self.game: KlaverjasGame | None = None
        self.game_thread: threading.Thread | None = None
        self.started = False
        self.game_mode: str = "score_limit"
        self.score_limit: int = 500
        self.team_names: list[str] = ["Team 0", "Team 1"]

        # Per-room game state for Hands / History viewers
        self.cur_round_tricks: list[dict] = []
        self.cur_trick_cards: dict[int, dict] = {}
        self.round_history: list[dict] = []
        self.cur_roem = [0, 0]
        self.cur_tricks = [0, 0]

    def next_free_seat(self) -> int | None:
        for i in range(4):
            if i not in self.seats:
                return i
        return None

    def seat_for_sid(self, sid: str) -> int | None:
        for seat, info in self.seats.items():
            if info["sid"] == sid:
                return seat
        return None

    def player_names(self) -> dict[int, str]:
        """Return {seat_idx: display_name} for all 4 seats."""
        names = {}
        for i in range(4):
            if i in self.seats:
                names[i] = self.seats[i]["name"]
            else:
                names[i] = f"AI {SEAT_DEFAULTS[i]}"
        return names

    def lobby_state(self) -> dict:
        """State for the lobby waiting room."""
        return {
            "code": self.code,
            "seats": {
                str(i): {
                    "name": self.seats[i]["name"] if i in self.seats else None,
                    "team": SEAT_TEAMS[i],
                    "is_human": i in self.seats,
                }
                for i in range(4)
            },
            "started": self.started,
        }


# sid → room code (for fast lookup on disconnect)
sid_to_room: dict[str, str] = {}
rooms: dict[str, Room] = {}


def generate_code() -> str:
    while True:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
        if code not in rooms:
            return code


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─── Lobby events ─────────────────────────────────────────────────────────────

@socketio.on("create_room")
def handle_create_room(data):
    sid = request.sid
    name = (data.get("name") or "Player").strip()[:16] or "Player"

    # Leave any existing room
    _leave_current_room(sid)

    code = generate_code()
    room = Room(code, sid, name)
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

    if room.started:
        # Check if this is a reconnect
        reconnect_seat = None
        for seat, info in room.seats.items():
            if not info["connected"] and info["name"] == name:
                reconnect_seat = seat
                break

        if reconnect_seat is not None:
            _reconnect_player(room, reconnect_seat, sid)
            return
        else:
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

    # Leave any existing room
    _leave_current_room(sid)

    room.seats[seat] = {"sid": sid, "name": name, "connected": True}
    sid_to_room[sid] = code

    join_room(code)
    emit("room_joined", {"code": code, "seat": seat, "lobby": room.lobby_state()})
    # Notify others
    socketio.emit("lobby_update", room.lobby_state(), room=code)


@socketio.on("peek_room")
def handle_peek_room(data):
    """Return lobby state for a room code so the client can show seat picker."""
    code = (data.get("code") or "").strip().upper()
    if code not in rooms:
        emit("join_error", {"key": "error.room_not_found"})
        return
    room = rooms[code]
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
    if room.creator_sid != sid:
        emit("error", {"key": "error.only_creator"})
        return
    if room.started:
        return

    if data:
        mode = data.get("mode", "score_limit")
        if mode in ("score_limit", "boom", "free_play"):
            room.game_mode = mode
        limit = data.get("score_limit")
        if isinstance(limit, int) and 50 <= limit <= 5000:
            room.score_limit = limit
        names = data.get("team_names")
        if isinstance(names, list) and len(names) == 2:
            room.team_names = [str(n).strip()[:16] or f"Team {i}" for i, n in enumerate(names)]

    _start_room_game(room)


@socketio.on("leave_room")
def handle_leave_room(_data=None):
    _leave_current_room(request.sid)


# ─── Game input events ────────────────────────────────────────────────────────

@socketio.on("play_card")
def handle_play_card(data):
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        return
    room = rooms[code]
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
    if room.creator_sid != sid:
        return

    # Interrupt existing game
    if room.game:
        room.game.signal_next_round()  # unblock if waiting between rounds
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

    _start_room_game(room)


@socketio.on("next_round")
def handle_next_round(_data=None):
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        return
    room = rooms[code]
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


@socketio.on("get_hands")
def handle_get_hands(_data=None):
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        return
    room = rooms[code]
    emit("hands_data", {
        "tricks": room.cur_round_tricks,
        "players": room.player_names(),
    })


@socketio.on("get_history")
def handle_get_history(_data=None):
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        return
    room = rooms[code]
    emit("history_data", {"rounds": room.round_history})


# ─── Connection lifecycle ─────────────────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    pass  # Player will create or join a room explicitly


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    code = sid_to_room.get(sid)
    if not code or code not in rooms:
        sid_to_room.pop(sid, None)
        return

    room = rooms[code]
    seat = room.seat_for_sid(sid)
    if seat is None:
        sid_to_room.pop(sid, None)
        return

    if room.started:
        # Mark disconnected — game will pause when it's this player's turn
        room.seats[seat]["connected"] = False
        player = room.game.players[seat] if room.game else None
        if isinstance(player, HumanPlayer):
            player.set_disconnected()
        socketio.emit("player_disconnected", {
            "seat": seat,
            "name": room.seats[seat]["name"],
        }, room=code)
    else:
        # Not started — remove from lobby
        del room.seats[seat]
        sid_to_room.pop(sid, None)
        leave_room(code)

        if not room.seats:
            # Room is empty — clean up
            del rooms[code]
        else:
            # If creator left, reassign
            if room.creator_sid == sid:
                first_seat = min(room.seats.keys())
                room.creator_sid = room.seats[first_seat]["sid"]
            socketio.emit("lobby_update", room.lobby_state(), room=code)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _leave_current_room(sid: str) -> None:
    """Remove a player from whatever room they're in."""
    code = sid_to_room.pop(sid, None)
    if not code or code not in rooms:
        return
    room = rooms[code]
    seat = room.seat_for_sid(sid)
    if seat is not None and not room.started:
        del room.seats[seat]
    leave_room(code)

    if not room.seats:
        rooms.pop(code, None)
    else:
        if room.creator_sid == sid and not room.started:
            first_seat = min(room.seats.keys())
            room.creator_sid = room.seats[first_seat]["sid"]
        socketio.emit("lobby_update", room.lobby_state(), room=code)


def _reconnect_player(room: Room, seat: int, new_sid: str) -> None:
    """Handle a player reconnecting to a running game."""
    old_sid = room.seats[seat]["sid"]
    room.seats[seat]["sid"] = new_sid
    room.seats[seat]["connected"] = True
    sid_to_room[new_sid] = room.code

    # Restore creator status if this was the host
    if room.creator_sid == old_sid:
        room.creator_sid = new_sid

    join_room(room.code)

    player = room.game.players[seat] if room.game else None
    if isinstance(player, HumanPlayer):
        player.set_reconnected()

    # Send current game state to the reconnected player
    state = {
        "seat": seat,
        "code": room.code,
        "scores": list(room.game.scores) if room.game else [0, 0],
        "player_names": room.player_names(),
        "cur_tricks": list(room.cur_tricks),
        "cur_roem": list(room.cur_roem),
        "is_creator": room.creator_sid == new_sid,
        "team_names": room.team_names,
    }
    # Send their current hand
    if room.game:
        p = room.game.players[seat]
        state["hand"] = [c.to_dict() for c in p.hand]
        state["card_counts"] = {
            str(i): len(room.game.players[i].hand) for i in range(4) if i != seat
        }

    emit("reconnected", state)
    socketio.emit("player_reconnected", {
        "seat": seat,
        "name": room.seats[seat]["name"],
    }, room=room.code)


def _start_room_game(room: Room) -> None:
    """Start the game for a room, filling empty seats with AI."""
    room.started = True
    room.round_history.clear()
    room.cur_round_tricks.clear()
    room.cur_trick_cards.clear()
    room.cur_roem[:] = [0, 0]
    room.cur_tricks[:] = [0, 0]

    human_seats = {seat: info["name"] for seat, info in room.seats.items()}

    room.game = KlaverjasGame(
        human_seats=human_seats,
        log_fn=lambda msg, tag="": _room_log(room, msg, tag),
        state_fn=lambda event, data: _room_state(room, event, data),
        game_mode=room.game_mode,
        score_limit=room.score_limit,
    )

    # Wire up callbacks for each human player
    for seat in human_seats:
        player = room.game.players[seat]
        if isinstance(player, HumanPlayer):
            player._on_move_request = lambda seat_idx, legal, r=room: _on_move_request(r, seat_idx, legal)
            player._on_bid_request = lambda seat_idx, suit, forced, r=room: _on_bid_request(r, seat_idx, suit, forced)
            player._on_disconnect_pause = lambda seat_idx, r=room: _on_disconnect_pause(r, seat_idx)
            player.reset_interrupt()

    # Notify all players the game is starting
    for seat, info in room.seats.items():
        socketio.emit("game_starting", {
            "seat": seat,
            "player_names": room.player_names(),
            "team_names": room.team_names,
        }, to=info["sid"])

    def run():
        try:
            room.game.play()
        except GameInterrupt:
            pass

    room.game_thread = threading.Thread(target=run, daemon=True)
    room.game_thread.start()


# ─── Game event relay (per-room) ──────────────────────────────────────────────

def _room_log(room: Room, msg: str, tag: str = "") -> None:
    socketio.emit("log", {"msg": msg, "tag": tag}, room=room.code)


def _room_state(room: Room, event: str, data: dict) -> None:
    """Relay game events to all players in the room."""
    send_data = dict(data)

    if event == "deal_done":
        room.cur_roem[:] = [0, 0]
        room.cur_tricks[:] = [0, 0]
        room.cur_round_tricks.clear()
        room.cur_trick_cards.clear()

        # Send each human their own hand privately
        if room.game:
            for seat, info in room.seats.items():
                player = room.game.players[seat]
                card_counts = {
                    str(i): len(room.game.players[i].hand)
                    for i in range(4) if i != seat
                }
                socketio.emit("deal_done", {
                    "hand": [c.to_dict() for c in player.hand],
                    "card_counts": card_counts,
                    "my_seat": seat,
                }, to=info["sid"])
        return  # already sent per-player

    elif event == "trump_set":
        # Resend hands to each human
        if room.game:
            for seat, info in room.seats.items():
                player = room.game.players[seat]
                socketio.emit("trump_set", {
                    **send_data,
                    "hand": [c.to_dict() for c in player.hand],
                }, to=info["sid"])
        return

    elif event == "trick_played":
        card = data["card"]
        pidx = data["player_idx"]
        room.cur_trick_cards[pidx] = card.to_dict()
        send_data["card"] = card.to_dict()

        if room.game:
            # Send card counts and updated hands per-player
            for seat, info in room.seats.items():
                player = room.game.players[seat]
                card_counts = {
                    str(i): len(room.game.players[i].hand)
                    for i in range(4) if i != seat
                }
                socketio.emit("trick_played", {
                    **send_data,
                    "hand": [c.to_dict() for c in player.hand],
                    "card_counts": card_counts,
                }, to=info["sid"])
        return

    elif event == "trick_won":
        winner_idx = data["winner_idx"]
        pts = data["pts"]
        room.cur_round_tricks.append({
            "cards": dict(room.cur_trick_cards),
            "winner_idx": winner_idx,
            "winner_name": room.player_names().get(winner_idx, "?"),
            "pts": pts,
        })
        room.cur_trick_cards.clear()
        room.cur_tricks[:] = list(data["trick_pts"])
        room.cur_roem[:] = list(data["roem_pts"])
        send_data["cur_tricks"] = list(room.cur_tricks)
        send_data["cur_roem"] = list(room.cur_roem)

    elif event == "roem":
        room.cur_roem[:] = list(data["roem_pts"])
        send_data["items"] = [(desc, pts) for desc, pts in data["items"]]
        send_data["cur_roem"] = list(room.cur_roem)

    elif event == "round_done":
        room.cur_roem[:] = [0, 0]
        room.cur_tricks[:] = [0, 0]
        if "history" in data:
            room.round_history.append(data["history"])
        send_data.pop("history", None)
        send_data["cur_tricks"] = [0, 0]
        send_data["cur_roem"] = [0, 0]

    # Broadcast shared events to the whole room
    socketio.emit(event, send_data, room=room.code)


def _on_move_request(room: Room, seat_idx: int, legal_cards) -> None:
    """Game thread asks a human for a card."""
    info = room.seats.get(seat_idx)
    if info and info["connected"]:
        socketio.emit("request_move", {
            "legal": [str(c) for c in legal_cards],
        }, to=info["sid"])


def _on_bid_request(room: Room, seat_idx: int, suit: str, forced: bool) -> None:
    """Game thread asks a human to bid."""
    info = room.seats.get(seat_idx)
    if info and info["connected"]:
        socketio.emit("request_bid", {
            "suit": suit,
            "suit_name": SUIT_NAMES[suit],
            "forced": forced,
        }, to=info["sid"])


def _on_disconnect_pause(room: Room, seat_idx: int) -> None:
    """Notify all players that the game is paused waiting for a disconnected player."""
    name = room.seats.get(seat_idx, {}).get("name", "?")
    socketio.emit("game_paused", {
        "seat": seat_idx,
        "name": name,
        "key": "error.waiting_reconnect",
    }, room=room.code)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    socketio.run(app, host="0.0.0.0", port=5000, debug=debug, use_reloader=False)
