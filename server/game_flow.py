"""Game-flow helpers: snapshot building, reconnect application, event routing.

This module owns the in-game state that the reconnect snapshot needs, and
provides the `apply_reconnect` and `schedule_seat_auto_close` primitives used
by the app layer.
"""

import threading

import gevent

from config import CONFIG
from main import GameInterrupt, HumanPlayer, KlaverjasGame, SUIT_NAMES
from server.room_state import Room, rooms


# ─── Snapshot building ───────────────────────────────────────────────────


def build_game_state_snapshot(room: Room, seat: int) -> dict:
    """Assemble the complete game-state snapshot for a reconnecting player.

    This is the single authoritative source for "what the player needs to
    rebuild their UI from scratch". The client's `game_state_snapshot`
    handler pulls everything it needs from this dict and does not rely on
    any follow-up events.
    """
    snapshot = {
        "seat": seat,
        "code": room.code,
        "is_host": room.host_seat == seat,
        "host_seat": room.host_seat,
        "player_names": room.player_names(),
        "team_names": list(room.team_names),
        "scores": list(room.game.scores) if room.game else [0, 0],
        "cur_tricks": list(room.cur_tricks),
        "cur_roem": list(room.cur_roem),
        "trump": room.cur_trump,
        "declaring_player": room.cur_declaring_player,
        "declaring_player_idx": room.cur_declaring_player_idx,
        "declaring_team": room.cur_declaring_team,
        "trick_cards": {str(k): v for k, v in room.cur_trick_cards.items()},
        "trick_history": list(room.cur_round_tricks),
        "bid_history": list(room.cur_bid_history),
        "log_messages": list(room.cur_log_messages),
        "seat_reconnect_timeout_seconds": CONFIG.room.seat_reconnect_timeout_seconds,
        "seats_status": {
            str(i): {
                "name": room.seats[i]["name"] if i in room.seats else None,
                "is_human": i in room.seats,
                "connected": room.seats[i]["connected"] if i in room.seats else False,
            }
            for i in range(4)
        },
    }

    if room.game:
        player = room.game.players[seat]
        if isinstance(player, HumanPlayer):
            snapshot["hand"] = [c.to_dict() for c in player.hand]
            snapshot["card_counts"] = {
                str(i): len(room.game.players[i].hand)
                for i in range(4) if i != seat
            }
            snapshot["pending_request"] = player.get_pending_request()
        else:
            snapshot["hand"] = []
            snapshot["card_counts"] = {str(i): len(room.game.players[i].hand) for i in range(4) if i != seat}
            snapshot["pending_request"] = None
    else:
        snapshot["hand"] = []
        snapshot["card_counts"] = {}
        snapshot["pending_request"] = None

    return snapshot


# ─── Reconnect ────────────────────────────────────────────────────────────


def apply_reconnect(socketio, emit_fn, join_room_fn, room: Room, seat: int, new_sid: str) -> None:
    """Rebind a seat to a new sid and push the right resume payload to the client.

    Cancels any pending auto-abort greenlet and updates the seat's sid. Then:
      • If the room has an active game → emit `game_state_snapshot` and tell
        the HumanPlayer it's reconnected so the game thread can resume.
      • Otherwise (lobby state, e.g. the game was aborted while this seat was
        disconnected) → emit `room_joined` so the client lands in the lobby.
    Either way, broadcasts `seat_reconnected` so other players hide the
    paused overlay.
    """
    if seat not in room.seats:
        return

    room.cancel_close_greenlet(seat)
    room.attach_sid(seat, new_sid)
    room.mark_reconnected(seat)

    join_room_fn(room.code)

    if room.started and room.game is not None:
        snapshot = build_game_state_snapshot(room, seat)
        emit_fn("game_state_snapshot", snapshot)
        player = room.game.players[seat]
        if isinstance(player, HumanPlayer):
            player.set_reconnected()
    else:
        emit_fn("room_joined", {
            "code": room.code,
            "seat": seat,
            "lobby": room.lobby_state(),
            "is_host": room.host_seat == seat,
        })

    socketio.emit("seat_reconnected", {
        "seat": seat,
        "name": room.seats[seat]["name"],
    }, room=room.code)


# ─── Game abort ──────────────────────────────────────────────────────────


def abort_game(socketio, room: Room) -> None:
    """Tear down an in-progress game and reset the room back to lobby state.

    Triggers the game thread to exit via GameInterrupt; the thread's
    `run()` handler in `start_room_game` then resets `room.started`,
    clears game state, and emits `game_aborted` + `lobby_update` so all
    clients return to the lobby view together. Disconnected seats are NOT
    removed — they remain so the disconnected player can rejoin the lobby
    once they come back.
    """
    if not room.started or room.game is None:
        return

    # Make sure the run() handler does the cleanup (it skips when
    # _game_abort_handled is True, which is set by leave_game/new_game).
    room._game_abort_handled = False

    # Unblock any between-rounds wait, then interrupt every human player so
    # whatever the game thread is awaiting raises GameInterrupt.
    room.game.signal_next_round()
    for p in room.game.players:
        if isinstance(p, HumanPlayer):
            p.interrupt()


# ─── Auto-abort greenlet ─────────────────────────────────────────────────


def schedule_seat_auto_close(socketio, room: Room, seat: int) -> None:
    """Spawn an auto-abort countdown for a disconnected seat.

    If the seat does not reconnect within `seat_reconnect_timeout_seconds`,
    the entire game is aborted (rather than the seat being closed and the
    game continuing in a broken state). The function name is kept for
    backwards compat with callers but the behavior is now full-game abort.
    """
    if seat not in room.seats:
        return
    timeout = CONFIG.room.seat_reconnect_timeout_seconds

    def _worker():
        try:
            gevent.sleep(timeout)
        except gevent.GreenletExit:
            return
        # Re-check state after the sleep — the player may have reconnected
        if room.code not in rooms:
            return
        if seat not in room.seats:
            return
        if room.seats[seat].get("connected"):
            return  # reconnected in time
        # Clear our greenlet ref before triggering abort (we are this greenlet)
        if room.seats[seat].get("close_greenlet") is gevent.getcurrent():
            room.seats[seat]["close_greenlet"] = None

        if room.started and room.game is not None:
            abort_game(socketio, room)
        else:
            # Game wasn't started (lobby disconnect) — just clean up the seat
            room.close_seat(seat)
            socketio.emit("lobby_update", room.lobby_state(), room=room.code)
            if not room.seats:
                rooms.pop(room.code, None)

    g = gevent.spawn(_worker)
    room.seats[seat]["close_greenlet"] = g


# ─── Game start / lifecycle ──────────────────────────────────────────────


def start_room_game(socketio, room: Room) -> None:
    room.started = True
    room._game_abort_handled = False
    room.round_history.clear()
    room.cur_round_tricks.clear()
    room.cur_trick_cards.clear()
    room.cur_bid_history.clear()
    room.cur_log_messages.clear()
    room.cur_roem[:] = [0, 0]
    room.cur_tricks[:] = [0, 0]
    room.cur_trump = None
    room.cur_declaring_player = None
    room.cur_declaring_player_idx = None
    room.cur_declaring_team = None

    human_seats = {seat: info["name"] for seat, info in room.seats.items()}

    room.game = KlaverjasGame(
        human_seats=human_seats,
        log_fn=lambda msg, tag="": room_log(socketio, room, msg, tag),
        state_fn=lambda event, data: room_state(socketio, room, event, data),
        game_mode=room.game_mode,
        score_limit=room.score_limit,
        ai_strength=room.ai_strength,
        rules_variant=room.rules_variant,
    )

    for seat in human_seats:
        player = room.game.players[seat]
        if isinstance(player, HumanPlayer):
            player._on_move_request = lambda seat_idx, legal, r=room: on_move_request(socketio, r, seat_idx, legal)
            player._on_bid_request = lambda seat_idx, suit, forced, r=room: on_bid_request(socketio, r, seat_idx, suit, forced)
            player._on_forced_suit_request = lambda seat_idx, r=room: on_forced_suit_request(socketio, r, seat_idx)
            player.reset_interrupt()

    for seat, info in room.seats.items():
        sid = info.get("sid")
        if sid is None:
            continue
        socketio.emit("game_starting", {
            "seat": seat,
            "player_names": room.player_names(),
            "team_names": room.team_names,
        }, to=sid)

    def run():
        try:
            room.game.play()
        except GameInterrupt:
            # If nobody else handled the abort (e.g. player timeout, not
            # an intentional interrupt from new_game/leave_game), reset
            # the room back to lobby so remaining players aren't stuck.
            if not room._game_abort_handled and room.code in rooms and room.started:
                room.started = False
                room.game = None
                room.game_thread = None
                room.disconnected_at.clear()
                room.cur_bid_history.clear()
                room.cur_log_messages.clear()
                # Remove any seats that are still disconnected — after the
                # abort they have no live socket, and leaving them in the
                # lobby would cause the next game to hang waiting on a player
                # who isn't there. Their close_greenlets are killed too.
                for s in list(room.seats.keys()):
                    if not room.seats[s].get("connected"):
                        room.close_seat(s)
                if room.seats:
                    socketio.emit("game_aborted", {"key": "msg.game_aborted"}, room=room.code)
                    socketio.emit("lobby_update", room.lobby_state(), room=room.code)
                else:
                    rooms.pop(room.code, None)

    room.game_thread = threading.Thread(target=run, daemon=True)
    room.game_thread.start()


# ─── Log and state routing ───────────────────────────────────────────────


def room_log(socketio, room: Room, msg: str, tag: str = "") -> None:
    """Emit a log entry to everyone in the room AND store it on the room
    so it can be replayed to reconnecting players."""
    entry = {"msg": msg, "tag": tag}
    room.cur_log_messages.append(entry)
    socketio.emit("log", entry, room=room.code)


def room_state(socketio, room: Room, event: str, data: dict) -> None:
    send_data = dict(data)

    if event == "deal_done":
        room.cur_roem[:] = [0, 0]
        room.cur_tricks[:] = [0, 0]
        room.cur_round_tricks.clear()
        room.cur_trick_cards.clear()
        room.cur_bid_history.clear()
        room.cur_trump = None
        room.cur_declaring_player = None
        room.cur_declaring_player_idx = None
        room.cur_declaring_team = None
        if room.game:
            for seat, info in room.seats.items():
                sid = info.get("sid")
                if sid is None:
                    continue
                player = room.game.players[seat]
                card_counts = {
                    str(i): len(room.game.players[i].hand)
                    for i in range(4) if i != seat
                }
                socketio.emit("deal_done", {
                    "hand": [c.to_dict() for c in player.hand],
                    "card_counts": card_counts,
                    "my_seat": seat,
                }, to=sid)
        return

    if event == "trump_offered":
        # Record that a new bidding round started with this suit
        room.cur_bid_history.append({
            "kind": "offered",
            "suit": data.get("suit"),
            "round_num": data.get("round_num"),
        })
        socketio.emit(event, send_data, room=room.code)
        return

    if event == "bid":
        # Record each player's pass/declare
        room.cur_bid_history.append({
            "kind": "bid",
            "player_idx": data.get("player_idx"),
            "trump": data.get("trump"),
        })
        socketio.emit(event, send_data, room=room.code)
        return

    if event == "forced_pick":
        room.cur_bid_history.append({
            "kind": "forced_pick",
            "player_idx": data.get("player_idx"),
        })
        socketio.emit(event, send_data, room=room.code)
        return

    if event == "trump_set":
        room.cur_trump = data.get("trump")
        room.cur_declaring_player = data.get("declaring_player")
        room.cur_declaring_player_idx = data.get("declaring_player_idx")
        room.cur_declaring_team = data.get("declaring_team")
        if room.game:
            for seat, info in room.seats.items():
                sid = info.get("sid")
                if sid is None:
                    continue
                player = room.game.players[seat]
                socketio.emit("trump_set", {
                    **send_data,
                    "hand": [c.to_dict() for c in player.hand],
                }, to=sid)
        return

    if event == "trick_played":
        card = data["card"]
        pidx = data["player_idx"]
        room.cur_trick_cards[pidx] = card.to_dict()
        send_data["card"] = card.to_dict()
        if room.game:
            for seat, info in room.seats.items():
                sid = info.get("sid")
                if sid is None:
                    continue
                player = room.game.players[seat]
                card_counts = {
                    str(i): len(room.game.players[i].hand)
                    for i in range(4) if i != seat
                }
                socketio.emit("trick_played", {
                    **send_data,
                    "hand": [c.to_dict() for c in player.hand],
                    "card_counts": card_counts,
                }, to=sid)
        return

    if event == "trick_won":
        winner_idx = data["winner_idx"]
        pts = data["pts"]
        room.cur_round_tricks.append({
            "cards": dict(room.cur_trick_cards),
            "winner_idx": winner_idx,
            "winner_name": room.player_names().get(winner_idx, "?"),
            "pts": pts,
        })
        # Don't clear cur_trick_cards here — cards remain visible until trick_cleared
        room.cur_tricks[:] = list(data["trick_pts"])
        room.cur_roem[:] = list(data["roem_pts"])
        send_data["cur_tricks"] = list(room.cur_tricks)
        send_data["cur_roem"] = list(room.cur_roem)
    elif event == "trick_cleared":
        # Clear trick cards now that the UI animation is complete
        room.cur_trick_cards.clear()
        socketio.emit(event, send_data, room=room.code)
        return
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

    socketio.emit(event, send_data, room=room.code)


# ─── Human input request callbacks ───────────────────────────────────────


def on_move_request(socketio, room: Room, seat_idx: int, legal_cards) -> None:
    info = room.seats.get(seat_idx)
    if info and info.get("connected") and info.get("sid") is not None:
        socketio.emit("request_move", {"legal": [str(c) for c in legal_cards]}, to=info["sid"])


def on_bid_request(socketio, room: Room, seat_idx: int, suit: str, forced: bool) -> None:
    info = room.seats.get(seat_idx)
    if info and info.get("connected") and info.get("sid") is not None:
        leader_idx = room.game._current_leader_idx if room.game else None
        leader_name = room.game.players[leader_idx].name if leader_idx is not None else None
        socketio.emit("request_bid", {
            "suit": suit,
            "suit_name": SUIT_NAMES[suit],
            "forced": forced,
            "leader_idx": leader_idx,
            "leader_name": leader_name,
        }, to=info["sid"])


def on_forced_suit_request(socketio, room: Room, seat_idx: int) -> None:
    info = room.seats.get(seat_idx)
    if info and info.get("connected") and info.get("sid") is not None:
        socketio.emit("request_forced_suit", {}, to=info["sid"])
