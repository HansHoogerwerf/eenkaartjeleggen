"""Game-flow helpers: snapshot building, reconnect application, event routing.

This module owns the in-game state that the reconnect snapshot needs, and
provides the `apply_reconnect` and `schedule_seat_auto_close` primitives used
by the app layer.
"""

import threading
import time

import gevent

from config import CONFIG
from main import GameInterrupt, HumanPlayer, KlaverjasGame, SUIT_NAMES
from server.room_state import (
    PHASE_BETWEEN_ROUNDS,
    PHASE_BIDDING,
    PHASE_GAME_OVER,
    PHASE_LOBBY,
    PHASE_PLAYING,
    Room,
    rooms,
)


# ─── Snapshot building ───────────────────────────────────────────────────


def _current_turn(room: Room) -> int | None:
    """Seat whose card is awaited in the current trick, or None when the
    trick is complete (waiting for trick_cleared) or no trick is running."""
    if room.phase != PHASE_PLAYING or room.cur_trick_leader is None:
        return None
    played = len(room.cur_trick_cards)
    if played >= 4:
        return None
    return (room.cur_trick_leader + played) % 4


def build_game_state_snapshot(room: Room, seat: int) -> dict:
    """Assemble the complete game-state snapshot for a reconnecting player.

    This is the single authoritative source for "what the player needs to
    rebuild their UI from scratch". The client's `game_state_snapshot`
    handler pulls everything it needs from this dict and does not rely on
    any follow-up events. `phase` tells the client which screen to rebuild:
    the bid dialog, the table, the next-round banner or the game-over overlay.
    """
    snapshot = {
        "seat": seat,
        "code": room.code,
        "token": room.seat_token(seat),
        "is_host": room.host_seat == seat,
        "host_seat": room.host_seat,
        "player_names": room.player_names(),
        "team_names": list(room.team_names),
        "phase": room.phase,
        "scores": list(room.game.scores) if room.game else [0, 0],
        "cur_tricks": list(room.cur_tricks),
        "cur_roem": list(room.cur_roem),
        "trump": room.cur_trump,
        "offered_suit": room.cur_offered_suit,
        "declaring_player": room.cur_declaring_player,
        "declaring_player_idx": room.cur_declaring_player_idx,
        "declaring_team": room.cur_declaring_team,
        "trick_cards": {str(k): v for k, v in room.cur_trick_cards.items()},
        "trick_leader": room.cur_trick_leader,
        "current_turn": _current_turn(room),
        "trick_history": list(room.cur_round_tricks),
        "bid_history": list(room.cur_bid_history),
        "log_messages": list(room.cur_log_messages),
        "last_round_result": dict(room.last_round_result) if room.last_round_result else None,
        "game_result": dict(room.game_result) if room.game_result else None,
        "seat_reconnect_timeout_seconds": CONFIG.room.seat_reconnect_timeout_seconds,
        "reconnect_extend_seconds": CONFIG.room.reconnect_extend_seconds,
        "seats_status": {
            str(i): {
                "name": room.seats[i]["name"] if i in room.seats else None,
                "is_human": i in room.seats,
                "connected": room.seats[i]["connected"] if i in room.seats else False,
            }
            for i in range(4)
        },
        "disconnected_seats": {
            str(s): info for s, info in room.disconnected_seats().items() if s != seat
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
            pending = player.get_pending_request()
            if pending and pending.get("type") == "bid":
                leader_idx = getattr(room.game, "_current_leader_idx", None)
                pending["leader_idx"] = leader_idx
                pending["leader_name"] = (
                    room.game.players[leader_idx].name if leader_idx is not None else None
                )
            snapshot["pending_request"] = pending
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


def apply_reconnect(socketio, emit_fn, join_room_fn, room: Room, seat: int, new_sid: str,
                    announce: bool = True) -> None:
    """Rebind a seat to a new sid and push the right resume payload to the client.

    Cancels any pending auto-abort greenlet and updates the seat's sid. Then:
      • If the room has an active game → emit `game_state_snapshot` and tell
        the HumanPlayer it's reconnected so the game thread can resume.
      • Otherwise (lobby state, e.g. the game was aborted while this seat was
        disconnected) → emit `room_joined` so the client lands in the lobby.
    With `announce`, broadcasts `seat_reconnected` so other players hide the
    paused overlay. A same-socket resync or a token takeover of a seat that
    never looked disconnected passes `announce=False` to keep the log quiet.
    """
    if seat not in room.seats:
        return

    room.cancel_close_greenlet(seat)
    room.attach_sid(seat, new_sid)
    room.mark_reconnected(seat)
    host_restored = room.restore_host(seat)
    # Somebody is connected and waiting from now on: any other seat that
    # was paused without a countdown gets one (reflected in the snapshot).
    resume_countdowns(socketio, room, except_seat=seat)

    join_room_fn(room.code)

    if host_restored:
        socketio.emit("host_migrated", {
            "seat": room.host_seat,
            "name": room.seats[seat]["name"],
        }, room=room.code)

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
            "token": room.seat_token(seat),
            "lobby": room.lobby_state(),
            "is_host": room.host_seat == seat,
        })

    if announce:
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


def grace_timeout_for(room: Room) -> int:
    """Seconds a dropped seat is held for, depending on the room phase."""
    if room.started:
        return CONFIG.room.seat_reconnect_timeout_seconds
    return CONFIG.room.lobby_reconnect_timeout_seconds


def schedule_seat_auto_close(socketio, room: Room, seat: int, timeout: float | None = None) -> None:
    """Spawn an auto-close countdown for a disconnected seat.

    Mid-game: if the seat does not reconnect before its deadline, the entire
    game is aborted (rather than the seat being closed and the game
    continuing in a broken state). In the lobby: the seat is freed.

    The countdown is driven by `room.reconnect_deadline[seat]`, so the host
    can push the deadline back with `extend_seat_grace` while the greenlet
    is sleeping; the greenlet re-checks the deadline whenever it wakes.
    """
    if seat not in room.seats:
        return
    if timeout is None:
        timeout = grace_timeout_for(room)
    if seat not in room.reconnect_deadline:
        room.reconnect_deadline[seat] = time.time() + timeout

    def _worker():
        try:
            while True:
                remaining = room.seconds_remaining(seat)
                if remaining is None or remaining <= 0:
                    break
                gevent.sleep(max(remaining, 0.001))
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
        if room.connected_human_count() == 0:
            # Nobody is waiting for this player: keep the seat and let the
            # room pause (it idles out via the room TTL). The countdown is
            # restarted by resume_countdowns when somebody returns.
            room.reconnect_deadline.pop(seat, None)
            return

        if room.started and room.game is not None:
            abort_game(socketio, room)
        else:
            # Game wasn't started (lobby disconnect) — just clean up the seat
            old_host = room.host_seat
            room.close_seat(seat)
            if not room.seats:
                rooms.pop(room.code, None)
                return
            socketio.emit("lobby_update", room.lobby_state(), room=room.code)
            if room.host_seat != old_host:
                new_host_info = room.seats.get(room.host_seat)
                if new_host_info is not None:
                    socketio.emit("host_migrated", {
                        "seat": room.host_seat,
                        "name": new_host_info.get("name"),
                    }, room=room.code)

    g = gevent.spawn(_worker)
    room.seats[seat]["close_greenlet"] = g


def resume_countdowns(socketio, room: Room, except_seat: int | None = None) -> None:
    """Start the reconnect countdown for every disconnected seat that has
    none. Called when a player (re)connects: from then on somebody is
    waiting, so the absent players' grace periods run. Seats that already
    count down are left alone."""
    timeout = grace_timeout_for(room)
    for seat, info in room.seats.items():
        if seat == except_seat or info.get("connected"):
            continue
        if seat in room.reconnect_deadline or info.get("close_greenlet") is not None:
            continue
        schedule_seat_auto_close(socketio, room, seat, timeout)


def extend_seat_grace(room: Room, seat: int, seconds: float) -> float | None:
    """Give a disconnected seat more time. Returns the new remaining seconds
    or None if the seat is not waiting to reconnect."""
    return room.extend_deadline(seat, seconds)


# ─── Game start / lifecycle ──────────────────────────────────────────────


def start_room_game(socketio, room: Room) -> None:
    room.started = True
    room._game_abort_handled = False
    room.reset_game_state()
    room.phase = PHASE_BIDDING

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
            "token": info.get("token"),
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
                room.phase = PHASE_LOBBY
                room.game = None
                room.game_thread = None
                room.disconnected_at.clear()
                room.reconnect_deadline.clear()
                room.host_before_migration = None
                room.reset_game_state()
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
        room.reset_round_state()
        room.phase = PHASE_BIDDING
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
        room.cur_offered_suit = data.get("suit")
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
        room.phase = PHASE_PLAYING
        room.cur_trump = data.get("trump")
        room.cur_offered_suit = None
        room.cur_trick_leader = data.get("leader_idx")
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
        room.cur_trick_leader = data.get("next_leader", room.cur_trick_leader)
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
        room.last_round_result = {
            "t0": data.get("t0"),
            "t1": data.get("t1"),
            "scores": list(data.get("scores", [])),
            "round_num": data.get("round_num"),
            "total_rounds": data.get("total_rounds"),
        }
    elif event == "waiting_for_host":
        room.phase = PHASE_BETWEEN_ROUNDS
    elif event == "game_over":
        room.phase = PHASE_GAME_OVER
        room.game_result = {
            "winner": data.get("winner"),
            "scores": list(data.get("scores", [])),
        }

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
