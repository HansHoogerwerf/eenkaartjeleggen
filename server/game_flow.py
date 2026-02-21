import threading

from main import GameInterrupt, HumanPlayer, KlaverjasGame, SUIT_NAMES
from server.room_state import Room, rooms, sid_to_room


def leave_current_room(socketio, leave_room, sid: str) -> None:
    code = sid_to_room.pop(sid, None)
    if not code or code not in rooms:
        return
    room = rooms[code]
    seat = room.seat_for_sid(sid)
    if seat is not None and not room.started:
        room.remove_seat(seat)
    leave_room(code)

    if not room.seats:
        rooms.pop(code, None)
    else:
        if room.creator_sid == sid and not room.started:
            host_seat = room.host_migration_target()
            if host_seat is not None:
                room.creator_sid = room.seats[host_seat]["sid"]
                socketio.emit("host_migrated", {"seat": host_seat, "name": room.seats[host_seat]["name"]}, room=code)
        socketio.emit("lobby_update", room.lobby_state(), room=code)


def reconnect_player(socketio, emit, join_room, room: Room, seat: int, new_sid: str) -> None:
    old_sid = room.seats[seat]["sid"]
    room.seats[seat]["sid"] = new_sid
    room.seats[seat]["connected"] = True
    room.mark_reconnected(seat)
    sid_to_room[new_sid] = room.code

    if room.creator_sid == old_sid:
        room.creator_sid = new_sid

    join_room(room.code)

    state = {
        "seat": seat,
        "code": room.code,
        "scores": list(room.game.scores) if room.game else [0, 0],
        "player_names": room.player_names(),
        "cur_tricks": list(room.cur_tricks),
        "cur_roem": list(room.cur_roem),
        "is_creator": room.creator_sid == new_sid,
        "team_names": room.team_names,
        "trump": room.cur_trump,
        "declaring_player": room.cur_declaring_player,
        "declaring_player_idx": room.cur_declaring_player_idx,
        "declaring_team": room.cur_declaring_team,
        "trick_cards": dict(room.cur_trick_cards),
    }
    if room.game:
        p = room.game.players[seat]
        state["hand"] = [c.to_dict() for c in p.hand]
        state["card_counts"] = {
            str(i): len(room.game.players[i].hand) for i in range(4) if i != seat
        }

    emit("reconnected", state)

    player = room.game.players[seat] if room.game else None
    if isinstance(player, HumanPlayer):
        player.set_reconnected()

    socketio.emit("player_reconnected", {
        "seat": seat,
        "name": room.seats[seat]["name"],
    }, room=room.code)


def start_room_game(socketio, room: Room) -> None:
    room.started = True
    room.round_history.clear()
    room.cur_round_tricks.clear()
    room.cur_trick_cards.clear()
    room.cur_roem[:] = [0, 0]
    room.cur_tricks[:] = [0, 0]

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
            player.disconnect_timeout = room.reconnect_timeout_seconds
            player._on_move_request = lambda seat_idx, legal, r=room: on_move_request(socketio, r, seat_idx, legal)
            player._on_bid_request = lambda seat_idx, suit, forced, r=room: on_bid_request(socketio, r, seat_idx, suit, forced)
            player._on_disconnect_pause = lambda seat_idx, r=room: on_disconnect_pause(socketio, r, seat_idx)
            player.reset_interrupt()

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


def room_log(socketio, room: Room, msg: str, tag: str = "") -> None:
    socketio.emit("log", {"msg": msg, "tag": tag}, room=room.code)


def room_state(socketio, room: Room, event: str, data: dict) -> None:
    send_data = dict(data)

    if event == "deal_done":
        room.cur_roem[:] = [0, 0]
        room.cur_tricks[:] = [0, 0]
        room.cur_round_tricks.clear()
        room.cur_trick_cards.clear()
        room.cur_trump = None
        room.cur_declaring_player = None
        room.cur_declaring_player_idx = None
        room.cur_declaring_team = None
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
        return

    if event == "trump_set":
        room.cur_trump = data.get("trump")
        room.cur_declaring_player = data.get("declaring_player")
        room.cur_declaring_player_idx = data.get("declaring_player_idx")
        room.cur_declaring_team = data.get("declaring_team")
        if room.game:
            for seat, info in room.seats.items():
                player = room.game.players[seat]
                socketio.emit("trump_set", {
                    **send_data,
                    "hand": [c.to_dict() for c in player.hand],
                }, to=info["sid"])
        return

    if event == "trick_played":
        card = data["card"]
        pidx = data["player_idx"]
        room.cur_trick_cards[pidx] = card.to_dict()
        send_data["card"] = card.to_dict()
        if room.game:
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

    if event == "trick_won":
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

    socketio.emit(event, send_data, room=room.code)


def on_move_request(socketio, room: Room, seat_idx: int, legal_cards) -> None:
    info = room.seats.get(seat_idx)
    if info and info["connected"]:
        socketio.emit("request_move", {"legal": [str(c) for c in legal_cards]}, to=info["sid"])


def on_bid_request(socketio, room: Room, seat_idx: int, suit: str, forced: bool) -> None:
    info = room.seats.get(seat_idx)
    if info and info["connected"]:
        socketio.emit("request_bid", {
            "suit": suit,
            "suit_name": SUIT_NAMES[suit],
            "forced": forced,
        }, to=info["sid"])


def on_disconnect_pause(socketio, room: Room, seat_idx: int) -> None:
    name = room.seats.get(seat_idx, {}).get("name", "?")
    socketio.emit("game_paused", {
        "seat": seat_idx,
        "name": name,
        "key": "error.waiting_reconnect",
    }, room=room.code)
