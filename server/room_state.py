import random
import string
import threading
import time
import os

from config import CONFIG
from main import KlaverjasGame, SEAT_DEFAULTS, SEAT_TEAMS


class Room:
    """One game room with up to 4 human players."""

    def __init__(self, code: str, creator_sid: str, creator_name: str):
        self.code = code
        self.creator_sid = creator_sid

        # seat_idx -> {sid, name, connected}
        self.seats: dict[int, dict] = {}
        self.seats[0] = {"sid": creator_sid, "name": creator_name, "connected": True}
        self._seat_join_order: dict[int, int] = {0: 0}
        self._join_counter = 1

        self.game: KlaverjasGame | None = None
        self.game_thread: threading.Thread | None = None
        self.started = False
        self.game_mode: str = CONFIG.room.default_game_mode
        self.score_limit: int = CONFIG.room.default_score_limit
        self.ai_strength: str = CONFIG.room.default_ai_strength
        self.rules_variant: str = CONFIG.room.default_rules_variant
        self.team_names: list[str] = list(CONFIG.room.default_team_names)

        self.cur_round_tricks: list[dict] = []
        self.cur_trick_cards: dict[int, dict] = {}
        self.round_history: list[dict] = []
        self.cur_roem = [0, 0]
        self.cur_tricks = [0, 0]
        self.cur_trump: str | None = None
        self.cur_declaring_player: str | None = None
        self.cur_declaring_player_idx: int | None = None
        self.cur_declaring_team: int | None = None
        self.created_at = time.time()
        self.last_activity_at = self.created_at
        self.reconnect_timeout_seconds = int(os.environ.get("ROOM_RECONNECT_TIMEOUT_SECONDS", "300"))
        self.disconnected_at: dict[int, float] = {}

    def touch(self) -> None:
        self.last_activity_at = time.time()

    def add_seat(self, seat: int, sid: str, name: str, connected: bool = True) -> None:
        self.seats[seat] = {"sid": sid, "name": name, "connected": connected}
        self._seat_join_order[seat] = self._join_counter
        self._join_counter += 1
        self.touch()

    def remove_seat(self, seat: int) -> None:
        self.seats.pop(seat, None)
        self._seat_join_order.pop(seat, None)
        self.disconnected_at.pop(seat, None)
        self.touch()

    def mark_disconnected(self, seat: int) -> None:
        self.disconnected_at[seat] = time.time()
        self.touch()

    def mark_reconnected(self, seat: int) -> None:
        self.disconnected_at.pop(seat, None)
        self.touch()

    def host_migration_target(self) -> int | None:
        if not self.seats:
            return None
        connected = [s for s, info in self.seats.items() if info.get("connected")]
        candidates = connected or list(self.seats.keys())
        return min(candidates, key=lambda s: (self._seat_join_order.get(s, 10_000), s))

    def next_free_seat(self) -> int | None:
        for i in range(CONFIG.room.seat_count):
            if i not in self.seats:
                return i
        return None

    def seat_for_sid(self, sid: str) -> int | None:
        for seat, info in self.seats.items():
            if info["sid"] == sid:
                return seat
        return None

    def player_names(self) -> dict[int, str]:
        names = {}
        for i in range(CONFIG.room.seat_count):
            if i in self.seats:
                names[i] = self.seats[i]["name"]
            else:
                names[i] = f"AI {SEAT_DEFAULTS[i]}"
        return names

    def lobby_state(self) -> dict:
        self.touch()
        return {
            "code": self.code,
            "seats": {
                str(i): {
                    "name": self.seats[i]["name"] if i in self.seats else None,
                    "team": SEAT_TEAMS[i],
                    "is_human": i in self.seats,
                }
                for i in range(CONFIG.room.seat_count)
            },
            "started": self.started,
            "ai_strength": self.ai_strength,
            "rules_variant": self.rules_variant,
            "reconnect_timeout_seconds": self.reconnect_timeout_seconds,
        }

    def is_expired(self, now: float, lobby_ttl: int, started_ttl: int) -> bool:
        ttl = started_ttl if self.started else lobby_ttl
        if now - self.last_activity_at > ttl:
            return True
        # Only expire on disconnect if ALL human players are disconnected
        # beyond the reconnect timeout (don't kill a game because one player dropped)
        if self.started and self.disconnected_at:
            all_humans_disconnected = all(
                seat in self.disconnected_at
                and now - self.disconnected_at[seat] > self.reconnect_timeout_seconds
                for seat in self.seats
            )
            if all_humans_disconnected:
                return True
        return False


# sid -> room code (for fast lookup on disconnect)
sid_to_room: dict[str, str] = {}
rooms: dict[str, Room] = {}

ROOM_LOBBY_TTL_SECONDS = int(os.environ.get("ROOM_LOBBY_TTL_SECONDS", "3600"))
ROOM_STARTED_TTL_SECONDS = int(os.environ.get("ROOM_STARTED_TTL_SECONDS", "21600"))


def generate_code() -> str:
    while True:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=CONFIG.room.room_code_length))
        if code not in rooms:
            return code


def cleanup_expired_rooms() -> list[str]:
    now = time.time()
    expired_codes = [
        code for code, room in rooms.items()
        if room.is_expired(now, ROOM_LOBBY_TTL_SECONDS, ROOM_STARTED_TTL_SECONDS)
    ]
    for code in expired_codes:
        room = rooms.pop(code, None)
        if room is None:
            continue
        for info in room.seats.values():
            sid_to_room.pop(info.get("sid"), None)
    return expired_codes
