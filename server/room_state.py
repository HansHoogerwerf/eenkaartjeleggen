import random
import string
import threading
import time
from collections import deque

from config import CONFIG
from main import KlaverjasGame, SEAT_DEFAULTS, SEAT_TEAMS


class Room:
    """One game room with up to 4 human players.

    Identity model:
      - A seat is identified by its seat index (0..3).
      - A seat's occupant is identified by their *name*, not their Socket.IO sid.
      - The sid is transient — it changes across browser tabs, reconnects, and
        internet drops. The name+code tuple is the stable identity used for
        reconnection.
    """

    def __init__(self, code: str, creator_sid: str, creator_name: str):
        self.code = code

        # seat_idx -> {sid, name, connected, close_greenlet}
        #   sid: current socket id, or None if disconnected
        #   name: stable player name (the reconnect key)
        #   connected: True if a live socket is attached
        #   close_greenlet: the auto-close countdown greenlet, or None
        self.seats: dict[int, dict] = {}
        self.seats[0] = {
            "sid": creator_sid,
            "name": creator_name,
            "connected": True,
            "close_greenlet": None,
        }
        self._seat_join_order: dict[int, int] = {0: 0}
        self._join_counter = 1

        # The host seat (was previously creator_sid). Stable across reconnects.
        self.host_seat: int = 0

        self.game: KlaverjasGame | None = None
        self.game_thread: threading.Thread | None = None
        self.started = False
        self.game_mode: str = CONFIG.room.default_game_mode
        self.score_limit: int = CONFIG.room.default_score_limit
        self.ai_strength: str = CONFIG.room.default_ai_strength
        self.rules_variant: str = CONFIG.room.default_rules_variant
        self.team_names: list[str] = list(CONFIG.room.default_team_names)

        # Per-round state that must be restored on reconnect
        self.cur_round_tricks: list[dict] = []
        self.cur_trick_cards: dict[int, dict] = {}
        self.cur_bid_history: list[dict] = []
        self.cur_log_messages: deque = deque(maxlen=80)
        self.round_history: list[dict] = []
        self.cur_roem = [0, 0]
        self.cur_tricks = [0, 0]
        self.cur_trump: str | None = None
        self.cur_declaring_player: str | None = None
        self.cur_declaring_player_idx: int | None = None
        self.cur_declaring_team: int | None = None

        self.created_at = time.time()
        self.last_activity_at = self.created_at
        self.disconnected_at: dict[int, float] = {}

        self._game_abort_handled = False

    def touch(self) -> None:
        self.last_activity_at = time.time()

    # ─── Seat management ──────────────────────────────────────────────────

    def add_seat(self, seat: int, sid: str, name: str, connected: bool = True) -> None:
        self.seats[seat] = {
            "sid": sid,
            "name": name,
            "connected": connected,
            "close_greenlet": None,
        }
        self._seat_join_order[seat] = self._join_counter
        self._join_counter += 1
        self.touch()

    def close_seat(self, seat: int) -> dict | None:
        """Permanently remove a seat from the room.

        Kills any pending auto-close greenlet. If the closed seat was the host,
        picks a new host. Returns the seat info dict (or None if not present)
        so callers can read the name for emissions.
        """
        info = self.seats.pop(seat, None)
        self._seat_join_order.pop(seat, None)
        self.disconnected_at.pop(seat, None)
        if info is not None:
            g = info.get("close_greenlet")
            if g is not None:
                try:
                    g.kill()
                except Exception:
                    pass
        if seat == self.host_seat:
            self._migrate_host()
        self.touch()
        return info

    def attach_sid(self, seat: int, sid: str) -> None:
        """Bind a new socket to a seat (join or reconnect)."""
        if seat not in self.seats:
            return
        self.seats[seat]["sid"] = sid
        self.seats[seat]["connected"] = True
        self.touch()

    def detach_sid(self, seat: int) -> None:
        """Mark a seat as having no live socket. Used on disconnect."""
        if seat not in self.seats:
            return
        self.seats[seat]["sid"] = None
        self.seats[seat]["connected"] = False
        self.touch()

    def mark_disconnected(self, seat: int) -> None:
        self.disconnected_at[seat] = time.time()
        self.touch()

    def mark_reconnected(self, seat: int) -> None:
        self.disconnected_at.pop(seat, None)
        self.touch()

    def cancel_close_greenlet(self, seat: int) -> None:
        info = self.seats.get(seat)
        if info is None:
            return
        g = info.get("close_greenlet")
        if g is not None:
            try:
                g.kill()
            except Exception:
                pass
            info["close_greenlet"] = None

    # ─── Host management ──────────────────────────────────────────────────

    def host_migration_target(self) -> int | None:
        """Pick the next host: prefer connected seats, oldest join first."""
        if not self.seats:
            return None
        connected = [s for s, info in self.seats.items() if info.get("connected")]
        candidates = connected or list(self.seats.keys())
        return min(candidates, key=lambda s: (self._seat_join_order.get(s, 10_000), s))

    def _migrate_host(self) -> bool:
        """Pick a new host if the current host_seat is gone or disconnected.
        Returns True if host_seat changed."""
        old = self.host_seat
        target = self.host_migration_target()
        if target is None:
            return False
        # Only migrate if current host is missing or disconnected
        cur = self.seats.get(self.host_seat)
        if cur is None or not cur.get("connected"):
            self.host_seat = target
        return self.host_seat != old

    def host_sid(self) -> str | None:
        info = self.seats.get(self.host_seat)
        return info.get("sid") if info else None

    def is_host(self, sid: str) -> bool:
        return self.host_sid() == sid

    # ─── Lookups ──────────────────────────────────────────────────────────

    def next_free_seat(self) -> int | None:
        for i in range(CONFIG.room.seat_count):
            if i not in self.seats:
                return i
        return None

    def seat_for_sid(self, sid: str) -> int | None:
        for seat, info in self.seats.items():
            if info.get("sid") == sid:
                return seat
        return None

    def seat_for_name(self, name: str) -> int | None:
        """Find a seat by player name. This is the primary identity lookup for
        reconnection — a player is identified by (code, name), not by sid."""
        for seat, info in self.seats.items():
            if info.get("name") == name:
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

    # ─── State serialization ──────────────────────────────────────────────

    def lobby_state(self) -> dict:
        self.touch()
        return {
            "code": self.code,
            "host_seat": self.host_seat,
            "seats": {
                str(i): {
                    "name": self.seats[i]["name"] if i in self.seats else None,
                    "team": SEAT_TEAMS[i],
                    "is_human": i in self.seats,
                    "connected": self.seats[i]["connected"] if i in self.seats else False,
                    "disconnected": i in self.seats and not self.seats[i]["connected"],
                }
                for i in range(CONFIG.room.seat_count)
            },
            "started": self.started,
            "ai_strength": self.ai_strength,
            "rules_variant": self.rules_variant,
            "seat_reconnect_timeout_seconds": CONFIG.room.seat_reconnect_timeout_seconds,
        }

    def is_expired(self, now: float, lobby_ttl: int, started_ttl: int) -> bool:
        """Only the global TTL is checked here. Per-seat reconnect timeouts
        are handled by the auto-close greenlet scheduled on disconnect."""
        ttl = started_ttl if self.started else lobby_ttl
        return now - self.last_activity_at > ttl


# ─── Module-level tracking ───────────────────────────────────────────────

# sid -> (room_code, seat_idx). This is the fast-lookup path used by all
# in-game Socket.IO event handlers. Replaces the old sid_to_room dict which
# only knew the code and had to do another O(n) seat lookup per event.
sid_to_seat: dict[str, tuple[str, int]] = {}

rooms: dict[str, Room] = {}


def generate_code() -> str:
    while True:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=CONFIG.room.room_code_length))
        if code not in rooms:
            return code


def cleanup_expired_rooms() -> list[str]:
    now = time.time()
    lobby_ttl = CONFIG.room.lobby_ttl_seconds
    started_ttl = CONFIG.room.started_ttl_seconds
    expired_codes = [
        code for code, room in rooms.items()
        if room.is_expired(now, lobby_ttl, started_ttl)
    ]
    for code in expired_codes:
        room = rooms.pop(code, None)
        if room is None:
            continue
        for info in room.seats.values():
            sid = info.get("sid")
            if sid is not None:
                sid_to_seat.pop(sid, None)
            g = info.get("close_greenlet")
            if g is not None:
                try:
                    g.kill()
                except Exception:
                    pass
    return expired_codes
