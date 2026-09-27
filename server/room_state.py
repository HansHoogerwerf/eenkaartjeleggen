import random
import secrets
import string
import threading
import time
from collections import deque

from config import CONFIG
from main import KlaverjasGame, SEAT_DEFAULTS, SEAT_TEAMS


# Game phases as seen by the reconnect snapshot. The room tracks the phase
# from the engine's state_fn events so a reconnecting client can rebuild the
# right screen (bid dialog, table, next-round banner, game-over overlay).
PHASE_LOBBY = "lobby"
PHASE_BIDDING = "bidding"
PHASE_PLAYING = "playing"
PHASE_BETWEEN_ROUNDS = "between_rounds"
PHASE_GAME_OVER = "game_over"


def _new_token() -> str:
    return secrets.token_urlsafe(16)


class Room:
    """One game room with up to 4 human players.

    Identity model:
      - A seat is identified by its seat index (0..3).
      - A seat's occupant is identified by a server-issued *token* (handed to
        the client on join and on every snapshot) and, as a fallback for a
        client that lost its stored session, by their *name*.
      - The sid is transient — it changes across browser tabs, reconnects, and
        internet drops. A matching token may take a seat over even while an
        old socket still looks connected; a name may only reclaim a seat that
        is currently disconnected.
    """

    def __init__(self, code: str, creator_sid: str, creator_name: str):
        self.code = code

        # seat_idx -> {sid, name, token, connected, close_greenlet}
        #   sid: current socket id, or None if disconnected
        #   name: player name (the fallback reconnect key)
        #   token: secret per-seat reconnect token (the primary reconnect key)
        #   connected: True if a live socket is attached
        #   close_greenlet: the auto-close countdown greenlet, or None
        self.seats: dict[int, dict] = {}
        self.seats[0] = {
            "sid": creator_sid,
            "name": creator_name,
            "token": _new_token(),
            "connected": True,
            "close_greenlet": None,
        }
        self._seat_join_order: dict[int, int] = {0: 0}
        self._join_counter = 1

        # The host seat (was previously creator_sid). Stable across reconnects.
        self.host_seat: int = 0
        # Seat that held the host role before it was handed to a connected
        # player during a mid-game disconnect. When that seat reconnects in
        # time it gets the role back, so a reload does not demote the owner.
        self.host_before_migration: int | None = None

        self.game: KlaverjasGame | None = None
        self.game_thread: threading.Thread | None = None
        self.started = False
        self.phase: str = PHASE_LOBBY
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
        self.cur_offered_suit: str | None = None
        self.cur_trick_leader: int | None = None
        self.cur_declaring_player: str | None = None
        self.cur_declaring_player_idx: int | None = None
        self.cur_declaring_team: int | None = None
        # Result of the last finished round (shown on the next-round banner)
        # and of the whole game (shown on the game-over overlay).
        self.last_round_result: dict | None = None
        self.game_result: dict | None = None

        self.created_at = time.time()
        self.last_activity_at = self.created_at
        self.disconnected_at: dict[int, float] = {}
        # seat -> absolute time at which the auto-close fires. The host can
        # push this back ("wait longer") without re-spawning the greenlet.
        self.reconnect_deadline: dict[int, float] = {}

        self._game_abort_handled = False

    def touch(self) -> None:
        self.last_activity_at = time.time()

    # ─── State resets ─────────────────────────────────────────────────────

    def reset_round_state(self) -> None:
        """Forget everything about the current round (new deal)."""
        self.cur_roem[:] = [0, 0]
        self.cur_tricks[:] = [0, 0]
        self.cur_round_tricks.clear()
        self.cur_trick_cards.clear()
        self.cur_bid_history.clear()
        self.cur_trump = None
        self.cur_offered_suit = None
        self.cur_trick_leader = None
        self.cur_declaring_player = None
        self.cur_declaring_player_idx = None
        self.cur_declaring_team = None

    def reset_game_state(self) -> None:
        """Forget everything about the current game (new game / abort)."""
        self.reset_round_state()
        self.round_history.clear()
        self.cur_log_messages.clear()
        self.last_round_result = None
        self.game_result = None

    # ─── Seat management ──────────────────────────────────────────────────

    def add_seat(self, seat: int, sid: str, name: str, connected: bool = True) -> None:
        self.seats[seat] = {
            "sid": sid,
            "name": name,
            "token": _new_token(),
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
        self.reconnect_deadline.pop(seat, None)
        if self.host_before_migration == seat:
            self.host_before_migration = None
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

    def mark_disconnected(self, seat: int, timeout: float | None = None) -> None:
        now = time.time()
        self.disconnected_at[seat] = now
        if timeout is not None:
            self.reconnect_deadline[seat] = now + timeout
        self.touch()

    def mark_reconnected(self, seat: int) -> None:
        self.disconnected_at.pop(seat, None)
        self.reconnect_deadline.pop(seat, None)
        self.touch()

    def extend_deadline(self, seat: int, seconds: float) -> float | None:
        """Push a disconnected seat's auto-close deadline back. Returns the
        new remaining time in seconds, or None if the seat is not waiting."""
        if seat not in self.reconnect_deadline:
            return None
        self.reconnect_deadline[seat] = max(self.reconnect_deadline[seat], time.time()) + seconds
        self.touch()
        return self.seconds_remaining(seat)

    def seconds_remaining(self, seat: int, now: float | None = None) -> float | None:
        deadline = self.reconnect_deadline.get(seat)
        if deadline is None:
            return None
        return max(0.0, deadline - (time.time() if now is None else now))

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

    def migrate_host_away_from(self, seat: int) -> bool:
        """Hand the host role to a connected seat because `seat` dropped.
        Remembers the original holder so `restore_host` can give it back.
        Returns True if host_seat changed."""
        migrated = self._migrate_host()
        if migrated and self.host_before_migration is None:
            self.host_before_migration = seat
        return migrated

    def restore_host(self, seat: int) -> bool:
        """Give the host role back to a seat that lost it while disconnected.
        Returns True if host_seat changed."""
        if self.host_before_migration != seat:
            return False
        self.host_before_migration = None
        info = self.seats.get(seat)
        if info is None or not info.get("connected") or self.host_seat == seat:
            return False
        self.host_seat = seat
        self.touch()
        return True

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
        """Find a seat by player name. This is the fallback identity lookup
        for reconnection without a stored token (new device, cleared
        storage); it may only reclaim a seat that is currently disconnected."""
        for seat, info in self.seats.items():
            if info.get("name") == name:
                return seat
        return None

    def seat_for_token(self, token: str | None) -> int | None:
        """Find a seat by its reconnect token (the primary identity)."""
        if not token:
            return None
        for seat, info in self.seats.items():
            if info.get("token") and secrets.compare_digest(info["token"], token):
                return seat
        return None

    def seat_token(self, seat: int) -> str | None:
        info = self.seats.get(seat)
        return info.get("token") if info else None

    def name_in_use(self, name: str, exclude_seat: int | None = None) -> bool:
        """True if a *connected* seat already uses this name."""
        for seat, info in self.seats.items():
            if seat == exclude_seat:
                continue
            if info.get("name") == name and info.get("connected"):
                return True
        return False

    def player_names(self) -> dict[int, str]:
        names = {}
        for i in range(CONFIG.room.seat_count):
            if i in self.seats:
                names[i] = self.seats[i]["name"]
            else:
                names[i] = f"AI {SEAT_DEFAULTS[i]}"
        return names

    def disconnected_seats(self, now: float | None = None) -> dict[int, dict]:
        """Seats currently waiting to reconnect, with the remaining grace."""
        now = time.time() if now is None else now
        out = {}
        for seat, info in self.seats.items():
            if info.get("connected"):
                continue
            out[seat] = {
                "name": info.get("name"),
                "seconds_remaining": self.seconds_remaining(seat, now),
            }
        return out

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
            "lobby_reconnect_timeout_seconds": CONFIG.room.lobby_reconnect_timeout_seconds,
        }

    def is_expired(self, now: float, lobby_ttl: int, started_ttl: int) -> bool:
        """Only the global TTL is checked here. Per-seat reconnect timeouts
        are handled by the auto-close greenlet scheduled on disconnect."""
        ttl = started_ttl if self.started else lobby_ttl
        return now - self.last_activity_at > ttl


# ─── Module-level tracking ───────────────────────────────────────────────

# sid -> (room_code, seat_idx). This is the fast-lookup path used by all
# in-game Socket.IO event handlers. A sid that is not in here (never joined,
# left, or was superseded by a token takeover) is ignored on disconnect.
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
