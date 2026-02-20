import random
import string
import threading

from main import KlaverjasGame, SEAT_DEFAULTS, SEAT_TEAMS


class Room:
    """One game room with up to 4 human players."""

    def __init__(self, code: str, creator_sid: str, creator_name: str):
        self.code = code
        self.creator_sid = creator_sid

        # seat_idx -> {sid, name, connected}
        self.seats: dict[int, dict] = {}
        self.seats[0] = {"sid": creator_sid, "name": creator_name, "connected": True}

        self.game: KlaverjasGame | None = None
        self.game_thread: threading.Thread | None = None
        self.started = False
        self.game_mode: str = "score_limit"
        self.score_limit: int = 500
        self.ai_strength: str = "expert"
        self.team_names: list[str] = ["Team 0", "Team 1"]

        self.cur_round_tricks: list[dict] = []
        self.cur_trick_cards: dict[int, dict] = {}
        self.round_history: list[dict] = []
        self.cur_roem = [0, 0]
        self.cur_tricks = [0, 0]
        self.cur_trump: str | None = None
        self.cur_declaring_player: str | None = None
        self.cur_declaring_player_idx: int | None = None
        self.cur_declaring_team: int | None = None

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
        names = {}
        for i in range(4):
            if i in self.seats:
                names[i] = self.seats[i]["name"]
            else:
                names[i] = f"AI {SEAT_DEFAULTS[i]}"
        return names

    def lobby_state(self) -> dict:
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
            "ai_strength": self.ai_strength,
        }


# sid -> room code (for fast lookup on disconnect)
sid_to_room: dict[str, str] = {}
rooms: dict[str, Room] = {}


def generate_code() -> str:
    while True:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
        if code not in rooms:
            return code
