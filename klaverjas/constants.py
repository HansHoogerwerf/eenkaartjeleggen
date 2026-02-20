SUITS = ["♣", "♦", "♥", "♠"]
SUIT_NAMES = {"♣": "Clubs", "♦": "Diamonds", "♥": "Hearts", "♠": "Spades"}
RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]

NON_TRUMP_PTS = {"A": 11, "10": 10, "K": 4, "Q": 3, "J": 2, "9": 0, "8": 0, "7": 0}
TRUMP_PTS = {"J": 20, "9": 14, "A": 11, "10": 10, "K": 4, "Q": 3, "8": 0, "7": 0}

NON_TRUMP_ORDER = ["7", "8", "9", "J", "Q", "K", "10", "A"]
TRUMP_ORDER = ["7", "8", "Q", "K", "10", "A", "9", "J"]
TRUMP_STRONGEST_FIRST = list(reversed(TRUMP_ORDER))
SEQUENCE_ORDER = ["7", "8", "9", "10", "J", "Q", "K", "A"]

WIN_SCORE = 500
RED_SUITS = {"♦", "♥"}
TRICK_CARD_TOTAL = 162

AI_PLAY_DELAY = 0.7
AI_BID_DELAY = 2.0
TRICK_CLEAR_DELAY = 1.4

SEAT_DEFAULTS = {0: "South", 1: "West", 2: "North", 3: "East"}
SEAT_TEAMS = {0: 0, 1: 1, 2: 0, 3: 1}

