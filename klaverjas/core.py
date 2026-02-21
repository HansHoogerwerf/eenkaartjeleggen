import random
from typing import TYPE_CHECKING

from klaverjas.constants import (
    NON_TRUMP_ORDER,
    NON_TRUMP_PTS,
    RANKS,
    SEQUENCE_ORDER,
    SUITS,
    TRUMP_ORDER,
    TRUMP_PTS,
)

if TYPE_CHECKING:
    from main import Player


class Card:
    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank

    def points(self, trump: str) -> int:
        return TRUMP_PTS[self.rank] if self.suit == trump else NON_TRUMP_PTS[self.rank]

    def strength(self, trump: str) -> int:
        if self.suit == trump:
            return TRUMP_ORDER.index(self.rank)
        return NON_TRUMP_ORDER.index(self.rank)

    def to_dict(self) -> dict:
        return {"suit": self.suit, "rank": self.rank}

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return str(self)


class Deck:
    def __init__(self, rng: random.Random | None = None):
        self.cards = [Card(s, r) for s in SUITS for r in RANKS]
        if rng is None:
            random.shuffle(self.cards)
        else:
            rng.shuffle(self.cards)

    def deal(self, n_players: int, per_player: int) -> list[list[Card]]:
        hands: list[list[Card]] = [[] for _ in range(n_players)]
        for i in range(n_players * per_player):
            hands[i % n_players].append(self.cards[i])
        return hands


Trick = list[tuple["Player", Card]]


def trick_winner_index(trick: Trick, trump: str) -> int:
    best = 0
    best_card = trick[0][1]
    for i, (_, card) in enumerate(trick[1:], 1):
        if card.suit == trump and best_card.suit != trump:
            best, best_card = i, card
        elif card.suit == best_card.suit:
            if card.strength(trump) > best_card.strength(trump):
                best, best_card = i, card
    return best


def find_roem(hand: list[Card], trump: str) -> list[tuple[str, int]]:
    roem: list[tuple[str, int]] = []

    trump_ranks = {c.rank for c in hand if c.suit == trump}
    if "K" in trump_ranks and "Q" in trump_ranks:
        roem.append(("Stuk (K+Q trump)", 20))

    for suit in SUITS:
        indices = sorted(SEQUENCE_ORDER.index(c.rank) for c in hand if c.suit == suit)
        if len(indices) < 3:
            continue
        run_start = 0
        for i in range(1, len(indices) + 1):
            end_of_run = i == len(indices) or indices[i] != indices[i - 1] + 1
            if end_of_run:
                run_len = i - run_start
                if run_len >= 3:
                    lo = SEQUENCE_ORDER[indices[run_start]]
                    hi = SEQUENCE_ORDER[indices[i - 1]]
                    pts = 100 if run_len >= 5 else (50 if run_len == 4 else 20)
                    roem.append((f"Sequence {run_len} ({lo}–{hi}{suit})", pts))
                run_start = i

    for rank in RANKS:
        if sum(1 for c in hand if c.rank == rank) == 4:
            roem.append(("Four Jacks", 200) if rank == "J" else (f"Four {rank}s", 100))

    return roem
