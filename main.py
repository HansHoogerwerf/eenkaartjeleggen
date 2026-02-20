"""
Klaverjassen — Rotterdam rules, 1 human vs 3 AI  (game engine)

Teams:  Team 0 = You (South) + AI North
        Team 1 = AI West  + AI East

Rotterdam rules (vs Amsterdam):
  - You MUST trump even when your partner is winning the trick
  - You MUST overtrump in all situations
  - Bidding: a random suit is offered; players decide in turn (round 1);
    if all pass, a second random suit is offered (round 2);
    if all pass again, the first player is forced to declare the round-2 suit
  - Nat (pit): based on trick points only — must score > 81 trick pts
    If nat: declarer scores 0; opponents receive all 162 + all roem

Roem (honours) scored at start of each round:
  Stuk (K+Q of trump)        20 pts
  Sequence of 3 (same suit)  20 pts
  Sequence of 4 (same suit)  50 pts
  Sequence of 5+ (same suit) 100 pts
  Four of a kind (any rank)  100 pts
  Four Jacks                 200 pts

First team to reach 500 points wins.
"""

import random
import threading
import time
from abc import ABC, abstractmethod

# ─── Constants ────────────────────────────────────────────────────────────────

SUITS = ["♣", "♦", "♥", "♠"]
SUIT_NAMES = {"♣": "Clubs", "♦": "Diamonds", "♥": "Hearts", "♠": "Spades"}
RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]

NON_TRUMP_PTS = {"A": 11, "10": 10, "K": 4, "Q": 3, "J": 2, "9": 0, "8": 0, "7": 0}
TRUMP_PTS     = {"J": 20, "9": 14, "A": 11, "10": 10, "K": 4, "Q": 3, "8": 0, "7": 0}

NON_TRUMP_ORDER      = ["7", "8", "9", "J", "Q", "K", "10", "A"]
TRUMP_ORDER          = ["7", "8", "Q", "K", "10", "A", "9", "J"]
TRUMP_STRONGEST_FIRST = list(reversed(TRUMP_ORDER))   # ["J","9","A","10","K","Q","8","7"]
SEQUENCE_ORDER       = ["7", "8", "9", "10", "J", "Q", "K", "A"]

WIN_SCORE = 500
RED_SUITS = {"♦", "♥"}
TRICK_CARD_TOTAL = 162   # fixed: 152 card pts + 10 last-trick bonus

AI_PLAY_DELAY     = 1.5   # delay between AI moves to make them easier to follow (can be set to 0 for fast autoplay)
AI_BID_DELAY      = 2.0   # deliberate pause so players can read bid badges
TRICK_CLEAR_DELAY = 1.4


# ─── Exceptions ───────────────────────────────────────────────────────────────

class GameInterrupt(Exception):
    """Raised in the game thread when a restart or window-close is requested."""


# ─── Card ─────────────────────────────────────────────────────────────────────

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


# ─── Deck ─────────────────────────────────────────────────────────────────────

class Deck:
    def __init__(self):
        self.cards = [Card(s, r) for s in SUITS for r in RANKS]
        random.shuffle(self.cards)

    def deal(self, n_players: int, per_player: int) -> list[list[Card]]:
        hands: list[list[Card]] = [[] for _ in range(n_players)]
        for i in range(n_players * per_player):
            hands[i % n_players].append(self.cards[i])
        return hands


# ─── Trick helpers ────────────────────────────────────────────────────────────

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


# ─── Roem (honours) ──────────────────────────────────────────────────────────

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


# ─── Player (base) ────────────────────────────────────────────────────────────

class Player(ABC):
    def __init__(self, name: str, team: int):
        self.name = name
        self.team = team
        self.hand: list[Card] = []
        self.played_cards: set[str] = set()   # all cards seen this round

    def observe_card(self, card: Card) -> None:
        """Record a card that has been played this round (called for every card played)."""
        self.played_cards.add(str(card))

    def start_round(self) -> None:
        """Reset per-round state."""
        self.played_cards.clear()

    def receive_hand(self, cards: list[Card]) -> None:
        self.hand = sorted(cards, key=lambda c: (SUITS.index(c.suit), RANKS.index(c.rank)))

    def legal_moves(self, trick: Trick, trump: str) -> list[Card]:
        """Rotterdam rules: must trump and overtrump in all cases."""
        if not trick:
            return list(self.hand)

        lead_suit = trick[0][1].suit
        same_suit = [c for c in self.hand if c.suit == lead_suit]

        if same_suit:
            if lead_suit == trump:
                wi = trick_winner_index(trick, trump)
                highest_trump = trick[wi][1]
                over = [c for c in same_suit if c.strength(trump) > highest_trump.strength(trump)]
                return over if over else same_suit
            return same_suit

        # Cannot follow suit — Rotterdam: must trump regardless of partner
        trumps = [c for c in self.hand if c.suit == trump]
        if trumps:
            trick_trumps = [c for _, c in trick if c.suit == trump]
            if trick_trumps:
                highest = max(trick_trumps, key=lambda c: c.strength(trump))
                over = [c for c in trumps if c.strength(trump) > highest.strength(trump)]
                return over if over else trumps
            return trumps

        return list(self.hand)

    @abstractmethod
    def choose_card(self, trick: Trick, trump: str) -> Card: ...

    @abstractmethod
    def choose_trump(self, suit: str, forced: bool) -> bool:
        """Return True to declare *suit* as trump, False to pass (only allowed when not forced)."""
        ...


# ─── Human player ─────────────────────────────────────────────────────────────

DISCONNECT_TIMEOUT = 120   # seconds to wait for a disconnected player


class HumanPlayer(Player):
    """Human player that communicates via callback functions.

    The callbacks are set by the web server (app.py) and use threading.Event
    to block the game thread until the human responds.

    Supports disconnect/reconnect: when disconnected, the game thread pauses
    (blocks on the event) until the player reconnects or DISCONNECT_TIMEOUT
    expires (which interrupts the game).
    """

    def __init__(self, name: str, team: int, seat_idx: int = 0):
        super().__init__(name, team)
        self.seat_idx = seat_idx
        self._move_event = threading.Event()
        self._bid_event = threading.Event()
        self._chosen_card: Card | None = None
        self._bid_result: bool = False
        self._interrupted = False
        self.connected = True

        # Track in-flight requests so set_reconnected() can re-fire them
        self._pending_legal: list | None = None
        self._pending_bid_suit: str | None = None
        self._pending_bid_forced: bool = False

        # Callbacks set by the web server
        self._on_move_request = None   # fn(seat_idx, legal_cards)
        self._on_bid_request = None    # fn(seat_idx, suit, forced)
        self._on_disconnect_pause = None  # fn(seat_idx) — notify others of pause

    def interrupt(self) -> None:
        """Signal the player to stop waiting (game restart / disconnect timeout)."""
        self._interrupted = True
        self._move_event.set()
        self._bid_event.set()

    def reset_interrupt(self) -> None:
        self._interrupted = False

    def set_disconnected(self) -> None:
        """Mark player as disconnected. Game thread will pause at next input request."""
        self.connected = False

    def set_reconnected(self) -> None:
        """Mark player as reconnected. If the game is paused waiting for this player,
        re-fire the request callback so the browser gets the prompt again.
        Clears pending state after firing to prevent double-fire from the disconnect loop."""
        self.connected = True
        if self._pending_legal is not None and self._on_move_request:
            legal = self._pending_legal
            self._pending_legal = None
            self._on_move_request(self.seat_idx, legal)
        elif self._pending_bid_suit is not None and self._on_bid_request:
            suit, forced = self._pending_bid_suit, self._pending_bid_forced
            self._pending_bid_suit = None
            self._on_bid_request(self.seat_idx, suit, forced)

    def choose_card(self, trick: Trick, trump: str) -> Card:
        legal = self.legal_moves(trick, trump)
        self._chosen_card = None
        self._move_event.clear()
        self._pending_legal = legal

        if self.connected and self._on_move_request:
            self._on_move_request(self.seat_idx, legal)

        # If disconnected, notify others and wait for reconnect (with timeout)
        if not self.connected:
            if self._on_disconnect_pause:
                self._on_disconnect_pause(self.seat_idx)
            # Block until reconnect sets connected=True and re-fires move request
            while not self.connected and not self._interrupted:
                self._move_event.wait(timeout=1.0)
                self._move_event.clear()
            if self._interrupted:
                self._pending_legal = None
                raise GameInterrupt()
            # Player reconnected — fire the request again (if set_reconnected didn't already)
            if self._pending_legal is not None and self._on_move_request:
                self._on_move_request(self.seat_idx, legal)

        self._move_event.wait(timeout=DISCONNECT_TIMEOUT)
        self._pending_legal = None
        if self._interrupted or self._chosen_card is None:
            if not self._interrupted:
                self._interrupted = True  # timeout
            raise GameInterrupt()
        card = self._chosen_card
        self.hand.remove(card)
        return card

    def supply_card(self, card_str: str) -> bool:
        """Called by the web layer when the human picks a card."""
        for c in self.hand:
            if str(c) == card_str:
                self._chosen_card = c
                self._move_event.set()
                return True
        return False

    def choose_trump(self, suit: str, forced: bool) -> bool:
        self._bid_result = False
        self._bid_event.clear()
        self._pending_bid_suit = suit
        self._pending_bid_forced = forced

        if self.connected and self._on_bid_request:
            self._on_bid_request(self.seat_idx, suit, forced)

        if not self.connected:
            if self._on_disconnect_pause:
                self._on_disconnect_pause(self.seat_idx)
            while not self.connected and not self._interrupted:
                self._bid_event.wait(timeout=1.0)
                self._bid_event.clear()
            if self._interrupted:
                self._pending_bid_suit = None
                raise GameInterrupt()
            # Player reconnected — fire the request again (if set_reconnected didn't already)
            if self._pending_bid_suit is not None and self._on_bid_request:
                self._on_bid_request(self.seat_idx, suit, forced)

        self._bid_event.wait(timeout=DISCONNECT_TIMEOUT)
        self._pending_bid_suit = None
        if self._interrupted:
            raise GameInterrupt()
        return self._bid_result

    def supply_bid(self, declare: bool) -> None:
        """Called by the web layer when the human bids."""
        self._bid_result = declare
        self._bid_event.set()


# ─── AI player ────────────────────────────────────────────────────────────────

class AIPlayer(Player):
    # Minimum declaration score [0–1] needed to voluntarily declare a proposed suit.
    # Higher roem at stake (risk of nat forfeiting it) → AI is more conservative.
    DECLARATION_THRESHOLD = 0.25

    def __init__(self, name: str, team: int):
        super().__init__(name, team)
        # Updated by KlaverjasGame before each trick so AI can adapt its strategy
        self.trick_pts: list[int] = [0, 0]
        self.roem_pts: list[int] = [0, 0]
        self.declaring_team: int = -1

    def start_round(self) -> None:
        super().start_round()
        self.trick_pts = [0, 0]
        self.roem_pts = [0, 0]
        self.declaring_team = -1

    def choose_card(self, trick: Trick, trump: str) -> Card:
        legal = self.legal_moves(trick, trump)
        card = self._strategy(legal, trick, trump)
        self.hand.remove(card)
        return card

    def choose_trump(self, suit: str, forced: bool) -> bool:
        if forced:
            return True
        score = self._hand_declaration_score(suit)
        # Be more conservative when holding lots of roem: losing nat would forfeit it all
        roem_risk = sum(p for _, p in find_roem(self.hand, suit)) // 20
        threshold = self.DECLARATION_THRESHOLD + roem_risk * 0.025
        return score >= threshold

    def _card_win_prob(self, card: Card, trump: str) -> float:
        """Estimated probability this card wins a trick when declared as trump.

        For trump cards: 1.0 if all stronger trumps are already in my hand (sure trick),
        dropping sharply for each uncontrolled card above it.
        For non-trump: fixed empirical estimates reflecting trumping risk.
        """
        if card.suit == trump:
            pos = TRUMP_STRONGEST_FIRST.index(card.rank)
            higher_not_held = sum(
                1 for i in range(pos)
                if not any(
                    c.suit == trump and c.rank == TRUMP_STRONGEST_FIRST[i]
                    for c in self.hand
                )
            )
            if higher_not_held == 0:
                return 1.0   # sure trick — I hold every stronger trump
            if higher_not_held == 1:
                return 0.12  # one opponent has the single card above me
            return 0.04      # multiple higher trumps unaccounted for
        else:
            # Non-trump: can always be trumped; A is still valuable
            _PROB = {"A": 0.50, "K": 0.28, "Q": 0.11, "J": 0.05,
                     "10": 0.03, "9": 0.02, "8": 0.01, "7": 0.01}
            return _PROB[card.rank]

    def _hand_declaration_score(self, trump: str) -> float:
        """Declaration score in [0, 1].

        1.0 means holding all 8 trump cards — certain to win every trick.
        0.0 means no card has any realistic chance of winning.
        Uses per-card win-probability estimates summed and normalised to 8 tricks.
        """
        expected_wins = sum(self._card_win_prob(c, trump) for c in self.hand)
        return min(expected_wins / 8.0, 1.0)

    def _remaining_in_suit(self, suit: str) -> int:
        """Cards of this suit still in other players' hands (not mine, not yet played)."""
        total = 8  # 8 cards per suit in a 32-card deck
        in_my_hand = sum(1 for c in self.hand if c.suit == suit)
        already_played = sum(1 for cs in self.played_cards if cs[-1] == suit)
        return max(0, total - in_my_hand - already_played)

    def _strategy(self, legal: list[Card], trick: Trick, trump: str) -> Card:
        if not trick:
            return self._lead(legal, trump)
        wi = trick_winner_index(trick, trump)
        if trick[wi][0].team == self.team:
            return self._discard_for_partner(legal, trump)
        return self._try_win(legal, trick, trump)

    def _highest_remaining_trump(self, trump: str) -> str | None:
        """Return the rank of the highest trump not yet played and not in our hand,
        i.e. the strongest trump an opponent could still hold.  None if all are
        accounted for."""
        for rank in TRUMP_STRONGEST_FIRST:
            card_str = f"{rank}{trump}"
            in_my_hand = any(c.suit == trump and c.rank == rank for c in self.hand)
            already_played = card_str in self.played_cards
            if not in_my_hand and not already_played:
                return rank
        return None

    def _best_lead_from(self, cards: list[Card], trump: str) -> Card:
        """Pick the best card to lead from a group of same-suit non-trump cards.

        Returns the highest card where all higher cards in that suit are
        already accounted for (played or in our hand).  If none qualify,
        returns the lowest-point card to minimise loss when probing.

        This prevents the AI from leading the 10 into an outstanding Ace,
        which would waste 10 points (the 10 is the second-highest non-trump
        card in Klaverjassen but worth as many points as the trump Jack).
        """
        suit = cards[0].suit
        for card in sorted(cards, key=lambda c: c.strength(trump), reverse=True):
            unaccounted_higher = any(
                f"{rank}{suit}" not in self.played_cards
                and not any(c2.suit == suit and c2.rank == rank for c2 in self.hand)
                for rank in NON_TRUMP_ORDER
                if NON_TRUMP_ORDER.index(rank) > card.strength(trump)
            )
            if not unaccounted_higher:
                return card
        # No safe card — lead cheapest to probe
        return min(cards, key=lambda c: c.points(trump))

    def _lead(self, legal: list[Card], trump: str) -> Card:
        """Lead strategy.

        Declaring team: aggressively pull opponents' trumps early so that
        non-trump winners (A, 10) are safe later.
          1. If we hold the highest remaining trump → lead it (sure win, pulls trump).
          2. If opponents still have trumps but we don't hold the top one →
             lead our cheapest trump to force opponents to spend theirs.
          3. Once opponents are likely void of trumps, switch to non-trump winners.

        Non-declaring team: prefer long suits for control.
        """
        non_trump = [c for c in legal if c.suit != trump]
        trump_legal = [c for c in legal if c.suit == trump]
        remaining_opp_trumps = self._remaining_in_suit(trump)

        # ── Declaring team: pull trumps ──────────────────────────────────────
        if self.declaring_team == self.team and trump_legal and remaining_opp_trumps > 0:
            top_trump = max(trump_legal, key=lambda c: c.strength(trump))
            strongest_opp = self._highest_remaining_trump(trump)

            if strongest_opp is None or top_trump.strength(trump) > TRUMP_ORDER.index(strongest_opp):
                # We hold the highest remaining trump — lead it (sure win)
                return top_trump
            else:
                # We don't hold the top trump; lead our cheapest trump to
                # force opponents to spend theirs
                cheapest_trump = min(trump_legal, key=lambda c: c.points(trump))
                return cheapest_trump

        # ── Non-trump lead (or declaring team after trumps are pulled) ───────
        if non_trump:
            suit_groups: dict[str, list[Card]] = {}
            for c in non_trump:
                suit_groups.setdefault(c.suit, []).append(c)

            best_card: Card | None = None
            best_score: tuple[int, int] = (-1, -1)
            for suit, cards in suit_groups.items():
                top = self._best_lead_from(cards, trump)
                remaining = self._remaining_in_suit(suit)
                # Prefer: (1) longer suit in hand, (2) higher top card
                score = (len(cards), top.strength(trump))
                # Extra bonus when opponents are void — guaranteed win
                if remaining == 0:
                    score = (len(cards) + 10, top.strength(trump))
                if score > best_score:
                    best_score = score
                    best_card = top
            if best_card:
                return best_card

        # Only trump left — lead highest
        return max(legal, key=lambda c: c.strength(trump))

    def _discard_for_partner(self, legal: list[Card], trump: str) -> Card:
        """Partner is winning — dump from our shortest non-trump suit (void development)
        to set up future trump plays, playing the lowest-value card of that suit."""
        non_trump = [c for c in legal if c.suit != trump]
        if non_trump:
            suit_counts: dict[str, int] = {}
            for c in non_trump:
                suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
            shortest_suit = min(suit_counts, key=lambda s: suit_counts[s])
            candidates = [c for c in non_trump if c.suit == shortest_suit]
            return min(candidates, key=lambda c: c.points(trump))
        return min(legal, key=lambda c: c.points(trump))

    def _play_safe_discard(self, legal: list[Card], trump: str) -> Card:
        """Cannot win — pick the card that leaks the fewest points to opponents.
        Prefer discarding from a short non-trump suit (helps create voids)."""
        non_trump = [c for c in legal if c.suit != trump]
        pool = non_trump if non_trump else legal
        return min(pool, key=lambda c: c.points(trump))

    def _try_win(self, legal: list[Card], trick: Trick, trump: str) -> Card:
        """Opponent is winning — try to beat with minimum trump; else safe discard.

        Avoids wasting high-value cards when remaining opponents are likely to
        outplay the beater anyway (e.g. when unaccounted-for higher trumps or
        higher cards in the led suit are still out there).
        """
        wi = trick_winner_index(trick, trump)
        winning_card = trick[wi][1]
        beaters = [
            c for c in legal
            if (c.suit == trump and winning_card.suit != trump)
            or (c.suit == winning_card.suit and c.strength(trump) > winning_card.strength(trump))
        ]
        if not beaters:
            return self._play_safe_discard(legal, trump)

        # If we're the last player in the trick, our beater wins for sure
        if len(trick) == 3:
            return min(beaters, key=lambda c: c.points(trump))

        cheapest_beater = min(beaters, key=lambda c: c.points(trump))

        # Check if remaining opponents (players yet to act) could beat our beater
        if not self._beater_likely_holds(cheapest_beater, trick, trump):
            # Our cheapest beater is likely to be outplayed — don't waste it
            # unless it's a zero-value card (nothing to lose)
            if cheapest_beater.points(trump) > 0:
                return self._play_safe_discard(legal, trump)

        return cheapest_beater

    def _beater_likely_holds(self, card: Card, trick: Trick, trump: str) -> bool:
        """Estimate whether *card* will survive the remaining opponents in this trick.

        Returns True if we think the card is reasonably safe, False if opponents
        likely have something stronger.
        """
        players_left = 4 - len(trick) - 1  # opponents still to play after us
        if players_left <= 0:
            return True

        lead_suit = trick[0][1].suit

        if card.suit == trump:
            # Count unaccounted-for trumps that are stronger than ours
            stronger_trump_ranks = [
                r for r in TRUMP_STRONGEST_FIRST
                if TRUMP_ORDER.index(r) > card.strength(trump)
            ]
            threats = 0
            for rank in stronger_trump_ranks:
                card_str_candidates = [f"{rank}{trump}"]
                already_seen = any(cs in self.played_cards for cs in card_str_candidates)
                in_my_hand = any(c.suit == trump and c.rank == rank for c in self.hand)
                if not already_seen and not in_my_hand:
                    threats += 1
            # If no stronger trumps unaccounted for, we're safe
            if threats == 0:
                return True
            # With 1 opponent left and 1 threat: ~37% chance they have it (rough)
            # With 2 opponents left and 1 threat: ~62% chance one has it
            # Be conservative: if any threat exists with 2+ opponents, likely loses
            return threats == 1 and players_left == 1
        else:
            # Non-trump beater: vulnerable to trumping AND to higher cards in suit
            # Check if opponents might still have trumps
            remaining_trumps = self._remaining_in_suit(trump)
            if remaining_trumps > 0:
                # Opponents may trump our non-trump card — risky
                # Check if opponents can even follow the led suit;
                # if the led suit is exhausted they're more likely to trump
                remaining_lead = self._remaining_in_suit(lead_suit)
                if remaining_lead == 0:
                    # All remaining lead-suit cards accounted for, opponents
                    # will discard or trump — high chance of being trumped
                    return False
                # Some lead suit still out plus trumps out — moderate risk
                if players_left >= 2:
                    return False

            # Check for higher cards in the led suit that are unaccounted for
            stronger_ranks = [
                r for r in NON_TRUMP_ORDER
                if NON_TRUMP_ORDER.index(r) > card.strength(trump)
            ]
            threats = 0
            for rank in stronger_ranks:
                card_str = f"{rank}{card.suit}"
                already_seen = card_str in self.played_cards
                in_my_hand = any(c.suit == card.suit and c.rank == rank for c in self.hand)
                if not already_seen and not in_my_hand:
                    threats += 1

            if threats == 0 and remaining_trumps == 0:
                return True
            return threats == 0 and players_left == 1


# ─── Game ─────────────────────────────────────────────────────────────────────

SEAT_DEFAULTS = {0: "South", 1: "West", 2: "North", 3: "East"}
SEAT_TEAMS = {0: 0, 1: 1, 2: 0, 3: 1}


class KlaverjasGame:
    """
    Seat order:  0=South  1=West  2=North  3=East
    Teams:       0 → seats 0, 2          1 → seats 1, 3
    """

    def __init__(
        self,
        human_seats: dict[int, str] | None = None,
        log_fn=None,
        state_fn=None,
        game_mode: str = "score_limit",
        score_limit: int = 500,
    ):
        """
        Args:
            human_seats: mapping of seat_idx → player_name for human players.
                         Seats not in this dict become AI.
                         If None, defaults to {0: "You"} (single human).
            game_mode: "score_limit", "boom", or "free_play".
            score_limit: target score for score_limit mode (default 500).
        """
        if human_seats is None:
            human_seats = {0: "You"}

        self.players: list[Player] = []
        for seat in range(4):
            team = SEAT_TEAMS[seat]
            if seat in human_seats:
                name = human_seats[seat]
                self.players.append(HumanPlayer(name, team, seat_idx=seat))
            else:
                name = f"AI {SEAT_DEFAULTS[seat]}"
                self.players.append(AIPlayer(name, team))

        self.scores = [0, 0]
        self.log = log_fn or (lambda msg, tag="": print(msg))
        self.notify = state_fn or (lambda event, data: None)
        self._next_round_event = threading.Event()
        self.game_mode = game_mode
        self.score_limit = score_limit
        self.boom_rounds = 16

    def signal_next_round(self) -> None:
        """Called by the web layer when the host advances to the next round."""
        self._next_round_event.set()

    def _game_continues(self, round_num: int) -> bool:
        """Return True if more rounds should be played."""
        if self.game_mode == "score_limit":
            return max(self.scores) < self.score_limit
        if self.game_mode == "boom":
            return round_num < self.boom_rounds
        # free_play: always continues (ended only by GameInterrupt)
        return True

    def play(self) -> None:
        dealer = random.randint(0, 3)
        round_num = 0

        while self._game_continues(round_num):
            round_num += 1
            first_bidder = (dealer + 1) % 4
            self.log(f"\n{'='*40}", "round")
            self.log(f"Round {round_num}  (dealer: {self.players[dealer].name})", "round")

            t0, t1, leader, history = self._play_round(first_bidder)

            self.scores[0] += t0
            self.scores[1] += t1

            # Finalise history record with round number and cumulative scores
            history["round_num"] = round_num
            history["scores_after"] = list(self.scores)

            round_done_data = {
                "t0": t0, "t1": t1, "scores": list(self.scores),
                "history": history, "round_num": round_num,
            }
            if self.game_mode == "boom":
                round_done_data["total_rounds"] = self.boom_rounds

            self.notify("round_done", round_done_data)
            self.log(f"Round result  →  Team 0: +{t0}   Team 1: +{t1}")
            self.log(f"Running total →  Team 0: {self.scores[0]}   Team 1: {self.scores[1]}")
            dealer = (dealer + 1) % 4

            # Wait for the host to advance to the next round
            if self._game_continues(round_num):
                wait_data = {"round_num": round_num}
                if self.game_mode == "boom":
                    wait_data["total_rounds"] = self.boom_rounds
                self.notify("waiting_for_host", wait_data)
                self._next_round_event.wait()
                self._next_round_event.clear()

        if self.game_mode != "free_play":
            winner = 0 if self.scores[0] >= self.scores[1] else 1
            self.log(f"GAME OVER – Team {winner} wins!  ({self.scores[0]} – {self.scores[1]})")
            self.notify("game_over", {"winner": winner, "scores": list(self.scores)})

    def _bidding(self, first_bidder: int) -> tuple[int, str]:
        """Two-round random-suit bidding."""
        suits = random.sample(SUITS, 2)

        self.log("── Bidding ──")
        for round_num, suit in enumerate(suits, 1):
            self.notify("trump_offered", {"suit": suit, "round_num": round_num})
            self.log(f"  Offered: {suit} ({SUIT_NAMES[suit]})  [round {round_num}]")

            for i in range(4):
                bidder_idx = (first_bidder + i) % 4
                player = self.players[bidder_idx]

                declared = player.choose_trump(suit, False)

                if declared:
                    self.log(
                        f"  {player.name} declares: {suit} ({SUIT_NAMES[suit]})", "trump"
                    )
                    self.notify("bid", {"player_idx": bidder_idx, "trump": suit})
                    if not isinstance(player, HumanPlayer):
                        time.sleep(AI_BID_DELAY)
                    return bidder_idx, suit
                else:
                    self.log(f"  {player.name} passes")
                    self.notify("bid", {"player_idx": bidder_idx, "trump": None})
                    if not isinstance(player, HumanPlayer):
                        time.sleep(AI_BID_DELAY)

        # All 8 players passed both rounds — force first bidder on round-2 suit
        forced_suit = suits[1]
        player = self.players[first_bidder]
        player.choose_trump(forced_suit, True)
        self.log(
            f"  {player.name} is forced to declare: {forced_suit} ({SUIT_NAMES[forced_suit]})",
            "trump",
        )
        self.notify("bid", {"player_idx": first_bidder, "trump": forced_suit})
        return first_bidder, forced_suit

    def _play_round(self, first_leader: int) -> tuple[int, int, int, dict]:
        """Returns (team0_pts, team1_pts, last_trick_winner_idx, history_record)."""
        for p in self.players:
            p.start_round()

        deck = Deck()
        hands = deck.deal(4, 8)
        for i, p in enumerate(self.players):
            p.receive_hand(hands[i])

        self.notify("deal_done", {})
        declaring_player_idx, trump = self._bidding(first_leader)
        declaring_team = self.players[declaring_player_idx].team
        declaring_name = self.players[declaring_player_idx].name

        self.notify("trump_set", {"trump": trump, "declaring_team": declaring_team,
                                    "declaring_player": declaring_name,
                                    "declaring_player_idx": declaring_player_idx,
                                    "leader_idx": first_leader})
        self.log(
            f"Trump: {trump}  ({SUIT_NAMES[trump]})  –  {declaring_name}'s team declares",
            "trump",
        )

        for p in self.players:
            if isinstance(p, AIPlayer):
                p.declaring_team = declaring_team

        trick_pts = [0, 0]
        roem_pts = [0, 0]
        trick_records: list[dict] = []
        roem_records: list[dict] = []
        leader = first_leader

        for trick_num in range(8):
            for p in self.players:
                if isinstance(p, AIPlayer):
                    p.trick_pts = list(trick_pts)
                    p.roem_pts = list(roem_pts)

            self.log(f"Trick {trick_num + 1}")
            winner_idx, pts, trick_cards, cards_played = self._play_trick(leader, trump)
            winning_team = self.players[winner_idx].team
            bonus = 10 if trick_num == 7 else 0
            trick_pts[winning_team] += pts + bonus

            roem_items = find_roem(cards_played, trump)
            if roem_items:
                roem_total = sum(p for _, p in roem_items)
                roem_pts[winning_team] += roem_total
                roem_records.append({
                    "team": winning_team,
                    "trick_num": trick_num + 1,
                    "items": list(roem_items),
                    "total": roem_total,
                })
                self.notify("roem", {
                    "team": winning_team,
                    "items": roem_items,
                    "pts": roem_total,
                    "roem_pts": list(roem_pts),
                })
                detail = ", ".join(f"{desc} +{p}" for desc, p in roem_items)
                self.log(f"  Roem – Team {winning_team}: {detail}  [+{roem_total}]", "roem")

            trick_records.append({
                "cards": trick_cards,
                "winner": self.players[winner_idx].name,
                "pts": pts + bonus,
                "roem": list(roem_items) if roem_items else [],
            })

            self.notify("trick_won", {
                "winner_idx": winner_idx,
                "pts": pts + bonus,
                "trick_pts": list(trick_pts),
                "roem_pts": list(roem_pts),
            })
            self.log(
                f"  → {self.players[winner_idx].name} wins (+{pts + bonus} pts)", "winner",
            )

            time.sleep(TRICK_CLEAR_DELAY)
            self.notify("trick_cleared", {"next_leader": winner_idx})
            leader = winner_idx

        opposing_team = 1 - declaring_team
        total_points = trick_pts[0] + roem_pts[0] + trick_pts[1] + roem_pts[1]
        nat = (trick_pts[declaring_team] + roem_pts[declaring_team]) <= (trick_pts[opposing_team] + roem_pts[opposing_team])
        if not nat:
            round_pts = [trick_pts[0] + roem_pts[0], trick_pts[1] + roem_pts[1]]
        else:
            total_penalty = TRICK_CARD_TOTAL + roem_pts[0] + roem_pts[1]
            round_pts = [0, 0]
            round_pts[opposing_team] = total_penalty
            self.notify("nat", {"declaring_team": declaring_team, "total": total_penalty})
            self.log(
                f"NAT! {declaring_name}'s team goes nat – "
                f"Team {opposing_team} receives all {total_penalty} pts!", "nat",
            )

        history: dict = {
            "trump": trump,
            "declaring_player": declaring_name,
            "declaring_team": declaring_team,
            "roem_records": roem_records,
            "tricks": trick_records,
            "trick_pts": list(trick_pts),
            "roem_pts": list(roem_pts),
            "round_pts": list(round_pts),
            "nat": nat,
        }

        return round_pts[0], round_pts[1], leader, history

    def _play_trick(
        self, leader: int, trump: str
    ) -> tuple[int, int, list[tuple[str, str]], list[Card]]:
        """Play one trick.
        Returns (winner_idx, trick_pts, [(player_name, card_str)], [card_objects])."""
        trick: Trick = []
        trick_cards: list[tuple[str, str]] = []
        cards_played: list[Card] = []

        for offset in range(4):
            idx = (leader + offset) % 4
            player = self.players[idx]
            card = player.choose_card(trick, trump)
            trick.append((player, card))
            trick_cards.append((player.name, str(card)))
            cards_played.append(card)

            for p in self.players:
                p.observe_card(card)

            self.notify("trick_played", {"player_idx": idx, "card": card})
            if not isinstance(player, HumanPlayer):
                self.log(f"  {player.name} plays: {card}")
                time.sleep(AI_PLAY_DELAY)

        wi = trick_winner_index(trick, trump)
        winner_global = self.players.index(trick[wi][0])
        total_pts = sum(c.points(trump) for _, c in trick)
        return winner_global, total_pts, trick_cards, cards_played
