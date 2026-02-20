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

AI_PLAY_DELAY     = 0.7   # delay between AI moves to make them easier to follow (can be set to 0 for fast autoplay)
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

    def observe_trick_play(self, player_idx: int, card: Card, lead_suit: str | None) -> None:
        """Called after each card in a trick — subclasses can override for tracking."""

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
        self.trick_num: int = 0          # current trick number (0-7)
        self.game_scores: list[int] = [0, 0]  # cumulative game scores

        # Opponent void tracking: opponent_voids[player_idx] = set of suits they can't follow
        self.opponent_voids: dict[int, set[str]] = {i: set() for i in range(4)}

    def start_round(self) -> None:
        super().start_round()
        self.trick_pts = [0, 0]
        self.roem_pts = [0, 0]
        self.declaring_team = -1
        self.trick_num = 0
        self.opponent_voids = {i: set() for i in range(4)}

    def observe_trick_play(self, player_idx: int, card: Card, lead_suit: str | None) -> None:
        """Track when opponents fail to follow suit → they're void in that suit."""
        if lead_suit is not None and card.suit != lead_suit and player_idx != -1:
            self.opponent_voids.setdefault(player_idx, set()).add(lead_suit)

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

    def _hand_declaration_score(self, trump: str) -> float:
        """Evaluate hand strength for declaring the given trump suit.

        Returns a score in [0, 1] based on:
        - Trump length and quality (J, 9, A are the power cards)
        - Long non-trump suits with top cards (runnable after pulling trumps)
        - Short suits (can trump early)
        - Roem potential (bonus for declaring with strong roem)
        """
        trumps = [c for c in self.hand if c.suit == trump]
        non_trumps = [c for c in self.hand if c.suit != trump]
        n_trumps = len(trumps)
        trump_ranks = {c.rank for c in trumps}

        # ── Trump quality (0–4 points) ──────────────────────────────────────
        trump_score = 0.0
        # The three power trumps: J (20pts), 9 (14pts), A (11pts)
        if "J" in trump_ranks:
            trump_score += 1.5   # trump Jack is the strongest card in the game
        if "9" in trump_ranks:
            trump_score += 1.0   # trump 9 is second strongest
        if "A" in trump_ranks:
            trump_score += 0.7
        if "10" in trump_ranks:
            trump_score += 0.3
        # Trump length bonus: each trump beyond 2 adds control
        if n_trumps >= 3:
            trump_score += (n_trumps - 2) * 0.5  # +0.5 per extra trump

        # ── Non-trump winners (0–3 points) ──────────────────────────────────
        # Group non-trumps by suit
        nt_score = 0.0
        suit_groups: dict[str, list[Card]] = {}
        for c in non_trumps:
            suit_groups.setdefault(c.suit, []).append(c)

        for suit, cards in suit_groups.items():
            ranks = {c.rank for c in cards}
            suit_len = len(cards)
            # Aces in long suits are near-certain winners (opponents can't trump
            # if they must follow suit, and long suit = more follow-suit rounds)
            if "A" in ranks:
                ace_prob = 0.45 + suit_len * 0.08  # 3 cards → 0.69, 4 → 0.77
                nt_score += min(ace_prob, 0.95)
            if "10" in ranks:
                ten_prob = 0.15 + suit_len * 0.06
                # 10 is only safe if we also hold Ace (or Ace is played)
                if "A" in ranks:
                    ten_prob += 0.25
                nt_score += min(ten_prob, 0.70)
            if "K" in ranks:
                k_prob = 0.10 + suit_len * 0.04
                nt_score += min(k_prob, 0.40)

        # ── Void / short suit bonus (0–0.8 points) ─────────────────────────
        void_score = 0.0
        for suit in SUITS:
            if suit == trump:
                continue
            count = sum(1 for c in self.hand if c.suit == suit)
            if count == 0:
                void_score += 0.4  # void = can trump immediately
            elif count == 1:
                void_score += 0.15  # singleton = void after 1 trick

        # ── Roem bonus (0–0.5 points) ───────────────────────────────────────
        roem_items = find_roem(self.hand, trump)
        roem_total = sum(p for _, p in roem_items)
        roem_score = min(roem_total / 100.0, 0.5)  # cap at 0.5

        total = trump_score + nt_score + void_score + roem_score
        # Normalize: ~8 points would be a near-perfect hand
        return min(total / 7.0, 1.0)

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
            return self._discard_for_partner(legal, trick, trump)
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

    def _opponent_indices(self) -> list[int]:
        """Return seat indices of opponents (the two players not on our team)."""
        my_team = self.team
        return [i for i in range(4) if SEAT_TEAMS[i] != my_team]

    def _partner_index(self) -> int:
        """Return seat index of our partner."""
        my_team = self.team
        return [i for i in range(4) if SEAT_TEAMS[i] == my_team and i != self._my_seat_idx()][0]

    def _my_seat_idx(self) -> int:
        """Find our own seat index by matching team and checking which AI we are."""
        # AI players are named "AI North", "AI West", etc.
        seat_map = {"South": 0, "West": 1, "North": 2, "East": 3}
        for label, idx in seat_map.items():
            if label in self.name and SEAT_TEAMS[idx] == self.team:
                return idx
        # Fallback: return first seat matching our team
        return 0 if self.team == 0 else 1

    def _opp_void_in(self, suit: str) -> bool:
        """Return True if ANY opponent is known to be void in this suit."""
        for opp in self._opponent_indices():
            if suit in self.opponent_voids.get(opp, set()):
                return True
        return False

    def _lead(self, legal: list[Card], trump: str) -> Card:
        """Lead strategy with void awareness.

        Declaring team: aggressively pull opponents' trumps early so that
        non-trump winners (A, 10) are safe later.
          1. If we hold the highest remaining trump → lead it (sure win, pulls trump).
          2. If opponents still have trumps but we don't hold the top one →
             lead our cheapest trump to force opponents to spend theirs.
          3. Once opponents are likely void of trumps, switch to non-trump winners.

        Non-declaring team: try to lead suits where we can win or partner can trump.

        Void awareness: avoid leading suits where opponents are void (they'll trump
        our non-trump cards). Prefer suits where opponents must follow.
        """
        non_trump = [c for c in legal if c.suit != trump]
        trump_legal = [c for c in legal if c.suit == trump]
        remaining_opp_trumps = self._remaining_in_suit(trump)

        # ── Declaring team: pull trumps ──────────────────────────────────────
        if self.declaring_team == self.team and trump_legal and remaining_opp_trumps > 0:
            top_trump = max(trump_legal, key=lambda c: c.strength(trump))
            strongest_opp = self._highest_remaining_trump(trump)

            if strongest_opp is None or top_trump.strength(trump) > TRUMP_ORDER.index(strongest_opp):
                return top_trump
            else:
                cheapest_trump = min(trump_legal, key=lambda c: c.points(trump))
                return cheapest_trump

        # ── Non-declaring team: lead through partner ─────────────────────────
        # If partner is void in a suit but opponents aren't → lead that suit
        # so partner can trump while opponents must follow suit
        if self.declaring_team != self.team and non_trump and remaining_opp_trumps > 0:
            partner = self._partner_index()
            for suit in SUITS:
                if suit == trump:
                    continue
                partner_void = suit in self.opponent_voids.get(partner, set())
                opp_void = self._opp_void_in(suit)
                cards_in_suit = [c for c in non_trump if c.suit == suit]
                if partner_void and not opp_void and cards_in_suit:
                    # Partner will trump, opponents must follow — great!
                    return min(cards_in_suit, key=lambda c: c.points(trump))

        # ── Non-trump lead (both teams, after trump-pulling phase) ───────────
        if non_trump:
            suit_groups: dict[str, list[Card]] = {}
            for c in non_trump:
                suit_groups.setdefault(c.suit, []).append(c)

            best_card: Card | None = None
            best_score: tuple[float, int] = (-99.0, -1)
            for suit, cards in suit_groups.items():
                top = self._best_lead_from(cards, trump)
                remaining = self._remaining_in_suit(suit)

                # Base score: suit length + card strength
                score_len = float(len(cards))
                score_str = top.strength(trump)

                # Penalty: if an opponent is void in this suit, they'll trump us
                if self._opp_void_in(suit) and remaining_opp_trumps > 0:
                    score_len -= 5.0  # heavy penalty — card will be trumped

                # Bonus: no remaining cards in this suit → guaranteed win
                if remaining == 0:
                    score_len += 10.0

                score = (score_len, score_str)
                if score > best_score:
                    best_score = score
                    best_card = top
            if best_card:
                return best_card

        # Only trump left — lead highest
        return max(legal, key=lambda c: c.strength(trump))

    def _discard_for_partner(self, legal: list[Card], trick: Trick, trump: str) -> Card:
        """Partner is winning — decide between schmearing (feeding points) and void development.

        Schmear (play high-value card) when:
        - Partner's win is secure (last to play, or partner has top trump/card)
        - The trick already has decent points (worth feeding more)

        Develop voids when:
        - Early in the round (more tricks to benefit from void)
        - We have a singleton in a non-trump suit we want to void
        """
        non_trump = [c for c in legal if c.suit != trump]
        pool = non_trump if non_trump else legal

        # Check if partner's win is "safe" — are remaining opponents likely to beat it?
        partner_safe = len(trick) == 3  # we're last to play → partner wins for sure
        if not partner_safe and len(trick) == 2:
            # Only 1 opponent left after us — check if partner has top card
            wi = trick_winner_index(trick, trump)
            winning_card = trick[wi][1]
            if winning_card.suit == trump:
                # Partner holds a trump — check if any stronger trump is unaccounted for
                strongest_opp = self._highest_remaining_trump(trump)
                if strongest_opp is None or winning_card.strength(trump) > TRUMP_ORDER.index(strongest_opp):
                    partner_safe = True

        if partner_safe:
            # Schmear: feed our highest-value card from the suit we have most of
            # (keeping length in that suit for future control, but giving up
            # expendable point cards like 10s from short suits)
            #
            # Priority: play highest-point card from shortest suit (expendable points)
            if non_trump:
                suit_counts: dict[str, int] = {}
                for c in non_trump:
                    suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
                # Among cards with points, prefer those from short suits (expendable)
                point_cards = [c for c in non_trump if c.points(trump) > 0]
                if point_cards:
                    # Pick the highest-value card from the shortest suit
                    best = max(point_cards, key=lambda c: (c.points(trump), -suit_counts[c.suit]))
                    return best
            # No point cards to schmear — fall through to void development
            return max(pool, key=lambda c: c.points(trump))

        # Partner's win is not guaranteed — develop voids instead (play cheap from short suit)
        if non_trump:
            suit_counts = {}
            for c in non_trump:
                suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
            # Early in round: prefer voiding a suit; late: schmear moderately
            if self.trick_num <= 3:
                shortest_suit = min(suit_counts, key=lambda s: suit_counts[s])
                candidates = [c for c in non_trump if c.suit == shortest_suit]
                return min(candidates, key=lambda c: c.points(trump))
            else:
                # Late game: still prefer cheap cards but don't overthink
                return min(non_trump, key=lambda c: c.points(trump))
        return min(legal, key=lambda c: c.points(trump))

    def _play_safe_discard(self, legal: list[Card], trump: str) -> Card:
        """Cannot win — pick the card that leaks the fewest points to opponents.
        Prefer discarding from a short non-trump suit (helps create voids)."""
        non_trump = [c for c in legal if c.suit != trump]
        pool = non_trump if non_trump else legal
        return min(pool, key=lambda c: c.points(trump))

    def _trick_point_value(self, trick: Trick, trump: str) -> int:
        """Total points currently in the trick."""
        return sum(c.points(trump) for _, c in trick)

    def _try_win(self, legal: list[Card], trick: Trick, trump: str) -> Card:
        """Opponent is winning — try to beat with minimum cost; else safe discard.

        Decision factors:
        - Trick value: more willing to spend resources on high-value tricks
        - Last trick: +10 bonus makes it always worth fighting for
        - Remaining opponents: if we're last, any beater is a sure win
        - Nat risk: if we're the declaring team and close to going nat, fight harder
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

        trick_value = self._trick_point_value(trick, trump)
        is_last_trick = self.trick_num == 7  # 10-point bonus

        # If we're the last player in the trick, our beater wins for sure
        if len(trick) == 3:
            # Last trick or high-value trick: use cheapest beater always
            return min(beaters, key=lambda c: c.points(trump))

        cheapest_beater = min(beaters, key=lambda c: c.points(trump))

        # High-value trick threshold: be more aggressive for valuable tricks
        # Base: only fight if trick is worth 10+ pts; last trick: always fight
        worth_fighting = trick_value >= 10 or is_last_trick

        # Nat danger: declaring team must win tricks to avoid nat
        if self.declaring_team == self.team:
            my_total = self.trick_pts[self.team] + self.roem_pts[self.team]
            opp_total = self.trick_pts[1 - self.team] + self.roem_pts[1 - self.team]
            if my_total <= opp_total + 20:
                # Close to going nat — fight harder for every trick
                worth_fighting = True

        if not worth_fighting and cheapest_beater.points(trump) > 0:
            # Low-value trick and our beater costs points — don't bother
            if not self._beater_likely_holds(cheapest_beater, trick, trump):
                return self._play_safe_discard(legal, trump)

        # Check if remaining opponents can beat us
        if not self._beater_likely_holds(cheapest_beater, trick, trump):
            if cheapest_beater.points(trump) > 0 and not is_last_trick:
                return self._play_safe_discard(legal, trump)

        return cheapest_beater

    def _beater_likely_holds(self, card: Card, trick: Trick, trump: str) -> bool:
        """Estimate whether *card* will survive the remaining opponents in this trick.

        Uses void tracking: if we know an opponent is void in the lead suit,
        they'll trump — making non-trump beaters unsafe.
        Returns True if we think the card is reasonably safe.
        """
        players_left = 4 - len(trick) - 1  # opponents still to play after us
        if players_left <= 0:
            return True

        lead_suit = trick[0][1].suit

        # Identify which specific opponents are still to play
        played_indices = set()
        for p, _ in trick:
            for i in range(4):
                if SEAT_TEAMS[i] == p.team and (p.name.endswith(SEAT_DEFAULTS[i]) or p.name == SEAT_DEFAULTS[i]):
                    played_indices.add(i)
                    break

        if card.suit == trump:
            stronger_trump_ranks = [
                r for r in TRUMP_STRONGEST_FIRST
                if TRUMP_ORDER.index(r) > card.strength(trump)
            ]
            threats = 0
            for rank in stronger_trump_ranks:
                card_str = f"{rank}{trump}"
                already_seen = card_str in self.played_cards
                in_my_hand = any(c.suit == trump and c.rank == rank for c in self.hand)
                if not already_seen and not in_my_hand:
                    threats += 1
            if threats == 0:
                return True
            return threats == 1 and players_left == 1
        else:
            # Non-trump beater: check if remaining opponents are void in lead suit
            remaining_trumps = self._remaining_in_suit(trump)
            if remaining_trumps > 0:
                # Check if any remaining opponent is known void in lead suit
                for opp in self._opponent_indices():
                    if opp not in played_indices:
                        if lead_suit in self.opponent_voids.get(opp, set()):
                            return False  # this opponent WILL trump us

                remaining_lead = self._remaining_in_suit(lead_suit)
                if remaining_lead == 0:
                    return False
                if players_left >= 2:
                    return False

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
                    p.trick_num = trick_num
                    p.game_scores = list(self.scores)

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

            lead_suit = trick[0][1].suit if trick else None
            for p in self.players:
                p.observe_card(card)
                p.observe_trick_play(idx, card, lead_suit)

            self.notify("trick_played", {"player_idx": idx, "card": card})
            if not isinstance(player, HumanPlayer):
                self.log(f"  {player.name} plays: {card}")
                time.sleep(AI_PLAY_DELAY)

        wi = trick_winner_index(trick, trump)
        winner_global = self.players.index(trick[wi][0])
        total_pts = sum(c.points(trump) for _, c in trick)
        return winner_global, total_pts, trick_cards, cards_played
