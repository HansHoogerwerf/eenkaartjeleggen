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

import copy
import json
import random
import threading
import time
from abc import ABC, abstractmethod
from itertools import combinations
from types import SimpleNamespace

from klaverjas.constants import (
    AI_BID_DELAY,
    AI_PLAY_DELAY,
    NON_TRUMP_ORDER,
    RED_SUITS,
    RANKS,
    SEAT_DEFAULTS,
    SEAT_TEAMS,
    SUIT_NAMES,
    SUITS,
    TRICK_CARD_TOTAL,
    TRICK_CLEAR_DELAY,
    TRUMP_ORDER,
    TRUMP_STRONGEST_FIRST,
    WIN_SCORE,
)
from klaverjas.core import Card, Deck, Trick, find_roem, trick_winner_index


# ─── Exceptions ───────────────────────────────────────────────────────────────

class GameInterrupt(Exception):
    """Raised in the game thread when a restart or window-close is requested."""


# ─── Player (base) ────────────────────────────────────────────────────────────

class Player(ABC):
    def __init__(self, name: str, team: int, seat_idx: int):
        self.name = name
        self.team = team
        self.seat_idx = seat_idx
        self.hand: list[Card] = []
        self.played_cards: set[str] = set()   # all cards seen this round
        self.rules_variant: str = "rotterdam"

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
        """Return legal moves for the configured rule variant."""
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

        trumps = [c for c in self.hand if c.suit == trump]
        if not trumps:
            return list(self.hand)

        trick_winner = trick[trick_winner_index(trick, trump)][0]
        partner_winning = trick_winner.team == self.team
        trick_trumps = [c for _, c in trick if c.suit == trump]

        if self.rules_variant == "amsterdam" and partner_winning:
            return list(self.hand)

        if trick_trumps:
            highest = max(trick_trumps, key=lambda c: c.strength(trump))
            over = [c for c in trumps if c.strength(trump) > highest.strength(trump)]
            if over:
                return over
            if self.rules_variant == "amsterdam":
                return list(self.hand)
            return trumps

        return trumps

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
        super().__init__(name, team, seat_idx)
        self._move_event = threading.Event()
        self._bid_event = threading.Event()
        self._chosen_card: Card | None = None
        self._bid_result: bool = False
        self._interrupted = False
        self.connected = True
        self.disconnect_timeout = DISCONNECT_TIMEOUT

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

        self._move_event.wait(timeout=self.disconnect_timeout)
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

        self._bid_event.wait(timeout=self.disconnect_timeout)
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
    DECLARATION_BASE_THRESHOLD = 2.75
    TIE_BREAK_DELTA = 0.35
    SIGNAL_MAX_CONFIDENCE = 1.0
    TRICK_WIN_SIM_SAMPLES = 20
    LOOKAHEAD_BRANCH_LIMIT = 3

    BID_WEIGHTS = {
        "trump_j": 1.85,
        "trump_9": 1.30,
        "trump_a": 0.85,
        "trump_10": 0.55,
        "trump_len_3p": 0.65,
        "side_ace": 0.70,
        "side_10_with_ace": 0.30,
        "side_king": 0.15,
        "void": 0.35,
        "singleton": 0.17,
        "roem_scale": 0.75,
    }
    AI_STRENGTH_PROFILES = {
        "beginner": {
            "use_inference": False,
            "use_trick_prob": False,
            "use_endgame_solver": False,
            "use_lookahead": False,
            "lookahead_enhanced": False,
            "lookahead_depth": 0,
            "lookahead_samples": 0,
            "tie_break_delta": 0.95,
            "random_mistake_rate": 0.10,
            "declaration_bias": 0.65,
            "trick_win_sim_samples": 4,
        },
        "advanced": {
            "use_inference": True,
            "use_trick_prob": True,
            "use_endgame_solver": False,
            "use_lookahead": False,
            "lookahead_enhanced": False,
            "lookahead_depth": 0,
            "lookahead_samples": 0,
            "tie_break_delta": 0.55,
            "random_mistake_rate": 0.03,
            "declaration_bias": 0.25,
            "trick_win_sim_samples": 12,
        },
        "expert": {
            "use_inference": True,
            "use_trick_prob": True,
            "use_endgame_solver": True,
            "use_lookahead": False,
            "lookahead_enhanced": False,
            "lookahead_depth": 0,
            "lookahead_samples": 0,
            "tie_break_delta": 0.35,
            "random_mistake_rate": 0.0,
            "declaration_bias": 0.0,
            "trick_win_sim_samples": 20,
        },
        "expert_v2_base": {
            "use_inference": True,
            "use_trick_prob": True,
            "use_endgame_solver": True,
            "use_lookahead": True,
            "lookahead_enhanced": False,
            "lookahead_depth": 3,
            "lookahead_samples": 8,
            "tie_break_delta": 0.35,
            "random_mistake_rate": 0.0,
            "declaration_bias": 0.0,
            "trick_win_sim_samples": 20,
        },
        "expert_v2": {
            "use_inference": True,
            "use_trick_prob": True,
            "use_endgame_solver": True,
            "use_lookahead": True,
            "lookahead_enhanced": True,
            "lookahead_depth": 3,
            "lookahead_samples": 8,
            "tie_break_delta": 0.35,
            "random_mistake_rate": 0.0,
            "declaration_bias": 0.0,
            "trick_win_sim_samples": 20,
        },
    }

    def __init__(
        self,
        name: str,
        team: int,
        seat_idx: int,
        rng_seed: int | None = None,
        signal_profile: str = "core",
        ai_strength: str = "expert",
    ):
        super().__init__(name, team, seat_idx)
        self.signal_profile = signal_profile
        self.rng = random.Random(rng_seed)
        self.ai_strength = ai_strength if ai_strength in self.AI_STRENGTH_PROFILES else "expert"
        profile = self.AI_STRENGTH_PROFILES[self.ai_strength]
        self.use_inference = bool(profile["use_inference"])
        self.use_trick_prob = bool(profile["use_trick_prob"])
        self.use_endgame_solver = bool(profile["use_endgame_solver"])
        self.use_lookahead = bool(profile["use_lookahead"])
        self.lookahead_enhanced = bool(profile["lookahead_enhanced"])
        self.lookahead_depth = int(profile["lookahead_depth"])
        self.lookahead_samples = int(profile["lookahead_samples"])
        self.random_mistake_rate = float(profile["random_mistake_rate"])
        self.declaration_bias = float(profile["declaration_bias"])
        self.TIE_BREAK_DELTA = float(profile["tie_break_delta"])
        self.TRICK_WIN_SIM_SAMPLES = int(profile["trick_win_sim_samples"])

        # Updated by KlaverjasGame before each trick so AI can adapt its strategy
        self.trick_pts: list[int] = [0, 0]
        self.roem_pts: list[int] = [0, 0]
        self.declaring_team: int = -1
        self.trick_num: int = 0          # current trick number (0-7)
        self.game_scores: list[int] = [0, 0]  # cumulative game scores
        self.current_trump: str | None = None
        self.game_mode: str = "score_limit"
        self.score_limit: int = WIN_SCORE
        self.boom_rounds: int = 16
        self.round_num: int = 0

        # Opponent/partner tracking by seat index
        self.opponent_voids: dict[int, set[str]] = {i: set() for i in range(4)}
        self.partner_signals: dict[str, dict] = {}
        self.possible_cards_by_seat: dict[int, set[str]] = {}
        self._observed_trick_cards: list[tuple[int, Card]] = []

    def start_round(self) -> None:
        super().start_round()
        self.trick_pts = [0, 0]
        self.roem_pts = [0, 0]
        self.declaring_team = -1
        self.trick_num = 0
        self.current_trump = None
        self.opponent_voids = {i: set() for i in range(4)}
        self.partner_signals = {}
        self.possible_cards_by_seat = {}
        self._observed_trick_cards = []

    def receive_hand(self, cards: list[Card]) -> None:
        super().receive_hand(cards)
        self._init_possible_cards()

    def observe_card(self, card: Card) -> None:
        super().observe_card(card)
        cs = str(card)
        for seat in range(4):
            self.possible_cards_by_seat.setdefault(seat, set()).discard(cs)
        self.possible_cards_by_seat.setdefault(self.seat_idx, set()).discard(cs)

    def observe_trick_play(self, player_idx: int, card: Card, lead_suit: str | None) -> None:
        """Track voids and decode basic partner signals."""
        if len(self._observed_trick_cards) >= 4:
            self._observed_trick_cards.clear()
        prior_cards = list(self._observed_trick_cards)

        if self.use_inference:
            self._apply_inference_from_play(player_idx, card, lead_suit, prior_cards)

        if lead_suit is not None and card.suit != lead_suit and player_idx != -1:
            self.opponent_voids.setdefault(player_idx, set()).add(lead_suit)
        if player_idx == self._partner_index():
            self._decode_partner_signal(card, lead_suit)
        self._decay_signals()
        self._observed_trick_cards.append((player_idx, card))

    def choose_card(self, trick: Trick, trump: str) -> Card:
        legal = self.legal_moves(trick, trump)
        card = self._strategy(legal, trick, trump)
        self.hand.remove(card)
        return card

    def choose_trump(self, suit: str, forced: bool) -> bool:
        if forced:
            return True
        score = self._declaration_score(suit)
        roem_total = sum(p for _, p in find_roem(self.hand, suit))
        nat_risk = 0.0
        if self.declaring_team not in (-1, self.team):
            nat_risk += 0.15
        pressure = self._score_pressure()
        threshold = (
            self.DECLARATION_BASE_THRESHOLD
            + (roem_total / 100.0) * 0.2
            + nat_risk
            + self.declaration_bias
            - pressure * 0.35
        )
        return score >= threshold

    def _simulated_bid_score(self, trump: str) -> float:
        """Simulate a few quick trick sequences to estimate expected points.

        Plays 3 tricks with sampled opponent hands, 4 times, and returns
        the average point differential for our team.
        """
        # Build a fake empty trick to pass to _sample_hands.
        fake_trick: Trick = []
        total_delta = 0.0
        valid = 0

        for _ in range(4):
            hands = self._sample_hands(fake_trick)
            if hands is None:
                continue

            # Simulate 3 tricks of play using simple greedy heuristic per seat.
            delta = 0.0
            leader = self.seat_idx
            sim_hands = {seat: list(cards) for seat, cards in hands.items()}
            for _ in range(3):
                trick_cards: list[tuple[int, Card]] = []
                for offset in range(4):
                    seat = (leader + offset) % 4
                    hand = sim_hands[seat]
                    if not hand:
                        break
                    legal = self._legal_moves_for_cards(hand, trick_cards, trump)
                    if not legal:
                        break
                    # Simple greedy: play highest strength card.
                    card = max(legal, key=lambda c: c.strength(trump))
                    trick_cards.append((seat, card))
                    cs = str(card)
                    for i, hc in enumerate(hand):
                        if str(hc) == cs:
                            del hand[i]
                            break

                if len(trick_cards) == 4:
                    sim_trick = [
                        (SimpleNamespace(seat_idx=s, team=SEAT_TEAMS[s]), c)
                        for s, c in trick_cards
                    ]
                    wi = trick_winner_index(sim_trick, trump)
                    winner_seat = trick_cards[wi][0]
                    pts = sum(c.points(trump) for _, c in trick_cards)
                    if SEAT_TEAMS[winner_seat] == self.team:
                        delta += pts
                    else:
                        delta -= pts
                    leader = winner_seat

            total_delta += delta
            valid += 1

        return total_delta / valid if valid > 0 else 0.0

    def _declaration_score(self, trump: str) -> float:
        trumps = [c for c in self.hand if c.suit == trump]
        non_trumps = [c for c in self.hand if c.suit != trump]
        trump_ranks = {c.rank for c in trumps}
        score = 0.0

        if "J" in trump_ranks:
            score += self.BID_WEIGHTS["trump_j"]
        if "9" in trump_ranks:
            score += self.BID_WEIGHTS["trump_9"]
        if "A" in trump_ranks:
            score += self.BID_WEIGHTS["trump_a"]
        if "10" in trump_ranks:
            score += self.BID_WEIGHTS["trump_10"]
        if len(trumps) >= 3:
            score += (len(trumps) - 2) * self.BID_WEIGHTS["trump_len_3p"]

        suit_groups: dict[str, list[Card]] = {}
        for c in non_trumps:
            suit_groups.setdefault(c.suit, []).append(c)

        for cards in suit_groups.values():
            ranks = {c.rank for c in cards}
            if "A" in ranks:
                score += self.BID_WEIGHTS["side_ace"] + len(cards) * 0.05
            if "10" in ranks:
                if "A" in ranks:
                    score += self.BID_WEIGHTS["side_10_with_ace"]
            if "K" in ranks:
                score += self.BID_WEIGHTS["side_king"]

        for suit in SUITS:
            if suit == trump:
                continue
            count = sum(1 for c in self.hand if c.suit == suit)
            if count == 0:
                score += self.BID_WEIGHTS["void"]
            elif count == 1:
                score += self.BID_WEIGHTS["singleton"]

        roem_items = find_roem(self.hand, trump)
        roem_total = sum(p for _, p in roem_items)
        score += min(roem_total / 100.0, self.BID_WEIGHTS["roem_scale"])
        return score

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _score_pressure(self) -> float:
        """Return urgency in [-1, 1] from current score context.

        Positive values -> more aggressive (behind / nat danger / must catch up).
        Negative values -> more conservative (ahead / protect lead).
        """
        my_round = self.trick_pts[self.team] + self.roem_pts[self.team]
        opp_round = self.trick_pts[1 - self.team] + self.roem_pts[1 - self.team]
        my_game = self.game_scores[self.team]
        opp_game = self.game_scores[1 - self.team]

        pressure = 0.0
        pressure += self._clamp((opp_round - my_round) / 40.0, -0.6, 0.9)

        if self.declaring_team == self.team and my_round <= opp_round + 10:
            pressure += 0.55
        elif self.declaring_team != -1 and self.declaring_team != self.team and my_round >= opp_round + 25:
            pressure -= 0.15

        pressure += self._clamp((opp_game - my_game) / 220.0, -0.7, 0.7)

        if self.game_mode == "score_limit":
            target = self.score_limit
            my_to_win = target - my_game
            opp_to_win = target - opp_game
            if opp_to_win <= 80 and my_to_win > opp_to_win:
                pressure += 0.45
            if my_to_win <= 80 and my_to_win < opp_to_win:
                pressure -= 0.20
        elif self.game_mode == "boom":
            if self.round_num >= self.boom_rounds - 3:
                pressure += self._clamp((opp_game - my_game) / 180.0, -0.35, 0.35)

        if self.trick_num >= 6:
            pressure *= 1.10
        return self._clamp(pressure, -1.0, 1.0)

    def _all_card_strings(self) -> set[str]:
        return {f"{rank}{suit}" for suit in SUITS for rank in RANKS}

    def _card_from_str(self, card_str: str) -> Card:
        return Card(card_str[-1], card_str[:-1])

    def _init_possible_cards(self) -> None:
        all_cards = self._all_card_strings()
        my_cards = {str(c) for c in self.hand}
        unknown = all_cards - my_cards - self.played_cards
        self.possible_cards_by_seat = {}
        for seat in range(4):
            if seat == self.seat_idx:
                self.possible_cards_by_seat[seat] = set(my_cards)
            else:
                self.possible_cards_by_seat[seat] = set(unknown)

    def _remove_suit_from_possible(self, seat: int, suit: str) -> None:
        poss = self.possible_cards_by_seat.setdefault(seat, set())
        to_remove = {cs for cs in poss if cs.endswith(suit)}
        poss.difference_update(to_remove)

    def _remove_trump_stronger_than(self, seat: int, trump: str, strength_idx: int) -> None:
        poss = self.possible_cards_by_seat.setdefault(seat, set())
        to_remove = {
            f"{rank}{trump}"
            for rank in TRUMP_ORDER
            if TRUMP_ORDER.index(rank) > strength_idx
        }
        poss.difference_update(to_remove)

    def _apply_inference_from_play(
        self,
        player_idx: int,
        card: Card,
        lead_suit: str | None,
        prior_cards: list[tuple[int, Card]],
    ) -> None:
        if player_idx < 0 or player_idx > 3:
            return
        trump = self.current_trump
        if lead_suit is None or trump is None:
            return

        # Failed to follow suit: remove lead suit from that seat's possibilities.
        if card.suit != lead_suit:
            self._remove_suit_from_possible(player_idx, lead_suit)

            if card.suit != trump:
                # Rotterdam: if they could not follow and did not trump, they had no trump.
                self._remove_suit_from_possible(player_idx, trump)
            else:
                # They trumped. If they did not overtrump while required, stronger trumps are impossible.
                prior_trumps = [c for _, c in prior_cards if c.suit == trump]
                if prior_trumps:
                    highest = max(prior_trumps, key=lambda c: c.strength(trump))
                    if card.strength(trump) <= highest.strength(trump):
                        self._remove_trump_stronger_than(player_idx, trump, highest.strength(trump))

        # Lead suit was trump and they followed trump but did not overtrump.
        if lead_suit == trump and card.suit == trump:
            prior_trumps = [c for _, c in prior_cards if c.suit == trump]
            if prior_trumps:
                highest = max(prior_trumps, key=lambda c: c.strength(trump))
                if card.strength(trump) <= highest.strength(trump):
                    self._remove_trump_stronger_than(player_idx, trump, highest.strength(trump))

    def _remaining_in_suit(self, suit: str) -> int:
        """Cards of this suit still in other players' hands (not mine, not yet played)."""
        total = 8  # 8 cards per suit in a 32-card deck
        in_my_hand = sum(1 for c in self.hand if c.suit == suit)
        already_played = sum(1 for cs in self.played_cards if cs[-1] == suit)
        return max(0, total - in_my_hand - already_played)

    def _strategy(self, legal: list[Card], trick: Trick, trump: str) -> Card:
        self.current_trump = trump
        if self.use_endgame_solver and len(self.hand) <= 3:
            solved = self._endgame_exact_choice(legal, trick, trump)
            if solved is not None:
                return solved
        if self.use_lookahead and len(self.hand) > 3:
            lookahead = self._lookahead_choice(legal, trick, trump)
            if lookahead is not None:
                return lookahead
        if not trick:
            return self._lead(legal, trump)
        wi = trick_winner_index(trick, trump)
        if trick[wi][0].team == self.team:
            return self._discard_for_partner(legal, trick, trump)
        return self._try_win(legal, trick, trump)

    def _pick_card(self, scores: list[tuple[Card, float]]) -> Card:
        if self.random_mistake_rate > 0.0 and len(scores) > 1 and self.rng.random() < self.random_mistake_rate:
            return self.rng.choice([card for card, _ in scores])
        best = max(score for _, score in scores)
        near_best = [card for card, score in scores if best - score <= self.TIE_BREAK_DELTA]
        return self.rng.choice(near_best)

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

    def _higher_outstanding_count_in_suit(self, card: Card, trump: str) -> int:
        """How many stronger same-suit cards are still unaccounted for."""
        rank_order = TRUMP_ORDER if card.suit == trump else NON_TRUMP_ORDER
        card_idx = rank_order.index(card.rank)
        count = 0
        for rank in rank_order[card_idx + 1:]:
            card_str = f"{rank}{card.suit}"
            in_my_hand = any(c.suit == card.suit and c.rank == rank for c in self.hand)
            if card_str not in self.played_cards and not in_my_hand:
                count += 1
        return count

    def _point_leak_penalty(self, card: Card, trump: str) -> float:
        """Penalty for exposing point cards under outstanding higher cards.

        This is intentionally generic (not tied to specific ranks):
        any point card is penalized when stronger same-suit cards are unaccounted
        for, including trump point cards like 9 under an unseen trump Jack.
        """
        pts = card.points(trump)
        if pts <= 0:
            return 0.0

        higher_count = self._higher_outstanding_count_in_suit(card, trump)
        if higher_count == 0:
            return 0.0

        suit_risk_weight = 0.75 if card.suit == trump else 0.60
        return pts * suit_risk_weight + (higher_count - 1) * 0.8

    @staticmethod
    def _legal_moves_for_cards(hand_cards: list[Card], trick_cards: list[tuple[int, Card]], trump: str) -> list[Card]:
        if not trick_cards:
            return list(hand_cards)

        lead_suit = trick_cards[0][1].suit
        same_suit = [c for c in hand_cards if c.suit == lead_suit]

        if same_suit:
            if lead_suit == trump:
                trick_trumps = [c for _, c in trick_cards if c.suit == trump]
                highest = max(trick_trumps, key=lambda c: c.strength(trump))
                over = [c for c in same_suit if c.strength(trump) > highest.strength(trump)]
                return over if over else same_suit
            return same_suit

        trumps = [c for c in hand_cards if c.suit == trump]
        if trumps:
            trick_trumps = [c for _, c in trick_cards if c.suit == trump]
            if trick_trumps:
                highest = max(trick_trumps, key=lambda c: c.strength(trump))
                over = [c for c in trumps if c.strength(trump) > highest.strength(trump)]
                return over if over else trumps
            return trumps

        return list(hand_cards)

    def _possible_cards_for_seat(self, seat: int, used_cards: set[str]) -> list[Card]:
        if seat == self.seat_idx:
            own_cards = [c for c in self.hand if str(c) not in used_cards]
            return own_cards

        poss = self.possible_cards_by_seat.get(seat, set())
        cards = [self._card_from_str(cs) for cs in sorted(poss) if cs not in used_cards]
        if cards:
            return cards

        # Fallback if inference became too restrictive/inconsistent.
        unseen = self._all_card_strings() - self.played_cards - {str(c) for c in self.hand} - used_cards
        return [self._card_from_str(cs) for cs in sorted(unseen)]

    def _estimate_team_trick_win_prob(self, trick: Trick, my_card: Card, trump: str) -> float:
        if not self.use_trick_prob:
            return self._heuristic_team_trick_win_prob(trick, my_card, trump)

        trick_state: list[tuple[int, Card]] = [(p.seat_idx, c) for p, c in trick]
        trick_state.append((self.seat_idx, my_card))

        players_left = 4 - len(trick_state)
        if players_left <= 0:
            sim_trick = [
                (SimpleNamespace(seat_idx=seat, team=SEAT_TEAMS[seat]), card)
                for seat, card in trick_state
            ]
            wi = trick_winner_index(sim_trick, trump)
            winner_seat = trick_state[wi][0]
            return 1.0 if SEAT_TEAMS[winner_seat] == self.team else 0.0

        remaining_order = [(self.seat_idx + step) % 4 for step in range(1, players_left + 1)]
        sims = self.TRICK_WIN_SIM_SAMPLES if players_left >= 2 else max(8, self.TRICK_WIN_SIM_SAMPLES // 2)
        wins = 0
        finished = 0

        for _ in range(sims):
            used = {str(c) for _, c in trick_state}
            sim_cards = list(trick_state)
            valid = True
            for seat in remaining_order:
                pool = self._possible_cards_for_seat(seat, used)
                legal = self._legal_moves_for_cards(pool, sim_cards, trump)
                if not legal:
                    valid = False
                    break
                choice = self.rng.choice(legal)
                sim_cards.append((seat, choice))
                used.add(str(choice))
            if not valid:
                continue
            sim_trick = [
                (SimpleNamespace(seat_idx=seat, team=SEAT_TEAMS[seat]), card)
                for seat, card in sim_cards
            ]
            wi = trick_winner_index(sim_trick, trump)
            winner_seat = sim_cards[wi][0]
            if SEAT_TEAMS[winner_seat] == self.team:
                wins += 1
            finished += 1

        if finished == 0:
            return 0.5
        return wins / finished

    def _heuristic_team_trick_win_prob(self, trick: Trick, my_card: Card, trump: str) -> float:
        trick_state = [(p.seat_idx, c) for p, c in trick] + [(self.seat_idx, my_card)]
        sim_trick = [
            (SimpleNamespace(seat_idx=seat, team=SEAT_TEAMS[seat]), card)
            for seat, card in trick_state
        ]
        wi = trick_winner_index(sim_trick, trump)
        winner_seat = trick_state[wi][0]
        winner_team = SEAT_TEAMS[winner_seat]
        players_left = 4 - len(trick_state)

        if players_left <= 0:
            return 1.0 if winner_team == self.team else 0.0

        winning_card = trick_state[wi][1]
        base = 0.75 if winner_team == self.team else 0.25
        if winning_card.suit != trump and self._remaining_in_suit(trump) > 0:
            base -= 0.20 if winner_team == self.team else -0.15
        if my_card.suit == trump:
            base += 0.08
        if players_left >= 2:
            base -= 0.15 if winner_team == self.team else -0.08
        return self._clamp(base, 0.05, 0.95)

    def _opponent_indices(self) -> list[int]:
        """Return seat indices of opponents (the two players not on our team)."""
        return [i for i in range(4) if SEAT_TEAMS[i] != self.team]

    def _partner_index(self) -> int:
        """Return seat index of our partner."""
        return [i for i in range(4) if SEAT_TEAMS[i] == self.team and i != self.seat_idx][0]

    def _opp_void_in(self, suit: str) -> bool:
        """Return True if ANY opponent is known to be void in this suit."""
        for opp in self._opponent_indices():
            if suit in self.opponent_voids.get(opp, set()):
                return True
        return False

    def _lead(self, legal: list[Card], trump: str) -> Card:
        non_trump = [c for c in legal if c.suit != trump]
        remaining_opp_trumps = self._remaining_in_suit(trump)
        pressure = self._score_pressure()
        scores: list[tuple[Card, float]] = []
        for c in legal:
            score = float(c.strength(trump))
            if c.suit == trump:
                if self.declaring_team == self.team and remaining_opp_trumps > 0:
                    score += 3.5
                else:
                    score -= 0.9
                strongest_opp = self._highest_remaining_trump(trump)
                if strongest_opp is None or c.strength(trump) > TRUMP_ORDER.index(strongest_opp):
                    score += 1.0
                score += pressure * 1.0
            else:
                suit_count = sum(1 for hc in self.hand if hc.suit == c.suit)
                score += suit_count * 0.6
                if self._opp_void_in(c.suit) and remaining_opp_trumps > 0:
                    score -= 4.0
                if self._remaining_in_suit(c.suit) == 0:
                    score += 4.0
                if self._partner_signal_strength(c.suit) > 0:
                    score += self._partner_signal_strength(c.suit) * 1.2
                if self._is_opening_signal_card(c, trump):
                    score += 0.5
                score += pressure * 0.45
            score -= self._point_leak_penalty(c, trump) * (1.1 - 0.25 * pressure)
            scores.append((c, score))
        return self._pick_card(scores)

    def _discard_for_partner(self, legal: list[Card], trick: Trick, trump: str) -> Card:
        non_trump = [c for c in legal if c.suit != trump]
        pool = non_trump if non_trump else legal
        partner_safe = self._partner_win_secure(trick, trump)
        trick_value = self._trick_point_value(trick, trump)
        pressure = self._score_pressure()
        if not partner_safe:
            # Current winner is teammate but not secure yet: preserve points first.
            return self._play_safe_discard(pool, trump, trick_value=trick_value, trick=trick)
        scores: list[tuple[Card, float]] = []
        for c in pool:
            suit_count = sum(1 for hc in pool if hc.suit == c.suit)
            score = 0.0
            schmear_weight = (0.25 if trick_value < 8 else 0.38) + max(0.0, pressure) * 0.18
            score += c.points(trump) * schmear_weight
            score -= suit_count * 0.4
            if self._is_same_suit_signal_card(c, trump):
                score += 0.5
            scores.append((c, score))
        return self._pick_card(scores)

    def _play_safe_discard(
        self,
        legal: list[Card],
        trump: str,
        trick_value: int = 0,
        trick: Trick | None = None,
    ) -> Card:
        """Cannot win — pick the card that leaks the fewest points to opponents.
        Prefer discarding from a short non-trump suit (helps create voids)."""
        non_trump = [c for c in legal if c.suit != trump]
        pool = non_trump if non_trump else legal
        pressure = self._score_pressure()
        scores: list[tuple[Card, float]] = []
        for c in pool:
            suit_count = sum(1 for hc in pool if hc.suit == c.suit)
            score = 0.0
            win_prob = 0.5
            if trick is not None:
                win_prob = self._estimate_team_trick_win_prob(trick, c, trump)
                score += win_prob * (6.0 + trick_value * 0.25 + pressure * 1.3)
            # When we are likely losing this trick, leaking points is very costly.
            leak_weight = (2.2 + trick_value * 0.08) - pressure * 0.35
            score -= c.points(trump) * leak_weight
            score -= self._point_leak_penalty(c, trump) * (1.1 - 0.20 * pressure)
            score -= (1.0 - win_prob) * c.points(trump) * (0.8 - 0.15 * pressure)
            if self.trick_num <= 3:
                score += (3 - min(3, suit_count)) * 0.6
            else:
                score += (2 - min(2, suit_count)) * 0.25
            if self._is_same_suit_signal_card(c, trump):
                score += 0.2
            scores.append((c, score))
        return self._pick_card(scores)

    def _trick_point_value(self, trick: Trick, trump: str) -> int:
        """Total points currently in the trick."""
        return sum(c.points(trump) for _, c in trick)

    def _try_win(self, legal: list[Card], trick: Trick, trump: str) -> Card:
        wi = trick_winner_index(trick, trump)
        winning_card = trick[wi][1]
        beaters = [
            c for c in legal
            if (c.suit == trump and winning_card.suit != trump)
            or (c.suit == winning_card.suit and c.strength(trump) > winning_card.strength(trump))
        ]
        if not beaters:
            return self._play_safe_discard(
                legal,
                trump,
                trick_value=self._trick_point_value(trick, trump),
                trick=trick,
            )

        trick_value = self._trick_point_value(trick, trump)
        is_last_trick = self.trick_num == 7
        pressure = self._score_pressure()
        beater_scores: list[tuple[Card, float]] = []
        beater_probs: dict[str, float] = {}
        for c in beaters:
            win_prob = self._estimate_team_trick_win_prob(trick, c, trump)
            survives = self._beater_likely_holds(c, trick, trump)
            score = win_prob * (6.5 + trick_value * 0.30 + pressure * 1.4)
            score += trick_value * 0.20
            if survives:
                score += 1.6
            else:
                score -= 1.2
            score -= c.points(trump) * (0.60 - 0.12 * pressure)
            score -= self._point_leak_penalty(c, trump) * (1.05 - 0.20 * pressure)
            if is_last_trick:
                score += 2.5
            beater_scores.append((c, score))
            beater_probs[str(c)] = win_prob

        best_beater = self._pick_card(beater_scores)
        best_beater_prob = beater_probs.get(str(best_beater), 0.5)
        fight_threshold = max(3.0, 8.0 - pressure * 4.0)
        worth_fighting = trick_value >= fight_threshold or is_last_trick

        if self.declaring_team == self.team:
            my_total = self.trick_pts[self.team] + self.roem_pts[self.team]
            opp_total = self.trick_pts[1 - self.team] + self.roem_pts[1 - self.team]
            if my_total <= opp_total + 20:
                worth_fighting = True

        if not worth_fighting and best_beater.points(trump) > 0:
            return self._play_safe_discard(legal, trump, trick_value=trick_value, trick=trick)
        if best_beater_prob < 0.45 and best_beater.points(trump) > 0 and not is_last_trick:
            return self._play_safe_discard(legal, trump, trick_value=trick_value, trick=trick)
        if not self._beater_likely_holds(best_beater, trick, trump) and best_beater.points(trump) > 0 and not is_last_trick:
            return self._play_safe_discard(legal, trump, trick_value=trick_value, trick=trick)
        return best_beater

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

        played_indices = {p.seat_idx for p, _ in trick}
        remaining_after_me = [
            (self.seat_idx + step) % 4
            for step in range(1, players_left + 1)
        ]

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
                    if opp in remaining_after_me and opp not in played_indices:
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

    def _partner_win_secure(self, trick: Trick, trump: str) -> bool:
        if len(trick) == 3:
            return True
        if len(trick) != 2:
            return False
        wi = trick_winner_index(trick, trump)
        winning_card = trick[wi][1]
        if winning_card.suit != trump:
            return False
        strongest_opp = self._highest_remaining_trump(trump)
        return strongest_opp is None or winning_card.strength(trump) > TRUMP_ORDER.index(strongest_opp)

    def _is_same_suit_signal_card(self, card: Card, trump: str) -> bool:
        if self.signal_profile != "core":
            return False
        if card.suit == trump:
            return False
        ranks = {c.rank for c in self.hand if c.suit == card.suit}
        return card.rank in ("7", "8", "9") and ("A" in ranks or ("10" in ranks and "A" in self.played_cards))

    def _is_opening_signal_card(self, card: Card, trump: str) -> bool:
        if self.signal_profile != "core" or self.trick_num != 0:
            return False
        if card.suit == trump:
            return card.rank in ("J", "9", "A")
        return card.rank in ("A", "K")

    def _decode_partner_signal(self, card: Card, lead_suit: str | None) -> None:
        trump = self.current_trump
        if self.signal_profile != "core" or trump is None:
            return

        # Opening signal: partner's first lead tells us about their hand.
        if self.trick_num == 0 and lead_suit is None:
            if card.suit == trump and card.rank in ("J", "9", "A"):
                self.partner_signals["trump_pull"] = {"confidence": 0.9, "source_trick": self.trick_num}
            elif card.suit != trump:
                self.partner_signals["opening_suit"] = {"suit": card.suit, "confidence": 0.75, "source_trick": self.trick_num}

        # Same-suit control signal: low card in a non-trump suit.
        if lead_suit is not None and card.suit != trump and card.rank in ("7", "8", "9"):
            cur = self.partner_signals.get(card.suit, {"confidence": 0.0, "meaning": "same_suit_control"})
            cur["confidence"] = min(self.SIGNAL_MAX_CONFIDENCE, cur.get("confidence", 0.0) + 0.35)
            cur["source_trick"] = self.trick_num
            cur["meaning"] = "same_suit_control"
            self.partner_signals[card.suit] = cur

        # Enhanced signals (Grandmaster only): discard attitude and void tracking.
        if not self.lookahead_enhanced:
            return

        # Discard signal: when partner can't follow suit and doesn't trump,
        # the suit they discard FROM is one they're weak in (negative signal),
        # and a high discard encourages the discarded suit, low discourages.
        if lead_suit is not None and card.suit != lead_suit and card.suit != trump:
            discard_suit = card.suit
            key = f"discard_{discard_suit}"
            if card.rank in ("A", "10", "K"):
                # High discard = attitude signal: "I have strength in this suit"
                cur = self.partner_signals.get(discard_suit, {"confidence": 0.0, "meaning": "same_suit_control"})
                cur["confidence"] = min(self.SIGNAL_MAX_CONFIDENCE, cur.get("confidence", 0.0) + 0.25)
                cur["source_trick"] = self.trick_num
                cur["meaning"] = "same_suit_control"
                self.partner_signals[discard_suit] = cur
            elif card.rank in ("7", "8") and discard_suit not in self.partner_signals:
                # Low discard from a new suit = "I don't care about this suit"
                self.partner_signals[key] = {
                    "confidence": 0.4,
                    "source_trick": self.trick_num,
                    "meaning": "weak_suit",
                }

        # Void signal: partner trumped in → they are void in led suit.
        # Store as a negative signal so we avoid leading that suit to them.
        if lead_suit is not None and card.suit == trump and lead_suit != trump:
            void_key = f"partner_void_{lead_suit}"
            self.partner_signals[void_key] = {
                "confidence": 1.0,
                "source_trick": self.trick_num,
                "meaning": "partner_void",
            }

    def _decay_signals(self) -> None:
        to_drop: list[str] = []
        for key, signal in self.partner_signals.items():
            if isinstance(signal, dict) and "confidence" in signal:
                signal["confidence"] = max(0.0, signal["confidence"] - 0.015)
                if signal["confidence"] <= 0.05:
                    to_drop.append(key)
        for key in to_drop:
            self.partner_signals.pop(key, None)

    def _partner_signal_strength(self, suit: str) -> float:
        """Net signal strength for a suit: positive = partner has strength,
        negative = partner is weak or void."""
        # Basic signal (all profiles).
        signal = self.partner_signals.get(suit)
        if not self.lookahead_enhanced:
            if not signal:
                return 0.0
            return float(signal.get("confidence", 0.0))

        # Enhanced: combine positive, negative, and void signals.
        strength = 0.0

        if signal and signal.get("meaning") == "same_suit_control":
            strength += float(signal.get("confidence", 0.0))

        weak = self.partner_signals.get(f"discard_{suit}")
        if weak and weak.get("meaning") == "weak_suit":
            strength -= float(weak.get("confidence", 0.0)) * 0.6

        void_sig = self.partner_signals.get(f"partner_void_{suit}")
        if void_sig:
            strength -= float(void_sig.get("confidence", 0.0)) * 0.8

        return strength

    def _cards_left_by_seat(self, trick: Trick) -> dict[int, int]:
        base = 8 - self.trick_num
        already_played = {p.seat_idx for p, _ in trick}
        return {
            seat: base - (1 if seat in already_played else 0)
            for seat in range(4)
        }

    # ── Lookahead (3-trick depth-limited search) ────────────────────────────

    def _prune_moves(self, legal: list[Card], trick_cards: list[tuple[int, Card]], trump: str) -> list[Card]:
        """Keep only the top LOOKAHEAD_BRANCH_LIMIT moves by heuristic.

        When lookahead_enhanced is True, uses context-aware scoring
        (partner winning, schmear, safe discard).  Otherwise uses a
        simple strength + suit heuristic.
        """
        if not self.lookahead_enhanced:
            # Simple pruning (original behaviour).
            scored: list[tuple[float, Card]] = []
            for card in legal:
                s = float(card.strength(trump))
                if card.suit == trump:
                    s += 2.0
                if trick_cards:
                    lead_suit = trick_cards[0][1].suit
                    if card.suit == lead_suit:
                        s += 1.0
                s += card.points(trump) * 0.05
                scored.append((s, card))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [card for _, card in scored[: self.LOOKAHEAD_BRANCH_LIMIT]]

        # Context-aware pruning (enhanced).
        partner_winning = False
        if trick_cards:
            sim_trick = [
                (SimpleNamespace(seat_idx=seat, team=SEAT_TEAMS[seat]), card)
                for seat, card in trick_cards
            ]
            wi = trick_winner_index(sim_trick, trump)
            partner_winning = SEAT_TEAMS[trick_cards[wi][0]] == self.team

        scored: list[tuple[float, Card]] = []
        for card in legal:
            s = float(card.strength(trump))

            if card.suit == trump:
                s += 2.0
            if trick_cards:
                lead_suit = trick_cards[0][1].suit
                if card.suit == lead_suit:
                    s += 1.0
                if partner_winning:
                    s += card.points(trump) * 0.15
                else:
                    s -= card.points(trump) * 0.08
            else:
                s += card.points(trump) * 0.05

            if card.points(trump) == 0 and card.suit != trump:
                s += 0.5

            scored.append((s, card))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [card for _, card in scored[: self.LOOKAHEAD_BRANCH_LIMIT]]

    def _sample_hands(self, trick: Trick) -> dict[int, list[Card]] | None:
        """Randomly sample a consistent hand distribution for all seats.

        Unlike _determinize_endgame_hands (which uses backtracking for an exact
        solution with few cards), this shuffles the available pool and deals
        cards respecting the possible-cards constraints.  Returns None if a
        consistent deal cannot be found.
        """
        counts = self._cards_left_by_seat(trick)
        my_cards = {str(c) for c in self.hand}
        available = self._all_card_strings() - self.played_cards - my_cards

        hands: dict[int, list[Card]] = {self.seat_idx: list(self.hand)}
        targets = [seat for seat in range(4) if seat != self.seat_idx]

        # Build constrained pools per seat.
        pools: dict[int, list[str]] = {}
        for seat in targets:
            poss = self.possible_cards_by_seat.get(seat, set())
            pool = sorted(poss & available)
            if len(pool) < counts[seat]:
                pool = sorted(available)
            pools[seat] = pool

        # Greedy assignment: most-constrained seat first, shuffle to randomise.
        assigned: set[str] = set()
        order = sorted(targets, key=lambda s: len(pools[s]))
        for seat in order:
            pool = [cs for cs in pools[seat] if cs not in assigned]
            need = counts[seat]
            if len(pool) < need:
                return None
            self.rng.shuffle(pool)
            picked = pool[:need]
            hands[seat] = [self._card_from_str(cs) for cs in picked]
            assigned.update(picked)

        return hands

    def _leaf_eval(self, hands: dict[int, list[Card]], trump: str) -> float:
        """Positional heuristic for unsearched tricks beyond the lookahead.

        Returns a score from our team's perspective: positive is good for us.
        Kept cheap — called once per leaf node.
        """
        if all(len(h) == 0 for h in hands.values()):
            return 0.0

        my_team_strength = 0.0
        opp_team_strength = 0.0
        for seat in range(4):
            val = 0.0
            for c in hands.get(seat, []):
                # High cards are worth roughly their point value in expectation.
                val += c.points(trump) * 0.3
                # Trump cards are more valuable (control).
                if c.suit == trump:
                    val += c.strength(trump) * 0.5
                else:
                    val += c.strength(trump) * 0.15
            # Voids are advantageous (can trump in).
            suits_in_hand = {c.suit for c in hands.get(seat, [])}
            non_trump_suits = {s for s in SUITS if s != trump}
            voids = len(non_trump_suits - suits_in_hand)
            val += voids * 1.5

            if SEAT_TEAMS[seat] == self.team:
                my_team_strength += val
            else:
                opp_team_strength += val

        return (my_team_strength - opp_team_strength) * 0.15

    def _lookahead_minimax(
        self,
        hands: dict[int, list[Card]],
        trick_cards: list[tuple[int, Card]],
        next_seat: int,
        trump: str,
        tricks_remaining: int,
        memo: dict,
        alpha: float = float("-inf"),
        beta: float = float("inf"),
    ) -> float:
        """Depth-limited minimax with alpha-beta pruning.

        Searches up to *tricks_remaining* full tricks (4 cards each).  At the
        depth boundary a leaf evaluation is returned combining accumulated
        points with a positional heuristic for the remaining hand.
        """
        # A trick is complete — resolve it and start the next trick.
        if len(trick_cards) == 4:
            sim_trick = [
                (SimpleNamespace(seat_idx=seat, team=SEAT_TEAMS[seat]), card)
                for seat, card in trick_cards
            ]
            wi = trick_winner_index(sim_trick, trump)
            winner_seat = trick_cards[wi][0]
            trick_pts = sum(card.points(trump) for _, card in trick_cards)
            last_bonus = 10 if sum(len(h) for h in hands.values()) == 0 else 0
            gain = trick_pts + last_bonus
            delta = gain if SEAT_TEAMS[winner_seat] == self.team else -gain

            if tricks_remaining <= 1 or all(len(h) == 0 for h in hands.values()):
                return delta
            return delta + self._lookahead_minimax(
                hands, [], winner_seat, trump, tricks_remaining - 1, memo,
                alpha, beta,
            )

        # All hands empty — nothing left.
        if all(len(h) == 0 for h in hands.values()):
            return 0.0

        key = self._endgame_state_key(hands, trick_cards, next_seat)
        if key in memo:
            return memo[key]

        legal = self._legal_moves_for_cards(hands[next_seat], trick_cards, trump)
        if not legal:
            memo[key] = 0.0
            return 0.0

        # Prune to top-K moves to keep the search tractable.
        if len(legal) > self.LOOKAHEAD_BRANCH_LIMIT:
            # Always use simple pruning inside minimax — context-aware pruning
            # is too speculative and can exclude the optimal move.
            scored_p: list[tuple[float, Card]] = []
            for card in legal:
                s = float(card.strength(trump))
                if card.suit == trump:
                    s += 2.0
                if trick_cards:
                    lead_suit = trick_cards[0][1].suit
                    if card.suit == lead_suit:
                        s += 1.0
                s += card.points(trump) * 0.05
                scored_p.append((s, card))
            scored_p.sort(key=lambda x: x[0], reverse=True)
            legal = [card for _, card in scored_p[: self.LOOKAHEAD_BRANCH_LIMIT]]

        is_max = SEAT_TEAMS[next_seat] == self.team
        best = float("-inf") if is_max else float("inf")
        cutoff = False

        for card in legal:
            new_hands = {seat: list(cards) for seat, cards in hands.items()}
            cs = str(card)
            for i, c in enumerate(new_hands[next_seat]):
                if str(c) == cs:
                    del new_hands[next_seat][i]
                    break

            new_trick = trick_cards + [(next_seat, card)]
            nxt = (next_seat + 1) % 4
            val = self._lookahead_minimax(
                new_hands, new_trick, nxt, trump, tricks_remaining, memo,
                alpha, beta,
            )
            if is_max:
                best = max(best, val)
                if self.lookahead_enhanced:
                    alpha = max(alpha, best)
            else:
                best = min(best, val)
                if self.lookahead_enhanced:
                    beta = min(beta, best)
            if self.lookahead_enhanced and beta <= alpha:
                cutoff = True
                break

        # Only memo exact values; pruned results depend on the alpha-beta window.
        if not cutoff:
            memo[key] = best
        return best

    def _adaptive_lookahead_depth(self) -> int:
        """Return search depth (in tricks) based on hand size.

        Fewer cards → less branching → can search deeper.
        Never goes below the configured base depth.
        """
        n = len(self.hand)
        if n <= 4:
            return max(self.lookahead_depth, 4)
        if n <= 5:
            return max(self.lookahead_depth, 3)
        return self.lookahead_depth

    def _lookahead_choice(self, legal: list[Card], trick: Trick, trump: str) -> Card | None:
        """Pick a card by averaging depth-limited minimax over sampled hands.

        Returns None if sampling fails consistently (caller falls through to
        heuristic play).
        """
        trick_cards_prefix = [(p.seat_idx, c) for p, c in trick]
        next_after_me = (self.seat_idx + 1) % 4
        depth = self.lookahead_depth

        totals: dict[str, float] = {str(c): 0.0 for c in legal}
        counts: dict[str, int] = {str(c): 0 for c in legal}

        for _ in range(self.lookahead_samples):
            hands = self._sample_hands(trick)
            if hands is None:
                continue

            for card in legal:
                test_hands = {seat: list(cards) for seat, cards in hands.items()}
                cs = str(card)
                removed = False
                for i, c in enumerate(test_hands[self.seat_idx]):
                    if str(c) == cs:
                        del test_hands[self.seat_idx][i]
                        removed = True
                        break
                if not removed:
                    continue

                tc = trick_cards_prefix + [(self.seat_idx, card)]
                score = self._lookahead_minimax(
                    test_hands, tc, next_after_me, trump,
                    depth, memo={},
                )
                totals[cs] += score
                counts[cs] += 1

        scores: list[tuple[Card, float]] = []
        for card in legal:
            cs = str(card)
            if counts[cs] > 0:
                scores.append((card, totals[cs] / counts[cs]))

        if not scores:
            return None
        return self._pick_card(scores)

    def _determinize_endgame_hands(self, trick: Trick) -> dict[int, list[Card]] | None:
        counts = self._cards_left_by_seat(trick)
        hands: dict[int, list[Card]] = {self.seat_idx: list(self.hand)}

        available = self._all_card_strings() - self.played_cards - {str(c) for c in self.hand}
        targets = [seat for seat in range(4) if seat != self.seat_idx]

        options_by_seat: dict[int, set[str]] = {}
        for seat in targets:
            opts = set(self.possible_cards_by_seat.get(seat, set())) & set(available)
            if not opts:
                opts = set(available)
            options_by_seat[seat] = opts

        order = sorted(targets, key=lambda s: len(options_by_seat[s]))
        assigned: dict[int, set[str]] = {}

        def backtrack(i: int, avail: set[str]) -> bool:
            if i == len(order):
                return True
            seat = order[i]
            need = counts[seat]
            opts = sorted(options_by_seat[seat] & avail)
            if len(opts) < need:
                return False
            for combo in combinations(opts, need):
                combo_set = set(combo)
                assigned[seat] = combo_set
                if backtrack(i + 1, avail - combo_set):
                    return True
            assigned.pop(seat, None)
            return False

        if not backtrack(0, set(available)):
            return None

        for seat in targets:
            cards = [self._card_from_str(cs) for cs in sorted(assigned.get(seat, set()))]
            if len(cards) != counts[seat]:
                return None
            hands[seat] = cards
        return hands

    def _endgame_state_key(
        self,
        hands: dict[int, list[Card]],
        trick_cards: list[tuple[int, Card]],
        next_seat: int,
    ) -> tuple:
        hand_key = tuple(
            tuple(sorted(str(c) for c in hands[seat]))
            for seat in range(4)
        )
        trick_key = tuple((seat, str(card)) for seat, card in trick_cards)
        return hand_key, trick_key, next_seat

    def _endgame_minimax(
        self,
        hands: dict[int, list[Card]],
        trick_cards: list[tuple[int, Card]],
        next_seat: int,
        trump: str,
        memo: dict,
    ) -> float:
        if len(trick_cards) == 4:
            sim_trick = [
                (SimpleNamespace(seat_idx=seat, team=SEAT_TEAMS[seat]), card)
                for seat, card in trick_cards
            ]
            wi = trick_winner_index(sim_trick, trump)
            winner_seat = trick_cards[wi][0]
            trick_pts = sum(card.points(trump) for _, card in trick_cards)
            last_bonus = 10 if sum(len(h) for h in hands.values()) == 0 else 0
            gain = trick_pts + last_bonus
            delta = gain if SEAT_TEAMS[winner_seat] == self.team else -gain
            return delta + self._endgame_minimax(hands, [], winner_seat, trump, memo)

        if all(len(h) == 0 for h in hands.values()):
            return 0.0

        key = self._endgame_state_key(hands, trick_cards, next_seat)
        if key in memo:
            return memo[key]

        legal = self._legal_moves_for_cards(hands[next_seat], trick_cards, trump)
        if not legal:
            memo[key] = 0.0
            return 0.0

        is_max = SEAT_TEAMS[next_seat] == self.team
        best = float("-inf") if is_max else float("inf")

        for card in legal:
            new_hands = {seat: list(cards) for seat, cards in hands.items()}
            # Remove by card identity where possible; fall back to string match.
            if card in new_hands[next_seat]:
                new_hands[next_seat].remove(card)
            else:
                cs = str(card)
                for i, c in enumerate(new_hands[next_seat]):
                    if str(c) == cs:
                        del new_hands[next_seat][i]
                        break

            new_trick = trick_cards + [(next_seat, card)]
            nxt = (next_seat + 1) % 4
            val = self._endgame_minimax(new_hands, new_trick, nxt, trump, memo)
            if is_max:
                best = max(best, val)
            else:
                best = min(best, val)

        memo[key] = best
        return best

    def _endgame_exact_choice(self, legal: list[Card], trick: Trick, trump: str) -> Card | None:
        hands = self._determinize_endgame_hands(trick)
        if not hands:
            return None

        scores: list[tuple[Card, float]] = []
        for card in legal:
            test_hands = {seat: list(cards) for seat, cards in hands.items()}
            cs = str(card)
            removed = False
            for i, c in enumerate(test_hands[self.seat_idx]):
                if str(c) == cs:
                    del test_hands[self.seat_idx][i]
                    removed = True
                    break
            if not removed:
                continue
            trick_cards = [(p.seat_idx, c) for p, c in trick] + [(self.seat_idx, card)]
            score = self._endgame_minimax(test_hands, trick_cards, (self.seat_idx + 1) % 4, trump, memo={})
            scores.append((card, score))
        if not scores:
            return None
        return self._pick_card(scores)


# ─── Game ─────────────────────────────────────────────────────────────────────


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
        ai_seed_base: int | None = None,
        ai_signal_profile: str = "core",
        ai_strength: str = "expert",
        rules_variant: str = "rotterdam",
        game_seed: int | None = None,
        replay_output_path: str | None = None,
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

        self.game_seed = game_seed if game_seed is not None else random.randrange(1, 2**31)
        self.rng = random.Random(self.game_seed)
        self.ai_seed_base = ai_seed_base if ai_seed_base is not None else (self.game_seed * 17 + 11)
        self.ai_signal_profile = ai_signal_profile
        self.ai_strength = ai_strength
        self.rules_variant = rules_variant if rules_variant in {"rotterdam", "amsterdam"} else "rotterdam"
        self.replay_output_path = replay_output_path
        self._current_round_replay: dict | None = None

        self.players: list[Player] = []
        for seat in range(4):
            team = SEAT_TEAMS[seat]
            if seat in human_seats:
                name = human_seats[seat]
                self.players.append(HumanPlayer(name, team, seat_idx=seat))
            else:
                name = f"AI {SEAT_DEFAULTS[seat]}"
                self.players.append(
                    AIPlayer(
                        name,
                        team,
                        seat_idx=seat,
                        rng_seed=self.ai_seed_base + seat,
                        signal_profile=ai_signal_profile,
                        ai_strength=ai_strength,
                    )
                )

        for p in self.players:
            p.rules_variant = self.rules_variant

        self.scores = [0, 0]
        self.log = log_fn or (lambda msg, tag="": print(msg))
        self.notify = state_fn or (lambda event, data: None)
        self._next_round_event = threading.Event()
        self.game_mode = game_mode
        self.score_limit = score_limit
        self.boom_rounds = 16
        self.replay_data: dict = {
            "version": 1,
            "game_seed": self.game_seed,
            "ai_seed_base": self.ai_seed_base,
            "ai_strength": self.ai_strength,
            "ai_signal_profile": self.ai_signal_profile,
            "rules_variant": self.rules_variant,
            "game_mode": self.game_mode,
            "score_limit": self.score_limit,
            "rounds": [],
            "scores_after": [],
            "winner": None,
        }

    def signal_next_round(self) -> None:
        """Called by the web layer when the host advances to the next round."""
        self._next_round_event.set()

    def get_replay_data(self) -> dict:
        """Return a deep copy of the deterministic replay trace for this game."""
        return copy.deepcopy(self.replay_data)

    def save_replay(self, path: str) -> None:
        """Persist replay trace to JSON on disk."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.replay_data, f, ensure_ascii=False, indent=2)

    def _game_continues(self, round_num: int) -> bool:
        """Return True if more rounds should be played."""
        if self.game_mode == "score_limit":
            return max(self.scores) < self.score_limit
        if self.game_mode == "boom":
            return round_num < self.boom_rounds
        # free_play: always continues (ended only by GameInterrupt)
        return True

    def play(self) -> None:
        self.replay_data["boom_rounds"] = self.boom_rounds
        dealer = self.rng.randint(0, 3)
        round_num = 0

        while self._game_continues(round_num):
            round_num += 1
            first_bidder = (dealer + 1) % 4
            round_seed = self.rng.randrange(1, 2**31)
            self.log(f"\n{'='*40}", "round")
            self.log(f"Round {round_num}  (dealer: {self.players[dealer].name})", "round")

            for p in self.players:
                if isinstance(p, AIPlayer):
                    p.game_scores = list(self.scores)
                    p.game_mode = self.game_mode
                    p.score_limit = self.score_limit
                    p.boom_rounds = self.boom_rounds
                    p.round_num = round_num

            t0, t1, leader, history = self._play_round(
                first_bidder,
                round_seed=round_seed,
                dealer_idx=dealer,
                round_num=round_num,
            )

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
            self.log(f"Round result  →  Team A: +{t0}   Team B: +{t1}")
            self.log(f"Running total →  Team A: {self.scores[0]}   Team B: {self.scores[1]}")
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
            self.replay_data["winner"] = winner
            self.log(f"GAME OVER – Team {winner} wins!  ({self.scores[0]} – {self.scores[1]})")
            self.notify("game_over", {"winner": winner, "scores": list(self.scores)})
        self.replay_data["scores_after"] = list(self.scores)
        if self.replay_output_path:
            self.save_replay(self.replay_output_path)

    def _bidding(self, first_bidder: int, rng: random.Random) -> tuple[int, str]:
        """Two-round random-suit bidding."""
        suits = rng.sample(SUITS, 2)
        if self._current_round_replay is not None:
            self._current_round_replay["offered_suits"] = list(suits)

        self.log("── Bidding ──")
        for round_num, suit in enumerate(suits, 1):
            self.notify("trump_offered", {"suit": suit, "round_num": round_num})
            self.log(f"  Offered: {suit} ({SUIT_NAMES[suit]})  [round {round_num}]")

            for i in range(4):
                bidder_idx = (first_bidder + i) % 4
                player = self.players[bidder_idx]

                t_start = time.monotonic()
                declared = player.choose_trump(suit, False)
                elapsed = time.monotonic() - t_start
                if self._current_round_replay is not None:
                    self._current_round_replay["bids"].append({
                        "round_num": round_num,
                        "player_idx": bidder_idx,
                        "suit": suit,
                        "declare": bool(declared),
                        "forced": False,
                    })

                if declared:
                    self.log(
                        f"  {player.name} declares: {suit} ({SUIT_NAMES[suit]})", "trump"
                    )
                    self.notify("bid", {"player_idx": bidder_idx, "trump": suit})
                    if not isinstance(player, HumanPlayer):
                        remaining_delay = AI_BID_DELAY - elapsed
                        if remaining_delay > 0:
                            time.sleep(remaining_delay)
                    return bidder_idx, suit
                else:
                    self.log(f"  {player.name} passes")
                    self.notify("bid", {"player_idx": bidder_idx, "trump": None})
                    if not isinstance(player, HumanPlayer):
                        remaining_delay = AI_BID_DELAY - elapsed
                        if remaining_delay > 0:
                            time.sleep(remaining_delay)

        # All 8 players passed both rounds — force first bidder on round-2 suit
        forced_suit = suits[1]
        player = self.players[first_bidder]
        player.choose_trump(forced_suit, True)
        self.log(
            f"  {player.name} is forced to declare: {forced_suit} ({SUIT_NAMES[forced_suit]})",
            "trump",
        )
        self.notify("bid", {"player_idx": first_bidder, "trump": forced_suit})
        if self._current_round_replay is not None:
            self._current_round_replay["bids"].append({
                "round_num": 2,
                "player_idx": first_bidder,
                "suit": forced_suit,
                "declare": True,
                "forced": True,
            })
        return first_bidder, forced_suit

    def _play_round(
        self,
        first_leader: int,
        round_seed: int,
        dealer_idx: int,
        round_num: int,
    ) -> tuple[int, int, int, dict]:
        """Returns (team0_pts, team1_pts, last_trick_winner_idx, history_record)."""
        round_rng = random.Random(round_seed)
        round_replay = {
            "round_num": round_num,
            "round_seed": round_seed,
            "dealer_idx": dealer_idx,
            "first_bidder_idx": first_leader,
            "hands": {},
            "offered_suits": [],
            "bids": [],
            "tricks": [],
            "trump": None,
            "declaring_player_idx": None,
            "declaring_team": None,
            "trick_pts": [0, 0],
            "roem_pts": [0, 0],
            "round_pts": [0, 0],
            "nat": False,
            "scores_after": [0, 0],
        }
        self._current_round_replay = round_replay

        for p in self.players:
            p.start_round()

        deck = Deck(rng=round_rng)
        hands = deck.deal(4, 8)
        for i, p in enumerate(self.players):
            p.receive_hand(hands[i])
        round_replay["hands"] = {str(i): [str(c) for c in hands[i]] for i in range(4)}

        self.notify("deal_done", {})
        declaring_player_idx, trump = self._bidding(first_leader, round_rng)
        declaring_team = self.players[declaring_player_idx].team
        declaring_name = self.players[declaring_player_idx].name
        round_replay["trump"] = trump
        round_replay["declaring_player_idx"] = declaring_player_idx
        round_replay["declaring_team"] = declaring_team

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
                    p.current_trump = trump

            self.log(f"Trick {trick_num + 1}")
            winner_idx, pts, trick_cards, cards_played, trick_cards_by_seat = self._play_trick(leader, trump)
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
            round_replay["tricks"].append({
                "trick_num": trick_num + 1,
                "leader_idx": leader,
                "winner_idx": winner_idx,
                "points": pts + bonus,
                "cards": [
                    {"player_idx": player_idx, "card": card_str}
                    for player_idx, card_str in trick_cards_by_seat
                ],
                "roem": [
                    {"description": desc, "points": roem_pts_item}
                    for desc, roem_pts_item in roem_items
                ],
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

        round_replay["trick_pts"] = list(trick_pts)
        round_replay["roem_pts"] = list(roem_pts)
        round_replay["round_pts"] = list(round_pts)
        round_replay["nat"] = nat
        round_replay["scores_after"] = [self.scores[0] + round_pts[0], self.scores[1] + round_pts[1]]
        self.replay_data["rounds"].append(round_replay)
        self._current_round_replay = None

        return round_pts[0], round_pts[1], leader, history

    def _play_trick(
        self, leader: int, trump: str
    ) -> tuple[int, int, list[tuple[str, str]], list[Card], list[tuple[int, str]]]:
        """Play one trick.
        Returns (winner_idx, trick_pts, [(player_name, card_str)], [card_objects], [(player_idx, card_str)])."""
        trick: Trick = []
        trick_cards: list[tuple[str, str]] = []
        cards_played: list[Card] = []
        trick_cards_by_seat: list[tuple[int, str]] = []

        for offset in range(4):
            idx = (leader + offset) % 4
            player = self.players[idx]
            t_start = time.monotonic()
            card = player.choose_card(trick, trump)
            elapsed = time.monotonic() - t_start
            trick.append((player, card))
            trick_cards.append((player.name, str(card)))
            cards_played.append(card)
            trick_cards_by_seat.append((idx, str(card)))

            lead_suit = trick[0][1].suit if trick else None
            for p in self.players:
                p.observe_card(card)
                p.observe_trick_play(idx, card, lead_suit)

            self.notify("trick_played", {"player_idx": idx, "card": card})
            if not isinstance(player, HumanPlayer):
                self.log(f"  {player.name} plays: {card}")
                remaining_delay = AI_PLAY_DELAY - elapsed
                if remaining_delay > 0:
                    time.sleep(remaining_delay)

        wi = trick_winner_index(trick, trump)
        winner_global = self.players.index(trick[wi][0])
        total_pts = sum(c.points(trump) for _, c in trick)
        return winner_global, total_pts, trick_cards, cards_played, trick_cards_by_seat
