"""Encode Klaverjassen bidding state into a fixed-size feature vector."""

import numpy as np

from klaverjas.constants import RANKS, SUITS
from klaverjas.core import Card, find_roem

NUM_BID_FEATURES = 86


def encode_bid_state(
    hand: list[Card],
    trump_suit: str,
    bid_position: int,
    bid_round: int,
    seat_idx: int,
    team: int,
    game_scores: list[int],
    round_num: int,
    round_1_suit: str | None = None,
) -> np.ndarray:
    """Encode the bidding state into a float32 feature vector.

    Args:
        hand: The 8 cards in the player's hand.
        trump_suit: The suit being offered as trump.
        bid_position: 0-3 (0 = first to bid this round).
        bid_round: 1 or 2.
        seat_idx: Player's seat (0-3).
        team: Player's team (0 or 1).
        game_scores: [team0_score, team1_score].
        round_num: Current round number in the game.
        round_1_suit: The suit offered in round 1 (None if currently in round 1).
    """
    opp_team = 1 - team
    features = np.zeros(NUM_BID_FEATURES, dtype=np.float32)
    offset = 0

    # 1. Full hand encoding (32) — which cards are in hand
    for c in hand:
        si = SUITS.index(c.suit)
        ri = RANKS.index(c.rank)
        features[offset + si * 8 + ri] = 1.0
    offset += 32  # 32

    # 2. Trump-specific card indicators (4) — has J, 9, A, 10 of trump
    trump_ranks = {c.rank for c in hand if c.suit == trump_suit}
    for i, rank in enumerate(["J", "9", "A", "10"]):
        if rank in trump_ranks:
            features[offset + i] = 1.0
    offset += 4  # 36

    # 3. Trump count normalized (1)
    trump_count = sum(1 for c in hand if c.suit == trump_suit)
    features[offset] = trump_count / 8.0
    offset += 1  # 37

    # 4. Side suit structure (3 non-trump suits × 6 features = 18)
    #    Per suit: count/8, has_ace, has_10, has_king, is_void, is_singleton
    #    Sorted by canonical SUITS order, skipping trump
    side_suits = [s for s in SUITS if s != trump_suit]
    for suit in side_suits:
        cards_in_suit = [c for c in hand if c.suit == suit]
        ranks_in_suit = {c.rank for c in cards_in_suit}
        count = len(cards_in_suit)
        features[offset] = count / 8.0
        features[offset + 1] = 1.0 if "A" in ranks_in_suit else 0.0
        features[offset + 2] = 1.0 if "10" in ranks_in_suit else 0.0
        features[offset + 3] = 1.0 if "K" in ranks_in_suit else 0.0
        features[offset + 4] = 1.0 if count == 0 else 0.0
        features[offset + 5] = 1.0 if count == 1 else 0.0
        offset += 6
    # offset = 37 + 18 = 55

    # 5. Roem value normalized (1)
    roem_items = find_roem(hand, trump_suit)
    roem_total = sum(p for _, p in roem_items)
    features[offset] = min(roem_total / 200.0, 1.0)
    offset += 1  # 56

    # 6. Has stuk — K+Q of trump (1)
    features[offset] = 1.0 if {"K", "Q"}.issubset(trump_ranks) else 0.0
    offset += 1  # 57

    # 7. Bidding position one-hot (4)
    features[offset + min(bid_position, 3)] = 1.0
    offset += 4  # 61

    # 8. Bidding round one-hot (2)
    features[offset + min(bid_round - 1, 1)] = 1.0
    offset += 2  # 63

    # 9. Game scores normalized (2)
    features[offset] = game_scores[team] / 500.0
    features[offset + 1] = game_scores[opp_team] / 500.0
    offset += 2  # 65

    # 10. Score difference normalized (1)
    features[offset] = (game_scores[team] - game_scores[opp_team]) / 500.0
    offset += 1  # 66

    # 11. Round progress normalized (1)
    features[offset] = round_num / 16.0
    offset += 1  # 67

    # 12. Trump suit one-hot (4)
    features[offset + SUITS.index(trump_suit)] = 1.0
    offset += 4  # 71

    # 13. Seat one-hot (4)
    features[offset + seat_idx] = 1.0
    offset += 4  # 75

    # 14. Number of trump cards raw (1) — complements the normalized version
    features[offset] = trump_count
    offset += 1  # 76

    # 15. Has trump J+9 combo (1) — the two strongest trumps
    features[offset] = 1.0 if "J" in trump_ranks and "9" in trump_ranks else 0.0
    offset += 1  # 77

    # 16. Number of voids in side suits (1)
    voids = sum(1 for s in side_suits if not any(c.suit == s for c in hand))
    features[offset] = voids / 3.0
    offset += 1  # 78

    # 17. Number of side aces (1)
    side_aces = sum(1 for c in hand if c.suit != trump_suit and c.rank == "A")
    features[offset] = side_aces / 3.0
    offset += 1  # 79

    # 18. Round 1 rejected suit one-hot (4) — which suit everyone passed on
    #     All zeros if currently in round 1 (round_1_suit is None)
    if round_1_suit is not None:
        features[offset + SUITS.index(round_1_suit)] = 1.0
    offset += 4  # 83

    # 19. Unprotected 10 per side suit (3) — has 10 but no Ace
    for suit in side_suits:
        ranks_in_suit = {c.rank for c in hand if c.suit == suit}
        if "10" in ranks_in_suit and "A" not in ranks_in_suit:
            features[offset] = 1.0
        offset += 1
    # offset = 86

    assert offset == NUM_BID_FEATURES, f"Bid feature count mismatch: {offset} != {NUM_BID_FEATURES}"
    return features
