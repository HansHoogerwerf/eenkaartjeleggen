"""Encode Klaverjassen game state into a fixed-size feature vector for neural network input."""

import numpy as np

from klaverjas.constants import (
    NON_TRUMP_ORDER,
    RANKS,
    SEAT_TEAMS,
    SUITS,
    TRUMP_ORDER,
    TRUMP_STRONGEST_FIRST,
)
from klaverjas.core import Card, Trick, trick_roem_points

# Canonical card ordering: 32 cards indexed by (suit_idx * 8 + rank_idx).
CARD_INDEX: dict[str, int] = {}
for _si, _suit in enumerate(SUITS):
    for _ri, _rank in enumerate(RANKS):
        CARD_INDEX[f"{_rank}{_suit}"] = _si * 8 + _ri

NUM_CARDS = 32
NUM_SUITS = 4

# Feature layout versions.
#   v1 (267): the original encoding; every checkpoint trained before the
#             roem-aware retrain (models/neural_best.pt) expects this.
#   v2 (300): v1 + 33 trick-roem features appended at the end (see the
#             "34."/"35." blocks in encode_state).  Offsets of the v1 blocks —
#             in particular the legal-move mask at 190 — are unchanged, so
#             tooling that slices the mask keeps working for both versions.
NUM_FEATURES = 267
NUM_ROEM_FEATURES = NUM_CARDS + 1
NUM_FEATURES_V2 = NUM_FEATURES + NUM_ROEM_FEATURES
FEATURE_SIZES = {1: NUM_FEATURES, 2: NUM_FEATURES_V2}
LEGAL_MASK_OFFSET = 190
ROEM_NORM = 100.0


def feature_version_for_size(num_features: int) -> int:
    """Map an input width (e.g. from a checkpoint's first Linear layer) to a
    feature-layout version.  Raises ValueError for unknown widths."""
    for version, size in FEATURE_SIZES.items():
        if size == num_features:
            return version
    raise ValueError(f"Unknown feature width {num_features}; known: {FEATURE_SIZES}")


def card_to_idx(card: Card) -> int:
    return CARD_INDEX[str(card)]


def card_roem_deltas(
    trick_cards: list[Card],
    candidates: list[Card],
    trump: str,
) -> tuple[np.ndarray, int]:
    """Per-card roem each candidate would ADD to the trick if played now.

    Returns (deltas[32] in raw points, roem already on the table).  Cards not
    in *candidates* get 0.  When leading (empty trick) every delta is 0 — a
    single card never forms roem on its own.
    """
    deltas = np.zeros(NUM_CARDS, dtype=np.float32)
    if not trick_cards:
        return deltas, 0
    base = trick_roem_points(trick_cards, trump)
    for c in candidates:
        added = trick_roem_points(trick_cards + [c], trump) - base
        if added:
            deltas[card_to_idx(c)] = float(added)
    return deltas, base


def encode_state(
    hand: list[Card],
    trick: Trick,
    trump: str,
    played_cards: set[str],
    seat_idx: int,
    trick_num: int,
    trick_pts: list[int],
    roem_pts: list[int],
    declaring_team: int,
    opponent_voids: dict[int, set[str]],
    game_scores: list[int],
    round_num: int,
    legal_moves: list[Card],
    feature_version: int = 1,
) -> np.ndarray:
    """Encode the full game state into a float32 feature vector.

    ``feature_version`` selects the layout: 1 -> NUM_FEATURES (267),
    2 -> NUM_FEATURES_V2 (300, adds the trick-roem block).
    """
    if feature_version not in FEATURE_SIZES:
        raise ValueError(f"Unsupported feature_version {feature_version}")
    team = SEAT_TEAMS[seat_idx]
    opp_team = 1 - team
    features = np.zeros(FEATURE_SIZES[feature_version], dtype=np.float32)
    offset = 0

    # 1. My hand (32)
    for c in hand:
        features[offset + card_to_idx(c)] = 1.0
    offset += NUM_CARDS  # 32

    # 2. Cards played this round (32)
    for cs in played_cards:
        if cs in CARD_INDEX:
            features[offset + CARD_INDEX[cs]] = 1.0
    offset += NUM_CARDS  # 64

    # 3. Current trick cards — 3 positional slots × 32 (96)
    #    Slot 0 = 1st card played, slot 1 = 2nd, slot 2 = 3rd.
    for i, (_, card) in enumerate(trick):
        if i < 3:
            features[offset + i * NUM_CARDS + card_to_idx(card)] = 1.0
    offset += 3 * NUM_CARDS  # 160

    # 4. Trump suit one-hot (4)
    trump_idx = SUITS.index(trump)
    features[offset + trump_idx] = 1.0
    offset += NUM_SUITS  # 164

    # 5. My position in trick one-hot (4)
    pos_in_trick = len(trick)  # 0=lead, 1=2nd, 2=3rd, 3=4th
    features[offset + pos_in_trick] = 1.0
    offset += 4  # 168

    # 6. Trick number normalized (1)
    features[offset] = trick_num / 7.0
    offset += 1  # 169

    # 7. Team trick points normalized (2)
    features[offset] = trick_pts[team] / 162.0
    features[offset + 1] = trick_pts[opp_team] / 162.0
    offset += 2  # 171

    # 8. Team roem points normalized (2)
    features[offset] = roem_pts[team] / 200.0
    features[offset + 1] = roem_pts[opp_team] / 200.0
    offset += 2  # 173

    # 9. Declaring team one-hot (2)
    if declaring_team == team:
        features[offset] = 1.0
    elif declaring_team == opp_team:
        features[offset + 1] = 1.0
    offset += 2  # 175

    # 10. Opponent voids: 3 opponents × 4 suits (12)
    opp_seats = [s for s in range(4) if s != seat_idx]
    for oi, opp_seat in enumerate(opp_seats):
        voids = opponent_voids.get(opp_seat, set())
        for si, suit in enumerate(SUITS):
            if suit in voids:
                features[offset + oi * NUM_SUITS + si] = 1.0
    offset += 3 * NUM_SUITS  # 187

    # 11. Game scores normalized (2)
    features[offset] = game_scores[team] / 500.0
    features[offset + 1] = game_scores[opp_team] / 500.0
    offset += 2  # 189

    # 12. Round in game normalized (1)
    features[offset] = round_num / 16.0
    offset += 1  # 190

    # 13. Legal move mask (32)
    for c in legal_moves:
        features[offset + card_to_idx(c)] = 1.0
    offset += NUM_CARDS  # 222

    # 14. Cards in hand count normalized (1)
    features[offset] = len(hand) / 8.0
    offset += 1  # 223

    # 15. Trump cards in hand normalized (1)
    trump_count = sum(1 for c in hand if c.suit == trump)
    features[offset] = trump_count / 8.0
    offset += 1  # 224

    # 16. Highest outstanding trump one-hot (8)
    #     Which trump rank is the highest that's not in our hand and not yet played.
    for rank in TRUMP_STRONGEST_FIRST:
        card_str = f"{rank}{trump}"
        in_hand = any(c.suit == trump and c.rank == rank for c in hand)
        already_played = card_str in played_cards
        if not in_hand and not already_played:
            ri = RANKS.index(rank)
            features[offset + ri] = 1.0
            break
    offset += 8  # 232

    # 17. Partner is winning the trick (1)
    if trick:
        from klaverjas.core import trick_winner_index
        wi = trick_winner_index(trick, trump)
        if SEAT_TEAMS[trick[wi][0].seat_idx] == team:
            features[offset] = 1.0
    offset += 1  # 233

    # 18. Points on table normalized (1)
    pts_on_table = sum(c.points(trump) for _, c in trick)
    features[offset] = pts_on_table / 40.0
    offset += 1  # 234

    # 19. My seat one-hot (4)
    features[offset + seat_idx] = 1.0
    offset += 4  # 238

    # 20. Lead suit one-hot (4) — zero vector if we are leading
    if trick:
        lead_suit = trick[0][1].suit
        features[offset + SUITS.index(lead_suit)] = 1.0
    offset += 4  # 242

    # 21. Cards per suit in hand normalized (4)
    for si, suit in enumerate(SUITS):
        features[offset + si] = sum(1 for c in hand if c.suit == suit) / 8.0
    offset += 4  # 246

    # 22. Trump control: do I hold the highest remaining trump (1)
    my_trump_strengths = [c.strength(trump) for c in hand if c.suit == trump]
    if my_trump_strengths:
        my_best = max(my_trump_strengths)
        # Check if any higher trump is outstanding
        has_higher = False
        for rank in TRUMP_STRONGEST_FIRST:
            card_str = f"{rank}{trump}"
            if card_str in played_cards:
                continue
            if any(c.suit == trump and c.rank == rank for c in hand):
                continue
            # This rank is outstanding and not in our hand
            if TRUMP_ORDER.index(rank) > my_best:
                has_higher = True
            break
        if not has_higher and my_trump_strengths:
            features[offset] = 1.0
    offset += 1  # 247

    # 23. Trick cards count one-hot (4)
    if len(trick) < 4:
        features[offset + len(trick)] = 1.0
    offset += 4  # 251

    # 24. Suit winners in hand — count of master cards per suit (4)
    for si, suit in enumerate(SUITS):
        suit_cards = [c for c in hand if c.suit == suit]
        for c in suit_cards:
            # A card is a "master" if no higher card in that suit is outstanding
            is_master = True
            if suit == trump:
                order = TRUMP_ORDER
            else:
                order = NON_TRUMP_ORDER
            my_strength = order.index(c.rank)
            for rank in order[my_strength + 1:]:
                card_str = f"{rank}{suit}"
                if card_str not in played_cards and not any(
                    h.suit == suit and h.rank == rank for h in hand
                ):
                    is_master = False
                    break
            if is_master:
                features[offset + si] += 1.0
    offset += 4  # 255

    # 25. Point cards in hand normalized (1)
    point_sum = sum(c.points(trump) for c in hand)
    features[offset] = point_sum / 60.0
    offset += 1  # 256

    # 26. Suit lengths played — per suit how many have been played (4)
    for si, suit in enumerate(SUITS):
        played_in_suit = sum(
            1 for cs in played_cards if cs in CARD_INDEX and cs.endswith(suit)
        )
        features[offset + si] = played_in_suit / 8.0
    offset += 4  # 260

    # 27. Net score delta (1)
    features[offset] = (trick_pts[team] - trick_pts[opp_team]) / 162.0
    offset += 1  # 261

    # Extra padding to reach 267:
    # 28. Team is declaring (1)
    features[offset] = 1.0 if declaring_team == team else 0.0
    offset += 1  # 262

    # 29. Last trick (1) — is this the final trick of the round
    features[offset] = 1.0 if trick_num == 7 else 0.0
    offset += 1  # 263

    # 30. Tricks remaining normalized (1)
    features[offset] = (7 - trick_num) / 7.0
    offset += 1  # 264

    # 31. Score pressure: how close to winning/losing (1)
    max_score = max(game_scores[0], game_scores[1]) if game_scores else 0
    features[offset] = max_score / 500.0
    offset += 1  # 265

    # 32. Am I the declarer's team and leading (1)
    features[offset] = 1.0 if declaring_team == team and not trick else 0.0
    offset += 1  # 266

    # 33. Number of legal moves normalized (1)
    features[offset] = len(legal_moves) / 8.0
    offset += 1  # 267

    assert offset == NUM_FEATURES, f"Feature count mismatch: {offset} != {NUM_FEATURES}"
    if feature_version == 1:
        return features

    # ── v2: trick-roem awareness ─────────────────────────────────────────────
    # 34. Roem each legal card would ADD to the current trick (32), /100.
    #     e.g. K♥ onto Q♥+J♥ -> 0.20; K-trump onto Q-trump -> 0.20;
    #     10-J-Q-K of trump -> 0.70 (sequence 50 + stuk 20).
    trick_cards = [c for _, c in trick]
    deltas, on_table = card_roem_deltas(trick_cards, legal_moves, trump)
    features[offset:offset + NUM_CARDS] = deltas / ROEM_NORM
    offset += NUM_CARDS  # 299

    # 35. Roem already formed on the table (1), /100 — extra value at stake
    #     for whoever wins this trick.
    features[offset] = on_table / ROEM_NORM
    offset += 1  # 300

    assert offset == NUM_FEATURES_V2, f"Feature count mismatch: {offset} != {NUM_FEATURES_V2}"
    return features
