"""Configurable reward rules for RL training.

Each rule is a function that receives trick context and returns a reward delta
for the player who just played. Rules are applied per-card, after the trick
is fully resolved. Return 0.0 for no effect.

Add or modify rules below — the training script picks them up automatically.
"""

from klaverjas.core import Card, Trick, trick_winner_index
from klaverjas.constants import TRUMP_PTS, NON_TRUMP_PTS, SEAT_TEAMS, TRUMP_ORDER, NON_TRUMP_ORDER


# ─── Context passed to each rule ─────────────────────────────────────────────
#
#  ctx = {
#      "card":           Card             — the card this player played
#      "player_idx":     int              — seat index (0-3)
#      "team":           int              — team of the player (0 or 1)
#      "trick":          Trick            — full trick (4 cards), already resolved
#      "trick_num":      int              — 0-7, which trick in the round
#      "trump":          str              — trump suit
#      "winner_idx":     int              — global seat index of trick winner
#      "winning_team":   int              — team that won the trick
#      "trick_points":   int              — total points in this trick
#      "played_cards":   set[str]         — all cards played so far this round (e.g. {"AH", "10S"})
#      "cards_by_seat":  dict[int, list[str]]  — cards played per seat this round
#      "partner_seat":   int              — seat index of this player's teammate
#      "voids":          dict[int, set[str]]   — known suit voids per seat (e.g. {2: {"H", "S"}})
#      "declaring_team": int              — team that declared trump (-1 if no declaration yet)
#      "team_trick_pts": list[int]        — cumulative trick points per team [t0, t1] after this trick
#  }
#
# ──────────────────────────────────────────────────────────────────────────────


def last_trick_bonus(ctx: dict) -> float:
    """Bonus for winning the last trick (laatste slag)."""
    if ctx["trick_num"] == 7 and ctx["winning_team"] == ctx["team"]:
        return 0.15
    return 0.0


def wasted_points_penalty(ctx: dict) -> float:
    """Penalty for playing an expensive card that doesn't end up being the
    highest in the trick — even if a teammate wins.

    'Expensive' = card worth >= 10 points (A, 10, trump J, trump 9).
    """
    card: Card = ctx["card"]
    trump = ctx["trump"]
    pts = card.points(trump)
    if pts < 10:
        return 0.0

    trick: Trick = ctx["trick"]
    # Find the highest card in the trick
    wi = trick_winner_index(trick, trump)
    winning_card = trick[wi][1]

    # Is this player's card the winning card?
    if winning_card is card:
        return 0.0

    # Same card object check may fail, also check by identity in trick
    for i, (_, c) in enumerate(trick):
        if c is card and i == wi:
            return 0.0

    # Player played an expensive card but it wasn't the highest
    penalty = -0.05 * (pts / 10.0)  # scale by how expensive
    return penalty


def waste_trump_penalty(ctx: dict) -> float:
    """Penalty for playing trump when opponents have no trump left.

    If no opponent has trump cards remaining, there's no reason to spend
    trump — our non-trump cards will win just as well against them.
    Does not penalise if we were forced to follow suit (lead was trump).
    """
    card: Card = ctx["card"]
    trump = ctx["trump"]

    if card.suit != trump:
        return 0.0

    # Don't penalise if trump was led (must follow suit)
    trick: Trick = ctx["trick"]
    if trick[0][1].suit == trump:
        return 0.0

    # Check if all trump cards held by opponents have been played
    seat = ctx["player_idx"]
    partner = ctx["partner_seat"]
    voids = ctx["voids"]
    opponents = [s for s in range(4) if s != seat and s != partner]
    all_opp_void = all(trump in voids.get(opp, set()) for opp in opponents)

    if not all_opp_void:
        return 0.0

    return -0.10


def schmear_reward(ctx: dict) -> float:
    """Bonus for dumping a high-value card onto a trick won by your partner.

    Schmearing (smearen) is a key Klaverjassen skill: when partner securely
    wins the trick, discard your expensive cards so the points stay on your team.
    Fires when the player played a card worth >= 10 points and partner won,
    but only if the card is NOT the highest remaining in its suit — if it is the
    master card it would have won tricks on its own anyway and is not a true schmear.
    Scales with how valuable the card was.
    """
    card: Card = ctx["card"]
    trump = ctx["trump"]
    pts = card.points(trump)
    if pts < 10:
        return 0.0
    if ctx["winner_idx"] != ctx["partner_seat"]:
        return 0.0

    # Check whether a higher card in the same suit is still unplayed.
    # Include current trick cards in the played set to avoid false positives.
    played = ctx["played_cards"] | {str(c) for _, c in ctx["trick"]}
    order = TRUMP_ORDER if card.suit == trump else NON_TRUMP_ORDER
    rank_pos = order.index(card.rank)
    has_higher_remaining = any(
        f"{rank}{card.suit}" not in played for rank in order[rank_pos + 1:]
    )
    if not has_higher_remaining:
        return 0.0  # master card of the suit — not a genuine schmear

    return 0.08 * (pts / 10.0)


def win_valuable_trick_bonus(ctx: dict) -> float:
    """Bonus for your team winning a trick with high point value.

    The base game reward captures total round point differential, but gives no
    per-trick signal about which tricks matter most. This rule rewards winning
    tricks worth >= 15 points, scaled by their value, teaching the model to
    fight for high-value tricks rather than concede them.
    """
    if ctx["winning_team"] != ctx["team"]:
        return 0.0
    pts = ctx["trick_points"]
    if pts < 15:
        return 0.0
    return 0.06 * (pts / 30.0)


# ─── Rule registry ────────────────────────────────────────────────────────────
# Add your rules here. Each entry is (rule_function, weight).
# The final shaped reward = sum(rule(ctx) * weight for rule, weight in RULES).

RULES: list[tuple[callable, float]] = [
    (last_trick_bonus, 1.0),
    (wasted_points_penalty, 1.0),
    (waste_trump_penalty, 1.0),
    (schmear_reward, 1.0),
    (win_valuable_trick_bonus, 1.0),
]
