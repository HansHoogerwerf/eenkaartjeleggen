"""
showoff.py — head-to-head round-robin between the three model players.

Klaverjassen is 2-on-2, so each "match" pits one contender (both seats of
team 0) against another (both seats of team 1).  Every distinct pair plays a
series, so N contenders give N*(N-1)/2 pairings.

Contenders:
  * Opus, Mythos      — the model-written players (determinized double-dummy
    search + Monte-Carlo nat-aware bidding).
  * Neural+MythosBid  — the trained net (models/neural_best.pt) for card play,
    paired with Mythos's MC nat-aware bidder instead of the stock heuristic
    one.  The checkpoint is verified to load before the run so it can't
    silently degrade to the heuristic fallback.

Fairness
--------
For every game_seed we play the deal twice with the team assignment *swapped*
(model A on team 0, then model A on team 1).  Because the deal is a pure
function of game_seed, both orientations see the identical cards, dealer and
offered trumps — so any seat/dealer/declare advantage cancels out and the
score difference reflects skill, not luck.

Run:
    python model_players/showoff.py            # default: 3 seeds x 6 rounds
    python model_players/showoff.py 4 8        # 4 seeds x 8 boom rounds
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The board logs use suit glyphs and we print arrows; force UTF-8 on consoles
# (e.g. Windows cp1252) that would otherwise raise UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import main
from klaverjas.constants import SEAT_TEAMS


from model_players.opus_player import OpusPlayer
from model_players.mythos_player import MythosPlayer
from model_players.neural_mythos_player import NeuralMythosBidPlayer


# label -> factory(name, team, seat, seed) -> player instance.
# Ordered so output is deterministic.
MODELS = {
    "Opus": lambda name, team, seat, seed: OpusPlayer(
        name, team, seat_idx=seat, rng_seed=seed),
    "Mythos": lambda name, team, seat, seed: MythosPlayer(
        name, team, seat_idx=seat, rng_seed=seed),
    "Neural+MythosBid": lambda name, team, seat, seed: NeuralMythosBidPlayer(
        name, team, seat_idx=seat, rng_seed=seed),
}


def verify_neural_loads():
    """Confirm the trained checkpoint actually loads, so the Neural contender
    plays with the net rather than silently falling back to heuristics."""
    from neural.player import DEFAULT_MODEL_PATH, _get_model
    model = _get_model(DEFAULT_MODEL_PATH)
    if model is None:
        raise SystemExit(
            f"Neural model failed to load from {DEFAULT_MODEL_PATH} — aborting so "
            "the show-off doesn't misreport a heuristic fallback as 'Neural'."
        )
    print(f"Neural checkpoint loaded OK: {DEFAULT_MODEL_PATH}")


def play_match(team0_label, team1_label, game_seed, boom_rounds):
    """Play one game with `team0_label` on team 0 (seats 0,2) and
    `team1_label` on team 1 (seats 1,3).  Returns (team0_pts, team1_pts)."""
    fac0 = MODELS[team0_label]
    fac1 = MODELS[team1_label]

    holder = {"game": None}

    def on_event(event, data):
        if event == "waiting_for_host" and holder["game"] is not None:
            holder["game"].signal_next_round()

    game = main.KlaverjasGame(
        human_seats={},
        log_fn=lambda *a, **k: None,
        state_fn=on_event,
        game_mode="boom",
        game_seed=game_seed,
        ai_strength="advanced",
    )
    holder["game"] = game
    game.boom_rounds = boom_rounds

    seed_base = game.ai_seed_base
    players = []
    for seat in range(4):
        team = SEAT_TEAMS[seat]
        label = team0_label if team == 0 else team1_label
        fac = fac0 if team == 0 else fac1
        players.append(fac(f"{label} {seat}", team, seat, seed_base + seat))
    for p in players:
        p.rules_variant = game.rules_variant
    game.players = players

    game.play()
    return game.scores[0], game.scores[1]


def run(num_seeds=3, boom_rounds=6, base_seed=20260611):
    # Disable UI pacing for headless play.
    main.AI_BID_DELAY = 0.0
    main.AI_PLAY_DELAY = 0.0
    main.TRICK_CLEAR_DELAY = 0.0

    if any("Neural" in lbl for lbl in MODELS):
        verify_neural_loads()

    labels = list(MODELS)
    pairings = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]]

    # aggregate points scored by each model across every match it played
    points = {lbl: 0 for lbl in labels}
    # head-to-head record: wins[a][b] = games a beat b
    h2h = {a: {b: {"w": 0, "l": 0, "t": 0, "pf": 0, "pa": 0} for b in labels} for a in labels}
    match_wins = {lbl: 0 for lbl in labels}

    print(f"\nKlaverjassen show-off  —  {num_seeds} seed(s) x 2 orientations x "
          f"{boom_rounds} rounds per pairing")
    print("=" * 64)

    t_start = time.perf_counter()
    for a, b in pairings:
        a_pts = b_pts = 0
        a_games = b_games = ties = 0
        for s in range(num_seeds):
            seed = base_seed + s * 101
            # Orientation 1: A on team 0, B on team 1.
            p0, p1 = play_match(a, b, seed, boom_rounds)
            # Orientation 2: swap teams on the *same* deal.
            q0, q1 = play_match(b, a, seed, boom_rounds)

            ga = p0 + q1   # A's points across both orientations
            gb = p1 + q0   # B's points across both orientations
            a_pts += ga
            b_pts += gb
            for first, x, y in ((True, p0, p1), (False, q0, q1)):
                # winner of each individual game
                aw = x if first else y
                bw = y if first else x
                if aw > bw:
                    a_games += 1
                elif bw > aw:
                    b_games += 1
                else:
                    ties += 1

        points[a] += a_pts
        points[b] += b_pts
        h2h[a][b].update(w=a_games, l=b_games, t=ties, pf=a_pts, pa=b_pts)
        h2h[b][a].update(w=b_games, l=a_games, t=ties, pf=b_pts, pa=a_pts)
        if a_pts > b_pts:
            match_wins[a] += 1
            verdict = f"{a} wins the series"
        elif b_pts > a_pts:
            match_wins[b] += 1
            verdict = f"{b} wins the series"
        else:
            verdict = "series tied"

        print(f"\n{a}  vs  {b}")
        print(f"  total points : {a} {a_pts:>5}   |   {b} {b_pts:>5}")
        print(f"  games won    : {a} {a_games:>5}   |   {b} {b_games:>5}"
              + (f"   (ties {ties})" if ties else ""))
        print(f"  → {verdict}")

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 64)
    print("FINAL STANDINGS")
    print("=" * 64)
    ranked = sorted(labels, key=lambda l: (match_wins[l], points[l]), reverse=True)
    print(f"{'rank':<5}{'model':<18}{'series':<8}{'total pts':<11}")
    for i, lbl in enumerate(ranked, 1):
        print(f"{i:<5}{lbl:<18}{match_wins[lbl]:<8}{points[lbl]:<11}")
    print(f"\nplayed in {elapsed:.1f}s")
    return ranked, points, match_wins


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    run(num_seeds=seeds, boom_rounds=rounds)
