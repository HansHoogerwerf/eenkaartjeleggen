# AI Strategy — Klaverjassen (eenkaartjeleggen)

**Describes:** `AIPlayer` in [main.py](../main.py)
**Related tests:** [tests/test_ai_decisions.py](../tests/test_ai_decisions.py), [tests/test_ai_advanced_tactics.py](../tests/test_ai_advanced_tactics.py), [tests/test_ai_signaling.py](../tests/test_ai_signaling.py), [tests/test_ai_seat_logic.py](../tests/test_ai_seat_logic.py), [tests/test_ai_strength_levels.py](../tests/test_ai_strength_levels.py)

---

## Overview

The AI is a rule-based heuristic player with four optional advanced capabilities that are toggled based on the selected difficulty level:

| Capability | Beginner | Advanced | Expert | Expert v2 |
|---|---|---|---|---|
| Card inference (void/trump deduction) | No | Yes | Yes | Yes |
| Monte Carlo trick-win probability | No | Yes | Yes | Yes |
| Endgame exact minimax solver | No | No | Yes | Yes |
| 3-trick lookahead (sampled minimax) | No | No | No | Yes |

All difficulty levels share the same decision-making structure. The difference is in the precision of the information used and the probability of making deliberate mistakes.

---

## Difficulty Profiles

Defined in `AIPlayer.AI_STRENGTH_PROFILES` ([main.py:294](../main.py#L294)):

| Parameter | Beginner | Advanced | Expert | Expert v2 |
|---|---|---|---|---|
| `use_inference` | False | True | True | True |
| `use_trick_prob` | False | True | True | True |
| `use_endgame_solver` | False | False | True | True |
| `use_lookahead` | False | False | False | True |
| `lookahead_depth` | 0 | 0 | 0 | 3 |
| `lookahead_samples` | 0 | 0 | 0 | 5 |
| `tie_break_delta` | 0.95 | 0.55 | 0.35 | 0.35 |
| `random_mistake_rate` | 10% | 3% | 0% | 0% |
| `declaration_bias` | +0.65 | +0.25 | 0.0 | 0.0 |
| `trick_win_sim_samples` | 4 | 12 | 20 | 20 |

**`tie_break_delta`** — How close two card scores must be before they are considered equivalent and chosen randomly between. A high value means more moves "look similar" to the AI → more variance in play (feels more human-like for beginners).

**`random_mistake_rate`** — Probability of ignoring card scores entirely and playing any legal card at random.

**`declaration_bias`** — Positive offset added to the bid threshold. Higher values mean the AI is harder to persuade to declare trump (more conservative bidder for lower levels).

---

## Bidding (Trump Declaration)

Method: `choose_trump()` ([main.py:409](../main.py#L409))

When offered a suit, the AI evaluates the hand by computing a **declaration score** and comparing it to a dynamic **threshold**.

### Declaration Score (`_declaration_score`)

Points are accumulated for hand features with the following weights:

| Feature | Weight |
|---|---|
| Trump Jack (highest trump) | +1.85 |
| Trump 9 (second-highest trump) | +1.30 |
| Trump Ace | +0.85 |
| Trump 10 | +0.55 |
| Each extra trump beyond 2 | +0.65 |
| Side-suit Ace | +0.70 (+ 0.05 per extra card in suit) |
| Side-suit 10 when also holding Ace | +0.30 |
| Side-suit King | +0.15 |
| Void in a side suit | +0.35 |
| Singleton in a side suit | +0.17 |
| Roem (honour combinations) | up to +0.75 (scaled from point value) |

### Dynamic Threshold

```
threshold = 2.75
           + (roem_total / 100) × 0.2    # hands with roem are better
           + nat_risk                      # +0.15 if opponents declared first
           + declaration_bias             # per difficulty level
           - score_pressure × 0.35       # lower threshold when under pressure
```

The AI declares when `declaration_score >= threshold`.

**Score pressure** lowers the threshold when the team is behind (more willing to take risks when trailing), and raises it when comfortably ahead.

---

## Card Play Decision Tree

Method: `_strategy()` ([main.py:587](../main.py#L587))

```
Is this the last 3 cards and endgame solver enabled?
  → Yes: use exact minimax solver (_endgame_exact_choice)

Is lookahead enabled and more than 3 cards remain? (Expert v2 only)
  → Yes: use 3-trick sampled minimax (_lookahead_choice)

Is it our lead (trick is empty)?
  → Yes: use _lead()

Is a teammate currently winning the trick?
  → Yes: use _discard_for_partner()

Otherwise:
  → use _try_win()
```

---

## 1. Leading (`_lead`)

When the AI leads a new trick, cards are scored on several axes:

**Trump cards:**
- Base score from card strength
- +3.5 if we declared trump and opponents still have trumps (pull trumps aggressively)
- -0.9 otherwise (prefer not to lead trump for free)
- +1.0 if we hold the highest remaining trump (leading it is safe)
- Scaled by score pressure

**Non-trump cards:**
- Prefer leading from long suits (+0.6 per card in hand in the suit)
- -4.0 penalty if an opponent is known void in the suit AND trumps are still out (they will cut us)
- +4.0 bonus if no cards of this suit are left anywhere else (guaranteed to win)
- Scaled by partner signal strength: if partner signaled strength in a suit, prefer leading it
- Bonus for opening signals (trick 0 only): leading trump top cards or side Aces signals partner

**Point-leak penalty** applies to all cards: if a card has point value AND stronger cards in the same suit remain outstanding, the AI penalises choosing it (avoids gifting point cards to opponents).

---

## 2. Discarding for Partner (`_discard_for_partner`)

When our teammate is winning the trick, the AI plays a supporting discard.

**If the partner's win is not yet secure** (they could still be beaten by remaining trumps): fall through to safe-discard mode to protect points.

**If the partner's win is secure:**
- Score each non-trump card for its *schmear value* (dumping point cards on partner's trick):
  - Point cards are weighted by `schmear_weight` (0.25–0.38 depending on trick value, scaled up under pressure)
  - Prefer not to break long suits (penalty for short suits doesn't apply as strongly)
  - Small bonus for playing a "same-suit signal card" (7/8/9 while holding Ace or 10 in the suit)

The AI will **not** schmear trump cards unless it has no non-trumps.

---

## 3. Trying to Win (`_try_win`)

When opponents are winning and the AI can beat them:

**Find beater candidates** — cards that either:
- Are trump (when opponent's winning card is not trump)
- Are the same suit and stronger than the current winning card

**If no beaters exist:** fall through to safe-discard mode.

**Score each beater:**
- `win_prob × (6.5 + trick_value × 0.30 + pressure × 1.4)` — weighted by estimated chance the beater survives
- `trick_value × 0.20` — bonus for valuable tricks
- +1.6 if the beater is expected to hold (no known threats remain in suit)
- -1.2 if the beater is unlikely to hold
- `-card.points × (0.60 - 0.12 × pressure)` — penalty for using valuable trump to win cheap tricks
- Point-leak penalty

**Fight thresholds** — even with a good beater, the AI backs off if:
- The trick isn't worth fighting for (`trick_value < fight_threshold`, where `fight_threshold = max(3, 8 - pressure×4)`)
- Win probability is below 45% and the beater has point value
- The beater is unlikely to survive remaining opponents

Exception: when the AI's team declared trump and is at risk of nat, it will fight regardless.

---

## 4. Safe Discard (`_play_safe_discard`)

Used when the AI cannot or should not win the trick. Goal: minimise points leaked to opponents.

**Scoring per card:**
- `win_prob × (6.0 + trick_value × 0.25 + pressure × 1.3)` — marginal cases where we may still win
- `-card.points × leak_weight` — cost of giving away points (leak_weight ~2.2, higher for bigger tricks)
- Point-leak penalty (double-punishes risky point cards)
- Early game (tricks 0–3): bonus for discarding from short suits (create voids sooner)
- Late game: weaker suit-shortening bonus

---

## Score Pressure

Method: `_score_pressure()` ([main.py:476](../main.py#L476))

Returns a value in `[-1.0, 1.0]` representing urgency:

| Situation | Effect |
|---|---|
| Behind in current round | Positive (aggressive) |
| Declared but losing the round | +0.55 extra (nat danger) |
| Ahead in current round | Slightly negative |
| Behind in game score | Positive (aggressive) |
| Opponent close to winning target | +0.45 |
| We are close to winning target | -0.20 (protect lead) |
| Last 2 tricks (trick 6+) | ×1.10 amplification |
| Boom mode, last 3 rounds | Extra amplification |

Pressure modulates almost every scoring formula: it increases aggression (fight for tricks, lead trumps, take risks) when behind and increases caution when ahead.

---

## Card Inference

Enabled at Advanced and Expert level.

### Void Tracking

After every card played, the AI records **suit voids** per seat:
- If a player fails to follow the led suit → they are void in that suit.
- If a player fails to follow led suit AND does not trump → they are void in both that suit AND trump.

### Trump Strength Inference

Under Rotterdam rules, players must overtrump when trumped into. If a player trumps but plays a weaker trump than the current highest:
- The AI infers that player cannot hold any **stronger** trump cards.
- All stronger trumps are removed from that seat's possibility set.

### Possible-Cards Tracking

Each AI maintains `possible_cards_by_seat[seat]`: the set of cards that seat could still be holding. This is initialised at deal time (all unseen cards are distributed equally), then narrowed as cards are played and void/strength inferences are made.

This tracking feeds into:
1. Monte Carlo simulations (more accurate sampling)
2. Endgame solver (more accurate hand determinization)
3. `_beater_likely_holds()` (knowing opponent voids)

---

## Monte Carlo Trick-Win Probability

Enabled at Advanced and Expert level. Method: `_estimate_team_trick_win_prob()` ([main.py:714](../main.py#L714))

When the AI needs to estimate whether a card will win a trick that has not yet been fully played:

1. Record the current trick state (cards already played).
2. For each remaining seat in play order, sample a card from their `possible_cards_by_seat` pool, restricted to legal moves.
3. Determine the trick winner.
4. Repeat for `TRICK_WIN_SIM_SAMPLES` iterations (4 / 12 / 20 per difficulty level).
5. Return `wins / total_simulations`.

**Fallback for Beginner:** a simpler heuristic computes win probability based on whether the current leader's card is trump, how many trumps remain, and how many players are left.

---

## Endgame Exact Solver

Enabled at Expert level only. Activates when **≤3 cards** remain in hand.

### Hand Determinization (`_determinize_endgame_hands`)

When only a few cards remain, the AI attempts to reconstruct the exact hands of all opponents via a **backtracking constraint-satisfaction**:

1. Count how many cards each seat must still hold.
2. Use `possible_cards_by_seat` as the constraint set for each seat.
3. Assign cards greedily from most-constrained seat to least, backtracking on conflicts.
4. If determinization succeeds, a complete concrete hand distribution is returned.

### Minimax (`_endgame_minimax`)

With complete information determinized, the AI runs **full-tree minimax** over the remaining tricks:

- The AI's team maximizes point gain.
- Opponents minimize it.
- Terminal nodes: trick point total + 10-point last-trick bonus.
- Memoization is applied per (hand state, trick state, next seat) tuple.

### Card Selection

Each legal card is tried as the first move. The one with the highest minimax value is returned (with tie-breaking by `TIE_BREAK_DELTA` at Expert level = 0.35).

---

## 3-Trick Lookahead (Expert v2)

Enabled at Expert v2 level only. Activates when **>3 cards** remain in hand (the endgame solver handles the last 3).

This is the key difference between Expert and Expert v2: instead of relying on heuristic scoring for card play, the AI searches 3 full tricks into the future using sampled opponent hands and minimax.

### Hand Sampling (`_sample_hands`)

Unlike the endgame solver's exact backtracking (feasible with few cards), the lookahead uses **random sampling** to generate consistent opponent hand distributions:

1. Determine how many cards each seat must hold.
2. For each seat (most-constrained first), draw cards from their `possible_cards_by_seat` pool, shuffled randomly.
3. Assigned cards are removed from subsequent seats' pools to maintain consistency.

Multiple independent samples (default: 5) provide coverage over the uncertainty of opponent hands.

### Depth-Limited Minimax (`_lookahead_minimax`)

Identical in structure to the endgame minimax, but with two critical differences:

1. **Depth limit**: the search terminates after `lookahead_depth` complete tricks (default: 3) rather than playing to the end of the round. The accumulated point differential at that point is the leaf evaluation.
2. **Move pruning**: at each internal node, if there are more than 3 legal moves, only the top 3 are explored (ranked by a fast heuristic: card strength + suit matching + point value). This caps the branching factor at 3, making 3^12 = ~530K the worst-case tree size per sample — feasible with memoization.

### Card Selection (`_lookahead_choice`)

For each legal card the AI could play:
1. Try that card across all `lookahead_samples` hand samples.
2. Run `_lookahead_minimax` for each sample, collecting the point-differential score.
3. Average the scores across samples.

The card with the highest average score is selected (with standard `TIE_BREAK_DELTA` tie-breaking).

If all sampling attempts fail (inconsistent constraints), the method returns `None` and the AI falls through to heuristic play.

### Performance Characteristics

| Parameter | Value |
|---|---|
| Lookahead depth | 3 tricks (12 plies) |
| Samples per decision | 5 |
| Branch limit per node | 3 |
| Worst-case nodes per sample | ~530K |
| Approximate time per round | ~5-7 seconds |

The lookahead is significantly more expensive than heuristic play (~1ms per round). This is acceptable for AI-vs-AI benchmarking and analysis, but too slow for real-time multiplayer play without further optimization.

---

## Partner Signaling

Method: `_decode_partner_signal()` / `_is_opening_signal_card()` / `_is_same_suit_signal_card()` ([main.py:1029](../main.py#L1029))

The AI uses a simple two-signal convention:

### Opening Signal (Trick 0)

When the partner leads on trick 0:
- **Trump J / 9 / A**: signals "trump_pull" — partner wants us to lead trump (confidence 0.9)
- **Non-trump Ace or King**: signals the suit played as a strong suit (confidence 0.75)

The AI responds by preferring to lead into signaled suits.

### Same-Suit Control Signal (Any trick)

When partner plays a 7, 8, or 9 in a non-trump suit while holding (or having previously shown) an Ace or 10 in that suit: the AI interprets this as a strength signal in that suit (+0.35 confidence, capped at 1.0).

### Signal Decay

All signals decay by 0.015 confidence per trick and are discarded when confidence drops below 0.05. This means signals are most relevant immediately after they are given and gradually become irrelevant.

### Influence on Play

In `_lead()`, the AI adds `partner_signal_strength(suit) × 1.2` to the score of leading that suit. This softly biases the AI toward suits where the partner has shown strength, without forcing it.

---

## Safe-Lead Heuristic (`_best_lead_from`)

When leading a non-trump suit, the AI avoids "leading into outstanding higher cards":
- For each candidate card (sorted strongest first), check if any stronger card in that suit is still unaccounted for.
- Lead the strongest card for which no stronger card remains outstanding.
- If no such card exists, lead the cheapest card in the suit to probe safely.

This prevents the classic mistake of leading the 10 into an outstanding Ace (losing 10 points unnecessarily).

---

## Beater Survival Assessment (`_beater_likely_holds`)

Before committing a trump or high card to win a trick, the AI checks whether it will survive the remaining players:

**For trump beaters:**
- Count how many stronger trumps are still unaccounted for (not played, not in our hand).
- If zero threats: safe.
- If one threat and one opponent left: borderline safe.

**For non-trump beaters:**
- If any known-void opponent is still to play AND trumps remain: the card WILL be cut → unsafe.
- If lead-suit cards are exhausted elsewhere: unsafe (nothing left to follow → opponents free to trump).
- Otherwise check outstanding stronger cards in the same suit.

---

## Summary: Key Design Principles

1. **Score pressure is pervasive.** Almost every decision (bidding threshold, fight threshold, point-leak tolerance, suit preference) is modulated by the current score context.

2. **Prefer not to waste point cards.** The `_point_leak_penalty` function appears in leading, discarding, trying to win, and safe-discard paths. The AI actively avoids gifting Aces, 10s, and trump top-cards to opponents.

3. **Aggression scales with stakes.** Low-value tricks are often conceded; high-value tricks are fought over. Pressure from the game score amplifies both extremes.

4. **Inference narrows uncertainty progressively.** The possible-cards model starts broad (all unseen cards possible) and tightens each time a void or trump-strength inference fires. Better information → better Monte Carlo samples → better decisions.

5. **The endgame solver provides perfect play in the final tricks.** Once ≤3 cards remain, the AI switches from heuristics to exact minimax (subject to successful hand determinization). This is where Expert AI is strongest.

6. **Partner signaling is lightweight but meaningful.** The signal system does not rely on secret conventions — it observes natural plays (leading strong suits, playing small encouraging cards) and softly biases future leads toward partner's shown strength.
