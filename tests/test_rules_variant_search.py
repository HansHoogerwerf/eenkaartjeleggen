"""The AI's internal model must follow the rules variant of the game.

The lobby offers Rotterdam and Amsterdam rules.  Every place that simulates
play (the shared move generation of the base engine, Opus's search, the
batched engine behind the net-rollout search) and the card inference that
feeds the deal sampler has to apply the same rule as Player.legal_moves.
"""

import os
import random
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import torch
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

from klaverjas.constants import SEAT_TEAMS, SUITS
from klaverjas.core import Card
from main import AIPlayer

RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
DECK = [Card(s, r) for s in SUITS for r in RANKS]
VARIANTS = ("rotterdam", "amsterdam")


def _random_position(rng):
    """A hand, the cards on the table and the seat to move."""
    n_trick = rng.randint(1, 3)
    cards = rng.sample(DECK, 8 + n_trick)
    leader = rng.randrange(4)
    seat = (leader + n_trick) % 4
    hand = cards[:rng.randint(1, 8)]
    trick_cards = [((leader + i) % 4, c) for i, c in enumerate(cards[8:8 + n_trick])]
    return hand, trick_cards, seat, rng.choice(SUITS)


def _real_legal(hand, trick_cards, seat, trump, variant):
    p = AIPlayer("X", team=SEAT_TEAMS[seat], seat_idx=seat)
    p.rules_variant = variant
    p.hand = list(hand)
    trick = [(SimpleNamespace(seat_idx=s, team=SEAT_TEAMS[s]), c) for s, c in trick_cards]
    return sorted(str(c) for c in p.legal_moves(trick, trump))


class TestSharedMoveGeneration(unittest.TestCase):
    def test_matches_the_real_rule_in_both_variants(self):
        for variant in VARIANTS:
            rng = random.Random(7)
            for _ in range(4000):
                hand, trick_cards, seat, trump = _random_position(rng)
                sim = sorted(str(c) for c in AIPlayer._legal_moves_for_cards(
                    hand, trick_cards, trump, seat, variant))
                self.assertEqual(sim, _real_legal(hand, trick_cards, seat, trump, variant),
                                 (variant, [str(c) for c in hand], [(s, str(c)) for s, c in trick_cards], trump))

    def test_variants_differ_and_no_seat_means_rotterdam(self):
        # West (seat 1) leads hearts, North (seat 2, our partner) plays a higher
        # heart; South is void in hearts and holds a trump.  Partner is winning.
        hand = [Card("♠", "9"), Card("♣", "7")]
        trick_cards = [(1, Card("♥", "K")), (2, Card("♥", "A"))]
        rot = AIPlayer._legal_moves_for_cards(hand, trick_cards, "♠", 3, "rotterdam")
        ams = AIPlayer._legal_moves_for_cards(hand, trick_cards, "♠", 0, "amsterdam")
        self.assertEqual([str(c) for c in rot], ["9♠"])                       # must trump
        self.assertEqual(sorted(str(c) for c in ams), ["7♣", "9♠"])           # free: partner winning
        no_seat = AIPlayer._legal_moves_for_cards(hand, trick_cards, "♠", None, "amsterdam")
        self.assertEqual([str(c) for c in no_seat], ["9♠"])

    def test_opus_search_uses_the_same_rule(self):
        from model_players.opus_player import OpusPlayer
        for variant in VARIANTS:
            p = OpusPlayer("O", 0, 0, rng_seed=1)
            p.rules_variant = variant
            rng = random.Random(11)
            for _ in range(1500):
                hand, trick_cards, seat, trump = _random_position(rng)
                sim = sorted(str(c) for c in p._legal_moves_search(hand, trick_cards, trump, seat))
                self.assertEqual(sim, _real_legal(hand, trick_cards, seat, trump, variant))


class TestInferenceFollowsTheVariant(unittest.TestCase):
    """What a discard or an undertrump tells about a void player's trumps."""

    TRUMP = "♠"

    def _observer(self, variant):
        p = AIPlayer("South", team=0, seat_idx=0)
        p.rules_variant = variant
        p.use_inference = True
        p.start_round()
        p.receive_hand([Card("♦", "7"), Card("♦", "8"), Card("♦", "9"), Card("♦", "10"),
                        Card("♣", "7"), Card("♣", "8"), Card("♣", "9"), Card("♣", "10")])
        p.current_trump = self.TRUMP
        return p

    def _play(self, p, plays):
        lead_suit = plays[0][1].suit
        for seat, card in plays:
            p.observe_trick_play(seat, card, lead_suit)

    def _trumps_possible(self, p, seat):
        return sorted(cs[:-1] for cs in p.possible_cards_by_seat[seat] if cs.endswith(self.TRUMP))

    def test_discard_while_partner_is_winning(self):
        # East (3) leads the ♥A, South (0) discards, and West (1), void in
        # hearts, discards a club while partner East is winning the trick.
        plays = [(3, Card("♥", "A")), (0, Card("♦", "7")), (1, Card("♣", "J"))]
        rot = self._observer("rotterdam")
        self._play(rot, plays)
        self.assertEqual(self._trumps_possible(rot, 1), [])          # Rotterdam: had to trump
        ams = self._observer("amsterdam")
        self._play(ams, plays)
        self.assertEqual(len(self._trumps_possible(ams, 1)), 8)      # Amsterdam: free, nothing learned
        self.assertNotIn("K♥", ams.possible_cards_by_seat[1])        # the void in hearts is still learned

    def test_discard_when_the_opponents_win_without_a_trump_in_the_trick(self):
        # North (2) leads ♥A; East (3) is void in hearts and discards: East's
        # partner West has not played, the opponents are winning -> no trump.
        plays = [(2, Card("♥", "A")), (3, Card("♣", "J"))]
        for variant in VARIANTS:
            p = self._observer(variant)
            self._play(p, plays)
            self.assertEqual(self._trumps_possible(p, 3), [], variant)

    def test_discard_under_an_opponents_trump_that_cannot_be_beaten(self):
        # West (1) leads ♥K, North (2) ruffs with the trump 10; East (3), void
        # in hearts, discards.  Opponents are winning with a trump.
        plays = [(1, Card("♥", "K")), (2, Card("♠", "10")), (3, Card("♣", "J"))]
        rot = self._observer("rotterdam")
        self._play(rot, plays)
        self.assertEqual(self._trumps_possible(rot, 3), [])          # Rotterdam: would have undertrumped
        ams = self._observer("amsterdam")
        self._play(ams, plays)
        # Amsterdam: only the trumps that beat the 10 are ruled out (A, 9, J).
        self.assertEqual(self._trumps_possible(ams, 3), ["10", "7", "8", "K", "Q"])

    def test_undertrump_under_an_opponents_trump_rules_out_the_stronger_trumps(self):
        # East (3) leads the ♥K, South (0) discards, West (1) ruffs with the
        # trump 9 and North (2), void in hearts, plays the trump 7.  The
        # opponents are winning, so North could not beat the 9: no trump J.
        plays = [(3, Card("♥", "K")), (0, Card("♦", "7")), (1, Card("♠", "9")), (2, Card("♠", "7"))]
        for variant in VARIANTS:
            p = self._observer(variant)
            self._play(p, plays)
            self.assertNotIn("J", self._trumps_possible(p, 2), variant)

    def test_undertrump_while_partner_is_winning_says_nothing_in_amsterdam(self):
        # North (2) leads the ♥K, East (3) ruffs with the trump 9, South (0)
        # discards and West (1), void in hearts, plays the trump 7 under
        # partner East's winning 9.
        plays = [(2, Card("♥", "K")), (3, Card("♠", "9")), (0, Card("♦", "7")), (1, Card("♠", "7"))]
        rot = self._observer("rotterdam")
        self._play(rot, plays)
        self.assertNotIn("J", self._trumps_possible(rot, 1))         # Rotterdam: had to overtrump
        ams = self._observer("amsterdam")
        self._play(ams, plays)
        self.assertIn("J", self._trumps_possible(ams, 1))            # Amsterdam: partner East was winning


@unittest.skipUnless(_TORCH_OK, "PyTorch not importable")
class TestBatchedEngineVariant(unittest.TestCase):
    """tools/gpu_engine.py legal mask == Player.legal_moves, per variant."""

    def _mismatches(self, variant, steps=64, batch=128):
        from neural.features import CARD_INDEX
        from tools.gpu_engine import KlaverjasGPUEngine
        by_idx = {i: Card(cs[-1], cs[:-1]) for cs, i in CARD_INDEX.items()}
        eng = KlaverjasGPUEngine(batch_size=batch, device="cpu", rules_variant=variant)
        eng.reset()
        torch.manual_seed(5)
        wrong = differs_from_rotterdam = 0
        for _ in range(steps):
            masks = eng.legal_mask()
            for b in range(0, batch, 2):
                seat = int(eng.current_seat[b])
                hand = [by_idx[i] for i in torch.nonzero(eng.hands[b, seat]).flatten().tolist()]
                n = int(eng.cards_in_trick[b])
                trick_cards = [(int(eng.trick_seats[b, k]), by_idx[int(eng.trick_cards[b, k])]) for k in range(n)]
                trump = SUITS[int(eng.trump[b])]
                got = sorted(str(by_idx[i]) for i in torch.nonzero(masks[b]).flatten().tolist())
                wrong += got != _real_legal(hand, trick_cards, seat, trump, variant)
                differs_from_rotterdam += got != _real_legal(hand, trick_cards, seat, trump, "rotterdam")
            eng.step(torch.multinomial(masks / masks.sum(dim=1, keepdim=True), 1).squeeze(1))
        return wrong, differs_from_rotterdam

    def test_rotterdam_mask(self):
        wrong, _ = self._mismatches("rotterdam")
        self.assertEqual(wrong, 0)

    def test_amsterdam_mask(self):
        wrong, differs = self._mismatches("amsterdam")
        self.assertEqual(wrong, 0)
        self.assertGreater(differs, 0)     # the run really reached variant-specific positions

    def test_default_engine_is_rotterdam(self):
        from tools.gpu_engine import KlaverjasGPUEngine
        self.assertEqual(KlaverjasGPUEngine(batch_size=2, device="cpu").rules_variant, "rotterdam")


class TestSearchPlayerPassesTheVariant(unittest.TestCase):
    def test_one_evaluator_per_variant(self):
        import model_players.pimc_player as pp
        built = []

        class FakeEvaluator:
            def __init__(self, model_path, deals, max_candidates, rules_variant="rotterdam"):
                self.rules_variant = rules_variant
                built.append(rules_variant)

        with mock.patch.dict(pp._EVALUATORS, {}, clear=True), \
             mock.patch("neural.pimc.NetRolloutEvaluator", FakeEvaluator):
            a = pp._evaluator("m.pt", 64, 8, "rotterdam")
            b = pp._evaluator("m.pt", 64, 8, "amsterdam")
            self.assertIs(pp._evaluator("m.pt", 64, 8, "amsterdam"), b)
        self.assertIsNot(a, b)
        self.assertEqual(built, ["rotterdam", "amsterdam"])

    def test_search_asks_for_the_evaluator_of_its_game(self):
        import model_players.pimc_player as pp
        with mock.patch.dict(os.environ, {"NEURAL_PIMC_DEALS": "64"}):
            p = pp.PIMCNetPlayer("P", 0, 0, rng_seed=1)
        p.rules_variant = "amsterdam"
        legal = [Card("♠", "J"), Card("♥", "A")]
        ev = mock.Mock()
        ev.warm = True
        ev.evaluate.return_value = {str(c): 1.0 for c in legal}
        with mock.patch.object(pp, "_evaluator", return_value=ev) as get, \
             mock.patch("neural.player.neural_rank_cards", return_value=list(legal)):
            self.assertIsNotNone(p._pimc_choice(legal, [], "♠"))
        self.assertEqual(get.call_args.args[3], "amsterdam")


if __name__ == "__main__":
    unittest.main()
