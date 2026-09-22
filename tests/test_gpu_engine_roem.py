"""Parity checks between the CUDA batch engine and the reference rules.

The GPU engine (tools/gpu_engine.py) re-implements scoring and feature
encoding with tensors.  These tests pin it to the Python reference
implementations (klaverjas.core.find_roem, neural.features.encode_state) for
the roem-related parts that were added in the roem-aware retrain:

* per-trick roem for random 4-card tricks,
* the v2 feature block (roem each legal card would add, roem on the table),
* the nat rule counting roem.

They are skipped when PyTorch cannot be imported (e.g. a blocked torch.dll).
"""

import random
import unittest

import numpy as np

try:
    import torch
    _TORCH_OK = True
except Exception:  # ImportError, or OSError from a blocked DLL
    _TORCH_OK = False

from klaverjas.constants import RANKS, SUITS
from klaverjas.core import Card, trick_roem_points
from neural.features import CARD_INDEX, NUM_FEATURES, NUM_FEATURES_V2, card_roem_deltas


def _card(idx: int) -> Card:
    return Card(SUITS[idx // 8], RANKS[idx % 8])


@unittest.skipUnless(_TORCH_OK, "PyTorch not importable")
class TestGpuEngineRoemParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.gpu_engine import KlaverjasGPUEngine
        cls.engine = KlaverjasGPUEngine(batch_size=64, device="cpu", feature_version=2)
        cls.engine.reset()

    def test_roem_from_presence_matches_find_roem_on_random_tricks(self):
        rng = random.Random(7)
        sets, trumps, expected = [], [], []
        for _ in range(400):
            n = rng.choice([1, 2, 3, 4, 4, 4])
            cards = rng.sample(range(32), n)
            t = rng.randrange(4)
            sets.append(cards)
            trumps.append(t)
            expected.append(trick_roem_points([_card(c) for c in cards], SUITS[t]))
        presence = torch.zeros(len(sets), 32)
        for i, cards in enumerate(sets):
            presence[i, cards] = 1.0
        got = self.engine._roem_from_presence(presence, torch.tensor(trumps))
        self.assertEqual(got.tolist(), expected)

    def test_known_combinations(self):
        eng = self.engine
        def roem(cards, trump):
            p = torch.zeros(1, 32)
            p[0, [CARD_INDEX[c] for c in cards]] = 1.0
            return int(eng._roem_from_presence(p, torch.tensor([SUITS.index(trump)]))[0])
        self.assertEqual(roem(["10♠", "J♠", "Q♠", "K♠"], "♠"), 70)   # 4-seq + stuk
        self.assertEqual(roem(["10♠", "J♠", "Q♠", "K♠"], "♥"), 50)
        self.assertEqual(roem(["7♣", "7♦", "7♥", "7♠"], "♠"), 100)
        self.assertEqual(roem(["J♣", "J♦", "J♥", "J♠"], "♠"), 200)
        self.assertEqual(roem(["Q♥", "K♥", "8♣"], "♥"), 20)          # stuk on a partial trick
        self.assertEqual(roem(["7♦", "9♦", "8♦", "A♣"], "♠"), 20)    # unordered 7-8-9

    def test_v2_feature_block_matches_reference_encoder(self):
        eng = self.engine
        B = eng.B
        # Play a few random legal cards so tricks have 0-3 cards on the table.
        rng = torch.Generator().manual_seed(3)
        for _ in range(5):
            masks = eng.legal_mask()
            probs = masks / masks.sum(1, keepdim=True)
            actions = torch.multinomial(probs, 1, generator=rng).squeeze(1)
            eng.step(actions)
        masks = eng.legal_mask()
        feats = eng.encode_features(masks)
        self.assertEqual(feats.shape, (B, NUM_FEATURES_V2))

        for b in range(B):
            trump = SUITS[int(eng.trump[b])]
            n = int(eng.cards_in_trick[b])
            trick_cards = [_card(int(eng.trick_cards[b, i])) for i in range(n)]
            legal = [_card(i) for i in range(32) if masks[b, i] > 0]
            deltas, on_table = card_roem_deltas(trick_cards, legal, trump)
            np.testing.assert_allclose(
                feats[b, NUM_FEATURES:NUM_FEATURES + 32].numpy(), deltas / 100.0, atol=1e-6)
            self.assertAlmostEqual(float(feats[b, NUM_FEATURES_V2 - 1]), on_table / 100.0, places=6)

    def test_per_trick_roem_reaches_roem_pts_and_nat_counts_roem(self):
        from tools.gpu_engine import KlaverjasGPUEngine
        eng = KlaverjasGPUEngine(batch_size=1, device="cpu", feature_version=2)
        eng.reset()
        # Force a known state: trump ♠, South leads a 10-J-Q-K spade trick.
        eng.trump[0] = SUITS.index("♠")
        eng.declaring_team[0] = 0
        eng.trick_leader[0] = 0
        eng.current_seat[0] = 0
        eng.cards_in_trick[0] = 0
        eng.trick_cards[0] = -1
        eng.trick_seats[0] = -1
        eng.hands.zero_()
        for seat, card in enumerate(["10♠", "J♠", "Q♠", "K♠"]):
            eng.hands[0, seat, CARD_INDEX[card]] = 1.0
        for seat, card in enumerate(["10♠", "J♠", "Q♠", "K♠"]):
            self.assertEqual(int(eng.current_seat[0]), seat)
            eng.step(torch.tensor([CARD_INDEX[card]]))
        # J♠ (seat 1, team 1) wins: 10+20+3+4 = 37 points and 70 roem.
        self.assertEqual(int(eng.trick_pts[0, 1]), 37)
        self.assertEqual(int(eng.roem_pts[0, 1]), 70)
        self.assertEqual(int(eng.trick_pts[0, 0]), 0)
        self.assertEqual(int(eng.roem_pts[0, 0]), 0)

        # Nat rule: declarer (team 0) with 90 trick points but only 0 roem loses
        # to 72 + 70 roem for the opponents -> nat.
        eng.trick_pts[0] = torch.tensor([90, 72], dtype=torch.int32)
        eng.roem_pts[0] = torch.tensor([0, 70], dtype=torch.int32)
        eng.game_scores.zero_()
        eng.trick_num[0] = 8
        eng._score_round(torch.tensor([True]))
        self.assertEqual(int(eng._last_round_pts[0, 0]), 0)
        self.assertEqual(int(eng._last_round_pts[0, 1]), 162 + 70)

    def test_dense_rewards_sum_to_the_sparse_round_reward(self):
        from tools.gpu_engine import KlaverjasGPUEngine
        torch.manual_seed(11)
        eng = KlaverjasGPUEngine(batch_size=16, device="cpu", feature_version=2, dense_rewards=True)
        eng.reset()
        rng = torch.Generator().manual_seed(5)
        acc = torch.zeros(eng.B)
        checked = 0
        for _ in range(32 * 6):   # ~6 rounds per game
            masks = eng.legal_mask()
            actions = torch.multinomial(masks / masks.sum(1, keepdim=True), 1, generator=rng).squeeze(1)
            rewards, round_done = eng.step(actions)
            acc += rewards
            if round_done.any():
                sparse = (eng._last_round_pts[:, 0] - eng._last_round_pts[:, 1]).float() / 162.0
                for b in torch.nonzero(round_done).flatten().tolist():
                    self.assertAlmostEqual(float(acc[b]), float(sparse[b]), places=5)
                    acc[b] = 0.0
                    checked += 1
        self.assertGreater(checked, 16)
        # Sparse engine never pays before the round ends.
        eng2 = KlaverjasGPUEngine(batch_size=4, device="cpu", feature_version=2)
        eng2.reset()
        for _ in range(3):
            masks = eng2.legal_mask()
            actions = torch.multinomial(masks / masks.sum(1, keepdim=True), 1, generator=rng).squeeze(1)
            rewards, _ = eng2.step(actions)
            self.assertEqual(float(rewards.abs().sum()), 0.0)

    def test_v1_engine_still_emits_267_features(self):
        from tools.gpu_engine import KlaverjasGPUEngine
        eng = KlaverjasGPUEngine(batch_size=4, device="cpu", feature_version=1)
        eng.reset()
        feats, masks = eng.get_state()
        self.assertEqual(feats.shape, (4, NUM_FEATURES))


if __name__ == "__main__":
    unittest.main()
