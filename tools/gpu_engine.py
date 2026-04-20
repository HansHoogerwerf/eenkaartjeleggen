"""
GPU-accelerated Klaverjassen game engine for AI training.

Runs B games simultaneously as batched tensor operations on GPU.
No Python game loop overhead — all game logic is vectorized.

Card encoding:  card_idx = suit_idx * 8 + rank_idx
  Suits:  ♣=0  ♦=1  ♥=2  ♠=3
  Ranks:  7=0  8=1  9=2  10=3  J=4  Q=5  K=6  A=7

Rotterdam rules (strict):
  - Must follow lead suit if able
  - If lead suit is trump: must overtrump if possible
  - If can't follow suit: must trump (and overtrump existing trick trump if possible)

CUDA-graph friendliness:
  All hot-path mutations use in-place ops or `tensor.copy_(torch.where(...))`,
  and no host syncs (no `.item()`, `.any()` guards) occur in step()/get_state().
  Every branch always runs and is masked; dynamic-shape tensors are avoided.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# ─── Precomputed constant arrays (rank-indexed, shape [8]) ────────────────────

# Non-trump strength: NON_TRUMP_ORDER = ["7","8","9","J","Q","K","10","A"]
_NONTR_STR = [0, 1, 2, 6, 3, 4, 5, 7]

# Trump strength: TRUMP_ORDER = ["7","8","Q","K","10","A","9","J"]
_TRUMP_STR = [0, 1, 6, 4, 7, 2, 3, 5]

# Reverse: trump strength → rank index
_TRUMP_STR_TO_RANK = [0, 1, 5, 6, 3, 7, 2, 4]

# Non-trump points by rank
_NONTR_PTS = [0, 0, 0, 10, 2, 3, 4, 11]

# Trump points by rank
_TRUMP_PTS = [0, 0, 14, 10, 20, 3, 4, 11]

# Team of each seat: {0:0, 1:1, 2:0, 3:1}
_SEAT_TEAM = [0, 1, 0, 1]

NUM_CARDS = 32
NUM_FEATURES = 267


class KlaverjasGPUEngine:
    """Runs B Klaverjassen games simultaneously on GPU.

    All game state lives in GPU tensors.  A single call to step() advances
    every game by one card play.  Trick and round resolution happens
    automatically inside step().

    Usage:
        engine = KlaverjasGPUEngine(batch_size=512, device="cuda")
        engine.reset()
        while True:
            feats, masks = engine.get_state()
            actions = policy(feats, masks)
            rewards, round_done = engine.step(actions)
    """

    def __init__(
        self,
        batch_size: int,
        device: str | torch.device = "cuda",
        score_limit: int = 500,
        bid_threshold: float = 2.75,
    ):
        self.B = batch_size
        self.device = torch.device(device)
        self.score_limit = score_limit
        self.bid_threshold = bid_threshold

        # ── Constant lookup tensors (registered once) ────────────────────────
        def reg(lst, dtype=torch.long):
            return torch.tensor(lst, dtype=dtype, device=self.device)

        self.NONTR_STR = reg(_NONTR_STR, torch.long)   # [8]
        self.TRUMP_STR = reg(_TRUMP_STR, torch.long)   # [8]
        self.STR2RANK  = reg(_TRUMP_STR_TO_RANK, torch.long)  # [8]
        self.NONTR_PTS = reg(_NONTR_PTS, torch.int32)  # [8]
        self.TRUMP_PTS = reg(_TRUMP_PTS, torch.int32)  # [8]
        self.SEAT_TEAM = reg(_SEAT_TEAM, torch.long)   # [4]

        # SUIT_OF[c] = suit index of card c  (c // 8)
        self.SUIT_OF = torch.arange(NUM_CARDS, device=self.device) // 8  # [32]
        # RANK_OF[c] = rank index of card c  (c % 8)
        self.RANK_OF = torch.arange(NUM_CARDS, device=self.device) % 8   # [32]

        # 4-of-a-kind points per rank (200 for Jacks=rank4, 100 otherwise)
        self._FOAK_PTS = torch.tensor(
            [100, 100, 100, 100, 200, 100, 100, 100],
            dtype=torch.int32, device=self.device)

        # Opponent-seats lookup: OPP_SEATS[s] = 3 non-s seats in ascending order.
        # Used to avoid boolean indexing (dynamic-shape; blocks CUDA graph capture).
        self.OPP_SEATS = torch.tensor(
            [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
            dtype=torch.long, device=self.device)

        # Scalar-valued scratch tensors — advanced-index assignments with a
        # Python scalar source (e.g. `t[ar, idx] = 0.0`) aren't CUDA-graph
        # captureable; using tensor sources is.
        self._ZERO_B = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self._ONE_B  = torch.ones(batch_size, dtype=torch.float32, device=self.device)

        # ── Mutable game state (allocated once; only mutated in-place) ───────
        B = self.B
        Z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=self.device)
        I = lambda *shape: torch.zeros(*shape, dtype=torch.long,    device=self.device)
        T = lambda *shape: torch.zeros(*shape, dtype=torch.int32,   device=self.device)

        # Per-round state
        self.hands           = Z(B, 4, NUM_CARDS)   # [B,4,32] binary
        self.initial_hands   = Z(B, 4, NUM_CARDS)   # [B,4,32] for roem
        self.played          = Z(B, NUM_CARDS)       # [B,32] played this round
        self.opponent_voids  = Z(B, 4, 4)           # [B,seat,suit]
        self.roem_pts        = T(B, 2)
        self.trick_pts       = T(B, 2)
        self.game_scores     = T(B, 2)
        self.round_num       = I(B)
        self.dealer          = I(B)
        self.trump           = I(B)
        self.declaring_team  = I(B)

        # Per-trick state
        self.trick_cards      = torch.full((B, 4), -1, dtype=torch.long, device=self.device)
        self.trick_seats      = torch.full((B, 4), -1, dtype=torch.long, device=self.device)
        self.cards_in_trick   = I(B)
        self.trick_leader     = I(B)
        self.trick_num        = I(B)
        self.current_seat     = I(B)

        # Terminal flag
        self.done = torch.zeros(B, dtype=torch.bool, device=self.device)

        # Round reward storage (filled by _score_round, read by training loop)
        self._last_round_pts = T(B, 2)

        self._arange = torch.arange(B, device=self.device)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all B games to fresh state and deal first round."""
        B = self.B
        self.game_scores.zero_()
        self.round_num.zero_()
        self.dealer.random_(0, 4)
        self.done.zero_()
        self._last_round_pts.zero_()
        active = torch.ones(B, dtype=torch.bool, device=self.device)
        self._start_new_round(active)

    def get_state(self) -> tuple[Tensor, Tensor]:
        """Return (features [B,267], legal_mask [B,32]) for current players."""
        masks = self.legal_mask()
        feats = self.encode_features(masks)
        return feats, masks

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Play one card per game, advance state.

        Sync-free: every branch always runs; results are masked to the
        appropriate games. State updates are all in-place.

        Args:
            actions: [B] card indices (must be legal for each game).
        Returns:
            rewards:    [B] float32 — non-zero at round end (team point diff / 162)
            round_done: [B] bool    — True when a round just completed
        """
        ar = self._arange
        seat = self.current_seat.clone()  # capture BEFORE any mutations

        # ── Play the card (in-place advanced-index assign with tensor src) ──
        self.hands[ar, seat, actions] = self._ZERO_B
        self.played[ar, actions] = self._ONE_B

        # ── Void inference (always compute; masked update) ───────────────────
        lead_card   = self.trick_cards[:, 0].clamp(min=0)
        lead_suit   = self.SUIT_OF[lead_card]
        played_suit = self.SUIT_OF[actions]
        has_trick   = (self.cards_in_trick > 0)
        is_void     = has_trick & (played_suit != lead_suit)
        # Conditionally set opponent_voids[b, seat, lead_suit] = 1 where is_void;
        # otherwise restore the old value (no-op).
        cur_void = self.opponent_voids[ar, seat, lead_suit]
        self.opponent_voids[ar, seat, lead_suit] = torch.where(
            is_void, torch.ones_like(cur_void), cur_void)

        # ── Add card to current trick ────────────────────────────────────────
        pos = self.cards_in_trick.clone()  # [B] position in trick
        self.trick_cards[ar, pos] = actions
        self.trick_seats[ar, pos] = seat
        self.cards_in_trick.add_(1)

        # ── Resolve complete tricks; compute rewards (always run) ────────────
        trick_complete = (self.cards_in_trick == 4)
        round_done = self._resolve_trick(trick_complete)
        rewards    = self._compute_round_rewards(round_done, seat)

        # ── Advance current seat for incomplete tricks (always computed) ─────
        still_in_trick = ~trick_complete & ~self.done
        next_seat = (self.trick_leader + self.cards_in_trick) % 4
        self.current_seat.copy_(
            torch.where(still_in_trick, next_seat, self.current_seat))

        return rewards, round_done

    # ─────────────────────────────────────────────────────────────────────────
    # Legal moves
    # ─────────────────────────────────────────────────────────────────────────

    def legal_mask(self) -> Tensor:
        """Compute legal move mask for the current player in each game.
        Returns [B, 32] float32 (1=legal, 0=illegal).

        Runs the full trick-rules path unconditionally; the `no_trick` branch
        is folded into the final torch.where so the op graph is static.
        """
        B = self.B
        ar = self._arange
        seat = self.current_seat  # [B]
        trump_suit = self.trump   # [B]

        player_hand = self.hands[ar, seat]  # [B, 32]
        no_trick = (self.cards_in_trick == 0)

        # ── Active-trick computation (runs for all games) ────────────────────
        lead_card = self.trick_cards[:, 0].clamp(min=0)
        lead_suit  = self.SUIT_OF[lead_card]   # [B]

        card_suits = self.SUIT_OF.unsqueeze(0)  # [1, 32]

        # Trump cards in hand
        trump_in_hand = player_hand * (card_suits == trump_suit.unsqueeze(1)).float()
        has_trump = trump_in_hand.sum(1) > 0  # [B]

        # Lead-suit cards in hand
        lead_in_hand = player_hand * (card_suits == lead_suit.unsqueeze(1)).float()
        has_lead = lead_in_hand.sum(1) > 0  # [B]

        # Existing trumps in trick and their max strength
        tc_suits = self.SUIT_OF[self.trick_cards.clamp(min=0)]  # [B, 4]
        tc_ranks = self.trick_cards.clamp(min=0) % 8            # [B, 4]
        tc_is_trump = (tc_suits == trump_suit.unsqueeze(1)) & (self.trick_cards >= 0)
        tc_trump_str = self.TRUMP_STR[tc_ranks]
        tc_trump_str_m = torch.where(tc_is_trump, tc_trump_str,
                                     torch.full_like(tc_trump_str, -1))
        highest_trick_trump = tc_trump_str_m.max(dim=1).values   # [B]
        any_trump_in_trick = tc_is_trump.any(dim=1)              # [B] tensor (not .any().item())

        # My trump strengths per card
        my_trump_str = self.TRUMP_STR[self.RANK_OF].unsqueeze(0)  # [1, 32]

        # Over-trumpers: my trump cards stronger than current highest trick trump
        over_trump = (trump_in_hand > 0) & (my_trump_str > highest_trick_trump.unsqueeze(1))
        has_over_trump = over_trump.any(dim=1)  # [B]

        # ── Case 1: has lead-suit cards ──────────────────────────────────────
        lead_is_trump = (lead_suit == trump_suit)  # [B]

        # Case 1a: lead IS trump → must overtrump if possible
        case1a = torch.where(has_over_trump.unsqueeze(1),
                             over_trump.float(), lead_in_hand)

        # Case 1b: lead is NOT trump → play any lead-suit card
        case1b = lead_in_hand

        case1 = torch.where(lead_is_trump.unsqueeze(1), case1a, case1b)

        # ── Case 2: no lead-suit cards → must trump (or discard) ────────────
        case2_trump_over = torch.where(has_over_trump.unsqueeze(1),
                                       over_trump.float(), trump_in_hand)
        case2_with_trick_trump = torch.where(has_trump.unsqueeze(1),
                                             case2_trump_over, player_hand)
        case2_no_trick_trump   = torch.where(has_trump.unsqueeze(1),
                                             trump_in_hand, player_hand)
        case2 = torch.where(any_trump_in_trick.unsqueeze(1),
                            case2_with_trick_trump, case2_no_trick_trump)

        trick_legal = torch.where(has_lead.unsqueeze(1), case1, case2)

        # Leading games use full hand; non-leading games use trick_legal
        result = torch.where(no_trick.unsqueeze(1), player_hand, trick_legal)
        return result.float()

    # ─────────────────────────────────────────────────────────────────────────
    # Feature encoding
    # ─────────────────────────────────────────────────────────────────────────

    def encode_features(self, legal_mask: Tensor | None = None) -> Tensor:
        """Encode full game state as [B, 267] feature vectors.

        Matches the layout of neural/features.py encode_state() exactly.
        """
        B = self.B
        ar = self._arange
        seat = self.current_seat   # [B]
        team = self.SEAT_TEAM[seat]   # [B]
        opp_team = 1 - team           # [B]
        trump_suit = self.trump       # [B]

        player_hand = self.hands[ar, seat]   # [B, 32]
        card_suits = self.SUIT_OF.unsqueeze(0)   # [1, 32]

        feats = torch.zeros(B, NUM_FEATURES, dtype=torch.float32, device=self.device)
        o = 0  # offset

        # 1. My hand (32)
        feats[:, o:o+32] = player_hand
        o += 32

        # 2. Cards played this round (32)
        feats[:, o:o+32] = self.played
        o += 32

        # 3. Current trick cards — 3 positional slots × 32 (96)
        for i in range(3):
            valid = (self.cards_in_trick > i) & (self.trick_cards[:, i] >= 0)
            ci = self.trick_cards[:, i].clamp(min=0)
            oh = F.one_hot(ci, num_classes=32).float()
            feats[:, o + i*32 : o + (i+1)*32] = oh * valid.unsqueeze(1).float()
        o += 96

        # 4. Trump suit one-hot (4)
        feats[:, o:o+4] = F.one_hot(trump_suit, num_classes=4).float()
        o += 4

        # 5. Position in trick one-hot (4)
        feats[:, o:o+4] = F.one_hot(self.cards_in_trick.clamp(max=3), num_classes=4).float()
        o += 4

        # 6. Trick number normalized (1)
        feats[:, o] = self.trick_num.float() / 7.0
        o += 1

        # 7. Team trick points normalized (2)
        feats[:, o]   = self.trick_pts[ar, team].float() / 162.0
        feats[:, o+1] = self.trick_pts[ar, opp_team].float() / 162.0
        o += 2

        # 8. Team roem points normalized (2)
        feats[:, o]   = self.roem_pts[ar, team].float() / 200.0
        feats[:, o+1] = self.roem_pts[ar, opp_team].float() / 200.0
        o += 2

        # 9. Declaring team flags (2)
        feats[:, o]   = (self.declaring_team == team).float()
        feats[:, o+1] = (self.declaring_team == opp_team).float()
        o += 2

        # 10. Opponent voids: 3 opponents × 4 suits (12)
        # Static lookup instead of boolean indexing — keeps shape known to the
        # graph capturer.
        opp_seats = self.OPP_SEATS[seat]  # [B, 3]
        for oi in range(3):
            opp_s = opp_seats[:, oi]
            feats[:, o + oi*4 : o + oi*4 + 4] = self.opponent_voids[ar, opp_s]
        o += 12

        # 11. Game scores normalized (2)
        feats[:, o]   = self.game_scores[ar, team].float() / 500.0
        feats[:, o+1] = self.game_scores[ar, opp_team].float() / 500.0
        o += 2

        # 12. Round number normalized (1)
        feats[:, o] = self.round_num.float() / 16.0
        o += 1

        # 13. Legal move mask (32)
        if legal_mask is None:
            legal_mask = self.legal_mask()
        feats[:, o:o+32] = legal_mask
        o += 32

        # 14. Cards in hand count normalized (1)
        feats[:, o] = player_hand.sum(1) / 8.0
        o += 1

        # 15. Trump cards in hand normalized (1)
        trump_hand_mask = (card_suits == trump_suit.unsqueeze(1)).float()
        trump_count = (player_hand * trump_hand_mask).sum(1)
        feats[:, o] = trump_count / 8.0
        o += 1

        # 16. Highest outstanding trump one-hot (8)
        trump_card_base = trump_suit * 8
        rank_offsets = torch.arange(8, device=self.device)
        trump_card_idx = trump_card_base.unsqueeze(1) + rank_offsets.unsqueeze(0)
        hand_tp   = player_hand.gather(1, trump_card_idx)
        played_tp = self.played.gather(1, trump_card_idx)
        outstanding_tp = (1 - hand_tp - played_tp).clamp(min=0)

        tp_str = self.TRUMP_STR[rank_offsets].unsqueeze(0).expand(B, -1)
        out_str_m = torch.where(outstanding_tp.bool(), tp_str,
                                torch.full_like(tp_str, -1))
        highest_out_str = out_str_m.max(dim=1).values
        has_out = outstanding_tp.sum(1) > 0
        highest_out_rank = self.STR2RANK[highest_out_str.clamp(min=0)]
        hot_out = F.one_hot(highest_out_rank, num_classes=8).float()
        feats[:, o:o+8] = hot_out * has_out.unsqueeze(1).float()
        o += 8

        # 17. Partner winning current trick (1)
        win_pos = self._current_trick_winner_pos()
        has_trick = (self.cards_in_trick > 0)
        win_seat = self.trick_seats[ar, win_pos.clamp(min=0)]
        win_team = self.SEAT_TEAM[win_seat]
        partner_winning = has_trick & (win_team == team)
        feats[:, o] = partner_winning.float()
        o += 1

        # 18. Points on table normalized (1)
        tc_valid = (self.trick_cards >= 0)
        tc_suits_f = self.SUIT_OF[self.trick_cards.clamp(min=0)]
        tc_ranks_f = self.trick_cards.clamp(min=0) % 8
        tc_is_trump = (tc_suits_f == trump_suit.unsqueeze(1)) & tc_valid
        tc_pts = torch.where(tc_is_trump,
                             self.TRUMP_PTS[tc_ranks_f],
                             self.NONTR_PTS[tc_ranks_f])
        pts_on_table = (tc_pts * tc_valid.int()).sum(1).float()
        feats[:, o] = pts_on_table / 40.0
        o += 1

        # 19. My seat one-hot (4)
        feats[:, o:o+4] = F.one_hot(seat, num_classes=4).float()
        o += 4

        # 20. Lead suit one-hot (4)
        lead_card = self.trick_cards[:, 0].clamp(min=0)
        lead_suit  = self.SUIT_OF[lead_card]
        lead_oh = F.one_hot(lead_suit, num_classes=4).float()
        feats[:, o:o+4] = lead_oh * has_trick.unsqueeze(1).float()
        o += 4

        # 21. Cards per suit in hand normalized (4)
        for si in range(4):
            suit_mask = (card_suits == si).float()
            feats[:, o+si] = (player_hand * suit_mask).sum(1) / 8.0
        o += 4

        # 22. Trump control: hold highest remaining trump (1)
        my_tp_str = torch.where(hand_tp.bool(), tp_str,
                                torch.full_like(tp_str, -1))
        my_best = my_tp_str.max(dim=1).values
        has_my_trump = hand_tp.sum(1) > 0
        higher_out = outstanding_tp.bool() & (tp_str > my_best.unsqueeze(1))
        has_higher_out = higher_out.any(dim=1)
        trump_control = has_my_trump & ~has_higher_out
        feats[:, o] = trump_control.float()
        o += 1

        # 23. Trick cards count one-hot (4)
        feats[:, o:o+4] = F.one_hot(self.cards_in_trick.clamp(max=3), num_classes=4).float()
        o += 4

        # 24. Suit master cards per suit (4)
        for si in range(4):
            hand_s   = player_hand[:, si*8:(si+1)*8]
            played_s = self.played[:, si*8:(si+1)*8]
            out_s    = (1 - hand_s - played_s).clamp(min=0)

            is_tp_suit = (trump_suit == si)
            str_t = self.TRUMP_STR.unsqueeze(0).expand(B, -1)
            str_n = self.NONTR_STR.unsqueeze(0).expand(B, -1)
            strength = torch.where(is_tp_suit.unsqueeze(1), str_t, str_n)

            out_str_s = torch.where(out_s.bool(), strength,
                                    torch.full_like(strength, -1))
            max_out = out_str_s.max(dim=1).values

            is_master = (hand_s > 0) & (strength > max_out.unsqueeze(1))
            feats[:, o+si] = is_master.float().sum(1)
        o += 4

        # 25. Point cards in hand normalized (1)
        all_is_trump = (card_suits == trump_suit.unsqueeze(1)).expand(B, -1)
        card_pts = torch.where(all_is_trump,
                               self.TRUMP_PTS[self.RANK_OF].unsqueeze(0).expand(B, -1).float(),
                               self.NONTR_PTS[self.RANK_OF].unsqueeze(0).expand(B, -1).float())
        point_sum = (player_hand * card_pts).sum(1)
        feats[:, o] = point_sum / 60.0
        o += 1

        # 26. Suit lengths played this round (4)
        for si in range(4):
            feats[:, o+si] = self.played[:, si*8:(si+1)*8].sum(1) / 8.0
        o += 4

        # 27. Net score delta (1)
        feats[:, o] = (self.trick_pts[ar, team] - self.trick_pts[ar, opp_team]).float() / 162.0
        o += 1

        # 28. Team is declaring (1)
        feats[:, o] = (self.declaring_team == team).float()
        o += 1

        # 29. Last trick (1)
        feats[:, o] = (self.trick_num == 7).float()
        o += 1

        # 30. Tricks remaining normalized (1)
        feats[:, o] = (7 - self.trick_num).float().clamp(min=0) / 7.0
        o += 1

        # 31. Score pressure (1)
        feats[:, o] = self.game_scores.max(dim=1).values.float() / 500.0
        o += 1

        # 32. Declarer's team leading (1)
        feats[:, o] = ((self.declaring_team == team) & ~has_trick).float()
        o += 1

        # 33. Legal moves count normalized (1)
        feats[:, o] = legal_mask.sum(1) / 8.0
        o += 1

        assert o == NUM_FEATURES, f"Feature count mismatch: {o}"
        return feats

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _current_trick_winner_pos(self) -> Tensor:
        """Position (0-3) of current winning card. Returns -1 where no cards."""
        ar = self._arange
        trump_suit = self.trump
        has_cards = (self.cards_in_trick > 0)

        lead_card = self.trick_cards[:, 0].clamp(min=0)
        lead_suit  = self.SUIT_OF[lead_card]

        tc_suits = self.SUIT_OF[self.trick_cards.clamp(min=0)]
        tc_ranks = self.trick_cards.clamp(min=0) % 8
        tc_valid = (self.trick_cards >= 0)

        is_trump = (tc_suits == trump_suit.unsqueeze(1)) & tc_valid
        is_lead  = (tc_suits == lead_suit.unsqueeze(1)) & tc_valid

        tr_str = self.TRUMP_STR[tc_ranks]
        nt_str = self.NONTR_STR[tc_ranks]

        any_trump = is_trump.any(dim=1)

        trump_m = torch.where(is_trump, tr_str, torch.full_like(tr_str, -1))
        lead_m  = torch.where(is_lead,  nt_str, torch.full_like(nt_str, -1))

        trump_winner = trump_m.argmax(dim=1)
        lead_winner  = lead_m.argmax(dim=1)
        winner_pos = torch.where(any_trump, trump_winner, lead_winner)

        return torch.where(has_cards, winner_pos,
                           torch.full_like(winner_pos, -1))

    def _resolve_trick(self, active: Tensor) -> Tensor:
        """Resolve completed tricks for active games. Always runs; ops masked.
        Returns round_done: [B] bool."""
        ar = self._arange
        trump_suit = self.trump

        lead_card = self.trick_cards[:, 0].clamp(min=0)
        lead_suit  = self.SUIT_OF[lead_card]

        tc_suits = self.SUIT_OF[self.trick_cards.clamp(min=0)]
        tc_ranks = self.trick_cards.clamp(min=0) % 8

        is_trump = (tc_suits == trump_suit.unsqueeze(1))
        is_lead  = (tc_suits == lead_suit.unsqueeze(1))

        tr_str = self.TRUMP_STR[tc_ranks]
        nt_str = self.NONTR_STR[tc_ranks]

        any_trump = is_trump.any(dim=1)

        trump_m = torch.where(is_trump, tr_str, torch.full_like(tr_str, -1))
        lead_m  = torch.where(is_lead,  nt_str, torch.full_like(nt_str, -1))

        winner_pos = torch.where(any_trump, trump_m.argmax(1), lead_m.argmax(1))
        winner_seat = self.trick_seats[ar, winner_pos]
        winner_team = self.SEAT_TEAM[winner_seat]

        # Trick points (masked by is_trump)
        card_pts = torch.where(is_trump,
                               self.TRUMP_PTS[tc_ranks],
                               self.NONTR_PTS[tc_ranks])
        trick_total = card_pts.sum(dim=1)                        # [B]
        last_trick = (self.trick_num == 7)
        trick_total = trick_total + (last_trick.long() * 10).int()

        # Distribute trick points to winning team (in-place +=)
        for t in range(2):
            mask = active & (winner_team == t)
            self.trick_pts[ar, t] += torch.where(mask, trick_total,
                                                  torch.zeros_like(trick_total))

        # Advance trick counter (in-place add)
        self.trick_num.add_(active.long())

        # Reset trick buffers for active games (in-place masked fill)
        self.trick_cards.masked_fill_(active.unsqueeze(1), -1)
        self.trick_seats.masked_fill_(active.unsqueeze(1), -1)
        self.cards_in_trick.masked_fill_(active, 0)

        # Update leader & current seat to winner (in-place)
        self.trick_leader.copy_(torch.where(active, winner_seat, self.trick_leader))
        self.current_seat.copy_(torch.where(active, winner_seat, self.current_seat))

        # Round ends after 8 tricks (trick_num just incremented)
        round_done = active & (self.trick_num >= 8)
        # Always call (masked internally); handles all-False case cheaply.
        self._score_round(round_done)

        return round_done

    def _score_round(self, active: Tensor) -> None:
        """Apply round scoring, update game_scores, start new round or recycle.

        Always runs; if `active` is all-False the state updates are no-ops
        because every write is masked by `active` or `game_over ⊂ active`.
        """
        B = self.B
        ar = self._arange
        decl = self.declaring_team
        opp  = 1 - decl

        decl_trick = self.trick_pts[ar, decl]
        opp_trick  = self.trick_pts[ar, opp]
        decl_roem  = self.roem_pts[ar, decl]
        opp_roem   = self.roem_pts[ar, opp]

        # Pit bonus: +100 roem to whichever team took all 162 trick points
        # (matches main.py line 2163). Added as roem so it correctly propagates
        # into the nat penalty (nat_opp = 162 + total_roem) when applicable.
        pit_bonus  = torch.full_like(decl_roem, 100)
        zero_roem  = torch.zeros_like(decl_roem)
        decl_pit   = active & (decl_trick == 162)
        opp_pit    = active & (opp_trick  == 162)
        decl_roem  = decl_roem + torch.where(decl_pit, pit_bonus, zero_roem)
        opp_roem   = opp_roem  + torch.where(opp_pit,  pit_bonus, zero_roem)
        total_roem = decl_roem + opp_roem

        is_nat  = active & (decl_trick <= 81)

        normal_decl = decl_trick + decl_roem
        normal_opp  = opp_trick  + opp_roem
        nat_decl = torch.zeros(B, dtype=torch.int32, device=self.device)
        nat_opp  = (162 + total_roem).int()

        final_decl = torch.where(is_nat, nat_decl, normal_decl)
        final_opp  = torch.where(is_nat, nat_opp,  normal_opp)

        # Store last-round points (masked by active)
        zero_b = torch.zeros(B, dtype=torch.int32, device=self.device)
        self._last_round_pts[ar, decl] = torch.where(active, final_decl, zero_b)
        self._last_round_pts[ar, opp]  = torch.where(active, final_opp,  zero_b)

        # Add to game scores (+= with zero for inactive)
        self.game_scores[ar, decl] += torch.where(active, final_decl, zero_b)
        self.game_scores[ar, opp]  += torch.where(active, final_opp,  zero_b)

        # Detect game-over
        game_over = active & (self.game_scores.max(dim=1).values >= self.score_limit)

        # Recycle game_over games: reset scores, round counter, dealer (masked in-place)
        self.game_scores.copy_(torch.where(
            game_over.unsqueeze(1),
            torch.zeros_like(self.game_scores),
            self.game_scores))
        self.round_num.copy_(torch.where(
            game_over, torch.zeros_like(self.round_num), self.round_num))
        # Clear done for recycled games (done tracks terminal only transiently here)
        self.done.masked_fill_(game_over, False)
        new_dealer = torch.randint(0, 4, (B,), device=self.device, dtype=torch.long)
        self.dealer.copy_(torch.where(game_over, new_dealer, self.dealer))

        # Start a new round for all active games (continuing + just-recycled).
        # When `active` is all-False this is a cheap no-op via internal masking.
        self._start_new_round(active)

    def _compute_round_rewards(self, active: Tensor, seat: Tensor) -> Tensor:
        """Return per-game reward = (team0_pts - team1_pts) / 162.

        Team-0 perspective is invariant to which seat ended the round, so the
        signal is valid whether the training policy or the frozen opponent
        played the last card. Callers filter by `valid` mask to keep only the
        transitions that belong to the trained policy.
        """
        ar = self._arange
        team0_pts = self._last_round_pts[ar, 0].float()
        team1_pts = self._last_round_pts[ar, 1].float()
        reward = (team0_pts - team1_pts) / 162.0
        return torch.where(active, reward, torch.zeros_like(reward))

    def _start_new_round(self, active: Tensor) -> None:
        """Reset round state, deal hands, bid, compute roem for active games.
        Sync-free; all updates masked by `active`."""
        # Reset round counters in-place (masked)
        self.trick_num.masked_fill_(active, 0)
        self.cards_in_trick.masked_fill_(active, 0)
        self.trick_cards.masked_fill_(active.unsqueeze(1), -1)
        self.trick_seats.masked_fill_(active.unsqueeze(1), -1)
        self.played.masked_fill_(active.unsqueeze(1), 0.0)
        self.opponent_voids.masked_fill_(active.view(-1, 1, 1), 0.0)
        self.trick_pts.masked_fill_(active.unsqueeze(1), 0)
        self.roem_pts.masked_fill_(active.unsqueeze(1), 0)
        self.round_num.copy_(torch.where(
            active, self.round_num + 1, self.round_num))

        # Advance dealer (masked)
        new_dealer = (self.dealer + 1) % 4
        self.dealer.copy_(torch.where(active, new_dealer, self.dealer))

        # Deal hands (runs for all B; masked inside)
        self._deal_hands(active)

        # Bidding (runs unconditionally; masked inside)
        self._bidding(active)

        # Roem scoring
        roem = self._compute_roem_batch(self.initial_hands, self.trump)  # [B, 2]
        self.roem_pts.copy_(torch.where(
            active.unsqueeze(1), roem, self.roem_pts))

        # First trick leader = player left of dealer
        first_bidder = (self.dealer + 1) % 4
        self.trick_leader.copy_(torch.where(active, first_bidder, self.trick_leader))
        self.current_seat.copy_(torch.where(active, first_bidder, self.current_seat))

    def _deal_hands(self, active: Tensor) -> None:
        """Deal random hands to all B games; inactive games' hands unchanged.

        Always generates B hands (vs. previously `n = active.sum().item()` which
        was a host sync), then uses torch.where to merge.
        """
        B = self.B

        noise = torch.rand(B, NUM_CARDS, device=self.device)
        perm  = noise.argsort(dim=1)  # [B, 32]

        # Build hands via one-hot+sum (graph-capture-friendly; avoids scalar
        # scatter_ which isn't always captureable).
        new_hands = torch.zeros(B, 4, NUM_CARDS, dtype=torch.float32, device=self.device)
        for seat in range(4):
            seat_cards = perm[:, seat*8:(seat+1)*8]          # [B, 8]
            new_hands[:, seat] = F.one_hot(
                seat_cards, num_classes=NUM_CARDS).float().sum(dim=1)

        mask = active.view(B, 1, 1)
        self.hands.copy_(torch.where(mask, new_hands, self.hands))
        self.initial_hands.copy_(torch.where(mask, new_hands, self.initial_hands))

    def _bid_score(self, hand: Tensor, suit: Tensor) -> Tensor:
        """Compute bid strength score for each active game.

        Args:
            hand: [B, 32] binary hand tensor
            suit: [B] offered trump suit index
        Returns:
            score: [B] float
        """
        B = hand.shape[0]
        ar = torch.arange(B, device=self.device)

        # High trump card bonuses
        J_idx  = suit * 8 + 4
        N9_idx = suit * 8 + 2
        A_idx  = suit * 8 + 7
        T10_idx = suit * 8 + 3
        K_idx  = suit * 8 + 6
        Q_idx  = suit * 8 + 5

        score = (hand[ar, J_idx]   * 1.85
               + hand[ar, N9_idx]  * 1.30
               + hand[ar, A_idx]   * 0.85
               + hand[ar, T10_idx] * 0.55
               + hand[ar, K_idx]   * 0.25
               + hand[ar, Q_idx]   * 0.15)

        # Side-suit aces and void/singleton bonuses
        suit_counts = torch.zeros(B, 4, device=self.device)
        for s in range(4):
            suit_counts[:, s] = hand[:, s*8:(s+1)*8].sum(1)

        for offset in range(1, 4):
            s = (suit + offset) % 4
            cnt = suit_counts[ar, s]
            A_side = s * 8 + 7
            score = score + hand[ar, A_side] * 0.70
            score = score + (cnt == 0).float() * 0.35
            score = score + (cnt == 1).float() * 0.17

        # Stuk bonus (K+Q of trump)
        has_K = hand[ar, K_idx]
        has_Q = hand[ar, Q_idx]
        score = score + (has_K * has_Q) * 0.30

        return score

    def _bidding(self, active: Tensor) -> None:
        """Vectorized bidding. Always runs all 8 bid iterations (no early
        break) to keep the op sequence static for graph capture."""
        B = self.B
        ar = self._arange

        noise = torch.rand(B, 4, device=self.device)
        suit_perm = noise.argsort(dim=1)
        r1_suit = suit_perm[:, 0]
        r2_suit = suit_perm[:, 1]

        first_bidder = (self.dealer + 1) % 4
        declared = torch.zeros(B, dtype=torch.bool, device=self.device)
        trump = r1_suit.clone()
        decl_team = self.SEAT_TEAM[first_bidder]

        # Round 1 — 4 seats get a chance at r1_suit
        for i in range(4):
            bidder = (first_bidder + i) % 4
            hand_b = self.hands[ar, bidder]
            score = self._bid_score(hand_b, r1_suit)
            bids = active & ~declared & (score >= self.bid_threshold)
            trump = torch.where(bids, r1_suit, trump)
            decl_team = torch.where(bids, self.SEAT_TEAM[bidder], decl_team)
            declared = declared | bids

        # Round 2 — same 4 seats at r2_suit
        for i in range(4):
            bidder = (first_bidder + i) % 4
            hand_b = self.hands[ar, bidder]
            score = self._bid_score(hand_b, r2_suit)
            bids = active & ~declared & (score >= self.bid_threshold)
            trump = torch.where(bids, r2_suit, trump)
            decl_team = torch.where(bids, self.SEAT_TEAM[bidder], decl_team)
            declared = declared | bids

        # Forced declare: first bidder takes round-2 suit
        forced = active & ~declared
        trump = torch.where(forced, r2_suit, trump)
        decl_team = torch.where(forced, self.SEAT_TEAM[first_bidder], decl_team)

        # Commit to state (masked in-place)
        self.trump.copy_(torch.where(active, trump, self.trump))
        self.declaring_team.copy_(torch.where(active, decl_team, self.declaring_team))

    def _compute_roem_batch(self, hands: Tensor, trump: Tensor) -> Tensor:
        """Compute roem for all seats, return [B, 2] team totals."""
        B = hands.shape[0]
        ar = torch.arange(B, device=self.device)
        roem = torch.zeros(B, 2, dtype=torch.int32, device=self.device)

        K_idx = trump * 8 + 6
        Q_idx = trump * 8 + 5

        for seat in range(4):
            team = _SEAT_TEAM[seat]
            hand = hands[:, seat]  # [B, 32]

            # Stuk (K+Q of trump = 20 pts)
            stuk = (hand[ar, K_idx] * hand[ar, Q_idx] * 20).int()
            roem[:, team] += stuk

            # 4-of-a-kind (vectorized): [B,32] → [B,4,8] → sum over suits → [B,8]
            by_suit = hand.view(B, 4, 8)
            rank_count = by_suit.sum(dim=1)  # [B, 8]
            foak = ((rank_count == 4).int() * self._FOAK_PTS).sum(dim=1)
            roem[:, team] += foak

            # Sequences per suit (3+=20, 4=50, 5+=100)
            for suit_idx in range(4):
                suit_hand = hand[:, suit_idx*8:(suit_idx+1)*8].int()  # [B, 8]

                # Scan run lengths
                run_len = torch.zeros(B, 8, dtype=torch.int32, device=self.device)
                prev = torch.zeros(B, dtype=torch.int32, device=self.device)
                for r in range(8):
                    cur = suit_hand[:, r]
                    run_len[:, r] = torch.where(cur.bool(), prev + 1,
                                                torch.zeros_like(prev))
                    prev = run_len[:, r]

                # End-of-run indicator: card present, next card absent (or end)
                next_cur = torch.cat([
                    suit_hand[:, 1:],
                    torch.zeros(B, 1, dtype=torch.int32, device=self.device),
                ], dim=1)
                is_end = suit_hand.bool() & ~next_cur.bool()

                seq_pts = torch.where(run_len >= 5,
                             torch.full_like(run_len, 100),
                             torch.where(run_len == 4,
                                torch.full_like(run_len, 50),
                                torch.where(run_len == 3,
                                   torch.full_like(run_len, 20),
                                   torch.zeros_like(run_len))))
                awarded = (is_end.int() * seq_pts).sum(dim=1)
                roem[:, team] += awarded

        return roem
