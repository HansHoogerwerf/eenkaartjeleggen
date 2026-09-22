"""Net-rollout PIMC: evaluate candidate cards by playing out the rest of the
round with the neural net over sampled deals, batched on the GPU engine.

For a card-play decision this samples K deals of the unseen cards that are
consistent with everything the player has inferred, sets the GPU engine to
that state for every (candidate, deal) pair, plays the candidate, and lets
the net (greedy) play every seat to the end of the round.  The candidate's
value is its mean round outcome (team frame, nat/pit applied by the engine).

This is a search-based *improvement operator* over the net's own policy: the
net proposes, rollouts with the same net judge.  Used by PIMCNetPlayer for
play and for generating training targets that are stronger than the net.

Runs on CUDA when available (CUDA-graph captured) and on the CPU otherwise
(~0.5 s per decision at 32 deals with 4 threads).  NEURAL_PIMC_DEVICE forces
a device, NEURAL_PIMC_THREADS caps the CPU threads (set it to 1 in
multi-process benchmarks).  evaluate() is serialised by a lock, so one
evaluator can be shared by every AI seat of a server process.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import torch

from klaverjas.constants import SEAT_TEAMS, SUITS
from klaverjas.core import Card
from neural.features import CARD_INDEX, feature_version_for_size
from neural.model import KlaverjasNet
from tools.gpu_engine import KlaverjasGPUEngine


class NetRolloutEvaluator:
    """Owns one GPU engine + net and scores candidate cards by rollouts."""

    def __init__(self, model_path: str | Path, deals: int = 32, max_candidates: int = 8,
                 device: str | None = None):
        device = device or os.environ.get("NEURAL_PIMC_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        threads = int(os.environ.get("NEURAL_PIMC_THREADS") or 0)
        if threads > 0 and self.device.type == "cpu":
            torch.set_num_threads(threads)
        self._lock = threading.Lock()
        sd = torch.load(str(model_path), map_location=self.device, weights_only=True)
        self.net = KlaverjasNet.from_state_dict(sd).to(self.device).eval()
        self.feature_version = feature_version_for_size(self.net.in_features)
        self.deals = deals
        self.max_candidates = max_candidates
        self.B = deals * max_candidates
        self.engine = KlaverjasGPUEngine(batch_size=self.B, device=self.device,
                                         feature_version=self.feature_version)
        self.engine.reset()
        # Static buffers for the CUDA-graph-captured rollout (see _capture).
        self._first = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self._total = torch.zeros(self.B, device=self.device)
        self._graph = None
        self.use_graph = self.device.type == "cuda"

    # ── rollout (eager, and CUDA-graph captured) ───────────────────────────
    def _rollout_body(self) -> None:
        """One full rollout: candidate card, then 31 greedy net steps."""
        e = self.engine
        self._total.zero_()
        rewards, _ = e.step(self._first)
        self._total += rewards
        for _ in range(31):
            feats, masks = e.get_state()
            logits = self.net(feats).masked_fill(masks == 0, float("-inf"))
            rewards, _ = e.step(logits.argmax(dim=-1))
            self._total += rewards

    def _capture(self) -> None:
        """Capture the rollout as a CUDA graph (10-50x fewer launches).

        Warm-up runs on a side stream from the current (valid) engine state;
        the captured graph then reads whatever state _load_states copied into
        the engine tensors before each replay.
        """
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._rollout_body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self._graph):
            self._rollout_body()
        torch.cuda.synchronize()

    def _rollout(self) -> torch.Tensor:
        if self.use_graph:
            if self._graph is None:
                try:
                    self._capture()
                except Exception:
                    self.use_graph = False
            if self._graph is not None:
                self._graph.replay()
                return self._total
        with torch.no_grad():
            self._rollout_body()
        return self._total

    # ── state transfer ─────────────────────────────────────────────────────
    def _load_states(self, player, deals: list[dict[int, list[Card]]], trick, trump: str,
                     candidates: list[Card]) -> tuple[torch.Tensor, int]:
        """Fill the engine with len(candidates) x len(deals) copies of the state.

        Returns (first_actions [B] long, n_used).  Slot (c, k) = c*len(deals)+k.
        """
        e = self.engine
        B = self.B
        n = len(candidates) * len(deals)
        dev = self.device
        hands = torch.zeros(B, 4, 32, device=dev)
        played = torch.zeros(B, 32, device=dev)
        voids = torch.zeros(B, 4, 4, device=dev)
        trick_cards = torch.full((B, 4), -1, dtype=torch.long, device=dev)
        trick_seats = torch.full((B, 4), -1, dtype=torch.long, device=dev)
        first = torch.zeros(B, dtype=torch.long, device=dev)

        played_idx = [CARD_INDEX[cs] for cs in player.played_cards if cs in CARD_INDEX]
        trick_idx = [(p.seat_idx, CARD_INDEX[str(c)]) for p, c in trick]
        void_pairs = [(s, SUITS.index(su)) for s, suits in player.opponent_voids.items() for su in suits]
        leader = trick[0][0].seat_idx if trick else player.seat_idx

        deal_hands = []
        for hd in deals:
            h = torch.zeros(4, 32, device=dev)
            for seat in range(4):
                idx = [CARD_INDEX[str(c)] for c in hd[seat]]
                if idx:
                    h[seat, idx] = 1.0
            deal_hands.append(h)
        deal_stack = torch.stack(deal_hands)  # [K,4,32]

        for ci, card in enumerate(candidates):
            a = CARD_INDEX[str(card)]
            base = ci * len(deals)
            hands[base:base + len(deals)] = deal_stack
            first[base:base + len(deals)] = a
        if played_idx:
            played[:n, played_idx] = 1.0
        for i, (seat, ci_) in enumerate(trick_idx):
            trick_cards[:n, i] = ci_
            trick_seats[:n, i] = seat
        for seat, suit in void_pairs:
            voids[:n, seat, suit] = 1.0
        # Unused slots: copy slot 0 so every game is a valid state.
        if n < B:
            hands[n:] = hands[0]
            played[n:] = played[0]
            voids[n:] = voids[0]
            trick_cards[n:] = trick_cards[0]
            trick_seats[n:] = trick_seats[0]
            first[n:] = first[0]

        e.hands.copy_(hands)
        e.played.copy_(played)
        e.opponent_voids.copy_(voids)
        e.trick_cards.copy_(trick_cards)
        e.trick_seats.copy_(trick_seats)
        e.cards_in_trick.fill_(len(trick))
        e.trick_leader.fill_(leader)
        e.trick_num.fill_(player.trick_num)
        e.current_seat.fill_(player.seat_idx)
        e.trump.fill_(SUITS.index(trump))
        e.declaring_team.fill_(player.declaring_team if player.declaring_team in (0, 1) else 0)
        e.trick_pts.copy_(torch.tensor(player.trick_pts, dtype=torch.int32, device=dev).expand(B, 2))
        e.roem_pts.copy_(torch.tensor(player.roem_pts, dtype=torch.int32, device=dev).expand(B, 2))
        e.game_scores.copy_(torch.tensor(player.game_scores, dtype=torch.int32, device=dev).expand(B, 2))
        e.round_num.fill_(max(0, player.round_num))
        e.done.zero_()
        e._last_round_pts.zero_()
        e._last_trick_delta.zero_()
        e._round_dense_sum.zero_()
        return first, n

    # ── evaluation ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, player, legal: list[Card], trick, trump: str) -> dict[str, float]:
        """Mean round outcome (own-team points minus theirs) per legal card."""
        with self._lock:
            return self._evaluate(player, legal, trick, trump)

    def _evaluate(self, player, legal: list[Card], trick, trump: str) -> dict[str, float]:
        candidates = list(legal)[: self.max_candidates]
        deals = []
        for _ in range(self.deals * 2):
            if len(deals) >= self.deals:
                break
            hd = player._sample_hands(trick)
            if hd:
                deals.append(hd)
        if not deals:
            return {}
        first, n = self._load_states(player, deals, trick, trump, candidates)
        self._first.copy_(first)
        total = self._rollout()
        sign = 1.0 if SEAT_TEAMS[player.seat_idx] == 0 else -1.0
        vals = (total[:n] * sign * 162.0).view(len(candidates), len(deals)).mean(dim=1)
        return {str(c): float(v) for c, v in zip(candidates, vals.tolist())}
