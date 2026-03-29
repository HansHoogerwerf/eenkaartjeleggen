"""RL fine-tuning for neural bidding via REINFORCE with KL penalty.

Trains the bid model by playing full games (expert_v2 card play + RL bidding)
against expert_v2, using actual game outcomes as reward signal.
Only the bidding decisions differ -- card play is identical on both sides.
"""

import copy
import random
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

print = partial(print, flush=True)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import main as main_mod
from klaverjas.constants import SEAT_DEFAULTS, SEAT_TEAMS
from main import AIPlayer, KlaverjasGame
from neural.bid_features import encode_bid_state
from neural.model import KlaverjassBidNet

# ── Configuration ──────────────────────────────────────────────────────

@dataclass
class RLConfig:
    # Training
    epochs: int = 200
    games_per_epoch: int = 16
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    # Exploration -- temperature on the sigmoid logit
    temp_start: float = 1.5
    temp_end: float = 1.0

    # KL penalty against pretrained model
    kl_coeff: float = 0.5

    # Reward shaping
    declare_win_scale: float = 1.0     # reward = +pts / 162
    declare_nat_penalty: float = 1.5   # reward = -penalty / 162
    pass_reward_scale: float = 0.2     # reward = scale * round_outcome

    # Benchmarking
    benchmark_every: int = 25
    benchmark_rounds: int = 512

    # Model paths
    pretrained_path: str = str(ROOT / "models" / "bid_neural_v1.pt")
    output_path: str = str(ROOT / "models" / "bid_rl_v1.pt")

    # Game
    boom_rounds: int = 16
    seed: int = 42


# ── Bid decision record ───────────────────────────────────────────────

@dataclass
class BidRecord:
    features: np.ndarray
    logit: float
    action: int          # 1 = declare, 0 = pass
    log_prob: float
    team: int
    seat_idx: int


@dataclass
class RoundResult:
    declaring_team: int
    nat: bool
    round_pts: list[int]  # [team0, team1]


# ── RL Player ─────────────────────────────────────────────────────────

# Module-level storage for bid records (per-game, cleared each game)
_bid_records: list[BidRecord] = []
_round_results: list[RoundResult] = []

# Module-level references to the model and temperature (set before each game)
_rl_model: KlaverjassBidNet | None = None
_rl_temperature: float = 1.5


class HeuristicBidNeuralPlayer(AIPlayer):
    """Neural card play with expert heuristic bidding (no neural bidding)."""

    def choose_trump(self, suit: str, forced: bool) -> bool:
        if forced:
            return True
        return super().choose_trump(suit, forced)


class BidRLPlayer(AIPlayer):
    """AIPlayer with expert_v2 card play but RL-sampled bidding decisions.

    Uses expert_v2 for all card play so the only variable is bidding.
    When explore=True, samples from the policy; when False, uses greedy threshold.
    """

    explore: bool = True

    def choose_trump(self, suit: str, forced: bool) -> bool:
        if forced:
            return True

        model = _rl_model
        if model is None:
            return super().choose_trump(suit, forced)

        features = encode_bid_state(
            hand=list(self.hand),
            trump_suit=suit,
            bid_position=self.bid_position,
            bid_round=self.bid_round,
            seat_idx=self.seat_idx,
            team=self.team,
            game_scores=list(self.game_scores),
            round_num=self.round_num,
            round_1_suit=getattr(self, "round_1_suit", None),
        )

        x = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            logit = model(x).squeeze(-1).item()

        if self.explore:
            # Apply temperature and sample
            temp = _rl_temperature
            prob = torch.sigmoid(torch.tensor(logit / temp)).item()
            prob = max(0.01, min(0.99, prob))

            declare = random.random() < prob
            action = 1 if declare else 0

            log_prob = np.log(prob) if declare else np.log(1.0 - prob)

            _bid_records.append(BidRecord(
                features=features,
                logit=logit,
                action=action,
                log_prob=log_prob,
                team=self.team,
                seat_idx=self.seat_idx,
            ))
        else:
            # Greedy (for benchmarking)
            prob = torch.sigmoid(torch.tensor(logit)).item()
            declare = prob >= 0.5

        return declare


# ── Game runner ───────────────────────────────────────────────────────

def play_rl_game(
    game_seed: int,
    candidate_team: int,
) -> tuple[list[BidRecord], list[RoundResult], int, int]:
    """Play one game: neural card play on both sides.

    Candidate team: neural card play + RL bidding.
    Opponent team: neural card play + expert heuristic bidding.

    Returns (bid_records, round_results, points_for, points_against).
    """
    _bid_records.clear()
    _round_results.clear()

    game_ref: dict = {"game": None}

    def on_event(event: str, data: dict) -> None:
        if event == "round_done":
            history = data["history"]
            _round_results.append(RoundResult(
                declaring_team=history["declaring_team"],
                nat=history["nat"],
                round_pts=list(history["round_pts"]),
            ))
        elif event == "waiting_for_host" and game_ref["game"] is not None:
            game_ref["game"].signal_next_round()

    game = KlaverjasGame(
        human_seats={},
        log_fn=lambda *_args, **_kwargs: None,
        state_fn=on_event,
        game_mode="boom",
        game_seed=game_seed,
        ai_seed_base=game_seed * 31 + 7,
    )
    game_ref["game"] = game
    game.boom_rounds = 16

    players: list[AIPlayer] = []
    for seat in range(4):
        team = SEAT_TEAMS[seat]
        if team == candidate_team:
            p = BidRLPlayer(
                name=f"RL {SEAT_DEFAULTS[seat]}",
                team=team,
                seat_idx=seat,
                rng_seed=game_seed * 31 + 7 + seat,
                signal_profile="core",
                ai_strength="neural",
            )
        else:
            p = HeuristicBidNeuralPlayer(
                name=f"Exp {SEAT_DEFAULTS[seat]}",
                team=team,
                seat_idx=seat,
                rng_seed=game_seed * 31 + 7 + seat,
                signal_profile="core",
                ai_strength="neural",
            )
        players.append(p)

    game.players = players
    game.play()

    points_for = game.scores[candidate_team]
    points_against = game.scores[1 - candidate_team]

    return list(_bid_records), list(_round_results), points_for, points_against


# ── Reward computation ────────────────────────────────────────────────

def compute_rewards(
    bid_records: list[BidRecord],
    round_results: list[RoundResult],
    cfg: RLConfig,
) -> list[float]:
    """Assign a reward to each bid decision based on round outcomes.

    Bid decisions are matched to rounds by order (each round has up to 8
    bid decisions -- 4 per bidding round x up to 2 rounds).
    """
    rewards: list[float] = []

    # Match bids to rounds: bids occur sequentially, round boundaries
    # are when someone declares or all pass (forced).
    # We process round results in order; for each round, we assign rewards
    # to all bids that happened during that round's bidding phase.
    bid_idx = 0
    for rr in round_results:
        # Count how many bids belong to this round
        # Each round has at most 8 bids (4 per offer x 2 offers) + 1 forced
        # We count bids until we find the declaring one (action=1) or run out
        round_bids_start = bid_idx
        found_declare = False
        while bid_idx < len(bid_records):
            if bid_records[bid_idx].action == 1:
                bid_idx += 1
                found_declare = True
                break
            bid_idx += 1
            # Safety: if we've consumed too many without a declare, break
            if bid_idx - round_bids_start >= 8:
                break

        round_bids = bid_records[round_bids_start:bid_idx]

        for br in round_bids:
            if br.action == 1:
                # This player declared
                if rr.declaring_team == br.team:
                    if rr.nat:
                        # Nat -- declaring team loses everything
                        reward = -cfg.declare_nat_penalty * rr.round_pts[1 - br.team] / 162.0
                    else:
                        # Successful declare
                        pts = rr.round_pts[br.team]
                        reward = cfg.declare_win_scale * pts / 162.0
                else:
                    # Shouldn't happen -- the declarer's team should match
                    reward = 0.0
            else:
                # This player passed -- reward based on round outcome
                team_pts = rr.round_pts[br.team]
                opp_pts = rr.round_pts[1 - br.team]
                outcome = (team_pts - opp_pts) / 162.0
                reward = cfg.pass_reward_scale * outcome

            rewards.append(reward)

    # Handle any remaining unmatched bids (shouldn't happen normally)
    while len(rewards) < len(bid_records):
        rewards.append(0.0)

    return rewards


# ── Training loop ─────────────────────────────────────────────────────

def train(cfg: RLConfig) -> None:
    global _rl_model, _rl_temperature

    # Disable pacing
    main_mod.AI_BID_DELAY = 0.0
    main_mod.AI_PLAY_DELAY = 0.0
    main_mod.TRICK_CLEAR_DELAY = 0.0

    # Load pretrained model
    device = torch.device("cpu")
    model = KlaverjassBidNet()
    model.load_state_dict(torch.load(cfg.pretrained_path, map_location=device, weights_only=True))
    model.to(device)

    # Frozen copy for KL penalty
    pretrained = copy.deepcopy(model)
    pretrained.eval()
    for p in pretrained.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = random.Random(cfg.seed)

    print(f"RL Bid Fine-tuning: {cfg.epochs} epochs, {cfg.games_per_epoch} games/epoch")
    print(f"  LR={cfg.lr}, KL_coeff={cfg.kl_coeff}, temp={cfg.temp_start}->{cfg.temp_end}")
    print(f"  Benchmark every {cfg.benchmark_every} epochs ({cfg.benchmark_rounds} rounds)")
    print(f"  Pretrained: {cfg.pretrained_path}")
    print(f"  Output: {cfg.output_path}")
    print()

    best_winrate = 0.0
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        t_start = time.monotonic()

        # Temperature schedule (linear decay)
        progress = (epoch - 1) / max(cfg.epochs - 1, 1)
        temperature = cfg.temp_start + (cfg.temp_end - cfg.temp_start) * progress
        _rl_temperature = temperature

        # Set model for RL players
        model.eval()
        _rl_model = model

        # Collect experience from games
        all_features = []
        all_actions = []
        all_rewards = []
        all_old_logits = []
        epoch_wins = 0
        epoch_declares = 0
        epoch_nats = 0
        epoch_pts_for = 0
        epoch_pts_against = 0

        for g in range(cfg.games_per_epoch):
            game_seed = rng.randint(1, 10_000_000)
            candidate_team = g % 2  # alternate sides

            bid_records, round_results, pts_for, pts_against = play_rl_game(
                game_seed=game_seed,
                candidate_team=candidate_team,
            )

            rewards = compute_rewards(bid_records, round_results, cfg)

            for br, reward in zip(bid_records, rewards):
                # Only train on our team's decisions
                if br.team == candidate_team:
                    all_features.append(br.features)
                    all_actions.append(br.action)
                    all_rewards.append(reward)
                    all_old_logits.append(br.logit)

            epoch_pts_for += pts_for
            epoch_pts_against += pts_against
            if pts_for >= pts_against:
                epoch_wins += 1
            epoch_declares += sum(1 for br in bid_records if br.action == 1 and br.team == candidate_team)
            epoch_nats += sum(1 for rr in round_results if rr.nat and rr.declaring_team == candidate_team)

        if not all_features:
            print(f"Epoch {epoch:3d}: no bid decisions collected, skipping")
            continue

        # Convert to tensors
        features_t = torch.tensor(np.array(all_features), dtype=torch.float32)
        actions_t = torch.tensor(all_actions, dtype=torch.float32)
        rewards_np = np.array(all_rewards, dtype=np.float32)

        # Baseline: mean reward (REINFORCE with baseline)
        baseline = rewards_np.mean()
        advantages = rewards_np - baseline
        advantages_t = torch.tensor(advantages, dtype=torch.float32)

        # Forward pass through current model
        model.train()
        logits = model(features_t).squeeze(-1)
        probs = torch.sigmoid(logits / temperature)
        probs = probs.clamp(0.01, 0.99)

        # Log probabilities of taken actions
        log_probs = actions_t * torch.log(probs) + (1 - actions_t) * torch.log(1 - probs)

        # REINFORCE loss: -E[advantage * log_prob]
        policy_loss = -(advantages_t * log_probs).mean()

        # KL penalty against pretrained model
        with torch.no_grad():
            pretrained_logits = pretrained(features_t).squeeze(-1)
        pretrained_probs = torch.sigmoid(pretrained_logits)
        pretrained_probs = pretrained_probs.clamp(0.01, 0.99)
        current_probs = torch.sigmoid(logits)
        current_probs = current_probs.clamp(0.01, 0.99)

        # KL(current || pretrained) for Bernoulli
        kl = (current_probs * (torch.log(current_probs) - torch.log(pretrained_probs))
              + (1 - current_probs) * (torch.log(1 - current_probs) - torch.log(1 - pretrained_probs)))
        kl_loss = kl.mean()

        total_loss = policy_loss + cfg.kl_coeff * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        elapsed = time.monotonic() - t_start
        n_decisions = len(all_features)
        avg_reward = rewards_np.mean()
        winrate = epoch_wins / cfg.games_per_epoch

        print(
            f"Epoch {epoch:3d} | "
            f"t={elapsed:5.1f}s | "
            f"decisions={n_decisions:3d} | "
            f"reward={avg_reward:+.3f} | "
            f"policy_loss={policy_loss.item():.4f} | "
            f"kl={kl_loss.item():.4f} | "
            f"temp={temperature:.2f} | "
            f"win={winrate:.2f} | "
            f"decl={epoch_declares} nat={epoch_nats}"
        )

        # Benchmark
        if epoch % cfg.benchmark_every == 0 or epoch == 1:
            model.eval()
            _rl_model = model
            BidRLPlayer.explore = False  # greedy for benchmark

            bm_rng = random.Random(cfg.seed + epoch)
            bm_games = (cfg.benchmark_rounds + 15) // 16  # games needed
            bm_wins = 0
            bm_pts_for = 0
            bm_pts_against = 0
            bm_declares = 0
            bm_successful = 0
            bm_total_games = 0

            for bg in range(bm_games):
                gs = bm_rng.randint(1, 10_000_000)
                for ct in (0, 1):
                    _, rr_list, pf, pa = play_rl_game(gs, ct)
                    bm_total_games += 1
                    bm_pts_for += pf
                    bm_pts_against += pa
                    if pf >= pa:
                        bm_wins += 1
                    for rr in rr_list:
                        if rr.declaring_team == ct:
                            bm_declares += 1
                            if not rr.nat:
                                bm_successful += 1

            BidRLPlayer.explore = True  # restore exploration

            bm_winrate = bm_wins / bm_total_games if bm_total_games else 0.0
            bm_diff = (bm_pts_for - bm_pts_against) / bm_total_games if bm_total_games else 0.0
            bm_decl = bm_successful / bm_declares if bm_declares else 0.0

            print(
                f"  > Benchmark @ epoch {epoch}: "
                f"win_rate={bm_winrate:.4f} | "
                f"point_diff={bm_diff:+.1f} | "
                f"decl_success={bm_decl:.4f} | "
                f"games={bm_total_games}"
            )

            if bm_winrate > best_winrate:
                best_winrate = bm_winrate
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.output_path)
                print(f"  * New best! Saved to {cfg.output_path}")

    # Final save
    final_path = str(Path(cfg.output_path).with_stem(Path(cfg.output_path).stem + "_final"))
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete!")
    print(f"  Best win rate: {best_winrate:.4f} at epoch {best_epoch}")
    print(f"  Best model: {cfg.output_path}")
    print(f"  Final model: {final_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RL fine-tune bid model with REINFORCE + KL penalty")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--games-per-epoch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl-coeff", type=float, default=0.5)
    parser.add_argument("--temp-start", type=float, default=1.5)
    parser.add_argument("--temp-end", type=float, default=1.0)
    parser.add_argument("--benchmark-every", type=int, default=25)
    parser.add_argument("--benchmark-rounds", type=int, default=512)
    parser.add_argument("--nat-penalty", type=float, default=1.5, help="Penalty multiplier for nat (failed declaration)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", default=str(ROOT / "models" / "bid_neural_v1.pt"))
    parser.add_argument("--output", default=str(ROOT / "models" / "bid_rl_v1.pt"))
    args = parser.parse_args()

    cfg = RLConfig(
        epochs=args.epochs,
        games_per_epoch=args.games_per_epoch,
        lr=args.lr,
        kl_coeff=args.kl_coeff,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        declare_nat_penalty=args.nat_penalty,
        benchmark_every=args.benchmark_every,
        benchmark_rounds=args.benchmark_rounds,
        seed=args.seed,
        pretrained_path=args.pretrained,
        output_path=args.output,
    )
    train(cfg)
