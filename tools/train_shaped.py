"""Self-play PPO training with configurable shaped rewards.

Based on train_selfplay.py but adds per-trick reward shaping from reward_rules.py.
Edit reward_rules.py to add/modify/weight reward signals.
"""

import argparse
import builtins
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

print = partial(builtins.print, flush=True)


def _make_logger(log_path: str):
    """Return a timestamped print function that writes to stdout and a log file."""
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")

    def log(*args):
        ts = time.strftime("%H:%M:%S")
        msg = " ".join(str(a) for a in args)
        line = f"[{ts}] {msg}"
        builtins.print(line, flush=True)
        builtins.print(line, file=log_file, flush=True)

    return log, log_file

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from klaverjas.constants import SEAT_DEFAULTS, SEAT_TEAMS
from klaverjas.core import Card, Trick, trick_winner_index
from main import AIPlayer, KlaverjasGame
from neural.features import CARD_INDEX, NUM_FEATURES, encode_state
from neural.model import KlaverjasActorCritic, KlaverjasNet
from tools.reward_rules import RULES

IDX_TO_CARD_STR: dict[int, str] = {v: k for k, v in CARD_INDEX.items()}
PY = sys.executable


@dataclass
class Transition:
    state: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float = 0.0
    legal_mask: np.ndarray = field(default_factory=lambda: np.zeros(32, dtype=np.float32))


class PPOPlayer(AIPlayer):
    def __init__(self, name, team, seat_idx, model, device, temperature=1.0,
                 rng_seed=None, signal_profile="core", ai_strength="expert"):
        super().__init__(name, team, seat_idx, rng_seed, signal_profile, ai_strength)
        self.model = model
        self.device = device
        self.temperature = temperature
        self.transitions: list[Transition] = []
        # Track which card was played per transition for shaped rewards
        self.transition_cards: list[Card] = []

    def clear_transitions(self):
        self.transitions.clear()
        self.transition_cards.clear()

    def choose_card(self, trick: Trick, trump: str) -> Card:
        legal = self.legal_moves(trick, trump)

        if self.use_endgame_solver and len(self.hand) <= 3:
            self.current_trump = trump
            solved = self._endgame_exact_choice(legal, trick, trump)
            if solved is not None:
                self.hand.remove(solved)
                return solved

        features = encode_state(
            hand=list(self.hand),
            trick=trick,
            trump=trump,
            played_cards=set(self.played_cards),
            seat_idx=self.seat_idx,
            trick_num=self.trick_num,
            trick_pts=list(self.trick_pts),
            roem_pts=list(self.roem_pts),
            declaring_team=self.declaring_team,
            opponent_voids={k: set(v) for k, v in self.opponent_voids.items()},
            game_scores=list(self.game_scores),
            round_num=self.round_num,
            legal_moves=legal,
        )

        x = torch.from_numpy(features).unsqueeze(0).to(self.device)

        legal_mask_np = np.zeros(32, dtype=np.float32)
        for c in legal:
            legal_mask_np[CARD_INDEX[str(c)]] = 1.0
        legal_mask = torch.from_numpy(legal_mask_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.model(x)
            logits = logits.masked_fill(legal_mask == 0, float("-inf"))
            logits = logits / self.temperature
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        self.transitions.append(Transition(
            state=features,
            action=action.item(),
            log_prob=log_prob.item(),
            value=value.item(),
            legal_mask=legal_mask_np,
        ))

        chosen_idx = action.item()
        target_str = IDX_TO_CARD_STR[chosen_idx]
        chosen_card = None
        for c in legal:
            if str(c) == target_str:
                chosen_card = c
                break
        if chosen_card is None:
            chosen_card = legal[0]

        self.transition_cards.append(chosen_card)
        self.hand.remove(chosen_card)
        return chosen_card


def apply_shaped_rewards(
    trick: Trick,
    trick_num: int,
    trump: str,
    winner_idx: int,
    winning_team: int,
    trick_points: int,
    players: list[PPOPlayer],
    play_order: list[int],
    cards_by_seat: dict[int, list[str]] | None = None,
) -> dict[int, float]:
    """Apply all reward rules to each player's card in this trick.
    Returns {seat_idx: shaped_reward}."""
    rewards: dict[int, float] = {}
    for i, (player, card) in enumerate(trick):
        seat = play_order[i]
        partner_seat = (seat + 2) % 4
        ctx = {
            "card": card,
            "player_idx": seat,
            "team": SEAT_TEAMS[seat],
            "trick": trick,
            "trick_num": trick_num,
            "trump": trump,
            "winner_idx": winner_idx,
            "winning_team": winning_team,
            "trick_points": trick_points,
            "played_cards": set(player.played_cards),
            "cards_by_seat": {s: list(cs) for s, cs in cards_by_seat.items()} if cards_by_seat else {},
            "partner_seat": partner_seat,
            "voids": {s: set(v) for s, v in player.opponent_voids.items()},
            "declaring_team": player.declaring_team,
            "team_trick_pts": list(player.trick_pts),
        }
        shaped = sum(rule(ctx) * weight for rule, weight in RULES)
        rewards[seat] = shaped
    return rewards


def _play_ppo_game(model, device, game_seed, temperature, shaping_weight):
    """Play one self-play game, return (transitions, avg_round_reward)."""
    round_rewards_t0: list[float] = []
    round_rewards_t1: list[float] = []
    round_boundaries: list[list[int]] = [[] for _ in range(4)]
    shaped_rewards: dict[int, list[float]] = {s: [] for s in range(4)}
    trick_boundaries: dict[int, list[int]] = {s: [] for s in range(4)}
    game_ref: dict = {"game": None}
    cards_by_seat: dict[int, list[str]] = {s: [] for s in range(4)}

    def on_event(event: str, data: dict,
                 _rr0=round_rewards_t0, _rr1=round_rewards_t1,
                 _rb=round_boundaries, _gr=game_ref,
                 _sr=shaped_rewards, _tb=trick_boundaries,
                 _cbs=cards_by_seat) -> None:
        if event == "trick_done":
            trick = data["trick"]
            trump = data["trump"]
            trick_num = data["trick_num"]
            winner_idx = data["winner_idx"]
            winning_team = data["winning_team"]
            trick_points = data["trick_points"]
            play_order = data["play_order"]

            for seat_idx, card in data["trick_cards_by_seat"]:
                _cbs[seat_idx].append(card)

            rewards = apply_shaped_rewards(
                trick, trick_num, trump, winner_idx, winning_team,
                trick_points, _gr["game"].players, play_order,
                cards_by_seat=_cbs,
            )
            for seat, r in rewards.items():
                _sr[seat].append(r)
                p = _gr["game"].players[seat]
                if isinstance(p, PPOPlayer):
                    _tb[seat].append(len(p.transitions))

        elif event == "round_done":
            t0, t1 = data["t0"], data["t1"]
            _rr0.append((t0 - t1) / 162.0)
            _rr1.append((t1 - t0) / 162.0)
            for s in range(4):
                _cbs[s].clear()
            for seat in range(4):
                p = _gr["game"].players[seat]
                if isinstance(p, PPOPlayer):
                    _rb[seat].append(len(p.transitions))
        elif event == "waiting_for_host" and _gr["game"] is not None:
            _gr["game"].signal_next_round()

    game = KlaverjasGame(
        human_seats={},
        log_fn=lambda *_args, **_kwargs: None,
        state_fn=on_event,
        game_mode="boom",
        game_seed=game_seed,
        ai_seed_base=game_seed * 31 + 7,
        ai_strength="expert",
    )
    game_ref["game"] = game
    game.boom_rounds = 16

    original_play_trick = game._play_trick

    def patched_play_trick(leader: int, trump: str,
                           _orig=original_play_trick, _game=game) -> tuple:
        play_order = [(leader + offset) % 4 for offset in range(4)]
        result = _orig(leader, trump)
        winner_global, total_pts, trick_cards, cards_played, trick_cards_by_seat = result

        trick_num = 0
        for p in _game.players:
            if isinstance(p, PPOPlayer):
                trick_num = p.trick_num
                break

        trick: Trick = []
        for seat_idx, card_str in trick_cards_by_seat:
            for c in cards_played:
                if str(c) == card_str:
                    trick.append((_game.players[seat_idx], c))
                    break

        winning_team = _game.players[winner_global].team

        _game.notify("trick_done", {
            "trick": trick,
            "trump": trump,
            "trick_num": trick_num,
            "winner_idx": winner_global,
            "winning_team": winning_team,
            "trick_points": total_pts,
            "play_order": play_order,
            "trick_cards_by_seat": trick_cards_by_seat,
        })

        return result

    game._play_trick = patched_play_trick

    players = []
    for seat in range(4):
        team = SEAT_TEAMS[seat]
        p = PPOPlayer(
            name=f"SP {SEAT_DEFAULTS[seat]}",
            team=team,
            seat_idx=seat,
            model=model,
            device=device,
            temperature=temperature,
            rng_seed=game_seed * 31 + 7 + seat,
            signal_profile="core",
            ai_strength="expert",
        )
        players.append(p)
    game.players = players
    game.play()

    # Assign rewards: base (round point diff) + shaped (per-trick)
    all_transitions: list[Transition] = []
    for seat in range(4):
        p = players[seat]
        team = SEAT_TEAMS[seat]
        base_rewards = round_rewards_t0 if team == 0 else round_rewards_t1
        boundaries = [0] + round_boundaries[seat]

        for rnd_idx, reward in enumerate(base_rewards):
            start = boundaries[rnd_idx] if rnd_idx < len(boundaries) else 0
            end = boundaries[rnd_idx + 1] if rnd_idx + 1 < len(boundaries) else len(p.transitions)
            for t_idx in range(start, end):
                if t_idx < len(p.transitions):
                    p.transitions[t_idx].reward = reward

        t_bounds = trick_boundaries[seat]
        for trick_idx in range(len(t_bounds)):
            t_start = t_bounds[trick_idx - 1] if trick_idx > 0 else 0
            t_end = t_bounds[trick_idx]
            if t_end <= t_start:
                # No transition recorded for this trick (endgame solver bypassed choose_card)
                continue
            if trick_idx < len(shaped_rewards[seat]):
                shaped_r = shaped_rewards[seat][trick_idx] * shaping_weight
                if t_end > 0 and (t_end - 1) < len(p.transitions):
                    p.transitions[t_end - 1].reward += shaped_r

        all_transitions.extend(p.transitions)
        p.clear_transitions()

    avg_reward = float(np.mean(round_rewards_t0)) if round_rewards_t0 else 0.0
    return all_transitions, avg_reward


def _collect_game_batch_worker(args):
    """Multiprocessing worker: reconstruct model, play a batch of games."""
    state_dict, hidden_sizes, game_seeds, temperature, shaping_weight = args

    main.AI_BID_DELAY = 0.0
    main.AI_PLAY_DELAY = 0.0
    main.TRICK_CLEAR_DELAY = 0.0

    device = torch.device("cpu")
    policy_net = KlaverjasNet(hidden_sizes=hidden_sizes)
    policy_net.load_state_dict(state_dict)
    model = KlaverjasActorCritic.from_policy_net(policy_net)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    batch_transitions = []
    batch_rewards = []
    for seed in game_seeds:
        transitions, avg_reward = _play_ppo_game(model, device, seed, temperature, shaping_weight)
        for t in transitions:
            batch_transitions.append((t.state, t.action, t.log_prob, t.value, t.reward, t.legal_mask))
        batch_rewards.append(avg_reward)
    return batch_transitions, batch_rewards


def collect_games(
    inference_model: KlaverjasActorCritic,
    inference_device: torch.device,
    num_games: int,
    rng: random.Random,
    temperature: float,
    shaping_weight: float = 0.5,
    workers: int = 1,
) -> tuple[list[Transition], list[float]]:
    """Play self-play games with per-trick shaped rewards."""
    main.AI_BID_DELAY = 0.0
    main.AI_PLAY_DELAY = 0.0
    main.TRICK_CLEAR_DELAY = 0.0

    game_seeds = [rng.randint(1, 10_000_000) for _ in range(num_games)]

    if workers <= 1:
        # Sequential: use provided model directly
        all_transitions: list[Transition] = []
        game_rewards: list[float] = []
        for seed in game_seeds:
            transitions, avg_reward = _play_ppo_game(
                inference_model, inference_device, seed, temperature, shaping_weight
            )
            all_transitions.extend(transitions)
            game_rewards.append(avg_reward)
        return all_transitions, game_rewards

    # Parallel: export model weights, distribute to worker processes
    export_net = inference_model.export_policy_net()
    state_dict = {k: v.cpu() for k, v in export_net.state_dict().items()}
    src_layers = [m for m in export_net.net if isinstance(m, nn.Linear)]
    hidden_sizes = tuple(l.out_features for l in src_layers[:-1])

    # Split seeds into chunks (one per worker)
    chunk_size = max(1, (num_games + workers - 1) // workers)
    tasks = []
    for i in range(0, num_games, chunk_size):
        chunk_seeds = game_seeds[i:i + chunk_size]
        tasks.append((state_dict, hidden_sizes, chunk_seeds, temperature, shaping_weight))

    all_transitions: list[Transition] = []
    game_rewards: list[float] = []
    with Pool(processes=min(workers, len(tasks))) as pool:
        for batch_trans, batch_rew in pool.imap_unordered(_collect_game_batch_worker, tasks):
            for state, action, log_prob, value, reward, legal_mask in batch_trans:
                all_transitions.append(
                    Transition(state, action, log_prob, value, reward, legal_mask)
                )
            game_rewards.extend(batch_rew)

    return all_transitions, game_rewards


def compute_gae(
    transitions: list[Transition],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(transitions)
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)

    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = 0.0
        else:
            next_value = transitions[t + 1].value
        delta = transitions[t].reward + gamma * next_value - transitions[t].value
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + transitions[t].value

    return advantages, returns


def ppo_update(
    model: KlaverjasActorCritic,
    ref_model: KlaverjasActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    device: torch.device,
    clip_eps: float = 0.2,
    entropy_coeff: float = 0.02,
    value_coeff: float = 0.5,
    kl_coeff: float = 0.1,
    mini_batch_size: int = 256,
    ppo_epochs: int = 4,
) -> dict:
    advantages, returns = compute_gae(transitions)

    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    states = torch.from_numpy(np.stack([t.state for t in transitions])).to(device)
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=device)
    old_log_probs = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32, device=device)
    legal_masks = torch.from_numpy(np.stack([t.legal_mask for t in transitions])).to(device)
    adv_tensor = torch.from_numpy(advantages).to(device)
    ret_tensor = torch.from_numpy(returns).to(device)

    n = len(transitions)
    # Accumulate losses on GPU — sync to CPU only once at the end
    total_policy_loss = torch.tensor(0.0, device=device)
    total_value_loss = torch.tensor(0.0, device=device)
    total_entropy = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    num_updates = 0

    for _ in range(ppo_epochs):
        indices = torch.randperm(n, device=device)

        for start in range(0, n, mini_batch_size):
            end = min(start + mini_batch_size, n)
            mb_idx = indices[start:end]

            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_legal_masks = legal_masks[mb_idx]
            mb_advantages = adv_tensor[mb_idx]
            mb_returns = ret_tensor[mb_idx]

            logits, values = model(mb_states)
            logits = logits.masked_fill(mb_legal_masks == 0, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            with torch.no_grad():
                ref_logits, _ = ref_model(mb_states)
                ref_logits = ref_logits.masked_fill(mb_legal_masks == 0, float("-inf"))
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            cur_log_probs = F.log_softmax(logits, dim=-1)
            log_ratio = cur_log_probs - ref_log_probs
            log_ratio = log_ratio.nan_to_num(0.0)
            kl_div = (probs * log_ratio).sum(dim=-1).mean()

            ratio = (new_log_probs - mb_old_log_probs).exp()
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, mb_returns)

            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy + kl_coeff * kl_div

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_policy_loss += policy_loss.detach()
            total_value_loss += value_loss.detach()
            total_entropy += entropy.detach()
            total_kl += kl_div.detach()
            num_updates += 1

    n_upd = max(num_updates, 1)
    return {
        "policy_loss": (total_policy_loss / n_upd).item(),
        "value_loss": (total_value_loss / n_upd).item(),
        "entropy": (total_entropy / n_upd).item(),
        "kl_div": (total_kl / n_upd).item(),
    }


def benchmark_vs_expert_v2(model_path: str, rounds: int = 256) -> dict | None:
    result = subprocess.run(
        [PY, "tools/ai_benchmark.py",
         "--candidate-strength", "neural",
         "--baseline-strength", "expert_v2",
         "--rounds", str(rounds),
         "--workers", "14",
         "--model", model_path],
        cwd=str(ROOT),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    stats = {}
    for line in result.stdout.strip().split("\n"):
        if "=" in line:
            for part in line.split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        stats[k.strip()] = float(v.strip())
                    except ValueError:
                        stats[k.strip()] = v.strip()
    return stats


def train(
    model_path: str,
    output_path: str,
    num_epochs: int = 300,
    games_per_epoch: int = 256,
    lr: float = 1e-4,
    temperature: float = 1.15,
    shaping_weight: float = 0.25,
    clip_eps: float = 0.15,
    entropy_coeff: float = 0.02,
    kl_coeff: float = 0.1,
    value_coeff: float = 0.5,
    ppo_epochs: int = 4,
    benchmark_interval: int = 25,
    seed: int = 42,
    workers: int = 0,
) -> None:
    log_path = output_path.replace(".pt", ".log")
    print, _log_file = _make_logger(log_path)
    print(f"Logging to {log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Reward rules: {len(RULES)} active")
    for rule_fn, weight in RULES:
        print(f"  - {rule_fn.__name__} (weight={weight})")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    linear_keys = sorted(
        (k for k in state_dict if k.endswith(".weight") and "net." in k),
        key=lambda k: int(k.split(".")[1]),
    )
    hidden_sizes = tuple(state_dict[k].shape[0] for k in linear_keys[:-1])
    policy_net = KlaverjasNet(hidden_sizes=hidden_sizes)
    policy_net.load_state_dict(state_dict)
    print(f"Loaded pretrained policy from {model_path} (architecture: {hidden_sizes})")

    model = KlaverjasActorCritic.from_policy_net(policy_net)
    model.to(device)

    ref_model = KlaverjasActorCritic.from_policy_net(policy_net)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # CPU model for game inference — batch size 1 is faster on CPU than GPU
    inference_device = torch.device("cpu")
    cpu_model = KlaverjasActorCritic.from_policy_net(policy_net)
    cpu_model.cpu()
    cpu_model.eval()
    for p in cpu_model.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Actor-Critic parameters: {total_params:,}")
    if workers <= 0:
        workers = max(1, cpu_count() - 1)
    print(f"Settings: lr={lr}, temp={temperature}, shaping_weight={shaping_weight}, "
          f"clip={clip_eps}, entropy={entropy_coeff}, kl={kl_coeff}, games/epoch={games_per_epoch}, "
          f"workers={workers}")

    print("\n--- Initial benchmark vs expert_v2 ---")
    stats = benchmark_vs_expert_v2(model_path, rounds=1024)
    initial_diff = float(stats.get("avg_point_diff", -9999)) if stats else -9999
    if stats:
        print(f"  win_rate={stats.get('win_rate', '?')}, avg_point_diff={initial_diff}")
    print("\n--- Finished initial benchmark vs expert_v2 ---")
    best_pt = str(ROOT / "models" / "neural_best.pt")
    if str(Path(model_path).resolve()) != str(Path(output_path).resolve()):
        shutil.copy(model_path, output_path)
    if str(Path(model_path).resolve()) != str(Path(best_pt).resolve()):
        shutil.copy(model_path, best_pt)
    print(f"  => Initial model saved as baseline (avg_point_diff={initial_diff:+.2f})")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)

    reward_history = []
    best_benchmark_diff = initial_diff
    best_reward_in_window = -float("inf")
    best_reward_epoch = 0
    window_path = output_path.replace(".pt", "_window_best.pt")
    current_path = output_path.replace(".pt", "_current.pt")

    for epoch in range(1, num_epochs + 1):
        model.eval()
        transitions, game_rewards = collect_games(
            inference_model=cpu_model,
            inference_device=inference_device,
            num_games=games_per_epoch,
            rng=rng,
            temperature=temperature,
            shaping_weight=shaping_weight,
            workers=workers,
        )
        model.train()

        avg_reward = np.mean(game_rewards)
        reward_history.append(avg_reward)
        recent_avg = np.mean(reward_history[-20:])

        if not transitions:
            print(f"Epoch {epoch}: no transitions, skipping")
            continue

        stats = ppo_update(
            model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            transitions=transitions,
            device=device,
            clip_eps=clip_eps,
            entropy_coeff=entropy_coeff,
            value_coeff=value_coeff,
            kl_coeff=kl_coeff,
            ppo_epochs=ppo_epochs,
        )
        # Sync updated GPU weights to CPU inference model
        cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})

        print(
            f"Epoch {epoch:3d}/{num_epochs}  "
            f"reward={avg_reward:+.4f}  "
            f"recent_20={recent_avg:+.4f}  "
            f"p_loss={stats['policy_loss']:.4f}  "
            f"v_loss={stats['value_loss']:.4f}  "
            f"entropy={stats['entropy']:.4f}  "
            f"kl={stats['kl_div']:.4f}  "
            f"trans={len(transitions)}"
        )

        if recent_avg > best_reward_in_window:
            best_reward_in_window = recent_avg
            best_reward_epoch = epoch
            export_net = model.export_policy_net()
            torch.save(export_net.state_dict(), window_path)
            print(f"  -> Window best (epoch {epoch}, recent_20={recent_avg:+.4f})")

        if epoch % benchmark_interval == 0:
            export_net = model.export_policy_net()
            torch.save(export_net.state_dict(), current_path)

            shutil.copy(current_path, str(ROOT / "models" / "neural_best.pt"))
            print(f"\n--- Benchmark vs expert_v2 (epoch {epoch}) ---")
            bench_current = benchmark_vs_expert_v2(current_path, rounds=1024)
            current_diff = float(bench_current.get("avg_point_diff", -9999)) if bench_current else -9999
            if bench_current:
                print(f"  win_rate={bench_current.get('win_rate', '?')}, "
                      f"avg_point_diff={current_diff}")

            best_diff = current_diff
            chosen = "current"
            chosen_path = current_path
            if best_reward_epoch != epoch and Path(window_path).exists():
                shutil.copy(window_path, str(ROOT / "models" / "neural_best.pt"))
                print(f"--- Benchmark window best (epoch {best_reward_epoch}) ---")
                bench_window = benchmark_vs_expert_v2(window_path, rounds=1024)
                window_diff = float(bench_window.get("avg_point_diff", -9999)) if bench_window else -9999
                if bench_window:
                    print(f"  win_rate={bench_window.get('win_rate', '?')}, "
                          f"avg_point_diff={window_diff}")
                if window_diff > current_diff:
                    best_diff = window_diff
                    chosen = f"window (epoch {best_reward_epoch})"
                    chosen_path = window_path

            if best_diff > best_benchmark_diff:
                best_benchmark_diff = best_diff
                shutil.copy(chosen_path, output_path)
                shutil.copy(chosen_path, str(ROOT / "models" / "neural_best.pt"))
                print(f"  => NEW BEST model saved from {chosen} "
                      f"(avg_point_diff={best_diff:+.2f})")
            else:
                shutil.copy(output_path, str(ROOT / "models" / "neural_best.pt"))
                print(f"  => Kept previous best (avg_point_diff={best_benchmark_diff:+.2f})")

            best_reward_in_window = -float("inf")
            best_reward_epoch = epoch
            print()

        if epoch % 75 == 0 and temperature > 1.0:
            temperature = max(1.0, temperature - 0.05)
            print(f"  -> Temperature: {temperature:.2f}")

    export_net = model.export_policy_net()
    final_path = output_path.replace(".pt", "_final.pt")
    torch.save(export_net.state_dict(), final_path)

    shutil.copy(final_path, str(ROOT / "models" / "neural_best.pt"))
    print(f"\n--- Final benchmark vs expert_v2 ---")
    bench = benchmark_vs_expert_v2(final_path, rounds=512)
    final_diff = float(bench.get("avg_point_diff", -9999)) if bench else -9999
    if bench:
        print(f"  win_rate={bench.get('win_rate', '?')}, avg_point_diff={final_diff}")

    if final_diff > best_benchmark_diff:
        shutil.copy(final_path, output_path)
        print(f"  => Final model is new best!")
    else:
        shutil.copy(output_path, str(ROOT / "models" / "neural_best.pt"))
        print(f"  => Keeping previous best (avg_point_diff={best_benchmark_diff:+.2f})")

    print(f"\nDone. Best model: {output_path} (benchmark diff={best_benchmark_diff:+.2f})")
    _log_file.close()


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Shaped-reward PPO training for neural Klaverjassen AI.")
    parser.add_argument("--model", default=str(ROOT / "models" / "neural_best.pt"), help="Pretrained policy model.")
    parser.add_argument("--output", default=str(ROOT / "models" / "neural_shaped_v1.pt"), help="Output model path.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--games-per-epoch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.15)
    parser.add_argument("--shaping-weight", type=float, default=0.5,
                        help="Weight of shaped rewards relative to base game rewards.")
    parser.add_argument("--clip-eps", type=float, default=0.15)
    parser.add_argument("--entropy-coeff", type=float, default=0.02)
    parser.add_argument("--kl-coeff", type=float, default=0.1)
    parser.add_argument("--benchmark-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="Parallel game collection workers (default: cpu_count-1).")
    args = parser.parse_args()

    train(
        model_path=args.model,
        output_path=args.output,
        num_epochs=args.epochs,
        games_per_epoch=args.games_per_epoch,
        lr=args.lr,
        temperature=args.temperature,
        shaping_weight=args.shaping_weight,
        clip_eps=args.clip_eps,
        entropy_coeff=args.entropy_coeff,
        kl_coeff=args.kl_coeff,
        benchmark_interval=args.benchmark_interval,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main_cli()
