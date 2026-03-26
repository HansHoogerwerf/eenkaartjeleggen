"""Generate imitation-learning training data for bidding from expert AI self-play.

Runs expert vs expert games and records every bidding decision as a
(feature_vector, declare_label) pair.  Output is saved as a compressed
NumPy .npz file.
"""

import argparse
import random
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from klaverjas.constants import SEAT_DEFAULTS, SEAT_TEAMS
from main import AIPlayer, KlaverjasGame
from neural.bid_features import NUM_BID_FEATURES, encode_bid_state

# Module-level list that workers append to (each process gets its own copy).
_worker_samples: list[tuple[np.ndarray, int]] = []


class BidRecordingAIPlayer(AIPlayer):
    """AIPlayer subclass that records every bidding decision."""

    def choose_trump(self, suit: str, forced: bool) -> bool:
        # Skip forced bids — no decision to learn
        if forced:
            return True

        # Encode state BEFORE the decision
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

        # Let the parent AI decide normally
        declared = super().choose_trump(suit, forced)
        label = 1 if declared else 0

        _worker_samples.append((features, label))
        return declared


def _collect_game(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Play one game and return all recorded bid samples."""
    game_seed, strength, boom_rounds = args

    main.AI_BID_DELAY = 0.0
    main.AI_PLAY_DELAY = 0.0
    main.TRICK_CLEAR_DELAY = 0.0

    _worker_samples.clear()

    game_ref: dict = {"game": None}

    def on_event(event: str, data: dict) -> None:
        if event == "waiting_for_host" and game_ref["game"] is not None:
            game_ref["game"].signal_next_round()

    game = KlaverjasGame(
        human_seats={},
        log_fn=lambda *_args, **_kwargs: None,
        state_fn=on_event,
        game_mode="boom",
        game_seed=game_seed,
        ai_seed_base=game_seed * 31 + 7,
        ai_strength=strength,
    )
    game_ref["game"] = game
    game.boom_rounds = boom_rounds

    game.players = [
        BidRecordingAIPlayer(
            name=f"Rec {SEAT_DEFAULTS[seat]}",
            team=SEAT_TEAMS[seat],
            seat_idx=seat,
            rng_seed=game_seed * 31 + 7 + seat,
            signal_profile="core",
            ai_strength=strength,
        )
        for seat in range(4)
    ]

    game.play()

    if not _worker_samples:
        return np.empty((0, NUM_BID_FEATURES), dtype=np.float32), np.empty((0,), dtype=np.int64)

    features = np.stack([s[0] for s in _worker_samples])
    labels = np.array([s[1] for s in _worker_samples], dtype=np.int64)
    return features, labels


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Generate bidding training data from AI self-play.")
    parser.add_argument("--rounds", type=int, default=1024, help="Target number of rounds to play.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--strength", default="expert_v2", help="AI strength to learn from.")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (default: cpu_count-1).")
    parser.add_argument("--output", default=str(ROOT / "models" / "bid_training_data.npz"), help="Output file path.")
    parser.add_argument("--boom-rounds", type=int, default=16, help="Rounds per game.")
    args = parser.parse_args()

    if args.workers <= 0:
        args.workers = max(1, cpu_count() - 1)

    rng = random.Random(args.seed)
    num_games = (args.rounds + args.boom_rounds - 1) // args.boom_rounds
    game_seeds = [rng.randint(1, 10_000_000) for _ in range(num_games)]
    tasks = [(gs, args.strength, args.boom_rounds) for gs in game_seeds]

    print(f"Generating bid training data: {num_games} games ({args.rounds} rounds target)")
    print(f"  Strength: {args.strength}")
    print(f"  Workers: {args.workers}")
    sys.stdout.flush()

    all_features = []
    all_labels = []
    completed = 0

    if args.workers == 1:
        for task in tasks:
            features, labels = _collect_game(task)
            all_features.append(features)
            all_labels.append(labels)
            completed += 1
            if completed % 5 == 0:
                total_samples = sum(f.shape[0] for f in all_features)
                declares = sum(l.sum() for l in all_labels)
                print(f"  {completed}/{num_games} games, {total_samples} samples ({declares} declares)")
                sys.stdout.flush()
    else:
        with Pool(processes=args.workers) as pool:
            for features, labels in pool.imap_unordered(_collect_game, tasks):
                all_features.append(features)
                all_labels.append(labels)
                completed += 1
                if completed % 5 == 0:
                    total_samples = sum(f.shape[0] for f in all_features)
                    declares = sum(int(l.sum()) for l in all_labels)
                    print(f"  {completed}/{num_games} games, {total_samples} samples ({declares} declares)")
                    sys.stdout.flush()

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    n_declare = int(y.sum())
    n_pass = len(y) - n_declare

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), features=X, labels=y)
    print(f"\nDone! {X.shape[0]} samples saved to {output_path}")
    print(f"  Feature shape: {X.shape}")
    print(f"  Declares: {n_declare} ({100*n_declare/len(y):.1f}%)")
    print(f"  Passes:   {n_pass} ({100*n_pass/len(y):.1f}%)")


if __name__ == "__main__":
    main_cli()
