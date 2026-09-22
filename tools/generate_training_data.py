"""Generate imitation-learning training data from AI self-play.

Runs teacher-vs-teacher games and records every non-forced card-play
decision as a (feature_vector, chosen_card_index) pair.  Output is saved as a
compressed NumPy .npz file with ``features``, ``labels`` and metadata
(``feature_version``, ``teacher``, ``rounds``).

Teachers
--------
  mythos            model_players/mythos_player.py - determinized alpha-beta
                    search that scores trick roem, nat and pit (default).
  opus              model_players/opus_player.py  - PIMC double-dummy search.
  neural_mythosbid  the current net + Mythos bidder (for distillation).
  pimc              the current net improved by net-rollout search on the
                    GPU (expert iteration; run with --workers 2-4).
  expert / advanced / beginner / expert_v2 / neural
                    engine-internal AIPlayer strength profiles.

Feature layout
--------------
``--feature-version 2`` (default) records the roem-aware 300-wide layout
(neural/features.py); ``1`` records the legacy 267-wide layout.

Uses a recording subclass of the teacher instead of monkey-patching, so it
works correctly with multiprocessing (spawn on Windows).

Example
-------
  python tools/generate_training_data.py --teacher mythos --rounds 12000
      --workers 15 --output models/training_data_mythos_v2.npz
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from klaverjas.constants import SEAT_DEFAULTS, SEAT_TEAMS
from klaverjas.core import Card, Trick
from main import AIPlayer, KlaverjasGame
from neural.features import CARD_INDEX, FEATURE_SIZES, encode_state

INTERNAL_PROFILES = tuple(AIPlayer.AI_STRENGTH_PROFILES)
DROP_IN_TEACHERS = ("mythos", "opus", "neural_mythosbid", "pimc")
TEACHERS = DROP_IN_TEACHERS + INTERNAL_PROFILES

# Module-level list that workers append to.  Each worker process gets its
# own copy, so there is no cross-process sharing.
_worker_samples: list[tuple[np.ndarray, int]] = []


class RecordingMixin:
    """Mixin that records every card-play decision of the wrapped teacher.

    Must precede the teacher class in the MRO so ``choose_card`` runs first:
    the state is encoded BEFORE the teacher removes the card from its hand.
    """

    feature_version: int = 2
    record_forced: bool = False

    def choose_card(self, trick: Trick, trump: str) -> Card:
        legal = self.legal_moves(trick, trump)
        features = None
        if len(legal) > 1 or self.record_forced:
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
                feature_version=self.feature_version,
            )

        card = super().choose_card(trick, trump)  # type: ignore[misc]

        if features is not None:
            _worker_samples.append((features, CARD_INDEX[str(card)]))
        return card


def _teacher_base_class(teacher: str):
    if teacher == "mythos":
        from model_players.mythos_player import MythosPlayer
        return MythosPlayer
    if teacher == "opus":
        from model_players.opus_player import OpusPlayer
        return OpusPlayer
    if teacher == "neural_mythosbid":
        from model_players.neural_mythos_player import NeuralMythosBidPlayer
        return NeuralMythosBidPlayer
    if teacher == "pimc":
        # Net-rollout search over the current net (GPU); use few workers, each
        # owns a CUDA engine.  NEURAL_PIMC_* env knobs apply.
        from model_players.pimc_player import PIMCNetPlayer
        return PIMCNetPlayer
    if teacher in INTERNAL_PROFILES:
        return AIPlayer
    raise ValueError(f"Unknown teacher {teacher!r}; choose from {TEACHERS}")


def make_recording_player(
    teacher: str,
    name: str,
    team: int,
    seat_idx: int,
    rng_seed: int,
    feature_version: int = 2,
    record_forced: bool = False,
):
    """Instantiate a teacher wrapped in RecordingMixin."""
    base = _teacher_base_class(teacher)
    cls = type(
        f"Recording{base.__name__}",
        (RecordingMixin, base),
        {"feature_version": feature_version, "record_forced": record_forced},
    )
    if teacher in INTERNAL_PROFILES:
        return cls(name, team, seat_idx=seat_idx, rng_seed=rng_seed,
                   signal_profile="core", ai_strength=teacher)
    return cls(name, team, seat_idx=seat_idx, rng_seed=rng_seed)


def _collect_game(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Play one game and return all recorded decision samples."""
    game_seed, teacher, boom_rounds, feature_version, record_forced, card_budget = args

    # Disable pacing delays
    main.AI_BID_DELAY = 0.0
    main.AI_PLAY_DELAY = 0.0
    main.TRICK_CLEAR_DELAY = 0.0
    if card_budget is not None:
        # Mythos/Opus read their per-card search budget from the environment.
        os.environ["AI_CARD_BUDGET"] = str(card_budget)

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
        ai_strength="advanced",
    )
    game_ref["game"] = game
    game.boom_rounds = boom_rounds

    players = [
        make_recording_player(
            teacher,
            name=f"Rec {SEAT_DEFAULTS[seat]}",
            team=SEAT_TEAMS[seat],
            seat_idx=seat,
            rng_seed=game_seed * 31 + 7 + seat,
            feature_version=feature_version,
            record_forced=record_forced,
        )
        for seat in range(4)
    ]
    for p in players:
        p.rules_variant = game.rules_variant
    game.players = players

    game.play()

    width = FEATURE_SIZES[feature_version]
    if not _worker_samples:
        return np.empty((0, width), dtype=np.float32), np.empty((0,), dtype=np.int64)

    features = np.stack([s[0] for s in _worker_samples])
    labels = np.array([s[1] for s in _worker_samples], dtype=np.int64)
    return features, labels


def _save(output_path: Path, feats: list[np.ndarray], labels: list[np.ndarray],
          feature_version: int, teacher: str, rounds_done: int) -> tuple[int, int]:
    width = FEATURE_SIZES[feature_version]
    X = np.concatenate(feats, axis=0) if feats else np.empty((0, width), np.float32)
    y = np.concatenate(labels, axis=0) if labels else np.empty((0,), np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.stem + ".tmp.npz")
    np.savez_compressed(
        str(tmp),
        features=X,
        labels=y,
        feature_version=np.int64(feature_version),
        teacher=np.array(teacher),
        rounds=np.int64(rounds_done),
    )
    os.replace(tmp, output_path)
    return X.shape[0], X.shape[1]


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Generate training data from AI self-play.")
    parser.add_argument("--rounds", type=int, default=1024, help="Target number of rounds to play.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--teacher", "--strength", dest="teacher", default="mythos",
                        choices=TEACHERS, help="Player to imitate (default: mythos).")
    parser.add_argument("--feature-version", type=int, default=2, choices=sorted(FEATURE_SIZES),
                        help="Feature layout to record (2 = roem-aware, default).")
    parser.add_argument("--record-forced", action="store_true",
                        help="Also record decisions with a single legal card (default: skip).")
    parser.add_argument("--card-budget", type=float, default=None,
                        help="AI_CARD_BUDGET seconds for search teachers (default: env / 1.0).")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (default: cpu_count-1).")
    parser.add_argument("--output", default=str(ROOT / "models" / "training_data.npz"), help="Output file path.")
    parser.add_argument("--boom-rounds", type=int, default=16, help="Rounds per game.")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Write the .npz every N finished games (0 = only at the end).")
    args = parser.parse_args()

    if args.workers <= 0:
        args.workers = max(1, cpu_count() - 1)

    rng = random.Random(args.seed)
    num_games = (args.rounds + args.boom_rounds - 1) // args.boom_rounds
    game_seeds = [rng.randint(1, 10_000_000) for _ in range(num_games)]
    tasks = [
        (gs, args.teacher, args.boom_rounds, args.feature_version, args.record_forced, args.card_budget)
        for gs in game_seeds
    ]
    output_path = Path(args.output)

    forced = "kept" if args.record_forced else "skipped"
    print(f"Generating training data: {num_games} games ({args.rounds} rounds target)")
    print(f"  Teacher: {args.teacher}   feature_version: {args.feature_version}   forced decisions: {forced}")
    print(f"  Workers: {args.workers}   output: {output_path}")
    sys.stdout.flush()

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    completed = 0
    t0 = time.perf_counter()

    def on_done(features: np.ndarray, labels: np.ndarray) -> None:
        nonlocal completed
        all_features.append(features)
        all_labels.append(labels)
        completed += 1
        if completed % 5 == 0 or completed == num_games:
            total_samples = sum(f.shape[0] for f in all_features)
            elapsed = time.perf_counter() - t0
            eta = elapsed / completed * (num_games - completed)
            print(f"  {completed}/{num_games} games, {total_samples} samples, "
                  f"{elapsed/60:.1f} min elapsed, ETA {eta/60:.1f} min")
            sys.stdout.flush()
        if args.checkpoint_every and completed % args.checkpoint_every == 0 and completed < num_games:
            n, w = _save(output_path, all_features, all_labels, args.feature_version,
                         args.teacher, completed * args.boom_rounds)
            print(f"  checkpoint: {n} samples x {w} features -> {output_path}")
            sys.stdout.flush()

    if args.workers == 1:
        for task in tasks:
            on_done(*_collect_game(task))
    else:
        with Pool(processes=args.workers) as pool:
            for features, labels in pool.imap_unordered(_collect_game, tasks):
                on_done(features, labels)

    n, w = _save(output_path, all_features, all_labels, args.feature_version,
                 args.teacher, completed * args.boom_rounds)
    print(f"\nDone! {n} samples x {w} features saved to {output_path}")
    print(f"  Elapsed: {(time.perf_counter() - t0)/60:.1f} min")


if __name__ == "__main__":
    main_cli()
