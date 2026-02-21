import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from main import AIPlayer, KlaverjasGame


@dataclass
class BenchStats:
    games: int = 0
    rounds: int = 0
    wins: int = 0
    points_for: int = 0
    points_against: int = 0
    nat_for: int = 0
    nat_against: int = 0
    declares: int = 0
    successful_declares: int = 0


def _make_players(
    candidate_team: int,
    seed_base: int,
    candidate_strength: str,
    baseline_strength: str,
) -> list[AIPlayer]:
    players: list[AIPlayer] = []
    for seat in range(4):
        team = main.SEAT_TEAMS[seat]
        strength = candidate_strength if team == candidate_team else baseline_strength
        p = AIPlayer(
            f"Bench {main.SEAT_DEFAULTS[seat]}",
            team,
            seat_idx=seat,
            rng_seed=seed_base + seat,
            signal_profile="core",
            ai_strength=strength,
        )
        players.append(p)
    return players


def _play_game(
    candidate_team: int,
    seed_base: int,
    candidate_strength: str,
    baseline_strength: str,
) -> tuple[int, dict]:
    event_state = {
        "rounds": 0,
        "nat_for": 0,
        "nat_against": 0,
        "declares": 0,
        "successful_declares": 0,
    }

    game_ref = {"game": None}

    def on_event(event: str, data: dict) -> None:
        if event == "round_done":
            event_state["rounds"] += 1
            declaring_team = data["history"]["declaring_team"]
            if declaring_team == candidate_team:
                event_state["declares"] += 1
                if not data["history"]["nat"]:
                    event_state["successful_declares"] += 1
        elif event == "nat":
            if data["declaring_team"] == candidate_team:
                event_state["nat_for"] += 1
            else:
                event_state["nat_against"] += 1
        elif event == "waiting_for_host" and game_ref["game"] is not None:
            game_ref["game"].signal_next_round()

    game = KlaverjasGame(
        human_seats={},
        log_fn=lambda *_args, **_kwargs: None,
        state_fn=on_event,
        game_mode="boom",
        game_seed=seed_base,
        ai_seed_base=seed_base * 31 + 7,
    )
    game_ref["game"] = game
    game.boom_rounds = 16
    game.players = _make_players(
        candidate_team=candidate_team,
        seed_base=seed_base * 43 + 5,
        candidate_strength=candidate_strength,
        baseline_strength=baseline_strength,
    )
    game.play()

    points_for = game.scores[candidate_team]
    points_against = game.scores[1 - candidate_team]
    winner = 1 if points_for >= points_against else 0
    result = {
        "rounds": event_state["rounds"],
        "nat_for": event_state["nat_for"],
        "nat_against": event_state["nat_against"],
        "declares": event_state["declares"],
        "successful_declares": event_state["successful_declares"],
        "points_for": points_for,
        "points_against": points_against,
    }
    return winner, result


def run_benchmark(
    target_rounds: int,
    seed: int,
    candidate_strength: str = "expert",
    baseline_strength: str = "advanced",
) -> BenchStats:
    rng = random.Random(seed)
    stats = BenchStats()

    # Remove pacing delays so benchmark is compute-bound.
    main.AI_BID_DELAY = 0.0
    main.AI_PLAY_DELAY = 0.0
    main.TRICK_CLEAR_DELAY = 0.0

    while stats.rounds < target_rounds:
        game_seed = rng.randint(1, 10_000_000)
        for candidate_team in (0, 1):
            winner, result = _play_game(
                candidate_team=candidate_team,
                seed_base=game_seed,
                candidate_strength=candidate_strength,
                baseline_strength=baseline_strength,
            )
            stats.games += 1
            stats.rounds += result["rounds"]
            stats.points_for += result["points_for"]
            stats.points_against += result["points_against"]
            stats.nat_for += result["nat_for"]
            stats.nat_against += result["nat_against"]
            stats.declares += result["declares"]
            stats.successful_declares += result["successful_declares"]
            stats.wins += winner
            if stats.rounds >= target_rounds:
                break

    return stats


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AI strengths against a baseline profile.")
    parser.add_argument("--rounds", type=int, default=10_000, help="Target number of played rounds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--candidate-strength", default="expert", choices=["beginner", "advanced", "expert"])
    parser.add_argument("--baseline-strength", default="advanced", choices=["beginner", "advanced", "expert"])
    args = parser.parse_args()

    stats = run_benchmark(
        target_rounds=args.rounds,
        seed=args.seed,
        candidate_strength=args.candidate_strength,
        baseline_strength=args.baseline_strength,
    )
    winrate = stats.wins / stats.games if stats.games else 0.0
    avg_diff = (stats.points_for - stats.points_against) / stats.games if stats.games else 0.0
    declare_rate = stats.successful_declares / stats.declares if stats.declares else 0.0

    print(f"games={stats.games}")
    print(f"rounds={stats.rounds}")
    print(f"win_rate={winrate:.4f}")
    print(f"avg_point_diff={avg_diff:.2f}")
    print(f"nat_for={stats.nat_for}")
    print(f"nat_against={stats.nat_against}")
    print(f"declare_success_rate={declare_rate:.4f}")


if __name__ == "__main__":
    main_cli()
