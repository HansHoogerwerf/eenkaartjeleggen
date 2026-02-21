import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from main import KlaverjasGame


def run_deterministic_game(
    *,
    game_seed: int,
    ai_seed_base: int | None,
    game_mode: str,
    score_limit: int,
    boom_rounds: int | None,
    ai_strength: str,
    ai_signal_profile: str,
) -> dict:
    game_ref: dict[str, KlaverjasGame] = {}

    def on_event(event: str, _data: dict) -> None:
        if event == "waiting_for_host":
            game_ref["game"].signal_next_round()

    main.AI_BID_DELAY = 0.0
    main.AI_PLAY_DELAY = 0.0
    main.TRICK_CLEAR_DELAY = 0.0

    game = KlaverjasGame(
        human_seats={},
        log_fn=lambda *_args, **_kwargs: None,
        state_fn=on_event,
        game_mode=game_mode,
        score_limit=score_limit,
        game_seed=game_seed,
        ai_seed_base=ai_seed_base,
        ai_strength=ai_strength,
        ai_signal_profile=ai_signal_profile,
    )
    game_ref["game"] = game
    if boom_rounds is not None:
        game.boom_rounds = boom_rounds
    game.play()
    return game.get_replay_data()


def capture_replay(
    *,
    output_path: Path,
    game_seed: int,
    ai_seed_base: int | None,
    game_mode: str,
    score_limit: int,
    boom_rounds: int | None,
    ai_strength: str,
    ai_signal_profile: str,
) -> dict:
    replay = run_deterministic_game(
        game_seed=game_seed,
        ai_seed_base=ai_seed_base,
        game_mode=game_mode,
        score_limit=score_limit,
        boom_rounds=boom_rounds,
        ai_strength=ai_strength,
        ai_signal_profile=ai_signal_profile,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")
    return replay


def verify_replay(replay_path: Path) -> bool:
    expected = json.loads(replay_path.read_text(encoding="utf-8"))
    actual = run_deterministic_game(
        game_seed=int(expected["game_seed"]),
        ai_seed_base=int(expected["ai_seed_base"]),
        game_mode=expected.get("game_mode", "score_limit"),
        score_limit=int(expected.get("score_limit", 500)),
        boom_rounds=int(expected["boom_rounds"]) if "boom_rounds" in expected else None,
        ai_strength=expected.get("ai_strength", "expert"),
        ai_signal_profile=expected.get("ai_signal_profile", "core"),
    )
    return expected == actual


def main_cli() -> int:
    parser = argparse.ArgumentParser(description="Capture or verify deterministic game replay traces.")
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Run a deterministic game and write replay JSON.")
    capture.add_argument("--out", required=True)
    capture.add_argument("--seed", type=int, required=True)
    capture.add_argument("--ai-seed-base", type=int, default=None)
    capture.add_argument("--mode", choices=["score_limit", "boom", "free_play"], default="boom")
    capture.add_argument("--score-limit", type=int, default=500)
    capture.add_argument("--boom-rounds", type=int, default=16)
    capture.add_argument("--ai-strength", choices=["beginner", "advanced", "expert"], default="expert")
    capture.add_argument("--ai-signal-profile", default="core")

    verify = sub.add_parser("verify", help="Re-run replay JSON and verify exact action equivalence.")
    verify.add_argument("--replay", required=True)

    args = parser.parse_args()
    if args.command == "capture":
        boom_rounds = args.boom_rounds if args.mode == "boom" else None
        replay = capture_replay(
            output_path=Path(args.out),
            game_seed=args.seed,
            ai_seed_base=args.ai_seed_base,
            game_mode=args.mode,
            score_limit=args.score_limit,
            boom_rounds=boom_rounds,
            ai_strength=args.ai_strength,
            ai_signal_profile=args.ai_signal_profile,
        )
        print(f"saved={args.out}")
        print(f"rounds={len(replay.get('rounds', []))}")
        print(f"scores={replay.get('scores_after')}")
        return 0

    ok = verify_replay(Path(args.replay))
    if ok:
        print("replay_verification=ok")
        return 0
    print("replay_verification=failed")
    return 1


if __name__ == "__main__":
    sys.exit(main_cli())
