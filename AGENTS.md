# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Testing

**Run single test file:**
```bash
python -m unittest tests/test_ai_decisions.py
```

**Run all tests:**
```bash
python -m unittest discover
```

**Browser E2E tests require Playwright:**
```bash
playwright install  # First time only
python -m unittest tests/test_browser_integration.py
```

## Critical Patterns

**Pacing delays MUST be zeroed in tools/benchmarks:**
All scripts in `tools/` that run automated games must set these at module level:
```python
main.AI_BID_DELAY = 0.0
main.AI_PLAY_DELAY = 0.0
main.TRICK_CLEAR_DELAY = 0.0
```
These are global mutable constants in [`klaverjas/constants.py`](klaverjas/constants.py:17-19) — tools patch them to disable UI pacing during automation.

**Gunicorn MUST use single worker (`-w 1`):**
Room state is in-memory and process-local. Multiple workers break WebSocket routing and state consistency.

**Neural model paths:**
- Production model: [`models/neural_best.pt`](models/neural_best.pt)
- GPU training output: [`models/neural_gpu.pt`](models/neural_gpu.pt)
- Neural player defaults to CPU inference (avoids gevent+CUDA conflicts)

**Thread offload hook:**
[`main.py`](main.py:63) defines `_thread_offload` — set by [`app.py`](app.py) to run AI computation in native threads under gevent, preventing event loop blocking.

**Reconnect identity is name-based:**
Server matches reconnecting clients by exact string equality on display name ([`app.py:75-96`](app.py:75-96)). No secret token — security gap documented in [`docs/claude-analysis.md`](docs/claude-analysis.md:583).

## Code Style

- Python 3.12+ with type hints (`str | None` union syntax)
- Modern imports: `from __future__ import annotations` only in [`config.py`](config.py:3), [`tools/gpu_engine.py`](tools/gpu_engine.py:22), [`tools/train_gpu_selfplay.py`](tools/train_gpu_selfplay.py:16)
- Frozen `@dataclass` for config objects ([`config.py`](config.py))
- `snake_case` for Python; `camelCase` for JavaScript

## Architecture Notes

**Game engine is stateful and blocking:**
[`main.py`](main.py) runs one daemon thread per room. Human players block on `threading.Event` waiting for card/bid input. Not async-friendly — scales to ~hundreds of concurrent games, not thousands.

**Hand determinization for endgame solver:**
Expert AI uses backtracking CSP to reconstruct opponent hands when ≤3 cards remain ([`main.py`](main.py) `_determinize_endgame_hands`). Fails gracefully if constraints are inconsistent.

**Lookahead uses sampling, not exact search:**
Expert v2 samples 5 random opponent hand distributions and runs depth-limited minimax (3 tricks deep). Too slow for real-time multiplayer (~5-7s per round) — intended for benchmarking only.
