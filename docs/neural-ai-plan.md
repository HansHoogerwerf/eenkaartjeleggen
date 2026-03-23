# Neural AI Implementation Plan

## Goal
Train a neural network via **imitation learning** to play Klaverjassen card play at expert level or better. The neural AI uses expert's bidding logic but replaces card play decisions with neural network inference.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Game Engine (main.py)              │
│                                                     │
│  NeuralAIPlayer(AIPlayer)                           │
│  ├── Bidding: inherited from AIPlayer (expert)   │
│  └── Card play: neural network inference            │
│       ├── Encode game state → tensor (267 features) │
│       ├── Forward pass through MLP                  │
│       └── Mask illegal moves, pick highest logit    │
└─────────────────────────────────────────────────────┘
```

---

## Step 1: Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

RTX 5080 should be supported by CUDA 12.4+ builds.

---

## Step 2: Feature Encoding (267 input features)

The game state is encoded as a fixed-size float tensor for each card-play decision:

| Feature Group             | Size | Description                                            |
|---------------------------|------|--------------------------------------------------------|
| **My hand**               | 32   | Binary: which of the 32 cards I hold                   |
| **Cards played (round)**  | 32   | Binary: which cards have been played this round        |
| **Current trick cards**   | 96   | 3 × 32 one-hot: cards played by positions 1st/2nd/3rd |
| **Trump suit**            | 4    | One-hot: which suit is trump                           |
| **My position in trick**  | 4    | One-hot: am I 1st/2nd/3rd/4th to play                 |
| **Trick number**          | 1    | Normalized 0-1 (trick_num / 7)                         |
| **Team trick points**     | 2    | Normalized: [my_team / 162, opp_team / 162]            |
| **Team roem points**      | 2    | Normalized: [my_team / 200, opp_team / 200]            |
| **Declaring team**        | 2    | One-hot: [my_team_declares, opp_team_declares]         |
| **Opponent voids**        | 12   | 3 opponents × 4 suits: known void flags               |
| **Game scores**           | 2    | Normalized: [my_team / 500, opp_team / 500]            |
| **Round in game**         | 1    | Normalized: round_num / 16                             |
| **Legal move mask**       | 32   | Binary: which cards are legal to play                  |
| **Cards in hand count**   | 1    | Normalized: len(hand) / 8                              |
| **Trump cards in hand**   | 1    | Normalized: trump_count / 8                            |
| **Highest trump alive**   | 8    | One-hot: which trump rank is the highest outstanding   |
| **Partner is winning**    | 1    | Binary: is partner currently winning the trick         |
| **Points on table**       | 1    | Normalized: points in current trick / 40               |
| **My seat**               | 4    | One-hot: which seat I'm sitting in                     |
| **Lead suit**             | 4    | One-hot: which suit was led (0 if leading)             |
| **Cards per suit in hand**| 4    | Normalized: count of each suit / 8                     |
| **Trump control**         | 1    | Binary: do I hold the highest remaining trump          |
| **Trick cards count**     | 4    | One-hot: how many cards in the current trick (0-3)     |
| **Suit winners in hand**  | 4    | Count of "top" cards per suit (master cards)           |
| **Point cards in hand**   | 1    | Sum of point values in hand / 60                       |
| **Suit lengths played**   | 4    | Per suit: how many have been played / 8                |
| **Net score delta**       | 1    | (my_team_pts - opp_pts) / 162                          |
| **Total**                 | **267** |                                                    |

**Output**: 32-dim logits (one per card). Masked to legal moves, then softmax → pick highest.

---

## Step 3: Neural Network Architecture

Simple MLP (multi-layer perceptron) — proven effective for tabular game state input:

```
Input (267) → Linear(512) → ReLU → Dropout(0.2)
           → Linear(256) → ReLU → Dropout(0.2)
           → Linear(128) → ReLU → Dropout(0.1)
           → Linear(32)  → (masked softmax)
```

- **Parameters**: ~220K (small, fast inference)
- **Loss**: Cross-entropy on expert's chosen card
- **Optimizer**: Adam, lr=1e-3 with cosine annealing

---

## Step 4: Training Data Generation

Script: `tools/generate_training_data.py`

1. Run expert vs expert self-play games
2. For each card-play decision, record:
   - The encoded game state (267 features)
   - The card index expert chose (0-31)
   - The legal move mask
3. Store as `.npz` file (compressed NumPy arrays)
4. Each 16-round game produces ~128 decisions (4 players × 8 tricks × ~4 rounds with decisions)
5. **1024 rounds ≈ ~16,000 decision samples** (starting point)

Uses multiprocessing with 16 workers for speed.

---

## Step 5: Training Script

Script: `tools/train_neural.py`

1. Load `.npz` training data
2. Split 90/10 train/validation
3. Train MLP with:
   - Batch size: 256
   - Epochs: 50 (with early stopping, patience=5)
   - Learning rate: 1e-3 → cosine decay
   - CUDA acceleration (RTX 5080)
4. Save best model to `models/neural_v1.pt`
5. Log training/validation accuracy and loss

---

## Step 6: Integration into Game Engine

File: `neural/player.py`

Create `NeuralAIPlayer` that extends `AIPlayer`:
- **Bidding**: Use parent `AIPlayer.choose_trump()` with expert profile
- **Card play**: Override `_strategy()`:
  1. Encode current game state → tensor
  2. Forward pass through loaded model
  3. Mask illegal moves
  4. Select card with highest logit
- **Fallback**: If model file not found, fall back to expert heuristic play

Add `"neural"` to `AI_STRENGTH_PROFILES` in `main.py` pointing to expert base settings + neural flag.

---

## Step 7: UI Integration

Add a new AI button in `templates/index.html` for the neural AI:
- Label: "Neural" (with i18n key `ai.neural`)
- Data attribute: `data-strength="neural"`

---

## Step 8: Benchmark

Run benchmark using `tools/ai_benchmark.py`:
```bash
py tools/ai_benchmark.py --candidate-strength neural --baseline-strength expert --rounds 1024 --workers 16
```

Compare win rate and point differential.

---

## Directory Structure (new files)

```
neural/
├── __init__.py
├── features.py      # State encoding (game state → tensor)
├── model.py         # MLP network definition
└── player.py        # NeuralAIPlayer class

models/              # Trained model weights
└── (neural_v1.pt)   # Generated after training

tools/
├── generate_training_data.py   # Data generation script
└── train_neural.py             # Training script
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| 1024 rounds may not produce enough data | Start small, evaluate, scale up later |
| Neural AI may play illegal moves | Hard mask on legal moves before selection |
| Model too slow for real-time play | Small MLP (~220K params), inference <1ms on GPU |
| CPU-only fallback needed for deployment | PyTorch CPU inference works fine for small models |
| Imitation learning has a ceiling (can't exceed teacher) | This is Phase 1; RL self-play can follow later |
