# Pitch Neural Net Training Guide

## Quick Start

```bash
cd training/scripts

# Option 1: Start fresh with human warm-start (recommended if you have human data)
python3 pitch_train_auto.py --human-warmstart --fresh

# Option 2: Start fresh self-play only (no human data needed)
python3 pitch_train_auto.py --fresh

# Option 3: Resume from where you left off
python3 pitch_train_auto.py
```

## Requirements

- Python 3.8+
- PyTorch (`pip install torch`)
- NumPy (`pip install numpy`)

Install on Mac: `pip3 install torch numpy`

## How It Works

The training pipeline runs in generations. Each generation:

1. **Generate games** — bots play thousands of hands against each other
2. **Train** — neural net learns from the game outcomes
3. **Validate** — test how well the model plays (P(make2) metric)
4. **Export** — save the best model as JSON for the web game
5. **Repeat** — each generation's model guides the next generation's games

### Generation 0
Uses pure heuristic bots (no NN). Creates a baseline of 50k games.

### Generation 1+
Mix of:
- **NN-rollout games** (500 games × 20 rollouts) — highest quality, uses current model
- **Heuristic self-play** (40k games) — fast, provides diversity
- **Hard cases** (5k games) — edge cases for jack protection, close bids

## Running 24/7

The script is designed to run unattended. To stop it gracefully:

```bash
# Create a STOP file — training will finish the current generation then exit
touch training/scripts/STOP
```

### Running with screen/tmux (recommended)
```bash
# Start a tmux session
tmux new -s pitch-training

# Run training
cd training/scripts
python3 pitch_train_auto.py --human-warmstart --fresh

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t pitch-training
```

### Running with an autonomous agent
Give the agent this task:

> Run the Pitch neural net training pipeline. cd into the training/scripts
> directory and run: python3 pitch_train_auto.py --human-warmstart --fresh
> Monitor the output for errors. If it crashes, restart it with just:
> python3 pitch_train_auto.py (no --fresh, so it resumes from checkpoint).
> The training log is at training/training_log_auto.txt.

## Monitoring Progress

- **Console output** — shows real-time progress
- **training/training_log_auto.txt** — full log with timestamps
- **Key metric: P(make2)** — probability the bidding team makes their contract
  - Random play: ~50%
  - Current model: ~88%
  - Good target: 92%+

## Output Files

| File | Description |
|------|-------------|
| `models/checkpoint.pt` | Latest training checkpoint (resume from here) |
| `models/model_genN.pt` | Individual generation model weights |
| `models/pitch_model_best.json` | Best model exported for web game |
| `models/pitch_model_genN.json` | Latest generation exported for web |
| `models/best_metric.json` | Tracks best P(make2) achieved |

## Collecting Human Data

1. Play the singleplayer game (`singleplayer/pitch_game.html`) with Solver ON
2. Click **Export Log** to download a JSON file
3. Move the JSON file to `training/data/`
4. Name it `human_gameplay_*.json` (any suffix works)
5. Rerun training with `--human-warmstart`

Human data gets 5x weight in training — even 200-500 hands help significantly.

## Tuning for Your Hardware

Edit the constants at the top of `pitch_train_auto.py`:

- **Slow laptop**: Reduce `GEN0_SELFPLAY_GAMES` to 20k, `GEN_SELFPLAY_GAMES` to 20k
- **Fast desktop/M-series Mac**: Increase to 100k+ for better data quality
- **GPU available**: PyTorch will auto-detect CUDA. Training is faster but data generation is CPU-bound

Typical generation time on an M1 MacBook: ~15-30 minutes.
