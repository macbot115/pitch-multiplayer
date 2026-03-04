"""
Pitch Autonomous Training Pipeline — V8.0
Designed to run unattended 24/7 for continuous self-play improvement.

Features:
  - Starts from human data warm-start (if available) or fresh
  - Runs self-play generations indefinitely
  - Each generation: generate games → train → validate → export → repeat
  - Auto-exports best model to JSON for web deployment
  - Detailed logging with timestamps
  - Graceful stop: create a file named STOP in the scripts dir to halt
  - Periodic checkpointing so you can resume anytime

Usage:
  python3 pitch_train_auto.py                   # Start/resume autonomous training
  python3 pitch_train_auto.py --fresh            # Start fresh from scratch
  python3 pitch_train_auto.py --human-warmstart  # Warm-start from human data, then self-play
  touch STOP                                     # Gracefully stop after current generation
"""

import os, sys, time, random, argparse, glob, json, shutil
import torch
import numpy as np

# Ensure we can import from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pitch_engine import (
    deal_hands, run_bidding, sim_hand, estimate_bid_level,
    generate_selfplay_games, generate_nn_rollout_games, generate_hard_cases,
    validate_pMake2, validate_pMake2_rollout, legal_plays
)
from pitch_nn import (
    PitchNet, INPUT_DIM, HIDDEN_DIMS, encode, encode_training_data,
    train_model, make_batched_nn_play_fn, save_checkpoint, load_checkpoint,
    encode_and_cache, load_cached_tensors, build_pool_from_cache,
    BidNet, BID_INPUT_DIM, BID_HIDDEN_DIMS, encode_bid_training_data, train_bid_model
)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION — Tune these for your hardware
# ══════════════════════════════════════════════════════════════

# How many generations to run (set high for 24/7 operation)
MAX_GENERATIONS = 100

# Data generation per generation
GEN0_SELFPLAY_GAMES = 50_000       # Gen 0: pure heuristic self-play
GEN_SELFPLAY_GAMES = 40_000        # Later gens: heuristic self-play
GEN_NN_ROLLOUT_GAMES = 500         # NN-guided rollout games (slower but higher quality)
GEN_NN_ROLLOUTS = 20               # Rollouts per NN game
HARD_CASE_GAMES = 5_000            # Edge cases (jack situations, close bids)

# Training hyperparameters
EPOCHS = 80
BATCH_SIZE = 4096
LR = 0.001
PATIENCE = 12
AUX_LOSS_WEIGHT = 0.3              # Weight for jack/low auxiliary heads

# Pool management
POOL_CAP = 2_000_000               # Max training samples in pool
ROLLING_WINDOW = 6                 # How many recent generations to include
ROLLING_DECAY = 0.85               # Older generations get less weight
HUMAN_WEIGHT = 5.0                 # Human data gets this multiplier

# Validation
VAL_HANDS = 2000                   # Quick validation hands
ROLLOUT_VAL_HANDS = 500            # Rollout validation (slower, more accurate)
ROLLOUT_VAL_ROLLOUTS = 50
ROLLOUT_VAL_EVERY = 3              # Full rollout validation every N gens

# Curriculum — gradually make training data harder
CURRICULUM = {
    0: 2.5, 1: 2.5, 2: 2.0, 3: 2.0,
    4: 1.5, 5: 1.5, 6: 1.0, 7: 1.0,
    8: 0.7, 9: 0.5, 10: 0.3
}

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CACHE_DIR = os.path.join(SCRIPT_DIR, 'tensor_cache')
LOG_FILE = os.path.join(SCRIPT_DIR, '..', 'training_log_auto.txt')
STOP_FILE = os.path.join(SCRIPT_DIR, 'STOP')
BEST_METRIC_FILE = os.path.join(MODELS_DIR, 'best_metric.json')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line + '\n')
    except:
        pass


def log_separator(title=""):
    log("=" * 60)
    if title:
        log(f"  {title}")
        log("=" * 60)


# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def should_stop():
    """Check if a STOP file exists — graceful shutdown signal."""
    return os.path.exists(STOP_FILE)


def threshold(gen):
    return CURRICULUM.get(gen, 0.3 if gen >= 10 else 1.0)


def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def load_best_metric():
    """Load the best P(make2) achieved so far."""
    if os.path.exists(BEST_METRIC_FILE):
        try:
            with open(BEST_METRIC_FILE) as f:
                data = json.load(f)
            return data.get('pMake2', 0), data.get('gen', -1)
        except:
            pass
    return 0.0, -1


def save_best_metric(pMake2, gen):
    with open(BEST_METRIC_FILE, 'w') as f:
        json.dump({'pMake2': pMake2, 'gen': gen, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f)


def export_model_to_json(model, output_path):
    """Export model to JSON for web inference."""
    try:
        # Import export logic
        model.eval()
        layers = []
        state = model.state_dict()

        # Get layer names and pair weights with biases
        keys = list(state.keys())
        i = 0
        while i < len(keys):
            key = keys[i]
            if 'weight' in key:
                w = state[key].cpu().detach().numpy()
                # Look for corresponding bias
                bias_key = key.replace('weight', 'bias')
                if bias_key in state:
                    b = state[bias_key].cpu().detach().numpy()
                else:
                    b = np.zeros(w.shape[0])

                # Check for batch norm (next keys)
                bn_weight_key = key.replace('weight', 'bn.weight') if 'layers' in key else None
                if bn_weight_key and bn_weight_key in state:
                    # Fuse batch norm
                    bn_w = state[bn_weight_key].cpu().detach().numpy()
                    bn_b = state[bn_weight_key.replace('weight', 'bias')].cpu().detach().numpy()
                    bn_mean = state[bn_weight_key.replace('weight', 'running_mean')].cpu().detach().numpy()
                    bn_var = state[bn_weight_key.replace('weight', 'running_var')].cpu().detach().numpy()
                    eps = 1e-5
                    scale = bn_w / np.sqrt(bn_var + eps)
                    w = w * scale[:, None]
                    b = (b - bn_mean) * scale + bn_b

                layers.append({
                    'type': 'linear',
                    'weight': w.tolist(),
                    'bias': b.tolist()
                })
            i += 1

        with open(output_path, 'w') as f:
            json.dump({'layers': layers}, f)
        log(f"  Exported model to {output_path}")
        return True
    except Exception as e:
        log(f"  WARNING: Model export failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# HUMAN DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_human_data():
    """Load and encode all human gameplay JSON files."""
    try:
        from pitch_human_loader import load_human_games
    except ImportError:
        log("  pitch_human_loader.py not found")
        return None

    human_states = load_human_games(os.path.join(DATA_DIR, 'human_gameplay_*.json'))
    if not human_states:
        return None

    log(f"  Encoding {len(human_states)} human decisions...")
    t0 = time.time()
    X, y_main, y_jack, y_low, weights = encode_training_data(human_states)
    log(f"  Human data encoded in {time.time()-t0:.1f}s")

    return {
        'X': X, 'y_main': y_main, 'y_jack': y_jack, 'y_low': y_low,
        'weights': weights, 'n_states': len(human_states)
    }


# ══════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def run_autonomous(args):
    device = get_device()
    rng = random.Random(42)

    log_separator("PITCH AUTONOMOUS TRAINING V8.0")
    log(f"  Device: {device}")
    log(f"  Input dim: {INPUT_DIM}, Hidden: {HIDDEN_DIMS}")
    log(f"  Max generations: {MAX_GENERATIONS}")
    log(f"  Self-play games/gen: {GEN_SELFPLAY_GAMES}")
    log(f"  NN rollout games/gen: {GEN_NN_ROLLOUT_GAMES} × {GEN_NN_ROLLOUTS}")
    log(f"  Pool cap: {POOL_CAP:,}, Rolling window: {ROLLING_WINDOW}")
    log(f"  To stop gracefully: touch {STOP_FILE}")
    log("")

    # ── Load or create model ──
    start_gen = 0
    checkpoint_path = os.path.join(MODELS_DIR, 'checkpoint.pt')

    if not args.fresh and os.path.exists(checkpoint_path):
        log("Loading existing checkpoint...")
        model, start_gen = load_checkpoint(checkpoint_path, device=device)
        log(f"  Resumed at generation {start_gen}")
    else:
        if args.fresh:
            log("Fresh start — clearing old cache and checkpoints")
            for f in glob.glob(os.path.join(CACHE_DIR, 'encoded_gen*.pt')):
                os.remove(f)
            for f in glob.glob(os.path.join(SCRIPT_DIR, 'training_data_gen*.pt')):
                os.remove(f)
        model = PitchNet(INPUT_DIM, HIDDEN_DIMS)
        log(f"Created fresh model ({INPUT_DIM}-dim, hidden={HIDDEN_DIMS})")

    model = model.to(device)

    # ── Load human data for warm-start ──
    log("Checking for human gameplay data...")
    human_tensors = load_human_data()
    if human_tensors:
        log(f"  ✓ Human data: {human_tensors['n_states']} decisions ({HUMAN_WEIGHT}x weighted)")
    else:
        log(f"  No human data found (optional — self-play will work without it)")

    # ── Human warm-start phase ──
    if args.human_warmstart and human_tensors and start_gen == 0:
        log_separator("HUMAN WARM-START PHASE")
        log("Training initial model on human data before self-play...")

        X = human_tensors['X']
        y_main = human_tensors['y_main']
        y_jack = human_tensors['y_jack']
        y_low = human_tensors['y_low']
        weights = human_tensors['weights']

        batch_size = min(BATCH_SIZE, max(32, X.shape[0] // 4))
        model, val_loss = train_model(
            model, X, y_main, y_jack=y_jack, y_low=y_low,
            weights=weights, epochs=200, batch_size=batch_size,
            lr=LR, patience=25, device=device, verbose=True
        )
        log(f"  Warm-start val loss: {val_loss:.4f}")

        # Quick validation
        nn_play_fn = make_batched_nn_play_fn(model, device=device)
        pMake2, avg_pts = validate_pMake2(VAL_HANDS, defense_play_fn=nn_play_fn, rng=rng)
        log(f"  ★ Warm-start P(make2) = {pMake2*100:.1f}%, avg bidder pts = {avg_pts:.2f}")

        # Save warm-start checkpoint
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        save_checkpoint(model, optimizer, 0, checkpoint_path)
        log(f"  Warm-start model saved")
        log("")

    # ── Load best metric ──
    best_pMake2, best_gen = load_best_metric()
    if best_pMake2 > 0:
        log(f"  Best so far: P(make2) = {best_pMake2*100:.1f}% (gen {best_gen})")

    # ══════════════════════════════════════════════════════════
    # GENERATION LOOP — runs until MAX_GENERATIONS or STOP file
    # ══════════════════════════════════════════════════════════

    total_start = time.time()

    for gen in range(start_gen, MAX_GENERATIONS):
        if should_stop():
            log("STOP file detected — shutting down gracefully")
            if os.path.exists(STOP_FILE):
                os.remove(STOP_FILE)
            break

        gen_start = time.time()
        thresh = threshold(gen)

        log_separator(f"GENERATION {gen}")
        log(f"  Curriculum threshold: {thresh}")
        log(f"  Elapsed total: {format_duration(time.time() - total_start)}")

        # ── Step 1: Generate training data ──
        cache_file = os.path.join(CACHE_DIR, f'encoded_gen{gen}.pt')
        data_file = os.path.join(SCRIPT_DIR, f'training_data_gen{gen}.pt')

        if os.path.exists(cache_file):
            log(f"  [DATA] Tensor cache exists for gen {gen}, skipping generation")
        else:
            states = []

            if gen == 0:
                # Gen 0: pure heuristic self-play (no NN needed)
                log(f"  [DATA] Generating {GEN0_SELFPLAY_GAMES:,} heuristic self-play games...")
                t0 = time.time()
                states = generate_selfplay_games(
                    GEN0_SELFPLAY_GAMES, rng=rng,
                    realistic_bids=True, curriculum_threshold=thresh
                )
                log(f"  [DATA] Generated {len(states):,} states in {format_duration(time.time()-t0)}")
            else:
                # Later gens: mix of NN-rollout + self-play + hard cases
                nn_play_fn = make_batched_nn_play_fn(model, device=device)

                # NN-guided rollout games (highest quality data)
                log(f"  [DATA] Generating {GEN_NN_ROLLOUT_GAMES} NN-rollout games ({GEN_NN_ROLLOUTS} rollouts)...")
                t0 = time.time()
                nn_states = generate_nn_rollout_games(
                    GEN_NN_ROLLOUT_GAMES, GEN_NN_ROLLOUTS,
                    nn_play_fn, rng=rng,
                    realistic_bids=True, curriculum_threshold=thresh
                )
                log(f"  [DATA] {len(nn_states):,} NN-rollout states in {format_duration(time.time()-t0)}")

                # Heuristic self-play (fast, diverse)
                log(f"  [DATA] Generating {GEN_SELFPLAY_GAMES:,} heuristic self-play games...")
                t0 = time.time()
                sp_states = generate_selfplay_games(
                    GEN_SELFPLAY_GAMES, rng=rng,
                    realistic_bids=True, curriculum_threshold=thresh
                )
                log(f"  [DATA] {len(sp_states):,} self-play states in {format_duration(time.time()-t0)}")

                # Hard cases (edge cases for jack, close games)
                log(f"  [DATA] Generating {HARD_CASE_GAMES:,} hard-case games...")
                t0 = time.time()
                hard_states = generate_hard_cases(HARD_CASE_GAMES, rng=rng)
                log(f"  [DATA] {len(hard_states):,} hard-case states in {format_duration(time.time()-t0)}")

                states = nn_states + sp_states + hard_states

            # Encode and cache
            log(f"  [DATA] Encoding {len(states):,} states to tensor cache...")
            t0 = time.time()
            encode_and_cache(gen, states, CACHE_DIR)
            log(f"  [DATA] Cached in {format_duration(time.time()-t0)}")
            del states  # Free memory

        # ── Step 2: Build training pool ──
        log(f"  [POOL] Building from gens {max(0, gen-ROLLING_WINDOW+1)}..{gen}")
        t0 = time.time()
        gens_to_load = list(range(max(0, gen - ROLLING_WINDOW + 1), gen + 1))
        X, y_main, y_jack, y_low, weights = build_pool_from_cache(
            gens_to_load, decay=ROLLING_DECAY, cache_dir=CACHE_DIR,
            human_tensors=human_tensors, pool_cap=POOL_CAP
        )
        log(f"  [POOL] {X.shape[0]:,} samples ready in {format_duration(time.time()-t0)}")

        if X.shape[0] == 0:
            log(f"  ERROR: Empty pool! Skipping generation.")
            continue

        # ── Step 3: Train ──
        log(f"  [TRAIN] Training ({EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}, patience={PATIENCE})")
        t0 = time.time()
        model, val_loss = train_model(
            model, X, y_main, y_jack=y_jack, y_low=y_low,
            weights=weights, epochs=EPOCHS, batch_size=BATCH_SIZE,
            lr=LR, patience=PATIENCE, device=device, verbose=True
        )
        log(f"  [TRAIN] Done in {format_duration(time.time()-t0)}, val_loss={val_loss:.4f}")

        del X, y_main, y_jack, y_low, weights

        # ── Step 4: Validate ──
        log(f"  [VAL] Quick validation ({VAL_HANDS} hands)...")
        nn_play_fn = make_batched_nn_play_fn(model, device=device)
        pMake2, avg_pts = validate_pMake2(VAL_HANDS, defense_play_fn=nn_play_fn, rng=rng)
        log(f"  ★ Gen {gen}: P(make2) = {pMake2*100:.1f}%, avg bidder pts = {avg_pts:.2f}")

        # Full rollout validation periodically
        if (gen + 1) % ROLLOUT_VAL_EVERY == 0 or gen == 0:
            log(f"  [VAL] Rollout validation ({ROLLOUT_VAL_HANDS} hands, {ROLLOUT_VAL_ROLLOUTS} rollouts)...")
            t0 = time.time()
            pMake2_ro, avg_pts_ro = validate_pMake2_rollout(
                ROLLOUT_VAL_HANDS, ROLLOUT_VAL_ROLLOUTS,
                defense_play_fn=nn_play_fn, rng=rng
            )
            log(f"  ★ Rollout: P(make2) = {pMake2_ro*100:.1f}%, avg pts = {avg_pts_ro:.2f} "
                f"({format_duration(time.time()-t0)})")

        # ── Step 5: Save checkpoint ──
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        save_checkpoint(model, optimizer, gen + 1, checkpoint_path)
        log(f"  [SAVE] Checkpoint saved (gen {gen+1})")

        # Save individual generation model
        gen_model_path = os.path.join(MODELS_DIR, f'model_gen{gen}.pt')
        torch.save(model.state_dict(), gen_model_path)

        # ── Step 6: Export best model to JSON ──
        if pMake2 > best_pMake2:
            best_pMake2 = pMake2
            best_gen = gen
            save_best_metric(pMake2, gen)

            best_json = os.path.join(MODELS_DIR, 'pitch_model_best.json')
            export_model_to_json(model, best_json)
            log(f"  ★★ NEW BEST: P(make2) = {pMake2*100:.1f}% — exported to pitch_model_best.json")

        # Also export latest regardless
        latest_json = os.path.join(MODELS_DIR, f'pitch_model_gen{gen}.json')
        export_model_to_json(model, latest_json)

        # Clean up old gen JSON exports (keep last 3 + best)
        old_jsons = sorted(glob.glob(os.path.join(MODELS_DIR, 'pitch_model_gen*.json')))
        if len(old_jsons) > 3:
            for old in old_jsons[:-3]:
                if 'best' not in old:
                    try:
                        os.remove(old)
                    except:
                        pass

        gen_time = time.time() - gen_start
        log(f"  [DONE] Generation {gen} complete in {format_duration(gen_time)}")
        log(f"  [DONE] Best so far: P(make2) = {best_pMake2*100:.1f}% (gen {best_gen})")
        log("")

    # ── Final summary ──
    total_time = time.time() - total_start
    log_separator("TRAINING COMPLETE")
    log(f"  Total time: {format_duration(total_time)}")
    log(f"  Generations completed: {min(gen + 1, MAX_GENERATIONS) - start_gen}")
    log(f"  Best P(make2): {best_pMake2*100:.1f}% (gen {best_gen})")
    log(f"  Best model: {os.path.join(MODELS_DIR, 'pitch_model_best.json')}")


def main():
    parser = argparse.ArgumentParser(description='Pitch Autonomous Training Pipeline V8.0')
    parser.add_argument('--fresh', action='store_true',
                       help='Start completely fresh (clears cache and checkpoints)')
    parser.add_argument('--human-warmstart', action='store_true',
                       help='Train on human data first, then begin self-play generations')
    args = parser.parse_args()

    run_autonomous(args)


if __name__ == '__main__':
    main()
