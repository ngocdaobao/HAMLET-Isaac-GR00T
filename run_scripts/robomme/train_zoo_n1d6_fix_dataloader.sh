#!/usr/bin/env bash
# GR00T N1.6 + HAMLET "zoo" fine-tune -- history-aware policy with a cross-iteration
# memory pool instead of a within-batch memory window.
# Usage: VIZ_BATCH_DIR=runs/robomme DATASET_PATH=data/robomme bash run_scripts/robomme/train_zoo_n1d6_fix_dataloader.sh
#   RoboMME modality (8-D abs-joint / 2-view) is preset (robomme_config.py).
#
# How zoo differs from the original HAMLET window (--memory-mode window):
#   window: each batch row loads K observations at once (video delta_indices spans the
#           window), so activation memory scales with K.
#   zoo:    each batch row loads exactly ONE observation (delta_indices=[0]). The window
#           is assembled across iterations from a per-episode pool of the most
#           "transitional" past observations, selected by the L1 distance between
#           consecutive moment->image attention maps. Activation memory is independent
#           of K, so GRAD_ACCUM does not need to grow with K here.
#           Pooled blocks are detached (they come from earlier iterations); gradient
#           reaches the backbone only through the current observation.
#
# zoo REQUIRES a strictly sequential, fixed-stride anchor stream *per episode*: an
# episode's consecutive appearances must be exactly MEMORY_STRIDE env-steps apart, or
# the cached observations do not form a coherent history. SEQUENTIAL_ANCHORS=1 and
# ANCHOR_STRIDE=MEMORY_STRIDE serve this; launch_finetune.py re-checks both and
# force-corrects them with a warning. The pool is keyed by EPISODE ID, not batch slot,
# so neither multi-worker loading nor slot reshuffling affects it.
#
# Single-stage by default: moment tokens are randomly initialized and trained end-to-end (no TCL-initialization).
# To use the optional two-stage paper recipe instead, first run a Stage-1 TCL job (--hamlet-mode tcl),
# then point LOAD_MOMENT_TOKENS_FROM at its checkpoint and set FREEZE_MOMENT_TOKENS=1.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# torchcodec's .so files have no RPATH, so LD_LIBRARY_PATH must point at the system
# FFmpeg 7 libs *before* python starts -- otherwise `import torchcodec` raises at load
# time and every dataloader worker silently falls back to a slower decoder backend.
# Sourced (not executed) so torchrun and its forked workers inherit the environment.
# source "$REPO_ROOT/torchcodec_setup.sh"

# config (override via env)
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to your benchmark dataset directory}"
MODALITY_CONFIG="${MODALITY_CONFIG:-gr00t/configs/data/robomme_config.py}"  # robomme_config.py | rmbench_config.py
OUTPUT_DIR="${OUTPUT_DIR:-runs/robomme/zoo_n1d6_fix_dataloader_11_history}"  # where to save checkpoints and logs
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"                  # zoo forwards 1 obs/row regardless of K, so this need not scale with K
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-60000}"
MASTER_PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 10000 ))}"

# The compute nodes have no outbound network, so wandb cannot be the run record --
# this file is. Everything the job prints from here on (config echo, the pre-flight
# zoo checks, every rank's stdout/stderr, the [mem] pool lines) is mirrored into
# OUTPUT_DIR while still streaming to the terminal. Timestamped so a rerun of the
# same OUTPUT_DIR does not overwrite the previous run's record; set LOG_FILE to pin it.
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
_TEE_PID=$!
# bash does not wait for a process substitution, so on exit the script can outrun tee
# and drop its last lines -- which are exactly the traceback explaining a crash. Close
# the pipe, then wait for tee to drain it.
trap 'exec 1>&- 2>&-; wait "$_TEE_PID" 2>/dev/null' EXIT
echo "[log] $LOG_FILE"

# HAMLET memory options
MEMORY_MODE="${MEMORY_MODE:-zoo}"             # zoo | window  (see header)
# Memory-transformer sequence length T = pool target. The window is
# selected(K-M) + recent(M-1) + current: the trailing M slots always hold the M newest
# observations and the pool selector fills the rest. K=1 leaves no room for history at
# all, so zoo needs K>=2; K=4 matches the HAMLET default window.
K="${K:-11}"                                   # memory window = history length
ZOO_RECENT_SLOTS="${ZOO_RECENT_SLOTS:-4}"     # M: reserved recency slots (1 = current only, K = plain FIFO)
ZOO_MAX_EPISODES="${ZOO_MAX_EPISODES:-1000}"  # LRU cap on how many episodes keep a pool
MEMORY_STRIDE="${MEMORY_STRIDE:-16}"          # env steps between snapshots; set equal to the eval n_action_steps
N_MOMENT_TOKENS="${N_MOMENT_TOKENS:-4}"       # moment tokens per step (n_q)
MEM_COND_TYPE="${MEM_COND_TYPE:-cross_attn}"  # cross_attn | adaln
MEMORY_TYPE="${MEMORY_TYPE:-moment_token}"    # moment_token | vision_feature
LOAD_MOMENT_TOKENS_FROM="${LOAD_MOMENT_TOKENS_FROM:-}"  # optional Stage-1 (TCL) ckpt; see README "Moment-token initialization"
FREEZE_MOMENT_TOKENS="${FREEZE_MOMENT_TOKENS:-0}"       # 1 = freeze moment tokens (paper recipe when TCL-initialized)
USE_KEY_MOMENT_GATE="${USE_KEY_MOMENT_GATE:-1}"        # 1 = zero memory on non-key-moment steps; 0 = plain HAMLET. Saved to checkpoint config -> eval inherits it.
DELTA_THRESHOLD="${DELTA_THRESHOLD:-100.0}"              # key-moment threshold on normalized-joint window-end delta (only used when gate on)
# Anchor ordering. SEQUENTIAL_ANCHORS=1 marches each batch slot forward through one
# demonstration: slot i at iteration t+1 holds the next anchor of the same episode it
# held at iteration t, so a (B, d) state cache stays row-aligned across iterations.
# Set SEQUENTIAL_ANCHORS=0 for the upstream GR00T recipe (i.i.d.-shuffled anchors).
SEQUENTIAL_ANCHORS="${SEQUENTIAL_ANCHORS:-1}"
# Env steps between anchors (sequential mode only). Keep equal to MEMORY_STRIDE so
# consecutive anchors of a slot are exactly one memory window apart -- what the
# key-moment gate pairs and what the eval rollout does (one call per n_action_steps).
ANCHOR_STRIDE="${ANCHOR_STRIDE:-$MEMORY_STRIDE}"
# Shard size must hold at least (GLOBAL_BATCH_SIZE / NUM_GPUS) whole episodes, or
# ShardedMixtureDataset._order_for_batch_slots silently falls back to plain sequential
# order and every slot in a batch ends up on the SAME episode. With ANCHOR_STRIDE=16 a
# 1024-anchor shard holds ~34 episodes for RoboMME (mean 481 steps), comfortably above 8.

ANCHOR_PHASES="${ANCHOR_PHASES:-5}"  # number of anchor phases (for multi-phase anchor streams, e.g. RoboMME's 2-view)
ANCHOR_CHUNK_SIZE="${ANCHOR_CHUNK_SIZE:-20}"  # number of consecutive anchors per phase (for multi-phase anchor streams, e.g. RoboMME's 2-view)
SHARD_SIZE="${SHARD_SIZE:-1024}"
# >1 worker round-robins whole batches across workers reading disjoint shards, so slot i
# at iteration t+1 would not follow slot i at iteration t. Keep at 1 in sequential mode.
# zoo is unaffected by this: its pool is keyed by episode id, not batch slot. With N
# workers a shard's batches land every Nth iteration, but _order_for_batch_slots still
# advances each episode by one anchor per appearance, so that episode's observations
# still arrive in temporal order ANCHOR_STRIDE apart -- which is all the pool needs.
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-5}"


# Memory pool selection hyperparameters (see gr00t_n1d6.py _pool_density)
ZOO_DENSITY_WEIGHT="${ZOO_DENSITY_WEIGHT:-0.6}"
ZOO_STEP_TAU="${ZOO_STEP_TAU:-0.15}"
ZOO_DIST_TAU="${ZOO_DIST_TAU:-1.0}"
ZOO_DENSITY_K="${ZOO_DENSITY_K:-5}"
ZOO_DENSITY_TEMP="${ZOO_DENSITY_TEMP:-2.0}"
# 1 = a candidate competes only within its temporal bucket (K-1 equal-width bins over the
# episode so far), so no single phase can own the pool. 0 = original global-argmin eviction.
ZOO_STRATIFIED="${ZOO_STRATIFIED:-0}"

if [ "$MEMORY_MODE" = "zoo" ]; then
    if [ "$K" -lt 2 ]; then
        echo "[zoo] ERROR: K=$K leaves no room for history (the window holds K-1 past observations). Use K>=2." >&2
        exit 1
    fi
    if [ "$ZOO_RECENT_SLOTS" -lt 1 ] || [ "$ZOO_RECENT_SLOTS" -gt "$K" ]; then
        echo "[zoo] ERROR: ZOO_RECENT_SLOTS=$ZOO_RECENT_SLOTS must be in [1, K=$K]; it reserves the trailing M slots of the window, leaving K-M for the pool selector." >&2
        exit 1
    fi
    if [ "$SEQUENTIAL_ANCHORS" != "1" ]; then
        echo "[zoo] ERROR: SEQUENTIAL_ANCHORS=$SEQUENTIAL_ANCHORS; zoo needs sequential anchors so consecutive iterations of an episode are MEMORY_STRIDE apart." >&2
        exit 1
    fi
    if [ "$ANCHOR_STRIDE" != "$MEMORY_STRIDE" ]; then
        echo "[zoo] ERROR: ANCHOR_STRIDE=$ANCHOR_STRIDE != MEMORY_STRIDE=$MEMORY_STRIDE; the gap between an episode's consecutive anchors must equal the memory stride." >&2
        exit 1
    fi
fi

MOMENT_ARGS=()
if [ "$FREEZE_MOMENT_TOKENS" = "1" ]; then MOMENT_ARGS+=(--freeze-moment-tokens); else MOMENT_ARGS+=(--no-freeze-moment-tokens); fi
if [ "$USE_KEY_MOMENT_GATE" = "1" ]; then MOMENT_ARGS+=(--use-key-moment-gate); else MOMENT_ARGS+=(--no-use-key-moment-gate); fi
MOMENT_ARGS+=(--delta-threshold "$DELTA_THRESHOLD")
if [ "$ZOO_STRATIFIED" = "1" ]; then MOMENT_ARGS+=(--zoo-stratified); else MOMENT_ARGS+=(--no-zoo-stratified); fi
[ -n "$LOAD_MOMENT_TOKENS_FROM" ] && MOMENT_ARGS+=(--load-moment-tokens-from "$LOAD_MOMENT_TOKENS_FROM")
if [ "$SEQUENTIAL_ANCHORS" = "1" ]; then
    MOMENT_ARGS+=(--sequential-anchors --anchor-stride "$ANCHOR_STRIDE")
fi

##Remove when no need to visualize image
# Dataloader image dump (debug): set VIZ_BATCH_DIR to save one figure per batch element,
# named iter_<i>_batch_<j>.png -- a row per camera view, timesteps left to right.
# Images are dumped as the model sees them (post-augmentation, pre-normalization).
if [ -n "${VIZ_BATCH_DIR:-}" ]; then
    export VIZ_BATCH_DIR
    export VIZ_BATCH_MAX_ITERS="${VIZ_BATCH_MAX_ITERS:-200}"   # stop dumping after this many iterations
fi
###
# The config wandb would have recorded. Printed after the env overrides have resolved
# so the log states what actually ran, not what the defaults say.
echo "[cfg] host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null || echo n/a) dataset=$DATASET_PATH base_model=$BASE_MODEL"
echo "[cfg] gpus=$NUM_GPUS batch=$GLOBAL_BATCH_SIZE grad_accum=$GRAD_ACCUM max_steps=$MAX_STEPS save_steps=$SAVE_STEPS"
echo "[cfg] memory_mode=$MEMORY_MODE K=$K stride=$MEMORY_STRIDE n_moment=$N_MOMENT_TOKENS cond=$MEM_COND_TYPE type=$MEMORY_TYPE gate=$USE_KEY_MOMENT_GATE delta=$DELTA_THRESHOLD"
echo "[cfg] zoo density_w=$ZOO_DENSITY_WEIGHT step_tau=$ZOO_STEP_TAU dist_tau=$ZOO_DIST_TAU density_k=$ZOO_DENSITY_K density_temp=$ZOO_DENSITY_TEMP max_episodes=$ZOO_MAX_EPISODES recent_slots=$ZOO_RECENT_SLOTS selected_slots=$((K - ZOO_RECENT_SLOTS))"

export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$MODALITY_CONFIG" \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --save-steps "$SAVE_STEPS" \
    --shard-size "$SHARD_SIZE" \
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
    --hamlet-mode finetune \
    --n-moment-tokens "$N_MOMENT_TOKENS" \
    --memory-window "$K" \
    --memory-stride "$MEMORY_STRIDE" \
    --memory-num-layers 2 \
    --mem-cond-type "$MEM_COND_TYPE" \
    --memory-type "$MEMORY_TYPE" \
    --memory-mode "$MEMORY_MODE" \
    --zoo-max-episodes "$ZOO_MAX_EPISODES" \
    --zoo-recent-slots "$ZOO_RECENT_SLOTS" \
    "${MOMENT_ARGS[@]}"
