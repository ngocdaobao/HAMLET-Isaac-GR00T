# Finetune config used for single node post-training.
from dataclasses import dataclass
from typing import Literal

from gr00t.data.embodiment_tags import EmbodimentTag


@dataclass
class FinetuneConfig:
    """ 
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """
 
    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path to the dataset root directory containing trajectory data for fine-tuning."""

    embodiment_tag: EmbodimentTag
    """Identifier specifying which embodiment (robot configuration) this fine-tuning run targets."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    state_dropout_prob: float = 0.0
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """
    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 1
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 5e-5
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str | None = None
    """Optional experiment name used as the W&B run name. Defaults to the output directory basename."""

    wandb_project: str = "finetune-gr00t-n1d6"
    """W&B project name to log runs to."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = True
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d6`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    sequential_anchors: bool = False
    """If True, yield anchors in temporal order within an episode instead of i.i.d.-shuffled,
    so each batch marches forward through one demonstration like the rolling memory cache at
    inference. Forces one contiguous anchor run per episode (episode_sampling_rate is ignored).
    Note this heavily correlates gradients within and across steps; the K-step memory window is
    rebuilt from disk per sample either way, so this does NOT carry memory across iterations."""

    anchor_stride: int = 1
    """Keep every Nth anchor. Set to memory_stride to match the inference snapshot spacing.
    Reduces the anchor count by ~N."""

    anchor_chunk_size: int = 0
    """Sequential mode only: cut each episode into contiguous runs of this many anchors,
    treat every run as its own episode (own memory/state cache key) and shuffle the runs.
    Without it every batch slot marches from step 0 in lockstep, so iteration t only ever
    shows states from phase t of the task; with it a batch mixes phases while each slot
    still walks forward in time. Costs a memory warm-up at every chunk boundary, so the
    chunk should be at least a couple of memory windows long.
    0 = auto: 2 * memory_window when sequential anchors are on, off otherwise.
    Negative = force off (one contiguous run per phase stream)."""

    anchor_phases: int = 3
    """Sequential mode only: how many of the `anchor_stride` phase offsets to sample.
    The stride keeps anchors 0, N, 2N, ... and throws the rest away, but offset o gives
    the equally valid, equally spaced stream o, o+N, o+2N, ..., so P offsets recover P/N
    of the discarded anchors -- each as its own virtual episode with its own memory pool.
    P = anchor_stride (or <=0) uses every frame of every episode and multiplies the shard
    count, and so the epoch length, by N. Offsets are spread evenly over [0, N), so
    neighbouring near-duplicate frames are only both used at large P."""

    skip_weight_loading: bool = False
    """If True, skip loading model weights from base_model_path (architecture only).
    Useful for CI/testing to skip the slow checkpoint shard loading."""

    # --- HAMLET (History-Aware Memory with Learned Tokens) ---
    hamlet_mode: Literal["off", "tcl", "finetune"] = "finetune"
    """HAMLET training mode.
    - "off": vanilla GR00T N1.6 finetune (no HAMLET).
    - "tcl": Stage 1 — time-contrastive pretraining of moment tokens.
    - "finetune": Stage 2 — HAMLET end-to-end fine-tune (memory module + action head).
    """

    n_moment_tokens: int = 4
    """Number of learnable moment tokens (n_q) appended to the VLM input."""

    memory_window: int = 4
    """History window length T — number of past moment-token sets fed to the memory transformer."""

    memory_stride: int = 16
    """Stride (in env steps) between consecutive past snapshots in the HAMLET memory window.
    Must equal `n_action_steps` (the inference replanning interval) so the cache, which is
    updated once per policy call, naturally holds snapshots at [t-(K-1)S, ..., t-S, t]."""

    memory_num_layers: int = 2
    """Depth of the HAMLET memory transformer (paper default: 2)."""

    mem_cond_type: Literal["cross_attn", "adaln"] = "cross_attn"
    """How memory conditions the action head.
    - "cross_attn" (default): memory-aggregated moment tokens replace the backbone
      moment-token tail and enter the DiT as cross-attention KV.
    - "adaln": the pooled memory vector goes through a zero-init Linear and is added to
      the DiT timestep embedding; the moment-token tail is sliced off the KV."""

    memory_mode: Literal["window", "zoo"] = "window"
    """How the memory window is populated.
    "window" (original HAMLET): each batch row loads K=memory_window observations at
    once (video delta_indices = [-(K-1)S, ..., -S, 0]) and the whole window is
    differentiable.
    "zoo": each batch row loads a SINGLE observation (delta_indices = [0]); the window is
    assembled across iterations from a per-episode cache of the most "transitional"
    past observations, selected by the L1 distance between consecutive moment->image
    attention maps. Requires sequential_anchors=True and anchor_stride == memory_stride
    so consecutive iterations of an episode really are memory_stride apart; both are
    forced in launch_finetune.py. Cached blocks are detached (they come from previous
    iterations), so gradient reaches the backbone only through the current observation."""

    zoo_max_episodes: int = 4096
    """Cap on how many episodes the zoo pool keeps; least-recently-seen entries are
    evicted. Each entry holds memory_window x n_moment_tokens x d activations (the
    K-zoo_recent_slots selected blocks, the reserved recency queue, and the staged
    candidate). At memory_window=12 that is ~192 KB/episode, so the default cap
    reserves ~790 MB.
    Only a few x the per-rank batch size is ever in flight -- lower it accordingly."""

    zoo_density_weight: float = 0.5
    """w in the pool eviction score: (1-w)*rank(attn) - w*softmax(log local_density).
    0 = keep the blocks the instruction attends to most, ignoring redundancy.
    1 = keep the blocks that best cover the trajectory, ignoring the instruction."""

    zoo_step_tau: float = 0.15
    """How hard the step gap weights the distance, via omega = 1 + dstep/tau on gaps
    normalized to [0,1]. tau is the gap at which a neighbour's distance counts double,
    and omega tops out at 1 + 1/tau.

    This is the knob for temporal spread vs. catching revisits. Small (0.05, omega up
    to 21x) heavily discounts anything far in time, so the pool spreads across the
    episode but a state revisited much later is no longer recognized as a duplicate.
    Large (1.0, omega up to 2x) barely discounts it, so revisits are caught but the
    pool is free to cluster in time."""

    zoo_dist_tau: float = 0.5
    """Bandwidth of the appearance kernel, exp(-d / tau_d). Tokens are L2-normalized
    before cdist, so d lies in [0, 2] -- roughly 0 for near-duplicate frames and 1.2-1.4
    for unrelated ones -- and this bandwidth keeps a fixed meaning as backbone features
    drift. Small -> only near-duplicates register as redundant."""

    zoo_density_k: int = 5
    """Neighbourhood size for the kNN density estimate, clamped to pool_size-1.
    Small -> only near-duplicates are penalized; large -> being anywhere in a
    crowded region is penalized."""

    zoo_density_temp: float = 2.0
    """Softmax temperature normalizing the density across the pool. After the max
    rescale this is exactly (sigma / sigma.max()) ** (1/temp), so <1 sharpens (only
    the densest block is penalized) and >1 flattens. Above 1 is usually right:
    sigma is 1/distance and therefore heavy-tailed, so at temp=1 a single pair of
    near-duplicate blocks saturates the term and every other block reads as 0."""

    zoo_recent_slots: int = 2
    """m: how many of the memory_window slots are reserved for the newest observations.

    mem_seq is `selected(K-m) + recent(m-1) + [current]`, oldest-first: the last m blocks
    are always the last m observations, verbatim, and only the leading K-m slots are
    filled by the pool selector. Reserving recency keeps short-horizon continuity without
    the selector having to spend slots on it -- the pool is then free to cover the rest
    of the episode. Blocks become eligible for the pool m steps after they were current,
    once they have left the reserved queue, so nothing appears in mem_seq twice.

    m=1 is the original behavior (only the current observation is reserved).
    m=memory_window turns the window into a plain FIFO of the last K observations.
    Clamped to [1, memory_window] at runtime."""

    zoo_stratified: bool = True
    """How a staged block competes for a pool slot.

    True: the pool's (memory_window - zoo_recent_slots) slots are treated as equal-width temporal bins
    over the episode's elapsed span, and a candidate competes only with residents of its
    own bin -- so one phase cannot own more of the pool than its share of the timeline.
    When the pool is full and the candidate's bin is empty, the slot is taken from the
    most crowded bin.

    False: the original global-argmin eviction. That rule cannot bound per-phase
    occupancy, because both terms of `pool_scores` are normalized within the pool
    (rank01 over attention, softmax over log-density): once the pool has collapsed onto
    one phase every block scores alike, the density term goes flat, and selection falls
    back to raw instruction saliency -- which prefers that same phase."""

    memory_type: Literal["moment_token", "vision_feature"] = "moment_token"
    """What flows through the memory module (action-head VLM conditioning is unchanged).
    "moment_token": learnable moment tokens' post-LLM hidden states.
    "vision_feature": primary view (first modality_key) image tokens, post-LLM, avg-pooled
    to 64/step (no moment tokens added). Supports both mem_cond_type values."""

    mem_ground_weight: float = 0.0
    """Weight of the memory-grounding hinge (RA-VLA's mse_r / margin term).

    Every training step the action head replays the DiT on the SAME noised trajectory
    with a MISMATCHED memory window and requires that pass to be at least
    `mem_ground_margin` worse in per-sample flow MSE:

        loss = mse + mem_ground_weight * relu(mem_ground_margin - (mse_r - mse))

    A policy that ignores memory predicts identically in both passes and pays the full
    margin, so the only way to reduce the term is to actually condition on what memory
    holds. 0 disables the second pass entirely (no extra compute); when on, expect
    roughly one extra DiT + memory-transformer forward/backward per step (the backbone
    is NOT re-run). Training-only."""

    mem_ground_margin: float = 0.0
    """How much worse the mismatched-memory pass must be before the hinge is satisfied.

    In the same units as the flow-matching MSE, so scale it against the observed
    `mse_loss`: a margin far above it saturates the hinge and the gradient just fights
    the main objective. Start around 5-20% of the running mse."""

    mem_ground_shuffle: Literal["batch_roll", "block_perm", "both"] = "batch_roll"
    """How the mismatched memory window is built.

    "batch_roll": each row is handed another batch row's window -- a different episode
        entirely. This is the mismatch RA-VLA applies to its retrieved neighbours, and
        it grounds the policy in memory CONTENT. Needs per_device_batch_size > 1.
    "block_perm": the row keeps its own blocks but in a random chronological order,
        with the current observation left in place. Grounds TEMPORAL structure: it only
        bites if the model reads the order of the past, not just its contents. Needs
        memory_window > 2.
    "both": permute the chronology and then swap rows."""

    use_key_moment_gate: bool = True
    """Key-moment gate. When True, memory is zeroed out on non-key-moment steps
    (window-end joint-state delta >= delta_threshold) in both training and inference;
    when False, memory is never gated (plain HAMLET). Persisted to the checkpoint
    config so evaluation inherits the same behavior."""

    delta_threshold: float = 0.2
    """L2 threshold on the normalized-joint-state delta between consecutive window
    ends; below it the step is a key moment (memory kept). Only used when
    use_key_moment_gate is True."""

    load_moment_tokens_from: str | None = None
    """Stage-2 entry. Path to a Stage-1 (TCL) checkpoint or `model.safetensors`
    from which the moment-token parameter is loaded."""

    freeze_moment_tokens: bool = False
    """Stage 2 freezes moment tokens by default (matches GR00T frozen-VLM recipe)."""

    tcl_tau: float = 0.07
    """InfoNCE temperature for the TCL stage."""

