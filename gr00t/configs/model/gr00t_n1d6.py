from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig

from . import register_model_config


@dataclass
class Gr00tN1d6Config(PretrainedConfig):
    """Unified configuration for Gr00tN1d6 model with backbone and action head."""

    # Model identification
    model_type: str = "Gr00tN1d6"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # backbone configuration
    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    backbone_model_type: str = "eagle"
    model_revision: str | None = None
    tune_top_llm_layers: int = 4  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 2048  # project_to_dim
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 16
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = True  # Enable BF16 loading
    collator_overwrite_image_inputs: bool = False  # Deprecated; use eagle_collator.
    eagle_collator: bool = (
        False  # this allows model to change image size in collator, needed for eagle any-res
    )
    backbone_trainable_params_fp32: bool = True

    ### Processing parameters
    image_crop_size: tuple[int, int] | None = None
    image_target_size: tuple[int, int] | None = None

    shortest_image_edge: int | None = 256
    crop_fraction: float | None = 0.95

    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    # Extra augmentation config (mask-based and others).
    extra_augmentation_config: dict | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_relative_action: bool = False

    # Action head configuration parameters
    max_state_dim: int = 29  # Default from state_shape
    max_action_dim: int = 29  # Default from action_shape
    action_horizon: int = 16
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # Global parameters from YAML
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    # Diffusion model type selection
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    # Diffusion model configuration with 32 layers (main difference from N15)
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 32,  # 32 layers instead of 16
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # State Augmentation parameters
    state_dropout_prob: float = 0.0  # State dropout probability
    state_additive_noise_scale: float = 0.0  # Scale for additive Gaussian noise on state features

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    # --- HAMLET ---
    # hamlet_mode in {"off", "tcl", "finetune"}.
    hamlet_mode: str = "finetune"
    n_moment_tokens: int = 4
    memory_window: int = 4
    memory_num_layers: int = 2
    # Env steps between cached memory snapshots; persisted to the checkpoint
    # config so evaluation can enforce n_action_steps == memory_stride.
    memory_stride: int = 16
    freeze_moment_tokens: bool = False
    # memory-to-action conditioning:
    #   "cross_attn" (default): memory-aggregated moment tokens replace the backbone
    #       moment-token tail and enter the DiT as cross-attention KV.
    #   "adaln": the pooled memory vector is added (zero-init) to the DiT timestep
    #       embedding and the moment-token tail is sliced off the KV (memory enters
    #       only via AdaLN).
    mem_cond_type: str = "cross_attn"
    # What flows through the memory module: {"moment_token", "vision_feature"}.
    memory_type: str = "moment_token"
    # How the memory window is filled: {"window", "zoo"}.
    #   "window": K observations per batch row (video delta_indices spans the window).
    #   "zoo": one observation per iteration; the window is assembled across iterations
    #       from a per-episode cache of the most transitional past observations.
    # Persisted to the checkpoint config so evaluation inherits the same behavior.
    memory_mode: str = "window"
    zoo_max_episodes: int = 4096
    # Zoo pool selection. score = (1-w)*rank(attn) - w*softmax(log local_density);
    # the lowest-scoring block is evicted. See Gr00tN1d6ActionHead.pool_scores.
    zoo_density_weight: float = 0.5  # w: 0 = attention only, 1 = density only
    zoo_step_tau: float = 0.15  # how far in time a neighbour still counts
    zoo_dist_tau: float = 0.5  # how close in appearance counts as redundant
    zoo_density_k: int = 4  # kNN neighbourhood size for the density estimate
    zoo_density_temp: float = 2.0  # softmax temperature normalizing density over the pool
    # m: trailing slots of the window reserved for the newest observations. mem_seq is
    # selected(K-m) + recent(m-1) + [current], so only K-m slots are selector-filled.
    # Persisted to the checkpoint config so evaluation assembles the same window.
    zoo_recent_slots: int = 2
    # True: a candidate competes only with pool blocks in its own temporal bucket, so
    # coverage of the episode is structural. False: original global-argmin eviction.
    zoo_stratified: bool = True
    # Memory-grounding auxiliary loss (RA-VLA's mse_r / margin, with HAMLET memory in
    # place of RA-VLA's retrieved neighbours). The action head runs the DiT a second
    # time on the SAME noised trajectory with a mismatched memory window and requires
    # that pass to be at least `mem_ground_margin` worse in per-sample flow MSE:
    #     loss = mse + mem_ground_weight * relu(mem_ground_margin - (mse_r - mse))
    # A policy that ignores memory scores identically in both passes and pays the full
    # margin, so the gap can only be earned by actually reading memory.
    # 0 disables the second pass entirely (no extra compute). Training-only.
    mem_ground_weight: float = 0.0
    mem_ground_margin: float = 0.0
    # How the mismatched memory window is built:
    #   "batch_roll": row i is given another batch row's window -- wrong episode
    #       entirely (this is what RA-VLA does to its retrieved set). Needs B > 1.
    #   "block_perm": the row's own past blocks in a random chronological order, with
    #       the current observation left in place. Grounds temporal structure rather
    #       than content. Needs memory_window > 2.
    #   "both": permute chronology and swap rows.
    #   "recent_only": ABLATION rather than corruption -- keep only the trailing
    #       zoo_recent_slots blocks (recent + current) and drop every selector-filled
    #       pool block, left-padding as the warm-up path does. The gap is then the value
    #       of the long-term pool over plain recency. Needs memory_window >
    #       zoo_recent_slots.
    mem_ground_shuffle: str = "batch_roll"
    # Upper edge of the gap band. The one-sided hinge only punishes too LITTLE memory
    # dependence, which lets the gap run away: the policy becomes hypersensitive, wrong
    # memory turns catastrophic, and training destabilizes until it retreats to ignoring
    # memory entirely (an absorbing state -- at exact invariance both passes are the same
    # function of theta, so the hinge gradient is zero and cannot escape). Penalizing
    # gap > mem_ground_gap_max keeps the dependence in a band instead. <= 0 disables the
    # upper edge, restoring the plain one-sided hinge.
    mem_ground_gap_max: float = 0.1
    # Steps of ZERO grounding weight at the start of training. Before the flow-matching
    # phase transition both passes predict the mean velocity and the gap is pure noise,
    # so the hinge only thrashes the memory representation. The second DiT pass is
    # skipped entirely while the weight is 0, so warmup costs nothing.
    mem_ground_warmup_steps: int = 2000
    # Steps to ramp the weight linearly from 0 to mem_ground_weight once warmup ends.
    # 0 = switch on at full weight.
    mem_ground_ramp_steps: int = 2000
    # Key-moment gate: when True, memory is zeroed out on non-key-moment steps
    # (window-end joint-state delta >= delta_threshold). When False, memory is
    # never gated -> plain HAMLET. Persisted to the checkpoint config so eval
    # matches training automatically.
    use_key_moment_gate: bool = False
    # L2 threshold on the normalized-joint-state delta between consecutive window
    # ends; below it the step is a key moment (memory kept). Only used when
    # use_key_moment_gate is True.
    delta_threshold: float = 0.2
    # If None, defaults to `action_horizon` at runtime.
    tcl_tau: float = 0.07

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            # PATCH: Backward compatibility for legacy argument "collator_overwrite_image_inputs"
            if key == "collator_overwrite_image_inputs":
                setattr(self, "eagle_collator", value)
            # /PATCH
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        """Return a dictionary representation of this config, optionally excluding augmentation keys."""
        if is_dataclass(self):
            cfg = asdict(self)
        else:
            cfg = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "use_albumentations_transforms",
                "formalize_language",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
            }
            cfg = {k: v for k, v in cfg.items() if k not in exclude_keys}

        return cfg

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        """Return a JSON string of this config, optionally excluding augmentation keys."""

        def default(o):
            if isinstance(o, (Path, torch.dtype, torch.device)):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )
 

register_model_config("GrootN1d6", Gr00tN1d6Config)
