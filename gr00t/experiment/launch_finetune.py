# Launch finetuning for N1.6 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ: 
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = False
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.sequential_anchors = ft_config.sequential_anchors
    config.data.anchor_stride = ft_config.anchor_stride
    config.data.anchor_chunk_size = ft_config.anchor_chunk_size
    config.data.anchor_phases = ft_config.anchor_phases

    config.training.skip_weight_loading = ft_config.skip_weight_loading

    # HAMLET configuration
    config.model.hamlet_mode = ft_config.hamlet_mode
    config.model.n_moment_tokens = ft_config.n_moment_tokens
    config.model.memory_window = ft_config.memory_window
    config.model.memory_num_layers = ft_config.memory_num_layers
    config.model.memory_stride = ft_config.memory_stride
    config.model.mem_cond_type = ft_config.mem_cond_type
    config.model.memory_type = ft_config.memory_type
    config.model.memory_mode = ft_config.memory_mode
    config.model.zoo_max_episodes = ft_config.zoo_max_episodes
    config.model.zoo_density_weight = ft_config.zoo_density_weight
    config.model.zoo_step_tau = ft_config.zoo_step_tau
    config.model.zoo_dist_tau = ft_config.zoo_dist_tau
    config.model.zoo_density_k = ft_config.zoo_density_k
    config.model.zoo_density_temp = ft_config.zoo_density_temp
    config.model.zoo_stratified = ft_config.zoo_stratified
    config.model.zoo_recent_slots = ft_config.zoo_recent_slots
    config.model.use_key_moment_gate = ft_config.use_key_moment_gate
    config.model.delta_threshold = ft_config.delta_threshold
    config.model.mem_ground_weight = ft_config.mem_ground_weight
    config.model.mem_ground_margin = ft_config.mem_ground_margin
    config.model.mem_ground_shuffle = ft_config.mem_ground_shuffle
    if ft_config.mem_ground_weight > 0:
        if ft_config.hamlet_mode != "finetune":
            raise ValueError(
                f"mem_ground_weight={ft_config.mem_ground_weight} needs hamlet_mode="
                f"'finetune' (there is no memory to mismatch under "
                f"hamlet_mode={ft_config.hamlet_mode!r})."
            )
        # A shuffle that cannot change anything makes the term a silent no-op for the
        # whole run. memory_window is known here; the per-device batch size is not
        # (it is derived from global_batch_size and the world size), so 'batch_roll'
        # is checked at runtime by Gr00tN1d6ActionHead instead.
        if ft_config.mem_ground_shuffle == "block_perm" and ft_config.memory_window <= 2:
            raise ValueError(
                f"mem_ground_shuffle='block_perm' reorders the past blocks of the "
                f"window, so it needs memory_window > 2 (got "
                f"{ft_config.memory_window}; only the current block would be left). "
                f"Use 'batch_roll' instead."
            )
        print(
            f"[HAMLET] memory grounding: weight={ft_config.mem_ground_weight} "
            f"margin={ft_config.mem_ground_margin} "
            f"shuffle={ft_config.mem_ground_shuffle}"
        )
    if (
        ft_config.hamlet_mode == "finetune"
        and ft_config.freeze_moment_tokens
        and not ft_config.load_moment_tokens_from
    ): 
        print(
            "[HAMLET][WARN] freeze_moment_tokens=True but no --load-moment-tokens-from "
            "was given: randomly initialized moment tokens would stay frozen for the "
            "whole run. Load TCL-pretrained tokens or pass --no-freeze-moment-tokens."
        )
    config.model.freeze_moment_tokens = ft_config.freeze_moment_tokens
    config.model.tcl_tau = ft_config.tcl_tau
    config.training.load_moment_tokens_from = ft_config.load_moment_tokens_from

    # HAMLET — override video delta_indices on the registered modality configs.
    if ft_config.hamlet_mode == "finetune" and ft_config.memory_mode == "zoo":
        # zoo: ONE observation per sample. The memory window is assembled across
        # iterations instead of within a batch row, so the temporal spacing has to come
        # from the anchor sampler: strictly sequential anchors, memory_stride apart.
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        for tag in MODALITY_CONFIGS:
            if "video" in MODALITY_CONFIGS[tag]:
                MODALITY_CONFIGS[tag]["video"].delta_indices = [0]
        config.data.allow_padding = True
        if not ft_config.sequential_anchors:
            print(
                "[HAMLET-ZOO][WARN] memory_mode='zoo' requires sequential_anchors=True "
                "(otherwise consecutive iterations are unrelated frames and the cache is "
                "meaningless) — forcing it on."
            )
            config.data.sequential_anchors = True
        if ft_config.anchor_stride != ft_config.memory_stride:
            print(
                f"[HAMLET-ZOO][WARN] anchor_stride={ft_config.anchor_stride} != "
                f"memory_stride={ft_config.memory_stride}; the gap between consecutive "
                f"iterations must equal memory_stride — forcing anchor_stride="
                f"{ft_config.memory_stride}."
            )
            config.data.anchor_stride = ft_config.memory_stride
        m = ft_config.zoo_recent_slots
        if not 1 <= m <= ft_config.memory_window:
            raise ValueError(
                f"zoo_recent_slots={m} must be in [1, memory_window="
                f"{ft_config.memory_window}]: it reserves the trailing m slots of the "
                f"window for the newest observations, leaving memory_window - m for the "
                f"pool selector."
            )
        print(
            f"[HAMLET-ZOO] single-obs batching: delta_indices=[0] "
            f"K_target={ft_config.memory_window} stride={ft_config.memory_stride} "
            f"recent_slots={m} selected_slots={ft_config.memory_window - m}"
        )
    elif ft_config.hamlet_mode == "finetune" and ft_config.memory_window > 1:
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        stride = ft_config.memory_stride
        K = ft_config.memory_window
        new_indices = [-(K - 1 - i) * stride for i in range(K)]
        # (-48,-32,-16,-0) for K=4, stride=16
        for tag in MODALITY_CONFIGS:
            if "video" in MODALITY_CONFIGS[tag]:
                MODALITY_CONFIGS[tag]["video"].delta_indices = new_indices
        config.data.allow_padding = True
        print(f"[HAMLET] K-step batching: stride={stride} window={(K-1)*stride} delta_indices={new_indices}")
    elif ft_config.hamlet_mode == "tcl":
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        new_indices = [0, -999]
        for tag in MODALITY_CONFIGS:
            if "video" in MODALITY_CONFIGS[tag]:
                MODALITY_CONFIGS[tag]["video"].delta_indices = new_indices
        config.data.allow_padding = True
        print(f"[HAMLET-TCL] video delta_indices = {new_indices}")

    # Resolve anchor chunking last: the zoo branch above can force sequential anchors on,
    # and chunking is a no-op without them.
    if config.data.sequential_anchors:
        if ft_config.anchor_chunk_size == 0:
            config.data.anchor_chunk_size = 2 * ft_config.memory_window
        elif ft_config.anchor_chunk_size < 0:
            config.data.anchor_chunk_size = 0
        if config.data.anchor_chunk_size > 0:
            print(
                f"[HAMLET] anchor chunking: {config.data.anchor_chunk_size} anchors per virtual "
                f"episode (memory_window={ft_config.memory_window}); chunks are shuffled, anchors "
                f"inside a chunk stay sequential."
            ) 
        else:
            print("[HAMLET] anchor chunking disabled: one contiguous anchor run per episode.")

        phases = config.data.anchor_stride if ft_config.anchor_phases <= 0 else ft_config.anchor_phases
        phases = min(phases, config.data.anchor_stride)
        config.data.anchor_phases = phases
        if phases > 1:
            print(
                f"[HAMLET] anchor phases: {phases} of stride {config.data.anchor_stride} "
                f"({phases}/{config.data.anchor_stride} of every episode's frames are used as "
                f"anchors, each offset as its own virtual episode) -> ~{phases}x the shards, "
                f"so an epoch is ~{phases}x longer."
            )

    run(config)
