from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.interfaces import ShardedDataset
from gr00t.data.types import EmbodimentTag, MessageType, ModalityConfig, VLAStepData

from .lerobot_episode_loader import LeRobotEpisodeLoader


def extract_step_data(
    episode_data: pd.DataFrame,
    step_index: int,
    modality_configs: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
    allow_padding: bool = False, 
) -> VLAStepData:
    step_data = {}
    traj_len = len(episode_data)

    # Extract data for each configured modality
    for modality, config in modality_configs.items():
        step_data[modality] = {}
        # Sample timesteps according to delta indices configuration
        indices_to_load = [step_index + delta_index for delta_index in config.delta_indices]

        # HAMLET-TCL: video modality with sentinel -999 -> resolve to a far-frame negative
        # within the same trajectory (|t' - anchor| >= 16).
        # Patterns supported:
        #   [0, -999]            -> (anchor, far_negative)
        #   [0, -8, -999]        -> (anchor, close_positive, far_negative)
        if modality == "video" and -999 in config.delta_indices:
            anchor = max(0, min(step_index, traj_len - 1))
            all_idxs = np.arange(traj_len)
            far_mask = np.abs(all_idxs - anchor) >= 16
            far_idxs = all_idxs[far_mask]
            if far_idxs.size > 0:
                neg = int(np.random.choice(far_idxs))
            else:
                neg = int(np.clip(anchor + 16, 0, traj_len - 1))
            if len(config.delta_indices) == 2 and config.delta_indices == [0, -999]:
                indices_to_load = [anchor, neg]
            elif (
                len(config.delta_indices) == 3
                and config.delta_indices[0] == 0
                and config.delta_indices[-1] == -999
            ):
                close_offset = int(config.delta_indices[1])  # typically -8
                close_mask = (np.abs(all_idxs - anchor) <= 8) & (all_idxs != anchor)
                close_idxs = all_idxs[close_mask]
                if close_idxs.size > 0:
                    target_close = anchor + close_offset
                    close = (
                        int(target_close)
                        if 0 <= target_close < traj_len and target_close != anchor
                        else int(np.random.choice(close_idxs))
                    )
                else:
                    close = int(np.clip(anchor + close_offset, 0, traj_len - 1))
                indices_to_load = [anchor, close, neg]
            else:
                indices_to_load = [max(0, min(idx, traj_len - 1)) for idx in indices_to_load]
        elif allow_padding:
            indices_to_load = [max(0, min(idx, traj_len - 1)) for idx in indices_to_load]
        for key in config.modality_keys:
            if f"{modality}.{key}" in episode_data.columns:
                modality_data = episode_data[f"{modality}.{key}"].iloc[indices_to_load]
            else:
                raise KeyError(
                    f"{modality}.{key} not found in episode data, available keys: {episode_data.columns}"
                )
            if modality in ["state", "action"]:
                # Stack arrays for numerical modalities
                step_data[modality][key] = np.vstack(
                    [
                        np.array(modality_data.iloc[i]).astype(np.float32)
                        for i in range(len(modality_data))
                    ]
                )
            else:
                # Keep as lists for other modalities (video, language)
                step_data[modality][key] = modality_data.tolist()

    # Parse extracted data into VLAStepData structure
    video_data = step_data.get("video", {})
    mask_data = step_data.get("mask", {})
    state_data = step_data.get("state", {})
    action_data = step_data.get("action", {})
    language_data = step_data.get("language", {})
    assert len(language_data) == 1, f"Expected 1 language, got {len(language_data)}"
    text = language_data[list(language_data.keys())[0]][0]

    vla_step_data = VLAStepData(
        images=video_data,
        masks=mask_data if mask_data else None,
        states=state_data,
        actions=action_data,
        text=text,
        embodiment=embodiment_tag,
    )
    return vla_step_data


class ShardedSingleStepDataset(ShardedDataset):
    """
    Single-step dataset that creates shards from individual timesteps across episodes.

    This dataset implementation provides step-level data access for VLA training by:
    1. Loading episodes using LeRobotEpisodeLoader
    2. Splitting episodes into individual timesteps
    3. Organizing timesteps into balanced shards for efficient loading
    4. Supporting episode subsampling for data efficiency

    The sharding strategy ensures balanced shard sizes while maintaining randomization
    across episodes and timesteps within episodes. Each shard contains a mix of
    timesteps from different episodes to improve training diversity.

    Key features:
    - Step-level data access (vs episode-level)
    - Balanced sharding for consistent batch sizes
    - Episode subsampling via sampling rate
    - Integration with LeRobot data format
    - Support for multi-modal data (video, state, action, language)

    Args:
        dataset_path: Path to LeRobot format dataset directory
        embodiment_tag: Embodiment identifier for cross-embodiment training
        modality_configs: Configuration for each modality (sampling, keys)
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for video backend
        shard_size: Target number of timesteps per shard
        episode_sampling_rate: Fraction of episode timesteps to use (for efficiency)
        seed: Random seed for reproducible sharding and sampling
        allow_padding: Whether to allow padding of indices to valid range [0, max_length - 1]
        sequential_anchors: Yield anchors in temporal order within an episode
        anchor_stride: Keep every Nth anchor
        anchor_chunk_size: Sequential mode only. Cut each episode into runs of this many
            anchors and treat every run as its own (virtual) episode, shuffling the runs.
            0 disables chunking, i.e. one contiguous run per phase stream.
        anchor_phases: Sequential mode only. How many of the `anchor_stride` phase
            offsets to sample: offset o yields anchors o, o+N, o+2N, ... as its own
            virtual episode. 1 keeps only offset 0 (the pre-existing behaviour);
            `anchor_stride` (or <=0) uses every anchor of the episode.

    Example:
        >>> dataset = ShardedSingleStepDataset(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     embodiment_tag=EmbodimentTag.FRANKA,
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(8)), modality_keys=["joint_velocities"]
        ...         ),
        ...     },
        ...     shard_size=1024,
        ...     episode_sampling_rate=0.1,
        ... )
        >>> shard_data = dataset.get_shard(0)  # Get first shard of processed timesteps
    """

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        shard_size: int = 2**10,  # 1024 steps
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
        allow_padding: bool = False,
        sequential_anchors: bool = False,
        anchor_stride: int = 1,
        anchor_chunk_size: int = 0,
        anchor_phases: int = 1,
    ):
        """Initialize single-step dataset with sharding configuration."""
        super().__init__(dataset_path)
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.shard_size = shard_size
        self.episode_sampling_rate = episode_sampling_rate
        self.seed = seed
        self.allow_padding = allow_padding
        self.sequential_anchors = sequential_anchors
        self.anchor_stride = max(1, anchor_stride)
        # Chunking and phase offsets only change anything in sequential mode -- shuffled
        # mode already breaks the temporal correlation they are meant to break, and a
        # phase offset there is just a smaller stride.
        self.anchor_chunk_size = anchor_chunk_size if sequential_anchors else 0
        # Phase offsets: the stride keeps only every Nth anchor, so offsets 1..N-1 are
        # dropped outright. Each offset is an equally valid anchor stream (still exactly
        # `anchor_stride` apart), so sampling P of them recovers P/N of the discarded
        # anchors as extra virtual episodes. <=0 means all N.
        phases = self.anchor_stride if anchor_phases <= 0 else anchor_phases
        self.anchor_phases = min(phases, self.anchor_stride) if sequential_anchors else 1
        self.run_id_stride = 1
        self.processor = None
        self.rng = np.random.default_rng(seed)
        action_delta_indices = modality_configs["action"].delta_indices
        self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1

        self.episode_loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
        )

        # Create balanced shards from episode timesteps
        self.shard_dataset()

    def shard_dataset(self):
        """
        Create balanced shards by distributing episode timesteps across shards.

        The sharding process:
        1. Shuffle episode order for randomization
        2. Split each episode into multiple sub-sequences based on sampling rate
           (sequential mode: split it into `anchor_phases` phase-offset streams, each
           cut into `anchor_chunk_size`-long chunks that become independent virtual
           episodes, instead)
        3. Distribute sub-sequences across shards to balance shard sizes
        4. Use greedy assignment to minimize shard size variance

        This approach ensures:
        - Balanced shard sizes for consistent training batches
        - Diversity within shards (mix of episodes and timesteps)
        - Reproducible sharding based on seed
        """
        shuffled_episode_indices = self.rng.permutation(len(self.episode_loader.episode_lengths))
        # Sequential mode keeps each episode's anchors in one contiguous run: splitting
        # them would interleave distant timesteps and break the temporal ordering.
        num_splits = 1 if self.sequential_anchors else int(1 / self.episode_sampling_rate)

        assert len(shuffled_episode_indices) > 0, (
            f"No valid trajectories found for dataset {self.dataset_path}"
        )

        # Resolve per-episode anchor indices first. Demo-phase anchors (is_demo=True)
        # are dropped so no action loss is computed on demonstration frames; their
        # moment tokens still populate the memory window via the K-step delta_indices.
        # Datasets without an `is_demo` column are unaffected. Doing this before
        # counting keeps the shard count in sync with the steps actually distributed.
        # The stride subsample is NOT applied here -- `_build_runs` does it once per
        # phase offset, so the anchors this drops can still be picked up by another
        # offset instead of being discarded for good.
        episode_anchor_indices: dict[int, np.ndarray] = {}
        for ep_idx in shuffled_episode_indices:
            step_indices = np.arange(0, self.get_effective_episode_length(ep_idx))
            # step_indices = Range of steps in an episode
            valid_mask = self._anchor_valid_mask(int(ep_idx), len(step_indices)) # Mask for valid anchors (is_demo=False)
            if valid_mask is not None:
                step_indices = step_indices[valid_mask]
            episode_anchor_indices[int(ep_idx)] = step_indices

        # Split each episode into phase-offset streams and cut those into fixed-length
        # chunks, each of which becomes an independent virtual episode. Without this,
        # sequential mode marches every batch slot from step 0 of its demonstration in
        # lockstep, so iteration t only ever shows the model states from phase t of the
        # task. Short chunks with a shuffled order put unrelated task phases in one batch
        # while each slot still walks forward in time.
        runs = self._build_runs(shuffled_episode_indices, episode_anchor_indices)

        # Calculate total timesteps and required number of shards
        total_steps = np.sum([len(step_indices) for _, _, step_indices in runs]).astype(int)
        num_shards = np.ceil(total_steps / self.shard_size).astype(int)

        # Initialize shard containers
        sharded_episodes = [[] for _ in range(num_shards)]
        shard_lengths = np.zeros(num_shards, dtype=int)

        # Distribute episode sub-sequences across shards
        for ep_idx, virtual_ep_idx, run_step_indices in runs:
            step_indices = run_step_indices.copy()
            if not self.sequential_anchors:
                self.rng.shuffle(step_indices)
            splits = [step_indices[i::num_splits] for i in range(num_splits)]
            for split_step_indices in splits:
                if len(split_step_indices) == 0:
                    continue
                # Assign to shard with minimum current length (greedy balancing)
                shard_index = np.argmin(shard_lengths)
                sharded_episodes[shard_index].append((ep_idx, virtual_ep_idx, split_step_indices))
                shard_lengths[shard_index] += len(split_step_indices)

        # Validate shard creation 
        assert all(shard_lengths[i] > 0 for i in range(num_shards)), (
            "All shards must have length greater than 0"
        )

        print(f"Generated {num_shards} shards for dataset {self.dataset_path}")
        print(
            f"Total steps: {total_steps}, average shard length: {total_steps / num_shards}, shard length std: {np.std(shard_lengths)}"
        )
        if self.run_id_stride > 1:
            print(
                f"Anchor runs: {len(runs)} virtual episodes from {len(episode_anchor_indices)} "
                f"episodes (phases={self.anchor_phases} of stride {self.anchor_stride}, "
                f"chunk={self.anchor_chunk_size or 'off'}; "
                f"virtual id = episode * {self.run_id_stride} + run)"
            )
        self.sharded_episodes = sharded_episodes
        self.shard_lengths = shard_lengths

    def _phase_offsets(self) -> np.ndarray:
        """Starting offsets of the phase-shifted anchor streams.

        With stride N and P phases these are P offsets spread evenly over [0, N), so at
        P == N every anchor of the episode belongs to exactly one stream (offset 0 gives
        0, N, 2N, ...; offset 1 gives 1, N+1, 2N+1, ...) and nothing is discarded. Lower
        P keeps the streams as far apart as possible, so the near-duplicate frames of
        neighbouring offsets are not both spent on a small budget.
        """
        if self.anchor_phases <= 1:
            return np.zeros(1, dtype=int)
        offsets = np.linspace(0, self.anchor_stride, self.anchor_phases, endpoint=False)
        return np.unique(offsets.astype(int))

    def _build_runs(
        self,
        shuffled_episode_indices: np.ndarray,
        episode_anchor_indices: dict[int, np.ndarray],
    ) -> list[tuple[int, int, np.ndarray]]:
        """Turn per-episode anchors into the contiguous runs the shards are built from.

        Each run is one phase-offset stream of one episode (`anchors[offset::stride]`),
        optionally cut into `anchor_chunk_size`-long chunks. Returns
        (episode_index, virtual_episode_index, step_indices) triples: the episode index
        is what `get_shard` loads frames from; the virtual index is what each datapoint
        is tagged with, and is what the model keys its per-episode state by -- two runs
        of one demonstration must not share a key, or the memory pool of the batch slot
        holding the second would be poisoned by the first.

        With one phase and no chunking there is one run per episode and the two indices
        coincide, which is the pre-existing behaviour.
        """
        offsets = self._phase_offsets()
        chunk_size = self.anchor_chunk_size

        # Group the runs by episode first so the packed virtual ids can be assigned once
        # the widest episode is known.
        runs_by_episode: dict[int, list[np.ndarray]] = {}
        for ep_idx in shuffled_episode_indices:
            anchors = episode_anchor_indices[int(ep_idx)]
            episode_runs = []
            for offset in offsets:
                stream = anchors[offset :: self.anchor_stride]
                if stream.size == 0:
                    continue
                if chunk_size <= 0:
                    episode_runs.append(stream)
                    continue
                for start in range(0, len(stream), chunk_size):
                    episode_runs.append(stream[start : start + chunk_size])
            if episode_runs:
                runs_by_episode[int(ep_idx)] = episode_runs

        max_runs = max((len(r) for r in runs_by_episode.values()), default=1)
        if max_runs <= 1:
            # One run per episode: the virtual index is just the episode index, and the
            # episode order is already shuffled, so nothing below applies.
            self.run_id_stride = 1
            return [(ep_idx, ep_idx, r[0]) for ep_idx, r in runs_by_episode.items()]

        # Power of ten so the packed id stays readable in the batch-image dumps and the
        # memory debug logs: episode 12, run 3 reads as 12003. Runs are numbered
        # phase-major, so run = phase_index * chunks_per_phase + chunk_index.
        self.run_id_stride = 10 ** max(3, len(str(max_runs)))

        runs = [
            (ep_idx, ep_idx * self.run_id_stride + run_id, steps)
            for ep_idx, episode_runs in runs_by_episode.items()
            for run_id, steps in enumerate(episode_runs)
        ]

        # Shuffle so runs of one episode are not dealt to neighbouring batch slots: the
        # greedy assignment below walks this list in order, so leaving it grouped would
        # put the same demonstration's phases side by side in a batch again.
        order = self.rng.permutation(len(runs))
        return [runs[i] for i in order]

    def get_effective_episode_length(self, episode_index: int) -> int:
        """Get the effective episode length accounting for action horizon."""
        original_length = self.episode_loader.get_episode_length(episode_index)
        return max(0, original_length - self.action_horizon + 1)

    def _anchor_valid_mask(self, episode_index: int, effective_len: int) -> np.ndarray | None:
        """Return a bool mask of anchor positions where is_demo=False.

        For datasets without an `is_demo` column, returns None so the caller falls
        back to using all anchors.
        """
        if effective_len <= 0:
            return None
        chunk_idx = episode_index // self.episode_loader.chunk_size
        parquet_filename = self.episode_loader.data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=episode_index
        )
        parquet_path = self.episode_loader.dataset_path / parquet_filename
        try:
            col_df = pd.read_parquet(parquet_path, columns=["is_demo"])
        except (KeyError, ValueError, FileNotFoundError):
            return None
        flags = col_df["is_demo"].to_numpy().astype(bool)
        if flags.ndim > 1:
            flags = flags.reshape(-1)
        # Restrict to effective length (drop tail accounting for action horizon).
        flags = flags[:effective_len]
        return ~flags

    def __len__(self):
        """Return the number of shards in the dataset."""
        return len(self.shard_lengths)

    def get_datapoint(self, episode_data: pd.DataFrame, step_index: int) -> dict:
        """
        Extract and process a single timestep from episode data.

        Converts raw episode data into a VLAStepData structure and applies
        the configured processor to create model-ready inputs.

        Args:
            episode_data: Complete episode DataFrame from LeRobotEpisodeLoader
            step_index: Timestep index within the episode to extract

        Returns:
            Processed datapoint ready for model training

        Raises:
            AssertionError: If processor is not set before calling this method
        """
        assert self.processor is not None, "Processor must be set before getting datapoints"
        vla_step_data = extract_step_data(
            episode_data,
            step_index,
            self.modality_configs,
            self.embodiment_tag,
            self.allow_padding,
        )
        # Apply processor to convert to model inputs
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        return self.processor(messages)

    def get_shard_length(self, idx: int) -> int:
        """Get the number of timesteps in a specific shard."""
        return self.shard_lengths[idx]

    def get_shard_run_lengths(self, idx: int) -> list[int]:
        """Lengths of the contiguous per-episode runs that `get_shard(idx)` concatenates.

        Pure metadata (no episode loading), so callers can recover the episode
        boundaries inside the flat datapoint list returned by `get_shard`.
        """
        return [len(step_indices) for _, _, step_indices in self.sharded_episodes[idx]]

    def get_shard(self, idx: int) -> list:
        """
        Load and process all timesteps in a specific shard.

        Loads the required episodes and extracts all timesteps assigned to this shard,
        applying the configured processor to each timestep.

        Args:
            idx: Shard index to load

        Returns:
            List of processed timesteps ready for model training
        """
        episodes = self.sharded_episodes[idx]
        # Tag each datapoint with its provenance. The batch image dump (VIZ_BATCH_DIR)
        # names files by it, and the model's key-moment gate keys its state cache by
        # episode and uses the step index to detect episode restarts (epoch wrap).
        # The tag is the VIRTUAL episode index (see `_build_runs`), so each chunk is a
        # separate demonstration as far as the model's per-episode state is concerned.
        # Step indices stay absolute within the real episode: they only have to be
        # strictly increasing inside a run, which chunking preserves.
        datapoints = []
        for ep_idx, virtual_ep_idx, step_indices in episodes:
            # Load episode data once per episode in shard
            episode_data = self.episode_loader[ep_idx]
            for step_index in step_indices:
                datapoint = self.get_datapoint(episode_data, step_index)
                datapoint["_viz_episode_index"] = int(virtual_ep_idx)
                datapoint["_viz_step_index"] = int(step_index)
                datapoints.append(datapoint)
        return datapoints

    def get_dataset_statistics(self) -> dict:
        """Get dataset statistics from the underlying episode loader."""
        return self.episode_loader.get_dataset_statistics()

    def get_initial_actions(self):
        """Get initial actions from the underlying episode loader."""
        return self.episode_loader.get_initial_actions()
