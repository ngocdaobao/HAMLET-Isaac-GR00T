import os

import torch
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature
import logging

logger = logging.getLogger(__name__)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary embedding on (B, H, L, Dh) with (B, 1, L, Dh) cos/sin."""
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


class EagleBackbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Eagle-Block2A-2B-v2",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict = {},
        n_moment_tokens: int = 0,
        freeze_moment_tokens: bool = False,
        memory_type: str = "moment_token",
        memory_mode: str = "window",
    ):
        """
        EagleBackbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Eagle-Block2A-2B-v2
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
            n_moment_tokens: HAMLET — if >0, append this many learnable moment tokens
                at the tail of the VLM input. They attend to image+text via causal
                self-attention and emerge in the VLM output as the m'_t slice.
        """

        super().__init__()

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            extra_kwargs["attn_implementation"] = "flash_attention_2"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        if model_name == "nvidia/Eagle-Block2A-2B-v2":
            assert use_flash_attention, (
                "nvidia/Eagle-Block2A-2B-v2 requires flash attention by default"
            )
            assert load_bf16, "nvidia/Eagle-Block2A-2B-v2 requires bfloat16 by default"
            eagle_path = os.path.join(os.path.dirname(__file__), "nvidia", "Eagle-Block2A-2B-v2")
            config = AutoConfig.from_pretrained(eagle_path, trust_remote_code=True)
            self.model = AutoModel.from_config(config, trust_remote_code=True)
        else:
            raise ValueError(f"Model {model_name} not supported")

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.model.layers) > select_layer:
            self.model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.memory_type = memory_type
        # "zoo": single-observation-per-iteration memory with an attention-saliency cache.
        # Only this mode pays for the extra last-layer attention read-out.
        self.memory_mode = memory_mode

        # HAMLET moment tokens, created here in __init__ so `from_pretrained` does not
        # leave them uninitialized. Stored on the backbone for use by the TCL (Stage-1)
        # and finetune (Stage-2) paths.
        self.n_moment_tokens = n_moment_tokens # number of moment tokens
        if n_moment_tokens > 0:
            hidden_size = self.model.config.text_config.hidden_size
            self.moment_tokens = torch.nn.Parameter(0.02 * torch.randn(n_moment_tokens, hidden_size))

        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if n_moment_tokens > 0 and freeze_moment_tokens:
            self.moment_tokens.requires_grad_(False)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    print(f"Casting trainable parameter {n} to fp32")

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.vision_model.requires_grad_(False)
            self.model.mlp1.requires_grad_(False)

        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.vision_model and not self.tune_visual:
                self.model.vision_model.eval()
                self.model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # 0. Set frozen module to eval
        keys_to_use = ["input_ids", "attention_mask", "pixel_values"]
        vl_input = {k: vl_input[k] for k in keys_to_use if k in vl_input}

        if self.n_moment_tokens > 0:
            return self._forward_with_moment_tokens(vl_input)

        outputs = self.model(**vl_input, output_hidden_states=True)
        outputs = outputs["hidden_states"][-1]
        image_mask = vl_input["input_ids"] == self.model.config.image_token_index
        attention_mask = vl_input["attention_mask"] == 1
        data = {
            "backbone_features": outputs,
            "backbone_attention_mask": attention_mask,
            "image_mask": image_mask,
        }
        if self.memory_type == "vision_feature" and vl_input.get("pixel_values") is not None:
            # HAMLET vision_feature path: expose the PRIMARY (first) view's post-LLM image
            # tokens, avg-pooled to 64/step, for the memory module. Eagle packs each view as
            # a separate image -> n_views = pixel_values rows / batch; tokens/view inferred
            # from the (uniform) per-row image-token count (square grid).
            from gr00t.model.modules.memory import pool_primary_view

            B = vl_input["input_ids"].shape[0]
            n_views = max(1, vl_input["pixel_values"].shape[0] // B)
            total = int(image_mask[0].sum().item())
            tpv = total // n_views
            side = int(round(tpv**0.5))
            if side * side == tpv and tpv > 0:
                data["primary_view_feature"] = pool_primary_view(outputs, image_mask, tpv, (side, side))
        return BatchFeature(data=data)  # [B, T2, hidden_size]

    def _forward_with_moment_tokens(self, vl_input: dict) -> BatchFeature:
        """HAMLET forward for the Eagle backbone.

        Follows the Eagle3-VL text/image splicing, then appends `moment_tokens` at the
        tail of the spliced embeddings before calling `language_model` directly with
        `inputs_embeds`. The last `n_moment_tokens` rows of the LM hidden state are the
        moment-token outputs m'_t.
        """
        input_ids = vl_input["input_ids"]
        attention_mask = vl_input["attention_mask"]
        pixel_values = vl_input["pixel_values"]
        # episode_idx = vl_input["_viz_episode_index"] 
        eagle = self.model  # Eagle3_VL

        # Embed text tokens.
        input_embeds = eagle.language_model.get_input_embeddings()(input_ids)

        # Extract visual features and splice them into the image placeholder slots,
        # matching Eagle3_VL.forward (modeling_eagle3_vl.py:231-255).
        vit_embeds = eagle.extract_feature(pixel_values)
        B, N, C = input_embeds.shape
        input_embeds_flat = input_embeds.reshape(B * N, C)
        input_ids_flat = input_ids.reshape(B * N)
        selected = input_ids_flat == eagle.image_token_index
        try:
            input_embeds_flat[selected] = input_embeds_flat[selected] * 0.0 + vit_embeds
        except Exception as e:
            print(
                f"warning: {e}, input_embeds_flat[selected].shape={input_embeds_flat[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds_flat[selected] = input_embeds_flat[selected] * 0.0 + vit_embeds[:n_token]
        input_embeds = input_embeds_flat.reshape(B, N, C)

        # Append moment tokens at the tail.
        n_q = self.n_moment_tokens
        meta = self.moment_tokens.unsqueeze(0).expand(B, -1, -1).to(input_embeds.dtype)
        input_embeds = torch.cat([input_embeds, meta], dim=1)

        extra_mask = torch.ones(B, n_q, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask_ext = torch.cat([attention_mask, extra_mask], dim=1)

        # Moment tokens sit after right-padding in sequence order, but their RoPE
        # positions continue from each sample's REAL length (= attention_mask.sum)
        # so train-time (padded batch) and inference-time (no padding) relative
        # distances to the content tokens match exactly. The prefix uses arange,
        # replicating HF's default when position_ids is None.
        seq_len = input_ids.shape[1]
        prefix_pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(B, -1)
        real_len = attention_mask.sum(dim=1, keepdim=True).to(torch.long)  # (B, 1)
        mq_pos = real_len + torch.arange(n_q, device=input_ids.device).unsqueeze(0)
        position_ids = torch.cat([prefix_pos, mq_pos], dim=1)

        outputs = eagle.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask_ext,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )

        last_hidden = outputs.hidden_states[-1]  # (B, N+n_q, d)

        image_mask = input_ids == eagle.image_token_index
        image_mask = torch.cat(
            [image_mask, torch.zeros(B, n_q, dtype=image_mask.dtype, device=image_mask.device)],
            dim=1,
        )

        data = {
            "backbone_features": last_hidden,
            "backbone_attention_mask": attention_mask_ext == 1,
            "image_mask": image_mask,
            "n_moment_tokens": n_q,
            # "episode_index": episode_idx,
        }

        if self.memory_mode == "zoo":
            # Saliency descriptor for the zoo memory cache: how the moment tokens spread
            # their last-layer attention over the image tokens.
            # NOTE: `output_attentions=True` is NOT usable here. Eagle3-VL asserts
            # flash_attention_2 for the LM, and FA2 never materializes the attention
            # matrix, so HF returns None for `outputs.attentions` (or silently falls back
            # to eager, which materializes a (B, H, L, L) tensor per layer -> OOM at these
            # sequence lengths). Instead recompute the final layer's attention exactly,
            # for the n_q moment-token query rows only.
            attn_map = self._moment_to_image_attention(
                hidden_in=outputs.hidden_states[-2],
                position_ids=position_ids,
                attention_mask_ext=attention_mask_ext,
                image_mask_ext=image_mask,
                n_q=n_q,
            )
            if attn_map is None:
                # Fallback descriptor: L2-normalized mean image-token hidden state. Not an
                # attention map, but it still measures "how much the observation changed".
                img_feat = last_hidden[image_mask.bool()].view(B, -1, last_hidden.shape[-1])
                attn_map = torch.nn.functional.normalize(
                    img_feat.float().mean(dim=1), dim=-1
                ).unsqueeze(1)
            data["mem_attn_score"] = attn_map.detach()  # (B, n_q, n_img) or (B, 1, d)

        return BatchFeature(data=data)

    @torch.no_grad()
    def _moment_to_image_attention(
        self,
        hidden_in: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask_ext: torch.Tensor,
        image_mask_ext: torch.Tensor,
        n_q: int,
    ):
        """Exact last-decoder-layer attention from the moment tokens to the image tokens.

        `hidden_in` is hidden_states[-2] -- the input to the final decoder layer -- so
        re-running that layer's input_layernorm and q/k projections reproduces its
        attention logits exactly. Only n_q query rows are computed, so this costs one
        (B, H, n_q, L) matmul instead of a full (B, H, L, L) attention matrix.

        Returns (B, n_q, n_img) probabilities renormalized over the image tokens, or None
        if the LM internals don't match the expected Qwen-style layout.
        """
        try:
            lm = self.model.language_model.model
            layer = lm.layers[-1]
            attn = layer.self_attn

            h = layer.input_layernorm(hidden_in)  # (B, L, C)
            B, L, _ = h.shape
            head_dim = getattr(attn, "head_dim", None)
            if head_dim is None:
                head_dim = attn.q_proj.out_features // attn.config.num_attention_heads

            q = attn.q_proj(h[:, -n_q:, :]).view(B, n_q, -1, head_dim)
            k = attn.k_proj(h).view(B, L, -1, head_dim)
            # Qwen3 normalizes each head before RoPE; Qwen2/Llama have no q_norm/k_norm.
            if getattr(attn, "q_norm", None) is not None:
                q = attn.q_norm(q)
            if getattr(attn, "k_norm", None) is not None:
                k = attn.k_norm(k)
            q = q.transpose(1, 2)  # (B, H, n_q, Dh)
            k = k.transpose(1, 2)  # (B, H_kv, L, Dh)

            rotary = getattr(lm, "rotary_emb", None)
            if rotary is not None:
                cos, sin = rotary(h, position_ids)  # (B, L, Dh)
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # (B, 1, L, Dh)
                q = _apply_rope(q, cos[:, :, -n_q:, :], sin[:, :, -n_q:, :])
                k = _apply_rope(k, cos, sin)

            # GQA: broadcast the kv heads up to the query heads.
            n_rep = q.shape[1] // k.shape[1]
            if n_rep > 1:
                k = k.repeat_interleave(n_rep, dim=1)

            logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / (head_dim**0.5)
            # The moment tokens sit after the right padding, so pad columns must be masked
            # out before the softmax or they steal probability mass.
            logits = logits.masked_fill(
                ~attention_mask_ext.bool()[:, None, None, :], torch.finfo(logits.dtype).min
            )
            probs = torch.softmax(logits, dim=-1).mean(dim=1)  # over heads -> (B, n_q, L)

            img_mask = image_mask_ext.bool()
            n_img = int(img_mask[0].sum().item())
            if n_img == 0 or not bool((img_mask.sum(dim=1) == n_img).all()):
                return None  # ragged image-token counts -> no fixed-width descriptor
            img_probs = probs[img_mask.unsqueeze(1).expand(-1, n_q, -1)].view(B, n_q, n_img)
            # Renormalize over image tokens so the L1 distance between two timesteps
            # reflects *where* attention moved, not the text/image mass ratio.
            return img_probs / (img_probs.sum(dim=-1, keepdim=True) + 1e-8)
        except Exception as e:  # depends on the installed transformers internals
            logger.warning(
                f"[zoo] moment->image attention read-out unavailable "
                f"({type(e).__name__}: {e}); using hidden-state descriptor instead."
            )
            return None
