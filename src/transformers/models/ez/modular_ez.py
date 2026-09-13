# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Callable

import torch
import torch.nn as nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, create_causal_mask, flash_attention_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..nanochat.modeling_nanochat import (
    NanoChatAttention,
    NanoChatDecoderLayer,
    NanoChatForCausalLM,
    NanoChatMLP,
    NanoChatModel,
    NanoChatPreTrainedModel,
    NanoChatRMSNorm,
    NanoChatRotaryEmbedding,
    eager_attention_forward,
)
from .configuration_ez import EZConfig


logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pass-through subclasses — inherited unchanged
# ---------------------------------------------------------------------------

class EZRMSNorm(NanoChatRMSNorm):
    pass


class EZRotaryEmbedding(NanoChatRotaryEmbedding):
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Override to match nanochat exact cos/sin shape: (B, T, 1, head_dim//2).
        NanoChatRotaryEmbedding (via LlamaRotaryEmbedding) returns (B, T, head_dim)
        via cat(freqs, freqs). We return raw freqs.cos()/sin() with the singleton
        head dim nanochat uses, so EZAttention broadcasts over (B, T, H, head_dim//2)
        without any slicing.
        """
        inv_freq = self.inv_freq.to(x.device).float()
        # Per-row positions: with left padding every row has its own offsets, so using
        # position_ids[0] for the whole batch would misplace every other row's tokens.
        freqs = position_ids.float()[:, :, None] * inv_freq[None, None, :]   # (B, T, head_dim//2)
        cos = (freqs.cos() * self.attention_scaling).to(dtype=x.dtype)
        sin = (freqs.sin() * self.attention_scaling).to(dtype=x.dtype)
        # (B, T, 1, head_dim//2) — broadcasts over (B, T, H, head_dim//2)
        return cos[:, :, None, :], sin[:, :, None, :]


class EZMLP(NanoChatMLP):
    pass


# ---------------------------------------------------------------------------
# Attention — adds Q/K * 1.2 scale after QK norm + per-layer sliding window
# ---------------------------------------------------------------------------

class EZAttention(NanoChatAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        window_size: tuple[int, int] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Keep in (B, T, H, D) layout to apply RoPE — matches nanochat's apply_rotary_emb
        query_states = self.q_proj(hidden_states).view(hidden_shape)   # (B, T, H, D)
        key_states = self.k_proj(hidden_states).view(hidden_shape)     # (B, T, H, D)
        value_states = self.v_proj(hidden_states).view(hidden_shape)   # (B, T, H, D)

        # Apply RoPE in (B, T, H, D) layout — matches nanochat apply_rotary_emb exactly.
        # EZRotaryEmbedding.forward returns cos/sin of shape (B, T, 1, head_dim//2),
        # which broadcasts directly over (B, T, H, head_dim//2) — no unsqueeze needed.
        cos, sin = position_embeddings   # each (B, T, 1, head_dim//2)
        half = cos.shape[-1]             # head_dim // 2
        q1, q2 = query_states[..., :half], query_states[..., half:]
        k1, k2 = key_states[..., :half], key_states[..., half:]
        query_states = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=3)
        key_states = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=3)

        # QK norm in (B, T, H, D) layout, then transpose to (B, H, T, D) for attention
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose to (B, H, T, D) for attention interface
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Sharper attention: split scale between Q and K
        query_states = query_states * 1.2
        key_states = key_states * 1.2

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Sliding window: keep only keys at most `left` positions before each query.
        # Matches nanochat FA3 window_size=(left, 0), whose bounds are inclusive.
        # The mask built here is itself causal, so it is safe even when the library handed
        # us None because sdpa could otherwise have taken its is_causal fast path.
        if window_size is not None:
            left, _ = window_size
            T_q, T_kv = query_states.shape[2], key_states.shape[2]
            if left > 0 and T_kv > left + 1:
                impl = (self.config._attn_implementation or "").removeprefix("paged|")
                mask_fn = ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.get(impl)
                # hub kernel names register lazily, so fall back to a name check
                using_flash = mask_fn is flash_attention_mask or "flash" in impl.lower()
                if using_flash:
                    # flash handles the window natively; HF maps sliding_window -> (sliding_window - 1, 0),
                    # so +1 lands on the inclusive (left, 0) window used in training
                    kwargs["sliding_window"] = left + 1
                elif attention_mask is None or isinstance(attention_mask, torch.Tensor):
                    device = query_states.device
                    q_positions = torch.arange(T_kv - T_q, T_kv, device=device)[:, None]
                    k_positions = torch.arange(T_kv, device=device)[None, :]
                    keep = ((k_positions <= q_positions) & ((q_positions - k_positions) <= left))[None, None]
                    if attention_mask is None:
                        attention_mask = torch.zeros(
                            1, 1, T_q, T_kv, dtype=query_states.dtype, device=device
                        ).masked_fill(~keep, torch.finfo(query_states.dtype).min)
                    elif attention_mask.dtype == torch.bool:
                        attention_mask = attention_mask & keep
                    else:
                        attention_mask = attention_mask.masked_fill(~keep, torch.finfo(attention_mask.dtype).min)
                # else: flex_attention BlockMask — left unchanged, i.e. full context on S layers

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# DecoderLayer — accepts window_size and passes it through to attention
# ---------------------------------------------------------------------------

class EZDecoderLayer(NanoChatDecoderLayer):
    def __init__(self, config: EZConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = EZAttention(config=config, layer_idx=layer_idx)
        self.mlp = EZMLP(config)
        self.input_layernorm = EZRMSNorm(eps=config.rms_norm_eps)
        self.post_attention_layernorm = EZRMSNorm(eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        window_size: tuple[int, int] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            window_size=window_size,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# PreTrainedModel — extends weight init to cover EZ scalar params
# ---------------------------------------------------------------------------

@auto_docstring
class EZPreTrainedModel(NanoChatPreTrainedModel):
    config_class = EZConfig
    _tied_weights_keys = []     # no weight tying (tie_word_embeddings = False)

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, nn.Linear) and getattr(module, "_is_smear_gate", False):
            nn.init.uniform_(module.weight, 0.0, 0.02)


# ---------------------------------------------------------------------------
# Model — full forward override to add smear, resid/x0 scalars, backout
# ---------------------------------------------------------------------------

@auto_docstring
class EZModel(NanoChatModel):
    def __init__(self, config: EZConfig):
        super().__init__(config)

        # Replace decoder layers with EZ variants
        self.layers = nn.ModuleList(
            [EZDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Per-layer learnable scalars
        self.resid_lambdas = nn.Parameter(torch.ones(config.num_hidden_layers))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.num_hidden_layers))

        # Smear: mix previous token embedding into current position (cheap bigram info)
        self.smear_gate = nn.Linear(config.smear_gate_in_features, 1, bias=False)
        self.smear_gate._is_smear_gate = True
        self.smear_lambda = nn.Parameter(torch.zeros(1))

        # Backout: subtract mid-layer residual before final norm
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

        # Per-layer window sizes derived from window_pattern
        self._window_sizes = self._compute_window_sizes(config)

    @staticmethod
    def _compute_window_sizes(config: EZConfig) -> list[tuple[int, int]]:
        """
        Returns list of (left, right) window size tuples, one per layer.
        left=-1 means full context. Last layer always gets full context.
        Short window is ceiling-rounded to 128 (FA3 tile alignment).
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), (
            f"Invalid window_pattern '{pattern}'. Only 'S' and 'L' are allowed."
        )
        long_win = config.max_position_embeddings
        short_win = math.ceil(long_win / 4 / 128) * 128
        char_to_win = {"L": (long_win, 0), "S": (short_win, 0)}
        sizes = [char_to_win[pattern[i % len(pattern)]] for i in range(config.num_hidden_layers)]
        sizes[-1] = (long_win, 0)   # last layer always full context
        return sizes

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            ).unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        # Initial norm before trunk (matches training)
        hidden_states = self.norm(inputs_embeds)

        # ------------------------------------------------------------------
        # Smear: mix previous token embedding into current position
        # ------------------------------------------------------------------
        # Applied at every position — prefill, cached decode and chunked prefill alike:
        # the predecessor of local position 0 comes from the cache when there is one.
        T = hidden_states.shape[1]
        prev = getattr(past_key_values, "ez_prev_embed", None) if past_key_values is not None else None
        if past_key_values is not None:
            # stash the PRE-smear normalised embedding of the last token, as training does
            past_key_values.ez_prev_embed = hidden_states[:, -1:].clone()
        if prev is None:
            prev = torch.zeros_like(hidden_states[:, :1])   # position 0 has no predecessor
        prev_all = torch.cat([prev.to(hidden_states.dtype), hidden_states[:, :-1]], dim=1)
        if (
            attention_mask is not None
            and isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim == 2
            and attention_mask.shape[1] >= T
        ):
            # left padding: never smear a pad token's embedding into the first real token
            mask_2d = attention_mask.to(hidden_states.dtype)
            start = mask_2d.shape[1] - T                    # global index of local position 0
            idx = torch.arange(start - 1, start + T - 1, device=mask_2d.device)
            valid = (idx >= 0).to(mask_2d.dtype)
            prev_all = prev_all * (mask_2d[:, idx.clamp(min=0)] * valid)[:, :, None]
        gate = self.smear_lambda.to(hidden_states.dtype) * torch.sigmoid(
            self.smear_gate(hidden_states[:, :, : self.config.smear_gate_in_features])
        )
        hidden_states = hidden_states + gate * prev_all

        # ------------------------------------------------------------------
        # Transformer trunk with per-layer resid/x0 scalars, window, backout
        # ------------------------------------------------------------------
        x0 = hidden_states
        backout_layer_idx = self.config.num_hidden_layers // 2
        x_backout = None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = (
                self.resid_lambdas[i].to(hidden_states.dtype) * hidden_states
                + self.x0_lambdas[i].to(hidden_states.dtype) * x0
            )

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                window_size=self._window_sizes[i],
                **kwargs,
            )

            if i == backout_layer_idx:
                x_backout = hidden_states

        # Backout: subtract mid-layer residual
        if x_backout is not None:
            hidden_states = hidden_states - self.backout_lambda.to(hidden_states.dtype) * x_backout

        # Final norm
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# ---------------------------------------------------------------------------
# ForCausalLM — inherited unchanged
# ---------------------------------------------------------------------------

@auto_docstring
class EZForCausalLM(NanoChatForCausalLM):
    _tied_weights_keys = []   # no weight tying — tie_word_embeddings = False

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained("7harshitsingh/ez1-e129")

        >>> tokenizer = AutoTokenizer.from_pretrained("7harshitsingh/ez1-e129")

        >>> conversation = [
        ...     {"role": "user", "content": "What is the capital of France?"},
        ... ]

        >>> inputs = tokenizer.apply_chat_template(
        ...     conversation, add_generation_prompt=True, tokenize=True,
        ...     return_dict=True, return_tensors="pt"
        ... ).to(device)

        >>> with torch.no_grad():
        ...     outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

        >>> generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        >>> output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # Upcast before the softcap, as training does: in bf16 the spacing at |x| ~ 15 is 0.0625,
        # which is the range softcapped logits live in.
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "EZPreTrainedModel",
    "EZModel",
    "EZForCausalLM",
]