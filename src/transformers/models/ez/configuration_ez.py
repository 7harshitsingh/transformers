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

from huggingface_hub.dataclasses import strict

from ...utils import auto_docstring
from ..nanochat.configuration_nanochat import NanoChatConfig


@auto_docstring(checkpoint="7harshitsingh/ez1-e129")
@strict
class EZConfig(NanoChatConfig):
    r"""
    Args:
    window_pattern (`str`, *optional*, defaults to `"SSSL"`):
        Sliding window attention pattern tiled across layers. 'L' = full context, 'S' = quarter context.
        Last layer always gets full context regardless of pattern.
    smear_gate_in_features (`int`, *optional*, defaults to `24`):
        Input feature width for the smear gate projection Linear(smear_gate_in_features, 1).

    Example:

    ```python
    >>> from transformers import EZModel, EZConfig

    >>> # Initializing an EZ style configuration
    >>> configuration = EZConfig()

    >>> # Initializing a model from the EZ style configuration
    >>> model = EZModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ez"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Inherited from NanoChatConfig and unchanged:
    #   vocab_size, hidden_size, intermediate_size, num_hidden_layers,
    #   num_attention_heads, num_key_value_heads, max_position_embeddings,
    #   hidden_act, attention_dropout, rms_norm_eps, initializer_range,
    #   rope_parameters, use_cache, final_logit_softcapping, attention_bias,
    #   bos_token_id, eos_token_id, pad_token_id, tie_word_embeddings

    # EZ-specific fields
    # head_dim is independent of hidden_size // num_attention_heads
    head_dim: int = 64
    # Sliding window attention pattern string, tiled across layers.
    # Characters: L = full context, S = quarter context.
    # Last layer always gets L regardless of pattern.
    # Examples: "L" = all full, "SL" = alternating, "SSSL" = three short then one long.
    window_pattern: str = "SSSL"
    # Input feature width for the smear gate projection: Linear(smear_gate_in_features, 1)
    smear_gate_in_features: int = 24

    # EZ overrides for defaults that differ from NanoChat
    vocab_size: int = 65536
    hidden_size: int = 576
    intermediate_size: int = 1728          # always 3 * hidden_size
    num_hidden_layers: int = 32
    num_attention_heads: int = 24
    num_key_value_heads: int | None = 4
    max_position_embeddings: int = 8192
    rope_parameters: RopeParameters | dict | None = None  # keep as inherited

    def __post_init__(self, **kwargs):
        if self.intermediate_size != 3 * self.hidden_size:
            raise ValueError(
                f"EZConfig requires intermediate_size == 3 * hidden_size, "
                f"got intermediate_size={self.intermediate_size} and hidden_size={self.hidden_size}."
            )
        # Set correct rope_theta if not already specified
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 100000.0}
        elif isinstance(self.rope_parameters, dict) and "rope_theta" not in self.rope_parameters:
            self.rope_parameters["rope_theta"] = 100000.0

        super().__post_init__(**kwargs)


__all__ = ["EZConfig"]