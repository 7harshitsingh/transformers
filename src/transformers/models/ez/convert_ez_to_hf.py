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

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from transformers import AutoTokenizer, EZConfig, EZForCausalLM


# ---------------------------------------------------------------------------
# Unchanged from NanoChat converter
# ---------------------------------------------------------------------------

def convert_layer(old_prefix: str, new_prefix: str) -> dict[str, str]:
    return {
        f"{old_prefix}.attn.c_q.weight": f"{new_prefix}.self_attn.q_proj.weight",
        f"{old_prefix}.attn.c_k.weight": f"{new_prefix}.self_attn.k_proj.weight",
        f"{old_prefix}.attn.c_v.weight": f"{new_prefix}.self_attn.v_proj.weight",
        f"{old_prefix}.attn.c_proj.weight": f"{new_prefix}.self_attn.o_proj.weight",
        f"{old_prefix}.mlp.c_fc.weight":   f"{new_prefix}.mlp.fc1.weight",
        f"{old_prefix}.mlp.c_proj.weight": f"{new_prefix}.mlp.fc2.weight",
    }


def assign(old_key, new_key, old_state, state_dict, rename_map):
    tensor = old_state.get(old_key)
    if tensor is None:
        return
    state_dict[new_key] = tensor.clone()
    rename_map[old_key] = new_key


# ---------------------------------------------------------------------------
# EZ-specific: reads model_config block which has all fields we need
# ---------------------------------------------------------------------------

def load_ez_config(input_path: Path) -> EZConfig:
    meta_files = list(input_path.glob("meta_*.json"))
    if not meta_files:
        raise ValueError(f"No meta_*.json found in {input_path}")

    meta_file = meta_files[0]
    print(f"Loading config from {meta_file.name}")
    with open(meta_file) as f:
        meta = json.load(f)

    mc = meta["model_config"]

    return EZConfig(
        vocab_size=mc["vocab_size"],
        hidden_size=mc["n_embd"],
        intermediate_size=3 * mc["n_embd"],    # always 3 * hidden_size for EZ
        num_hidden_layers=mc["n_layer"],
        num_attention_heads=mc["n_head"],
        num_key_value_heads=mc["n_kv_head"],
        max_position_embeddings=mc["sequence_len"],
        head_dim=mc["head_dim"],
        window_pattern=mc["window_pattern"],
    )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def write_model(input_dir, output_dir):
    print("Converting EZ model.")
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_dir)

    config = load_ez_config(input_path)
    print(
        f"Config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, "
        f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
        f"head_dim={config.head_dim}, window_pattern={config.window_pattern}"
    )

    # Load raw checkpoint
    checkpoint_files = list(input_path.glob("model_*.pt"))
    if not checkpoint_files:
        raise ValueError(f"No model_*.pt found in {input_path}")
    checkpoint_path = checkpoint_files[0]
    print(f"Loading checkpoint from {checkpoint_path}...")
    old_state = torch.load(checkpoint_path, map_location="cpu")

    # Cast fp32 tensors to bfloat16
    for key in old_state:
        if old_state[key].dtype == torch.float32:
            old_state[key] = old_state[key].to(torch.bfloat16)

    print("Remapping keys...")
    state_dict = {}
    rename_map = {}

    # Embedding and lm_head
    assign("transformer.wte.weight", "model.embed_tokens.weight", old_state, state_dict, rename_map)
    assign("lm_head.weight",         "lm_head.weight",            old_state, state_dict, rename_map)

    # Transformer layers
    for i in range(config.num_hidden_layers):
        for old_key, new_key in convert_layer(f"transformer.h.{i}", f"model.layers.{i}").items():
            assign(old_key, new_key, old_state, state_dict, rename_map)

    # EZ scalar parameters — names unchanged, just move under model.*
    ez_scalars = {
        "resid_lambdas":    "model.resid_lambdas",
        "x0_lambdas":       "model.x0_lambdas",
        "smear_lambda":     "model.smear_lambda",
        "smear_gate.weight":"model.smear_gate.weight",
        "backout_lambda":   "model.backout_lambda",
    }
    for old_key, new_key in ez_scalars.items():
        assign(old_key, new_key, old_state, state_dict, rename_map)

    skipped = [k for k in old_state if k not in rename_map]
    if skipped:
        print(f"Skipped {len(skipped)} keys with no EZ equivalent: {skipped}")

    del old_state
    gc.collect()

    config.torch_dtype = torch.bfloat16
    config.tie_word_embeddings = False

    print("Loading state dict into EZForCausalLM...")
    with torch.device("meta"):
        model = EZForCausalLM(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    print("Loaded successfully.")

    if hasattr(model.config, "_name_or_path"):
        del model.config._name_or_path

    print("Saving model...")
    model.save_pretrained(output_dir)
    del state_dict, model
    gc.collect()

    print("Reloading to verify...")
    EZForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


# ---------------------------------------------------------------------------
# Tokenizer — unchanged from NanoChat converter
# ---------------------------------------------------------------------------

def write_tokenizer(input_dir, output_dir):
    input_path = Path(input_dir)
    tokenizer_pkl = input_path / "tokenizer.pkl"

    if tokenizer_pkl.exists():
        try:
            import pickle
            from transformers.integrations.tiktoken import convert_tiktoken_to_fast
            with open(tokenizer_pkl, "rb") as f:
                tok_pkl = pickle.load(f)
            convert_tiktoken_to_fast(tok_pkl, output_dir)
            print("Converted tokenizer.pkl to HuggingFace format.")
        except Exception as e:
            print(f"Warning: Failed to convert tokenizer.pkl: {e}")
            for filename in ("tokenizer.json", "tokenizer_config.json"):
                src = input_path / filename
                if src.exists():
                    (Path(output_dir) / filename).write_bytes(src.read_bytes())
    else:
        for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src = input_path / filename
            if src.exists():
                (Path(output_dir) / filename).write_bytes(src.read_bytes())

    print("Tokenizer saved.")


# ---------------------------------------------------------------------------
# Quick generation test
# ---------------------------------------------------------------------------

def run_test(output_dir, prompt, max_new_tokens=64):
    print(f"Testing with prompt: {prompt}")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = EZForCausalLM.from_pretrained(output_dir, dtype=torch.bfloat16)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.25,
            top_k=40,
            repetition_penalty=1.5,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(generated, skip_special_tokens=False))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert EZ checkpoint to HuggingFace format")
    parser.add_argument("--input_dir",  required=True, help="Directory containing model_*.pt and meta_*.json")
    parser.add_argument("--output_dir", required=True, help="Where to write the HF model")
    parser.add_argument("--test_prompt", default=None, help="Optional prompt to test the converted model")
    args = parser.parse_args()

    write_model(args.input_dir, args.output_dir)
    write_tokenizer(args.input_dir, args.output_dir)

    if args.test_prompt:
        run_test(args.output_dir, args.test_prompt)


if __name__ == "__main__":
    main()