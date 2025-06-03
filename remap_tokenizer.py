#!/usr/bin/env python
"""
Remap a source Hugging Face causal-LM checkpoint so that its embedding/lm_head rows
follow the base tokenizer's vocabulary order. Any tokens missing from the base
tokenizer are either dropped (with --drop-extra) or appended to the end.

Usage
-----
python remap_tokenizer.py \
    --src_model  tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4 \
    --base_tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --out_dir   ./swallow-tokenfixed \
    [--drop-extra]
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(name_or_path: str) -> AutoTokenizer:
    """
    Enables remote code execution.
    """
    try:
        return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer '{name_or_path}'") from e


def main(
    src_model: str,
    base_tokenizer: str,
    out_dir: str,
    shard_size: str = "5GB",
    dtype_str: str = "bfloat16",
    device: str = "cpu",
    drop_extra: bool = False,
) -> None:
    """
    Remap the input embeddings and language modeling head of src_model so that
    the vocabulary order matches base_tokenizer. Save the remapped model and
    updated tokenizer to out_dir.
    """
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load tokenizers -------------------------------------------------------
    print(f"Loading tokenizers\n  source: {src_model}\n  base:   {base_tokenizer}")
    tokenizer_src = load_tokenizer(src_model)
    tokenizer_base = load_tokenizer(base_tokenizer)

    src_vocab = tokenizer_src.get_vocab()  # token -> id
    src_tokens = list(src_vocab.keys())
    base_vocab_set = set(tokenizer_base.get_vocab().keys())

    # Identify tokens in source but missing from base
    extra_tokens: List[str] = [tok for tok in src_tokens if tok not in base_vocab_set]

    if drop_extra:
        if extra_tokens:
            print(f"Pruning {len(extra_tokens)} tokens not present in base tokenizer.")
            print("Dropped tokens:")
            for tok in extra_tokens:
                print(f"  {tok}")
        else:
            print("No extra tokens found; vocabularies already aligned.")
    else:
        if extra_tokens:
            n_added = tokenizer_base.add_tokens(extra_tokens)
            print(f"Added {n_added} tokens to base tokenizer (new size = {len(tokenizer_base)}).")
        else:
            print("No extra tokens needed; vocabularies already aligned.")

    # 2. Load source model ------------------------------------------------------
    print("Loading source model (this may take a while)...")
    torch_dtype = getattr(torch, dtype_str)
    model = AutoModelForCausalLM.from_pretrained(
        src_model,
        torch_dtype=torch_dtype,
        device_map={"": device},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    embed_weights = model.get_input_embeddings().weight  # [orig_vocab_size, hidden_dim]
    lm_weights = model.lm_head.weight  # [orig_vocab_size, hidden_dim]
    hidden_dim = embed_weights.shape[1]

    # 3. Build new embedding and lm_head ----------------------------------------
    new_vocab_size = len(tokenizer_base)
    device_embed = embed_weights.device
    dtype_embed = embed_weights.dtype

    new_embed = torch.zeros(new_vocab_size, hidden_dim, dtype=dtype_embed, device=device_embed)
    new_lm = torch.zeros_like(new_embed)

    # Prepare unknown token vectors (average or unk token from source)
    if tokenizer_src.unk_token_id is None:
        unk_vec_embed = embed_weights.mean(dim=0)
        unk_vec_lm = lm_weights.mean(dim=0)
    else:
        unk_id = tokenizer_src.unk_token_id
        unk_vec_embed = embed_weights[unk_id].clone()
        unk_vec_lm = lm_weights[unk_id].clone()

    with torch.no_grad():
        if drop_extra:
            # For each base token ID, copy from source if exists; else assign unknown vector
            for base_id in range(new_vocab_size):
                token_str = tokenizer_base.convert_ids_to_tokens(base_id)
                src_id: Optional[int] = src_vocab.get(token_str, None)
                if src_id is None:
                    new_embed[base_id] = unk_vec_embed
                    new_lm[base_id] = unk_vec_lm
                else:
                    new_embed[base_id] = embed_weights[src_id]
                    new_lm[base_id] = lm_weights[src_id]
        else:
            # Build a mapping from source IDs to their new positions in base
            src_vocab_size = len(tokenizer_src)
            id_map = torch.full((src_vocab_size,), -1, dtype=torch.long, device=device_embed)
            for src_id in range(src_vocab_size):
                token_str = tokenizer_src.convert_ids_to_tokens(src_id)
                dst_id = tokenizer_base.convert_tokens_to_ids(token_str)
                if dst_id is None or dst_id < 0:
                    raise ValueError(
                        f"Token '{token_str}' (id={src_id}) cannot be mapped to base tokenizer."
                    )
                id_map[src_id] = dst_id

            # Initialize all rows with the unknown vector in case the base tokenizer
            # contains tokens that are absent from the source model
            new_embed[:] = unk_vec_embed
            new_lm[:] = unk_vec_lm

            # Scatter the embeddings and lm weights to new tensors
            new_embed[id_map] = embed_weights
            new_lm[id_map] = lm_weights

    # Replace model embeddings and lm_head with the remapped versions
    model.set_input_embeddings(torch.nn.Embedding.from_pretrained(new_embed, freeze=False))
    model.lm_head.weight = torch.nn.Parameter(new_lm)
    model.config.vocab_size = new_vocab_size

    # If the model ties word embeddings, re-tie them
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    # 4. Save remapped model and updated tokenizer --------------------------------
    print(f"Saving remapped model and tokenizer to '{out_dir}'...")
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size=shard_size)
    tokenizer_base.save_pretrained(output_path)

    # Write remapping information
    info = {
        "src_model": src_model,
        "base_tokenizer": base_tokenizer,
        "extra_tokens": 0 if drop_extra else len(extra_tokens),
        "dropped_tokens": len(extra_tokens) if drop_extra else 0,
    }
    if drop_extra and extra_tokens:
        info["dropped_token_list"] = extra_tokens

    info_path = output_path / "remap_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("Done! You can now merge this checkpoint with any other model sharing the same tokenizer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap HF causal-LM checkpoint embeddings to match a base tokenizer")
    parser.add_argument(
        "--src_model",
        required=True,
        help="Checkpoint (HF hub ID or local path) to be remapped",
    )
    parser.add_argument(
        "--base_tokenizer",
        required=True,
        help="Hugging Face tokenizer whose vocab order will become canonical",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for remapped model and tokenizer",
    )
    parser.add_argument(
        "--shard_size",
        default="5GB",
        help="Max shard size for safetensors (e.g., '5GB')",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="'cpu', 'cuda', or accelerator device ID",
    )
    parser.add_argument(
        "--drop-extra",
        action="store_true",
        help="Do NOT add missing tokens to base tokenizer; instead drop them from the model",
    )

    args = parser.parse_args()

    main(
        src_model=args.src_model,
        base_tokenizer=args.base_tokenizer,
        out_dir=args.out_dir,
        shard_size=args.shard_size,
        dtype_str=args.dtype,
        device=args.device,
        drop_extra=args.drop_extra,
    )
