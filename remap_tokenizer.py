#!/usr/bin/env python
"""
Remap a source HF causal-LM checkpoint so that its embedding / lm_head rows
follow *base_tokenizer*'s vocabulary order.  Any tokens missing from the base
tokenizer are appended at the end and their trained vectors are copied over,
so no information is lost.

Usage
-----
python remap_tokenizer.py \
    --src_model  tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4 \
    --base_tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --out_dir   ./swallow-tokenfixed
"""

from typing import List
import argparse, os, json, torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)

def load_tokenizer(name_or_path: str):
    """HF AutoTokenizer wrapper with remote-code enabled & fast=False fallback."""
    try:
        tok = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True, use_fast=False
        )
    except TypeError:  # old transformers w/o use_fast
        tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    return tok

def main(
    src_model: str,
    base_tokenizer: str,
    out_dir: str,
    shard_size: str = "5GB",
    dtype_str: str = "bfloat16",
    device: str = "cpu",
):
    os.makedirs(out_dir, exist_ok=True)

    # 1. tokenizers ----------------------------------------------------------------
    print(f"Loading tokenizers\n   src  = {src_model}\n   base = {base_tokenizer}")
    tok_src  = load_tokenizer(src_model)
    tok_base = load_tokenizer(base_tokenizer)

    # 2. extend base tokenizer with tokens missing from it -------------------------
    src_vocab = list(tok_src.get_vocab().keys())
    base_vocab_set = set(tok_base.get_vocab().keys())
    extra_tokens: List[str] = [t for t in src_vocab if t not in base_vocab_set]

    if extra_tokens:
        n_added = tok_base.add_tokens(extra_tokens)
        print(f"Added {n_added} tokens to base tokenizer (new size={len(tok_base)})")
    else:
        print("No extra tokens needed ? vocabularies already aligned.")

    # Mapping: src_id -> new_base_id
    id_map = torch.empty(len(tok_src), dtype=torch.long)
    for src_id in range(len(tok_src)):
        token = tok_src.convert_ids_to_tokens(src_id)
        id_map[src_id] = tok_base.convert_tokens_to_ids(token)
    assert (id_map >= 0).all(), "Some tokens still unresolved!"

    # 3. load source model ----------------------------------------------------------
    print("Loading source model … (may take a while)")
    torch_dtype = getattr(torch, dtype_str)
    model = AutoModelForCausalLM.from_pretrained(
        src_model,
        torch_dtype=torch_dtype,
        device_map={"": device},  # cpu or single gpu
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    embed = model.get_input_embeddings().weight
    lm    = model.lm_head.weight
    hidden_dim = embed.shape[1]

    # 4. build new embedding / lm_head ---------------------------------------------
    new_vocab_size = len(tok_base)
    new_embed = torch.zeros(new_vocab_size, hidden_dim, dtype=embed.dtype)
    new_lm    = torch.zeros_like(new_embed)

    # copy rows from source to their new positions
    with torch.no_grad():
        new_embed[id_map] = embed
        new_lm[id_map]    = lm

        # for tokens that were *added* to base (extra_tokens),
        # we already copied their vectors above because they exist in src.
        model.set_input_embeddings(
            torch.nn.Embedding.from_pretrained(new_embed, freeze=False)
        )
        model.lm_head.weight = torch.nn.Parameter(new_lm)
        model.config.vocab_size = new_vocab_size

    # 5. save ----------------------------------------------------------------------
    print(f"Saving remapped model + tokenizer to {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size=args.shard_size)
    tok_base.save_pretrained(out_dir)

    # also store a tiny manifest for reproducibility
    with open(os.path.join(out_dir, "remap_info.json"), "w") as f:
        json.dump(
            dict(
                src_model=src_model,
                base_tokenizer=base_tokenizer,
                extra_tokens=len(extra_tokens),
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Done!  You can now merge this checkpoint with any other model that "
          "shares the same tokenizer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_model", required=True,
                        help="checkpoint (HF hub id or local path) to be remapped")
    parser.add_argument("--base_tokenizer", required=True,
                        help="HF tokenizer whose vocab order will become canonical")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--shard_size", default="5GB",
                        help='max shard size for safetensors, e.g. "5GB"')
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cpu",
                        help="'cpu', 'cuda', or accelerator device id")
    args = parser.parse_args()
    main(args.src_model, args.base_tokenizer, args.out_dir, args.shard_size, args.dtype, args.device)
