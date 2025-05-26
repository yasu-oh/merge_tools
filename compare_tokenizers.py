#!/usr/bin/env python3
"""
compare_tokenizers.py

This script loads the tokenizers from two specified Hugging Face models or local directories and:
- Checks whether their vocabulary sizes match
- Reports lines where the ID-to-token mapping differs

Example:
    python compare_tokenizers.py meta-llama/Llama-3.3-70B-Instruct \
                                 tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4 \
                                 --head 20
"""

import argparse, sys
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_a", help="Reference model (Hugging Face ID or local path)")
    parser.add_argument("model_b", help="Model to compare")
    parser.add_argument("--head", type=int, default=0,
                        help="Number of mismatched lines to display (default: show all)")
    args = parser.parse_args()

    tok_a = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    tok_b = AutoTokenizer.from_pretrained(args.model_b, trust_remote_code=True)

    if len(tok_a) != len(tok_b):
        print(f"Vocabulary sizes do not match: {len(tok_a)} vs {len(tok_b)}", file=sys.stderr)
        sys.exit(1)

    mismatch = [i for i in range(len(tok_a)) if tok_a.convert_ids_to_tokens(i) != tok_b.convert_ids_to_tokens(i)]

    print(f"Mismatch count: {len(mismatch)}")

    if args.head == 0:
        args.head = len(mismatch)

    for i in mismatch[:args.head]:
        token_a = tok_a.convert_ids_to_tokens(i)
        token_b = tok_b.convert_ids_to_tokens(i)
        print(f"{i:6d}  {token_a:<30}  {token_b}")

if __name__ == "__main__":
    main()
