# merge_tools

Utilities for **tokenizer‐level sanity-checks and remapping** when combining large-scale Hugging Face Llama-family checkpoints (or any causal-LMs that use a word-piece / BPE vocabulary).  
Typical use-case: you want to **linearly merge** two or more model weights (e.g. with [MergeKit](https://github.com/arcee-ai/mergekit)) but need all source checkpoints to share **one canonical tokenizer order** first.

Currently the repo contains two self-contained CLI scripts:

| File | Purpose |
|------|---------|
| `compare_tokenizers.py` | Report ID-to-token mismatches between two tokenizers |
| `remap_tokenizer.py`    | Realign a model’s `embedding` + `lm_head` rows to match a base tokenizer (with an option to **drop** or **append** extra tokens) |

Both are pure-Python and depend only on `transformers >= 4.37` and `torch`.

## Installation

```bash
# create a fresh environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate

pip install --upgrade pip
pip install torch transformers==4.*  # pick the CUDA build you need
```

No other packages are required.

## 1. Quick tokenizer sanity-check

```bash
python compare_tokenizers.py \
    meta-llama/Llama-3.3-70B-Instruct \
    tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4 \
    --head 20            # show the first 20 mismatches only
```

The script exits **non-zero** if the two vocab sizes differ, so you can embed it in shell pipelines / CI.

## 2. Remap a checkpoint to a canonical tokenizer

### 2.1 Append-mode (default)

Keeps **all** original vectors; missing tokens are appended to the base tokenizer’s vocab:

```bash
python remap_tokenizer.py \
    --src_model tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4 \
    --base_tokenizer perplexity-ai/r1-1776-distill-llama-70b \
    --out_dir ./swallow-r1-tokenfixed
```

Result:

* `./swallow-r1-tokenfixed/` contains a re-sharded `safetensors` model and a **patched tokenizer** whose vocab size ≥ R1.
* Base tokens that were missing from the source model are initialized with the `<unk>` vector.

### 2.2 Drop-mode (`--drop-extra`)

Use this when you **must** keep the final vocab identical to the base tokenizer (e.g. you plan to discard any Swallow-specific special tokens):

```bash
python remap_tokenizer.py \
    --src_model tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4 \
    --base_tokenizer perplexity-ai/r1-1776-distill-llama-70b \
    --out_dir ./swallow-pruned \
    --drop-extra
```

* Rows that have no counterpart in the base tokenizer are replaced with the `<unk>` embedding (or the mean vector when `<unk>` is absent).
* The resulting `config.vocab_size` **exactly matches** R-1, so you can load the model with **R-1’s original tokenizer** without `<unk>` spam.

## 3. End-to-end linear merge with MergeKit

After remapping both Swallow and Meta checkpoints to R-1’s tokenizer:

```yaml
# merge.yaml
merge_method: linear
dtype: bfloat16

models:
  - model: perplexity-ai/r1-1776-distill-llama-70b
    parameters: { weight: 1.0 }

  # 0.4 * (Swallow - Meta)
  - model: ./swallow-pruned          # or ./swallow-r1-tokenfixed
    parameters: { weight: 0.4 }

  - model: ./meta-pruned             # Meta Llama remapped the same way
    parameters: { weight: -0.4 }

parameters:
  normalize: true
```

```bash
mergekit-yaml --cuda --copy-tokenizer merge.yaml ./merged-r1-70b
```

The final checkpoint **shares the tokenizer and chat templates of R-1**—no extra files to ship.

## 4 · FAQ

### Why not keep the original tokenizer and add a JSON mapping?
HF `transformers` does not support arbitrary ID remapping at runtime; the fast-tokenizer assumes contiguous IDs. Weight-level realignment is therefore the simplest, 100 % compatible solution.

### What happens if the model samples a token that the tokenizer doesn’t know?
With `--drop-extra` this cannot happen—the vocab sizes are identical. Without it, unknown IDs will be decoded as `<unk>`, leading to garbled output.

## 5 · License

The scripts are released under the **MIT License**. Individual checkpoints mentioned above are subject to their respective licenses (Meta Llama 3, Swallow, R-1, etc.); consult the original model cards.
