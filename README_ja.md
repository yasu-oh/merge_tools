# merge_tools

**Hugging Face 上の Llama 系チェックポイント（または BPE 語彙を使用する任意の因果言語モデル）を統合する際の、トークナイザーの整合性チェックおよびリマッピングのためのユーティリティ集**です。
典型的なユースケースは、[MergeKit](https://github.com/arcee-ai/mergekit) などを用いて複数のモデル重みを**線形結合（linear merge）**したい場合で、その前段として**すべてのモデルが同一の語彙順序（トークナイザー）を共有**する必要があります。

現在、このリポジトリには以下の 2 つの自己完結型 CLI スクリプトが含まれています：

| ファイル名                   | 目的                                                                              |
| ----------------------- | ------------------------------------------------------------------------------- |
| `compare_tokenizers.py` | 2つのトークナイザー間のトークンIDと語彙の不一致を報告します                                                 |
| `remap_tokenizer.py`    | モデルの `embedding` と `lm_head` を、基準トークナイザーに一致するよう再配置します（**余分なトークンの削除**や**追加**も可能） |

どちらも純粋な Python スクリプトで、必要な依存は `transformers >= 4.37` と `torch` のみです。

## インストール

```bash
# （推奨）仮想環境の作成
python -m venv .venv && source .venv/bin/activate

pip install --upgrade pip
pip install torch transformers==4.*  # CUDA ビルドは必要に応じて選択
```

他に必要なパッケージはありません。

## 1. トークナイザーの整合性チェック

```bash
python compare_tokenizers.py \
    meta-llama/Llama-3.3-70B-Instruct \
    tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4 \
    --head 20            # 最初の 20 件のみ表示
```

2つの語彙サイズが異なる場合、このスクリプトは**非ゼロで終了**するため、シェルスクリプトや CI に組み込むことも可能です。

## 2. チェックポイントを基準トークナイザーにリマップする

### 2.1 Append モード（デフォルト）

すべてのベクトルを保持し、基準トークナイザーにないトークンは語彙末尾に追加：

```bash
python remap_tokenizer.py \
    --src_model tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4 \
    --base_tokenizer perplexity-ai/r1-1776-distill-llama-70b \
    --out_dir ./swallow-r1-tokenfixed
```

結果：

* `./swallow-r1-tokenfixed/` にリシャード済みの `safetensors` モデルと、
* 語彙サイズが R1 以上の**パッチ済みトークナイザー**が保存されます。

### 2.2 Drop モード（`--drop-extra`）

最終的な語彙を基準トークナイザーに**完全一致**させる必要がある場合（Swallow 固有の特殊トークンを削除したい場合など）に使用：

```bash
python remap_tokenizer.py \
    --src_model tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4 \
    --base_tokenizer perplexity-ai/r1-1776-distill-llama-70b \
    --out_dir ./swallow-pruned \
    --drop-extra
```

* 基準トークナイザーに存在しない行は、`<unk>` 埋め込みで置換されます（`<unk>` がない場合は平均ベクトル）。
* 結果として得られる `config.vocab_size` は **R-1 と完全一致**し、**R-1 のオリジナルのトークナイザーで安全に推論**できます。

## 3. MergeKit を用いた線形マージの実行例

Swallow および Meta の両モデルを R-1 のトークナイザーに合わせてリマップした後：

```yaml
# merge.yaml
merge_method: linear
dtype: bfloat16

models:
  - model: perplexity-ai/r1-1776-distill-llama-70b
    parameters: { weight: 1.0 }

  # 0.4 * (Swallow - Meta)
  - model: ./swallow-pruned          # または ./swallow-r1-tokenfixed
    parameters: { weight: 0.4 }

  - model: ./meta-pruned             # Meta Llama を同様にリマップしたもの
    parameters: { weight: -0.4 }

parameters:
  normalize: true
```

```bash
mergekit-yaml --cuda --copy-tokenizer merge.yaml ./merged-r1-70b
```

最終的なチェックポイントは **R-1 のトークナイザーおよびチャットテンプレートを共有**します。追加ファイルは不要です。

## 4 · FAQ（よくある質問）

### Q. オリジナルトークナイザーのまま、ID マッピング用 JSON を追加してはダメなの？
A. HF `transformers` は実行時に任意の ID 再マッピングをサポートしていません。FastTokenizer は連続した ID を前提とするため、**重みレベルで再配置するのが最もシンプルで完全互換な方法**です。

### Q. トークナイザーが知らないトークンをモデルがサンプリングしたらどうなる？
A. `--drop-extra` を使った場合はこの問題は発生しません（語彙サイズが一致するため）。未使用なら `<unk>` としてデコードされ、**出力が崩れる**可能性があります。

## 5 · ライセンス
このスクリプト群は **MIT License** のもとで公開されています。上記で言及されている個別のチェックポイント（Meta Llama 3, Swallow, R-1 など）は、それぞれのモデルカードに記載のライセンスに従ってください。
