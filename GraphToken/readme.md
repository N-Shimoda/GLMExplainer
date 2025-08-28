# GraphToken

## Usage

### Qwen

```shell
# 1) 依存関係
pip install -U "transformers>=4.51.0" "trl>=0.10.0" peft bitsandbytes datasets accelerate sentencepiece protobuf flash_attn

# 2) 学習（例: cycle_check の zero_shot_*）
python finetune_graphqa_qwen3_4b.py \
  --subset cycle_check \
  --do_train \
  --output_dir ./qwen3-4b-graphqa-qlora \
  --do_eval \
  --wandb
```

### Llama

```shell
# 1) 依存関係
pip install -U "transformers>=4.43" "trl>=0.9.6" peft bitsandbytes datasets accelerate evaluate sentencepiece protobuf

# 2) 環境変数（HF_TOKEN は Hugging Face のアクセストークン）
export HF_TOKEN=hf_xxx

# 3) 学習（デフォルト: meta-llama/Llama-3.2-3B-Instruct, connected_nodes の zero_shot_* ）
python finetune_graphqa_llama32.py \
  --output_dir ./llama32-3b-graphqa-qlora \
  --subset connected_nodes \
  --train_split zero_shot_train \
  --eval_split zero_shot_validation
```
