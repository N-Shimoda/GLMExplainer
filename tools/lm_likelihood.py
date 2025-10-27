import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデル読み込み
model_name = "Qwen/Qwen3-4B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# テキスト
text = "The capital of France is Paris."

# トークン化
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 推論
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # [1, seq_len, vocab_size]
    probs = torch.softmax(logits, dim=-1)  # 確率分布に変換

# 各トークンのその位置での生成確率を取得
input_ids = inputs["input_ids"]
token_probs = []

# 位置 t=1〜N-1 に対して、P(x_t | x_<t>) を計算
for t in range(1, input_ids.size(1)):
    token_id = input_ids[0, t]
    prob = probs[0, t - 1, token_id].item()  # 1ステップ前の出力で次トークンを評価
    token_probs.append(prob)

# 結果を対応トークンとともに表示
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
clean_tokens = [t.replace("Ġ", " ") for t in tokens]  # 空白トークンの整形
for t, p in zip(clean_tokens[1:], token_probs):  # 先頭は確率なし
    print(f"{t:>15s}: {p:.6f}")

# 平均対数確率とパープレキシティの計算
log_probs = [math.log(p) for p in token_probs]
avg_log_prob = sum(log_probs) / len(log_probs)
ppl = math.exp(-avg_log_prob)
print(f"Average log-prob: {avg_log_prob:.4f}, Perplexity: {ppl:.4f}")
