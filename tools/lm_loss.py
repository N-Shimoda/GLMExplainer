import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザのロード
model_name = "Qwen/Qwen3-4B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# 入力文
X = "The capital of France is"
Y = " Paris."

# X+Y を連結
XY = X + Y

# トークン化
inputs = tokenizer(XY, return_tensors="pt").to(model.device)
with torch.no_grad():
    # X部分だけをトークン化して長さを取得
    X_len = len(tokenizer(X, return_tensors="pt")["input_ids"][0])

    # 損失を計算
    labels = inputs["input_ids"].clone()
    labels[:, :X_len] = -100  # Xの部分の損失は無視
    outputs = model(**inputs, labels=labels)

    loss = outputs.loss  # Y部分の平均負の対数尤度
    ppl = torch.exp(loss)  # perplexity（任意）

print(f"Loss: {loss.item():.4f}")
print(f"Perplexity: {ppl.item():.4f}")

