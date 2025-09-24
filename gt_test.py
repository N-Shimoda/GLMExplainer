from pprint import pprint

import torch
from transformers import AutoTokenizer

from src.glm import GraphTokenLM

ckpt_path = "outputs/edge_count/0924-1416/checkpoint-189"
model = GraphTokenLM.from_pretrained(ckpt_path)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

# text = ["My name is Naoki Shimoda. Nice to see you." for _ in range(4)]
text = "My name is Naoki Shimoda. Nice to see you."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=8)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    pprint(decoded)
