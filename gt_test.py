from pprint import pprint

import torch
from transformers import AutoTokenizer, GenerationConfig

from src.glm import GraphTokenLM

ckpt_path = "outputs/edge_count/0924-1525/checkpoint-189"
model = GraphTokenLM.from_pretrained(ckpt_path)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

# text = ["My name is Naoki Shimoda. Nice to see you." for _ in range(4)]
text = "My name is Naoki Shimoda. Nice to see you."
inputs = tokenizer(text, return_tensors="pt")
print("Initial inputs:", inputs)

with torch.no_grad():
    gen_cfg = GenerationConfig(max_new_tokens=4, num_beams=3, do_sample=True)
    out = model.generate(**inputs, generation_config=gen_cfg)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    pprint(decoded)
