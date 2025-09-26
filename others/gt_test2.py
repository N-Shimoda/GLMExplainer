from pprint import pprint

import torch
from torch_geometric.data import Data as PygData
from transformers import AutoTokenizer, GenerationConfig

from src.glm import GraphTokenLM

ckpt_path = "outputs/edge_count/0924-1525/checkpoint-189"
model = GraphTokenLM.from_pretrained(ckpt_path)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

graph_data = PygData(
    x=torch.tensor(
        [
            [-0.4482, -0.1706, 0.0703, -0.4915],
            [-0.1169, 0.2367, 0.1772, 0.7088],
            [0.5460, 0.1553, 0.3236, -0.1829],
            [0.4826, -0.5033, 0.1592, -0.1594],
            [0.2468, 0.5684, -0.1919, -0.1584],
            [-0.1408, 0.1317, -0.6063, -0.2199],
            [-0.4168, 0.0313, 0.5748, -0.1145],
            [-0.0010, -0.5440, -0.3141, 0.3325],
        ]
    ),
    edge_index=torch.tensor(
        [
            [0, 1, 0, 5, 0, 6, 0, 7, 1, 2, 1, 4, 1, 5, 1, 6, 1, 7, 2, 3, 2, 4, 3, 7, 4, 5, 5, 7],
            [1, 0, 5, 0, 6, 0, 7, 0, 2, 1, 4, 1, 5, 1, 6, 1, 7, 1, 3, 2, 4, 2, 7, 3, 5, 4, 7, 5],
        ]
    ),
    batch=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
)

# text = ["Q: How many edges are in this graph?\nA: " for _ in range(1)]
text = "user: How many edges are in this graph? assistant: "
inputs = tokenizer(text, return_tensors="pt")
print("Initial inputs:", inputs)

with torch.no_grad():
    gen_cfg = GenerationConfig(max_new_tokens=8, do_sample=True)
    out = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        graph=graph_data,
        generation_config=gen_cfg,
    )
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    pprint(decoded)
