# 推論時
from transformers import AutoTokenizer

from src.glm import GraphTokenLM

# ckpt_path = "outputs/edge_count/0922-1827/checkpoint-189"
ckpt_path = "outputs/edge_count/0922-2128/checkpoint-final"
model = GraphTokenLM.from_pretrained(ckpt_path)
tok = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
# tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True)
v = len(tok)

# 学習したcheckpointを読み込んだ後（または学習ログから）
emb_n = model.llm.get_input_embeddings().num_embeddings
# lm_head も（あれば）
head_n = model.llm.get_output_embeddings().weight.shape[0]
print(v, emb_n, head_n)
