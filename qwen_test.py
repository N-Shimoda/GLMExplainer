import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

llm_name = "Qwen/Qwen3-4B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(llm_name, device_map="auto", trust_remote_code=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(llm_name)

# text = ["My name is Naoki Shimoda. Nice to see you." for _ in range(8)]
text = "My name is Naoki Shimoda. Nice to see you."
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(inputs)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    print(decoded)
