import json

json_path = "results/maximum_flow/0924-1537_train.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

pred_list = [item["preds"] for item in data]
unique_preds = set(pred_list)

for unique in unique_preds:
    print(f"- {repr(unique)}: {pred_list.count(unique)}")
