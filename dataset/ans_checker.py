import json
import os


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def check_answer(jsonl_path):
    print(f"Checking answers in {jsonl_path}...")
    records = list(load_jsonl(jsonl_path))

    mismatch = 0
    for rec in records:
        ans_val = int(rec["answer"].split(".")[0])
        if len(rec["triangles"]) != ans_val:
            print("Answer mismatch for id:", records.index(rec))
            mismatch += 1

    print(f"Total mismatches found: {mismatch} out of {len(records)}")


if __name__ == "__main__":
    dataset_path = "dataset/triangle_counting"
    for split in ["train", "eval", "test"]:
        jsonl_path = os.path.join(dataset_path, f"{split}.jsonl")
        check_answer(jsonl_path)
