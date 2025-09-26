import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Count unique 'preds' values in a results JSON file (list of dicts).")
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to the results JSON file (e.g., results/edge_count/run.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    json_path = args.json_path

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: JSON root must be a list of objects having a 'preds' key.", file=sys.stderr)
        sys.exit(1)

    pred_list = [item.get("preds") for item in data if isinstance(item, dict) and "preds" in item]
    unique_preds = sorted(set(pred_list), key=lambda x: (str(x)))

    for unique in unique_preds:
        print(f"- {repr(unique)}: {pred_list.count(unique)}")


if __name__ == "__main__":
    main()
