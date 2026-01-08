import argparse

from transformers import AutoTokenizer

from src.ckpt import _resolve_ckpt_path
from src.constants import MOTIFQA_SUBSETS
from src.explanation.preprocess import build_dataset, filter_dataset
from src.glm import GraphTokenLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", type=str, required=True, choices=MOTIFQA_SUBSETS)
    p.add_argument("--sample-idx", type=int, required=True)
    p.add_argument("--model-path", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path, run_name = _resolve_ckpt_path(args.model_path)
    model = GraphTokenLM.from_pretrained(ckpt_path)
    tok = AutoTokenizer.from_pretrained(model.config.llm_name, trust_remote_code=True)
    print("Loaded model from {}".format(ckpt_path))

    dataset = build_dataset("MotifQA", args.subset, "test", node_feat_dim=model.config.node_feat_dim)
    dataset = filter_dataset(dataset, "MotifQA", sample_idx=args.sample_idx)
    print(dataset)
    print(dataset[0])


if __name__ == "__main__":
    main()
