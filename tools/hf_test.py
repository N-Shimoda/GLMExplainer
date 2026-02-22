import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_torch_dtype(dtype_str: str):
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {dtype_str}. " "Choose from: auto, float16, bfloat16, float32.")
    return dtype_map[dtype_str]


def load_hf_model(torch_dtype: str = "auto"):
    """Examine if the Hugging Face export of GraphTokenLM can be loaded without errors."""
    repo_id = "naos-ku/GraphTokenLM"
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        trust_remote_code=True,
        dtype=_parse_torch_dtype(torch_dtype),
    )
    tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    return model, tok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype used when loading model weights.",
    )
    args = parser.parse_args()
    model, tok = load_hf_model(torch_dtype=args.torch_dtype)
    print(model)
    print(tok)
