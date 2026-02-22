import argparse

import torch
from torchinfo import summary
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hf_model(torch_dtype: str):
    """Examine if the Hugging Face export of GraphTokenLM can be loaded without errors."""
    repo_id = "naos-ku/GraphTokenLM"
    dtype_map = {
        "auto": "auto",
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision="main",
        trust_remote_code=True,
        dtype=dtype_map[torch_dtype],
    )
    tok = AutoTokenizer.from_pretrained(repo_id, revision="main", trust_remote_code=True)
    return model, tok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Torch dtype used when loading model weights.",
    )
    args = parser.parse_args()
    model, tok = load_hf_model(torch_dtype=args.torch_dtype)
    print("\nModel:")
    summary(model)
    print("\nTokenizer:\n", tok)
