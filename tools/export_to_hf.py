import argparse
import json
import os
import sys

from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ckpt import _resolve_ckpt_path  # noqa: E402
from src.glm import GraphTokenLM  # noqa: E402


def build_args():
    p = argparse.ArgumentParser(description="Export the trained GraphTokenLM to Hugging Face format.")
    p.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the trained checkpoint directory (e.g. output/checkpoint-xxxx)",
    )
    p.add_argument(
        "--export-dir",
        type=str,
        default="GraphTokenLM-HF",
        help="Directory to save the exported model (default: GraphTokenLM-HF)",
    )
    p.add_argument(
        "--packages",
        nargs="+",
        default=["torch", "transformers", "huggingface_hub", "safetensors", "torch-geometric"],
        help="Additional packages to add to requirements.txt",
    )

    args = p.parse_args()
    args.ckpt_path, _ = _resolve_ckpt_path(args.ckpt_path)  # Validate and resolve checkpoint path
    return args


def insert_auto_map(config_path: str):
    """Insert the auto_map field to config.json for Hugging Face compatibility."""
    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "glm.GraphTokenLMConfig",
        "AutoModelForCausalLM": "glm.GraphTokenLM",
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def add_requirements(export_dir: str, packages: list[str]):
    """Add pckages to requirements.txt if not already present."""
    req_file = os.path.join(export_dir, "requirements.txt")
    existing_packages = set()
    if os.path.exists(req_file):
        with open(req_file, "r") as f:
            existing_packages = set(line.strip() for line in f if line.strip())
    with open(req_file, "a") as f:
        for pkg in packages:
            if pkg not in existing_packages:
                f.write(pkg + "\n")


args = build_args()
os.makedirs(args.export_dir, exist_ok=True)

# 1) Save model weights and config in Hugging Face format
model = GraphTokenLM.from_pretrained(args.ckpt_path, load_llm_weights=True)
model.save_pretrained(args.export_dir, safe_serialization=True)

# 2) Save tokenizer of the base LM
tokenizer = AutoTokenizer.from_pretrained(model.config.base_model, trust_remote_code=True)
tokenizer.save_pretrained(args.export_dir)

print("Saved to:", args.export_dir)

# 3) Add auto_map to config.json
config_path = os.path.join(args.export_dir, "config.json")
insert_auto_map(config_path)

# 4) Add transformers to requirements.txt
add_requirements(args.export_dir, args.packages)
