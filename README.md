# GraphToken

Reproductive experiment of Perozzi et al., ["Let Your Graph Do the Talking: Encoding Structured Data for LLMs"](https://arxiv.org/abs/2402.05862) (arXiv, Feb. 2024).

## Train GraphToken model

```shell
torchrun --nproc_per_node=2 train.py --wandb
```

## Files & Directories

- `fine-tuning`: Codes for fine-tuning `Qwen3-4B-Instruct-2507` on Graph-level tasks of GraphQA dataset.
- `notebook`: Sample codes from Hugging Face.
