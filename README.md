# GraphToken

Reproductive experiment of Perozzi et al., ["Let Your Graph Do the Talking: Encoding Structured Data for LLMs"](https://arxiv.org/abs/2402.05862) (arXiv, Feb. 2024).

## Train GraphToken model

```shell
torchrun --nproc_per_node=2 train.py --subset "${subset}" \
    --base_model "Qwen/Qwen3-4B-Base" \
    --node_feat_dim 8 \
    --gnn_hidden_dim 256 --gnn_out_dim 512 --num_gnn_layers 4 \
    --epochs 3 --lr 0.01 \
    --do_eval --wandb
```

Choice of the subsets are "node_count", "edge_count", "cycle_check", "triangle_counting", and "maximum_flow".

## Files & Directories

- `fine-tuning`: Codes for fine-tuning `Qwen3-4B-Instruct-2507` on Graph-level tasks of GraphQA dataset.
- `notebook`: Sample codes from Hugging Face.
