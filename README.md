# GraphToken

## Setup Environment

### Devices with CUDA (_reccomended_)

For devices with CUDA compatible GPUs, we recommend using `environment.yml` to build an environment.
This file includes the version index of PyTorch to ensure the reproducibility, and `flash_attn` library for fine-tuning efficiency.

```bash
conda env create -f environment.yml
```

### Others

For other devices, the following script will install the necessary packages to run the code.

```bash
conda create -n graphtoken python=3.12 scipy matplotlib rich openai pip pytest ninja
conda activate graphtoken
pip install -r requirements.txt
```

## Construction of GraphToken model

This repository provides the PyTorch implementation of GraphToken and scripts for training and evaluation.

### GraphToken (`src/glm.py`)

GraphToken is a pioneering Graph-Language Model (GLM) method proposed by Perozzi et al. (2024) that encodes graph-structured data into a soft-prompt vectors to be consumed by a pre-trained language model.

### Training of GraphToken (`train.py`)

To train a GraphToken model on graph QA datasets, run `train.py` in the following format.
The supported datasets include our [MotifQA](https://huggingface.co/datasets/naos-ku/motif-qa) and Fatemi et al.'s [GraphQA](https://huggingface.co/datasets/baharef/GraphQA) datasets.

```bash
torchrun --nproc_per_node=NUM_GPUS train.py \
   --dataset MotifQA \
   --subset ba_shapes ba_two_motifs tree_cycle tree_grid_v2 \
   --lpe-dim 8 --pos-emb-dim 8 \
   --gnn-type GIN \
   --gnn-hidden-dim 64 --gnn-out-dim 64 \
   --num-gnn-layers 3 --graph-pooling mean \
   --num-proj-layers 2 --num-graph-tokens 4 \
   --epochs 32 \
   --optim adamw --lr 5e-3 --weight-decay 1e-2 \
   --lr-scheduler-type cosine --warmup-ratio 0.05 \
   --wandb
```

This script utilizes `SFTTrainer` to train the weights of GNN encoder and projection layers,
while keeping the pre-trained language model frozen.
If needed, fine-tuning of the language model is applicable by setting `--use-lora` flag.
You can find out details of the other hyperparameters by running `python train.py --help`.

Our best model on [MotifQA dataset](https://huggingface.co/datasets/naos-ku/motif-qa) is available as [`naos-ku/GraphTokenLM`](https://huggingface.co/naos-ku/GraphTokenLM) on Hugging Face Hub.

### Evaluation (`eval.py`)

To evaluate the model, please run `eval.py` in the following format.

```bash
torchrun --nproc_per_node=2 eval.py \
	--dataset MotifQA --subset ba_shapes tree_cycle \
	--model-path naos-ku/GraphTokenLM \
	--num-trials 5 --per-device-batch-size 5
```

By specifying multiple subset names, this script evaluates the model and reports accuracy for each subset.

## Applying proposed method

In order to apply our explanation method to a trained GraphToken model, run `explain.py` with following arguments.

```bash
torchrun --nproc_per_node=2 explain.py \
   --dataset MotifQA --subset ba_shapes \
   --model-path "naos-ku/GraphTokenLM" \
   --target-pos-samples --num-trials 5 \
   --llr-threshold 1.0 --baseline-graph complete \
   --epochs 200 --lr 0.1 \
   --edge-size 3 --edge-ent 1.0 \
   --wandb
```

## Search of Optimal Hyperparameters

> [!TIP]
> Using a suitable hyperparameters in GNNExplainer achieves better explanation accuracy in our method.

### Hyperparameters in GNNExplainer

When using GNNExplainer for computing edge importance, the optimization process has four hyperparameters: `edge_size`, `edge_ent`, `lr`, and `epochs`.

For details, please refer to the original paper of GNNExplainer[^1] and the PyTorch Geometric documentation[^2].

[^1]: https://papers.nips.cc/paper_files/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html

[^2]: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.explain.algorithm.GNNExplainer.html

### Procedure

1. First, find the best `edge_size` setting for each MotifQA subset.

   ```bash
   bash scripts/explain/edge_size.sh
   ```

````

You can summarize the results in a table by running `tools/writing/edge_size_study.py` when using W&B logging.

1. For each optimal `edge_size` setting, find the best `lr`.

   ```bash
   bash scripts/explain/lr.sh
   ```

1. Finally, find the best `epochs` setting for each `edge_size` and `lr` combination.

   ```bash
   bash scripts/explain/epochs.sh
   ```
````
