# GraphToken

## Setup Environment

### Devices with CUDA (_recommended_)

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

## GraphToken model

> [!NOTE]
> **GraphToken** is a pioneering Graph-Language Model (GLM) method proposed by Perozzi et al. (2024) that encodes graph-structured data into a soft-prompt vectors to be consumed by a pre-trained language model.

### Training

To train a GraphToken model on graph QA datasets, run `train.py` in the following format.
The supported datasets include our [MotifQA](https://huggingface.co/datasets/naos-ku/motif-qa) and Fatemi et al. (2024)'s [GraphQA](https://huggingface.co/datasets/baharef/GraphQA) datasets.

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

The above setting trains GNN encoder and projection layers, while keeping the pre-trained language model frozen.
If needed, one can apply parameter efficient fine-tuning (LoRA) to LLM by setting `--use-lora` flag.
To find more details, please refer to `python train.py --help`.

### Evaluation

To evaluate the model, please run `eval.py` in the following format.

```bash
torchrun --nproc_per_node=2 eval.py \
	--dataset MotifQA --subset ba_shapes tree_cycle \
	--model-path "naos-ku/GraphTokenLM" \
	--num-trials 5 --per-device-batch-size 5
```

By specifying multiple subsets, this script reports the answer accuracy per subset.

### Hugging Face Model

Our best model trained on MotifQA dataset is available as [naos-ku/GraphTokenLM](https://huggingface.co/naos-ku/GraphTokenLM) on Hugging Face Hub.
This model can be loaded with `AutoModelForCausalLM`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
	"naos-ku/GraphTokenLM",
	trust_remote_code=True,
	load_llm_weights=False,  # skip loading LLM weights from original HF repo (Qwen/Qwen3-4B-Base).
)
tokenizer = AutoTokenizer.from_pretrained("naos-ku/GraphTokenLM", trust_remote_code=True)
```

## Explain model outputs

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
> Using a suitable hyperparameter configuration in GNNExplainer achieves a better explanation accuracy in our method.

### Hyperparameters

When using GNNExplainer for computing edge importance, the optimization process has four hyperparameters: `edge_size`, `edge_ent`, `lr`, and `epochs`.

For details, please refer to
the [original paper of GNNExplainer](https://papers.nips.cc/paper_files/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html)
and the [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.explain.algorithm.GNNExplainer.html).

### Procedure

1. First, find the best `edge_size` setting for each MotifQA subset.

   ```bash
   bash scripts/params/edge_size.sh
   python tools/writing/edge_size_study.py
   ```

   You can summarize the results in a table by running `/writing/edge_size_study.py` when using W&B logging.

1. For each optimal `edge_size` setting, find the best `lr`.

   ```bash
   bash scripts/params/lr.sh
   ```

1. Finally, find the best `epochs` setting for each `edge_size` and `lr` combination.

   ```bash
   bash scripts/params/epochs.sh
   ```
