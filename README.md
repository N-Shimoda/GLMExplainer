# GLMExplainer

## Setup Environment

This project uses [uv](https://docs.astral.sh/uv/) to manage its Python environment.
Every dependency is pinned in `uv.lock`, so the exact same versions are reproduced on any machine.

Which PyTorch build gets installed is selected by an extra, `cpu` or `cuda`.
The two are mutually exclusive, and **one of them must always be given**.

> [!IMPORTANT]
> Do not run a bare `uv sync`. Without an extra, PyTorch is pulled from PyPI as a transitive
> dependency, which on Linux drags in a full CUDA runtime that neither setup below intends.

### Devices with CUDA (_recommended_)

The `cuda` extra installs PyTorch from the CUDA 12.6 wheel index, together with the `flash_attn`
library for fine-tuning efficiency.

```bash
uv sync --extra cuda --group build --no-install-package flash-attn
uv sync --extra cuda --group build
```

<details>
<summary>Detailed Tips</summary>

- `flash_attn` imports PyTorch inside its own `setup.py`, so it has to be built with build isolation
  disabled — which in turn means PyTorch must already be present in the environment. Hence the two
  steps required: the first prepares the environment, the second builds `flash_attn` against it.
- To target a different CUDA version, change the `pytorch-cuda` index URL in `pyproject.toml`
  (e.g. `https://download.pytorch.org/whl/cu130`) and re-run `uv lock`.

</details>

### Others

For devices without a CUDA compatible GPU, the `cpu` extra installs the CPU build of PyTorch.
On Apple Silicon this is the same wheel that provides MPS support.

```bash
uv sync --extra cpu
```

`flash_attn` is not part of this setup, as it is only distributed for Linux with CUDA.

### Running the scripts

Activate the environment once, and the examples in the rest of this README can be run as written.

```bash
source .venv/bin/activate
pytest src/tests
```

`uv run --no-sync` is equivalent and needs no activation.

```bash
uv run --no-sync pytest src/tests
```

> [!WARNING]
> `--no-sync` matters here. A plain `uv run` re-syncs the environment first, and since it carries no
> extra it would replace your PyTorch build with the one described in the note above.

### Legacy setup with conda and pip

<details>
<summary>The conda / pip instructions used before the migration to uv</summary>

`environment.yml` and `requirements.txt` are kept in the repository for reference. They are no
longer the recommended path, and are not covered by `uv.lock`, but they still describe a working
environment.

For devices with CUDA compatible GPUs:

```bash
conda env create -f environment.yml
```

For other devices:

```bash
conda create -n glmexplainer python=3.12 scipy matplotlib rich openai pip pytest ninja
conda activate glmexplainer
pip install -r requirements.txt
```

Note that the `--extra-index-url .../cu124` line in `environment.yml` no longer has any effect:
the CUDA 12.4 index stops at PyTorch 2.6.0, so the `torch>=2.9.0` requirement cannot be satisfied
from it, and `pip` silently falls back to the default wheel on PyPI. The uv setup pins an index
explicitly to avoid this class of surprise.

</details>

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

Our model trained on MotifQA dataset is available as [naos-ku/GraphTokenLM](https://huggingface.co/naos-ku/GraphTokenLM) on Hugging Face Hub.
The architecture of the model is as follows:

- **Pre-trained LLM**: [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base)
- **GNN encoder**: 3-layer GIN with hidden dimension of 64.
- **Projection layers**: 2-layer MLP that maps 64-dim GNN output to 2560-dim GraphToken vectors.

You can load this model with `AutoModelForCausalLM`.

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
the [original paper](https://papers.nips.cc/paper_files/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html) of GNNExplainer
and the PyTorch Geometric [documentation](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.explain.algorithm.GNNExplainer.html).

### Procedure

1. First, find the best `edge_size` setting for each MotifQA subset.

   ```bash
   bash scripts/params/edge_size.sh
   python tools/writing/edge_size_study.py
   ```

   You can summarize the results in a table by running `tools/writing/edge_size_study.py` when using W&B logging.

1. For each optimal `edge_size` setting, find the best `lr`.

   ```bash
   bash scripts/params/lr.sh
   ```

1. Finally, find the best `epochs` setting for each `edge_size` and `lr` combination.

   ```bash
   bash scripts/params/epochs.sh
   ```
