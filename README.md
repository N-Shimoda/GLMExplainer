# GraphToken

## Setup Environment

### Devices with CUDA (_reccomended_)

For devices with CUDA compatible GPUs, we recommend using the `environment.yml` to build an environment.
This file includes the version index of PyTorch to ensure the reproducibility.

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

## Search of Optimal Hyperparameters

> [!TIP]
> Using a suitable hyperparameters in GNNExplainer achieves better explanation accuracy in our method.

When using GNNExplainer for computing edge importance, the optimization process has four hyperparameters: `edge_size`, `edge_ent`, `lr`, and `epochs`.

### Procedure

1. First, find the best `edge_size` setting for each MotifQA subset.

   ```bash
   bash scripts/explain/edge_size.sh
   ```

   You can summarize the results in a table by running `tools/writing/edge_size_study.py` when using W&B logging.

1. For each optimal `edge_size` setting, find the best `lr`.

   ```bash
   bash scripts/explain/lr.sh
   ```

1. Finally, find the best `epochs` setting for each `edge_size` and `lr` combination.

   ```bash
   bash scripts/explain/epochs.sh
   ```
