# GraphToken

## Setup Environment

### Devices with CUDA (e.g., NVIDIA GPUs)

```bash
conda env create -f environment.yml
```

### Others (e.g., Apple Silicon)

```bash
conda create -n graphtoken python=3.12 scipy matplotlib pytest pip ninja rich openai
conda activate graphtoken
pip install -r requirements.txt
```
