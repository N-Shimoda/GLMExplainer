# Fine-tuning Qwen3-4B on GraphQA

This directory contains the standalone scripts for LoRA fine-tuning and evaluation of Qwen/Qwen3-4B on the GraphQA tasks.

## 1. Multi-GPU setup with Accelerate

1. Install the dependencies defined at the repository root (`environment.yml`), including `accelerate`.
2. Review `accelerate_config.yaml` (two GPUs, bf16, NCCL). Adjust `gpu_ids`, `num_processes`, or other fields if your hardware differs.
   - Older Accelerate releases accept only the keys already present in this file; if you regenerate the config with a newer CLI, remove any unsupported keys before launching.
   - Alternatively, regenerate it with `accelerate config --config_file fine-tuning/accelerate_config.yaml`.

## 2. Fine-tuning

- **Single subset**

  ```bash
  cd fine-tuning
  accelerate launch --config_file accelerate_config.yaml \
    ft_qwen3_4b.py \
    --subset cycle_check \
    --epochs 3 \
    --do-eval \
    --wandb
  ```

  Key arguments:

  - `--subset`: one of `node_count`, `edge_count`, `cycle_check`, `triangle_counting`, `maximum_flow`
  - `--per-device-train-batch-size`, `--grad-accum-steps`, etc. to control the global batch (`world_size × per-device × grad-accum`)
  - `--base-model`: defaults to `Qwen/Qwen3-4B-Base`, but the script accepts any compatible checkpoint

- **All subsets (batch run)**

  ```bash
  cd fine-tuning
  bash scripts/ft_all.sh
  ```

  The script launches `accelerate` for each subset listed in the loop (`node_count`, `edge_count`, `cycle_check`, `triangle_counting` by default), logs progress to `../logs/ft_all.log`, and reuses the same arguments as above.  
  Override the Accelerate binary or config with environment variables:

  ```bash
  ACCELERATE_BIN=/path/to/accelerate \
  ACCELERATE_CONFIG=custom_config.yaml \
    bash scripts/ft_all.sh
  ```

## 3. Evaluation

- **Single subset**

  ```bash
  cd fine-tuning
  python eval_ft.py \
    --subset cycle_check \
    --model_path "./models/cycle_check/<run_timestamp>/checkpoint-final" \
    --batch-size 8
  ```

- **Evaluate all fine-tuned GraphToken models**

  Use the repository-level helper to iterate over saved checkpoints:

  ```bash
  cd ..
  bash scripts/eval_gt_all.sh --split test
  ```

  Logs are written to `logs/eval_gt_all.log` and include per-subset summaries.
