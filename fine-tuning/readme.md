# GraphToken

## Usage

### Fine-tuning

You can fine-tune all subsets at once and log results using the provided script:

```shell
bash scripts/ft_all.sh
```

This will run `ft_qwen3_4b.py` for each subset (node_count, edge_count, cycle_check, triangle_counting, maximum_flow) and log the results to `../logs/ft_all.log`.

To run fine-tuning for a single subset manually:

```shell
python ft_qwen3_4b.py \
  --subset [subset] \
  --epochs [epochs] \
  --wandb
```

### Evaluation

You can evaluate all subsets at once using the provided script:

```shell
bash scripts/eval_all.sh [--local] [--quick]

TQDM_DISABLE=1 bash ./scripts/eval_all.sh > ../logs/eval_all.log 2>&1 &. # Execute in background
```

- `--local`: Evaluate the latest fine-tuned model for each subset. The script automatically detects the latest checkpoint directory (e.g., `./models/[subset]/[date]/checkpoint-final`).
- `--quick`: Passes the `--quick` flag to `eval.py` for faster evaluation (if supported).

If `--local` is not specified, the script evaluates the pre-trained model.

To evaluate a single subset manually:

```shell
# Pre-trained model
python eval.py --subset [subset]

# Fine-tuned model (replace [date] with the actual directory name, e.g., 0901_2115)
python eval.py \
  --subset [subset] \
  --model_path "./models/[subset]/[date]/checkpoint-final"
```
