# GraphToken

## Usage

### Fine-tuning

Execute fine-tuning using QLoRA.

```shell
python ft_qwen3_4b.py \
  --subset [subset] \
  --wandb
```

### Evaluation

(i) Evaluate the **pre-trained** model:

```shell
python eval.py --subset [subset]
```

(ii) Evaluate the **fine-tuned** model:

`[date]` contains `MMDDhhmm` format of the date on which the training was executed.

```shell
python eval.py \
  --subset [subset] \
  --model_path "./qwen3-4b-[subset]/[date]/checkpoint-final"  # path to the checkpoint
```
