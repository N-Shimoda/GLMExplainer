# GraphToken

## Usage

### Fine-tuning

Execute fine-tuning using QLoRA.

```shell
python ft_qwen3_4b.py \
  --do_train \
  --subset [subset] \
  --output_dir "qwen3-4b-[subset] \
  --wandb
```

### Evaluation

(i) Evaluate the **fine-tuned** model:

```shell
python eval.py \
  --subset [subset] \
  --model_path "./qwen3-4b-[subset]/checkpoint-final"
```

(ii) Evaluate the **pre-trained** model:

```shell
python eval.py --subset [subset]
```
