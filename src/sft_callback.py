import math

from transformers import TrainerCallback


class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # During training: convert loss to train_ppl
        if logs is not None and "loss" in logs and logs["loss"] is not None:
            try:
                logs["train_ppl"] = math.exp(float(logs["loss"]))
            except (OverflowError, ValueError):
                logs["train_ppl"] = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # During evaluation: convert eval_loss to eval_ppl (also used for best model selection)
        if metrics is not None and "eval_loss" in metrics and metrics["eval_loss"] is not None:
            try:
                metrics["eval_ppl"] = math.exp(float(metrics["eval_loss"]))
            except (OverflowError, ValueError):
                metrics["eval_ppl"] = float("inf")
        # No need to return metrics here; it's updated by reference and recorded by Trainer as-is.
