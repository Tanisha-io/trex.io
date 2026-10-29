"""Transformers TrainingArguments that work on Colab's mixed library versions."""
from __future__ import annotations

import inspect

import torch
from transformers import TrainingArguments


def make_training_args(output_dir: str, epochs: int, batch_size: int, lr: float, **extra) -> TrainingArguments:
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=50,
        report_to=[],
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
        save_total_limit=1,  # avoid filling disk with every epoch checkpoint
    )
    kwargs.update(extra)
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy", "epoch")
    else:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy", "epoch")
    if "save_total_limit" not in params:
        kwargs.pop("save_total_limit", None)
    return TrainingArguments(**kwargs)
