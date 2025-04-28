import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    SequentialSampler,
    RandomSampler
)
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import (
    EvalPrediction,
    EvalLoopOutput,
    PredictionOutput,
    denumpify_detensorize,
    speed_metrics,
)
from multiprocessing import Process, Queue
try:
    from transformers.file_utils import is_torch_xla_available
except ImportError:
    from transformers.file_utils import is_torch_tpu_available as is_torch_xla_available
import math
import time
import random
import numpy as np
from tqdm import tqdm

class NARTrainer(Seq2SeqTrainer):
    def training_step(self, *args, **kwargs):
        if hasattr(self.model, 'debug_next_step'):
            debug_interval = 1000
            if self.state.global_step % debug_interval == 0 and self.model.device.index == 0:
                self.model.debug_next_step = True
            elif self.state.global_step % debug_interval == 1 and self.model.device.index == 0:
                self.log(self.model.debug_results)

        return super().training_step(*args, **kwargs)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, reduce_loss=True):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """

        inputs["do_training"] = True
        inputs["max_seq_length"] = self.args.max_target_length
        inputs["label_smoothing"] = self.args.label_smoothing_factor
        inputs["max_pad_length"] = self.args.max_pad_length

        outputs = model(**inputs)
        loss = outputs[1]

        self.loss_components = loss

        total_loss = loss
        if isinstance(loss, torch.Tensor):
            total_loss = loss
        elif isinstance(loss, tuple):
            total_loss = sum(loss)
        elif isinstance(loss, dict):
            total_loss = sum(loss.values())

        return (total_loss, outputs) if return_outputs else total_loss

    def _gather_mean(self, val):
        return self._nested_gather(val).mean().item()

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            if isinstance(self.loss_components, tuple):
                for i in range(len(self.loss_components)):
                    logs[f"loss_{i}"] = self._gather_mean(self.loss_components[i])
            elif isinstance(self.loss_components, dict):
                for key in self.loss_components.keys():
                    logs[key] = self._gather_mean(self.loss_components[key])

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )
