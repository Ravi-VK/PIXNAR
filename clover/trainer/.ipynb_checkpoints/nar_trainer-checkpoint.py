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
from transformers.trainer_utils import (
    EvalPrediction,
    EvalLoopOutput,
    PredictionOutput,
    denumpify_detensorize,
    speed_metrics,
)

from .dvae_generator_trainer import DVAEGenerationTrainer
from .base_trainer import BaseTrainer
from multiprocessing import Process, Queue
from clover.modeling.data.postprocess import (
    IOProcess,
    NARPostProcess,
    GENERATE_FINISHED,
    POSTPROCESS_FINISHED,
)
try:
    from transformers.file_utils import is_torch_xla_available
except ImportError:
    from transformers.file_utils import is_torch_tpu_available as is_torch_xla_available
import math
import time
import random
import numpy as np
from tqdm import tqdm

class NARTrainer(BaseTrainer):
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

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        beam_topk: int = 300,
        async_write: Optional[bool] = False,
        input_texts: Optional[List[str]] = None,
        output_fname: Optional[str] = None,
        postprocess_workers: Optional[int] = 1,
    ) -> PredictionOutput:
        self.beam_topk = beam_topk
        self.async_write = async_write
        self.output_fname = output_fname
        self.num_return_sequences = 1

        if self.is_world_process_zero() and self.async_write:
            print("Starting Async Write to {}".format(self.output_fname))
            self.fout = open(self.output_fname, "w")
            self.data_queue = Queue()
            self.msg_queue = Queue()
            self.p_list = []

            for _ in range(postprocess_workers):
                p = NARPostProcess(self.data_queue, self.msg_queue)
                self.p_list.append(p)
                p.start()

            self.io_process = IOProcess(self.msg_queue, self.fout, input_texts, 1)
            self.io_process.start()
            print("=" * 30)

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            test_dataloader,
            description="Prediction",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        num_samples = output.num_samples
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=math.ceil(num_samples / total_batch_size),
            )
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(
            predictions=output.predictions,
            label_ids=output.label_ids,
            metrics=output.metrics,
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        reduce_loss: Optional[bool] = True,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using obj:*inputs*.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.autocast_smart_context_manager():
                logits = model(**inputs)[0]
                if type(logits) is tuple:
                    topk_values_and_indices = logits
                else:
                    topk_values, topk_indices = torch.topk(logits, self.beam_topk, dim=-1)
                    topk_values_and_indices = (topk_values, topk_indices)
        return (None, topk_values_and_indices, None)
