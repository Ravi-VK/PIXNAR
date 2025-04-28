# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import warnings
import os
import math
import sys
import csv
import glob
from dataclasses import dataclass, field
from typing import Optional, Union, Any, Dict, List, NamedTuple, Sequence, Tuple

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric
from scipy.sparse import load_npz, save_npz, csr_matrix

import tokenmonster
import json
import pickle
import base64

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    M2M100Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    BatchEncoding,
)
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl
from clover.modeling.tokenizers import EmptyWrapper, AutoTokenizerWrapper, TokenMonsterWrapper, PreTrainedTokenizerFastWrapper, DebertaV2TokenizerWrapper
import dataclasses

from clover.modeling.nar import DeBertaNARBoWClusterInference, DeBertaNARBoWClusterTrain, DeBertaNARBoWShortlist
from clover.trainer import NARTrainer
from transformers.file_utils import is_offline_mode, PaddingStrategy
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.modeling_utils import load_sharded_checkpoint, load_state_dict
from safetensors.torch import load_file as load_safetensors_state_dict, load_file
from torch.optim.lr_scheduler import _LRScheduler

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.16.0.dev0")

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["NCCL_DEBUG"] = "WARN"

MULTILINGUAL_TOKENIZERS = [MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]

require_version("datasets>=1.8.0", "To fix: pip install datasets>=1.8.0")

logger = logging.getLogger(__name__)
logging.getLogger("datasets").setLevel(logging.ERROR)
datasets.logging.set_verbosity_error()
warnings.simplefilter(action='ignore', category=FutureWarning)

# import faulthandler
# faulthandler.dump_traceback_later(30, repeat=True)

def load_split_safetensors(path, device="cpu"):
    state_dict = {}
    if glob.glob(f"{path}/model-*.safetensors"):  # Split files exist
        for file in sorted(glob.glob(f"{path}/model-*.safetensors")):
            state_dict.update(load_file(file, device=device))
    else:  # Single file
        state_dict = load_file(f"{path}/model.safetensors", device=device)
    return state_dict


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    model_type: str = field(
        metadata={
            "help": "Type of the encoder model (default: bert) choices: [deberta]"
        },
        default="bert",
    )
    target_tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained target tokenizer name or path if not the same as model_name. If None same as source tokenizer"
        },
    )
    source_tokenizer_lib: Optional[str] = field(
        default="hf",
        metadata={"help": "Target tokenizer library (hf/tokenizers) default: hf"},
    )
    target_tokenizer_lib: Optional[str] = field(
        default="hf",
        metadata={"help": "Target tokenizer library (hf/tokenizers) default: hf"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    no_el_attention: bool = field(
        default=False,
        metadata={
            "help": "Do not use the clover-modified transfomer (uses public transformer)"
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )
    num_vocabs: Optional[int] = field(
        default=2,
        metadata={
            "help": "How many vocabs the matryoshka model expects"
        }
    )
    token_graph_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the scipy csr_matrix stored as a npz file."
        }
    )
    phrase_encoder_input_ids_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the phrase encoder input ids npy file."
        }
    )
    phrase_encoder_attention_mask_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the phrase encoder attention mask npy file."
        }
    )
    window_size: Optional[int] = field(
        default=128,
        metadata={
            "help": "Window size for screening matrix."
        }
    )
    num_clusters: Optional[int] = field(
        default=4096,
        metadata={
            "help": "Number of clusters."
        }
    )
    cluster_vectors_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to cluster vectors npy file (num_clusters, hidden_dim). If not provided, will be initialized randomly."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    data_schema: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma seperated list of columns in train/validation/test files."
        },
    )
    input_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the input texts (e.g. search queries)."
        },
    )
    target_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing target texts (e.g. close variant keywords)."
        },
    )
    score_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing main scores."
        },
    )
    extra_score_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing additional scores"
        }
    )
    input_column_to_output: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the input texts to write "
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (tsv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    pretokenized: bool = field(
        default=False,
        metadata={"help": "Add this flag if your data is already tokenized (columns contain comma separated ids)"},
    )
    source_pad_token_id: Optional[int] = field(
        default=-100,
        metadata={
            "help": "Input tokenizer pad token id"
        }
    )
    target_pad_token_id: Optional[int] = field(
        default=-100,
        metadata={
            "help": "Output tokenizer pad token id"
        }
    )
    preprocessed: bool = field(
        default=False,
        metadata={"help": "Add this flag if your data is already preprocessed and stored in arrow format"},
    )
    use_streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use streaming for dataset loading."},
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=16,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=16,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    max_pad_length: Optional[int] = field(
        default=3,
        metadata={"help": "The maximum tokens to pad during training"},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    lang: str = field(default="en_XX", metadata={"help": "Language id for seq2seq."})

    source_prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )
    forced_bos_token: Optional[str] = field(
        default="en_XX",
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    num_keywords_per_query: Optional[int] = field(
        default=100,
        metadata={"help": "Number of keywords per query."},
    )

    def to_dict(self):
        return dataclasses.asdict(self)

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

@dataclass
class NARTrainingArguments(Seq2SeqTrainingArguments):
    """
    Arguments pertaining to the training loop.
    """
    num_negatives: Optional[int] = field(
        default = None,
        metadata={
            "help": "Number of random negatives considered while computing InfoNCE loss"
        }
    )
    vocab_balancing_factor: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Controls which is the more important objective - larger vocab generation or smaller vocab generation. Set to value between 0 and 1"
        },
    )
    norm_loss_scaling_factor: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "The scaling factor for the norm loss"
        },
    )
    sequence_loss_scaling_factor: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "The scaling factor for the sequence loss"
        },
    )
    extra_sequence_loss_scaling_factor: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "The scaling factor for the additional sequence loss"
        },
    )
    bow_loss_scaling_factor: Optional[float] = field(
        default=0.25,
        metadata={
            "help": "The scaling factor for the sequence loss"
        },
    )
    label_score_threshold: Optional[float] = field(
        default=0.7,
        metadata={
            "help": "Labels with a score higher than this are considered positives"
        }
    )
    length_norm_factor: Optional[float] = field(
        default=2.5,
        metadata={
            "help": "The length normalization factor for calculating the model score for a given input"
        },
    )
    negative_class_weight: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "The weight for the BCE loss for non-label tokens"
        }
    )
    predict_unique_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Predict the unique label tokens at each position as a one vs. all classification problem"
        }
    )
    screen_size: Optional[int] = field(
        default=1000,
        metadata={
            "help": "Number of shortlisted tokens."
        }
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Temperature for training screening matrix."
        }
    )
    epochs_per_cycle: Optional[int] = field(
        default=1,
        metadata={"help": "Number of epochs for the first restart."},
    )
    lr_min: Optional[float] = field(
        default=1e-6,
        metadata={"help": "Minimum learning rate."},
    )
    cycle_mul: Optional[float] = field(
        default=1.0,
        metadata={"help": "Multiplier for cycle length after each restart."},
    )
    cycle_decay: Optional[float] = field(
        default=1.0,
        metadata={"help": "Decay factor for cycle length after each restart."},
    )
    cycle_limit: Optional[int] = field(
        default=1,
        metadata={"help": "Number of restarts."},
    )


@dataclass
class NARTrainingArgumentsExtended(NARTrainingArguments):
    """
    Arguments pertaining to the training loop.
    """
    max_target_length: Optional[int] = field(
        default=16,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_pad_length: Optional[int] = field(
        default=3,
        metadata={"help": "The maximum tokens to pad during training"},
    )


@dataclass
class PredictionArguments:
    beam_topk: Optional[int] = field(
        default=300,
        metadata={
            "help": "The number of highest probability vocabulary tokens to keep for top-k beam search. Between 1 and vocab_size. Default to 300"
        },
    )
    apply_activation_at_inference: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Specifies with SoftMax, Sigmoid or some other activation function will be applied."
        },
    )
    use_bow_shortlisting: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether or not to use the shortlist embedding for inference."
        },
    )
    cluster_size: Optional[int] = field(
        default=20000,
        metadata={
            "help": "Number of tokens in each cluster subset of the vocabulary."
        }
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            NARTrainingArguments,
            PredictionArguments,
        )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, prediction_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            prediction_args,
        ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if not model_args.no_el_attention:
        from clover import modeling

        logger.info(f"Using clover-modified huggingface")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            logger.warn(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.input_column_to_output is None:
        data_args.input_column_to_output = data_args.input_column

    # Get the datasets: you can either provide your own TSV training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For TSV files this script will use the first column for the input texts and the second column for the
    # target text (unless you specify column names for this with the `input_column` and `target_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            streaming=data_args.use_streaming
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = glob.glob(data_args.train_file)
        if data_args.validation_file is not None:
            data_files["validation"] = glob.glob(data_args.validation_file)
        if data_args.test_file is not None:
            data_files["test"] = glob.glob(data_args.test_file)

        if data_args.preprocessed:
            raw_datasets = load_dataset(
                "arrow",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                streaming=data_args.use_streaming
            )
        else:
            kwargs = {}
            if not data_args.use_streaming:
                kwargs['num_proc'] = data_args.preprocessing_num_workers
            raw_datasets = load_dataset(
                "csv",
                column_names=data_args.data_schema.split(","),
                sep="\t",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                quoting=csv.QUOTE_NONE,
                streaming=data_args.use_streaming,
                **kwargs
            )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    model_cls_dict = {
        "deberta_bow_shortlist": DeBertaNARBoWShortlist,
        "deberta_bow_cluster_train": DeBertaNARBoWClusterTrain,
        "deberta_bow_cluster_inference": DeBertaNARBoWClusterInference
    }
    assert (
        model_args.model_type in model_cls_dict
    ), "Model type must be one of {}".format(",".join(model_cls_dict.keys()))
    model_cls = model_cls_dict[model_args.model_type]

    if model_args.source_tokenizer_lib == "tokenizers":
        tokenizer = PreTrainedTokenizerFastWrapper(tokenizer_file=model_args.tokenizer_name)
        tokenizer.add_special_tokens(
            {
                "sep_token": "[SEP]",
                "eos_token": "[SEP]",
                "bos_token": "[CLS]",
                "cls_token": "[CLS]",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
            }
        )
    elif model_args.source_tokenizer_lib == "tokenmonster":
        tokenizer = TokenMonsterWrapper(model_args.tokenizer_name)
    elif model_args.source_tokenizer_lib == "deberta_spm":
        tokenizer = DebertaV2TokenizerWrapper(model_args.tokenizer_name)
    elif model_args.source_tokenizer_lib == "bypass_tokenizer":
        assert data_args.pretokenized
        tokenizer = EmptyWrapper(data_args.source_pad_token_id)
    else:
        tokenizer = AutoTokenizerWrapper(model_args.tokenizer_name)

    if model_args.target_tokenizer_name is not None:
        if model_args.target_tokenizer_lib == "tokenizers":
            target_tokenizer = PreTrainedTokenizerFastWrapper(tokenizer_file=model_args.target_tokenizer_name)
            target_tokenizer.add_special_tokens(
                {
                    "sep_token": "[SEP]",
                    "eos_token": "[SEP]",
                    "bos_token": "[CLS]",
                    "cls_token": "[CLS]",
                    "unk_token": "[UNK]",
                    "pad_token": "[PAD]",
                }
            )
            target_tokenizer.add_special_tokens_to_label = False
        elif model_args.target_tokenizer_lib == "tokenmonster":
            target_tokenizer = TokenMonsterWrapper(model_args.target_tokenizer_name)
        elif model_args.target_tokenizer_lib == "deberta_spm":
            target_tokenizer = DebertaV2TokenizerWrapper(model_args.target_tokenizer_name)
            target_tokenizer.add_special_tokens_to_label = False
        elif model_args.target_tokenizer_lib == "bypass_tokenizer":
            assert data_args.pretokenized
            target_tokenizer = EmptyWrapper(data_args.target_pad_token_id)
        else:
            target_tokenizer = AutoTokenizerWrapper(model_args.target_tokenizer_name)
            target_tokenizer.add_special_tokens_to_label = False
    else:
        target_tokenizer = tokenizer
    
    # print('Source vocab size', len(tokenizer.tokenizer))
    # print('Target vocab size', len(target_tokenizer.tokenizer))

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if model_args.model_type == 'deberta_bow_shortlist':
        config.norm_loss_scaling_factor = training_args.norm_loss_scaling_factor
        config.bow_loss_scaling_factor = training_args.bow_loss_scaling_factor
        config.screen_size = training_args.screen_size
        config.use_bow_shortlisting = prediction_args.use_bow_shortlisting
    if model_args.model_type == 'deberta_bow_cluster_train':
        config.num_clusters = model_args.num_clusters
    if model_args.model_type == 'deberta_bow_cluster_inference':
        if not hasattr(config, "num_clusters"):
            config.num_clusters = model_args.num_clusters
        config.screen_size = training_args.screen_size
        config.use_bow_shortlisting = prediction_args.use_bow_shortlisting

    config.topk = prediction_args.beam_topk
    config.apply_activation_at_inference = prediction_args.apply_activation_at_inference
    if (
        os.path.isfile(model_args.model_name_or_path)
        and os.path.splitext(model_args.model_name_or_path)[1] == ".json"
    ):
        model = model_cls(config)
    else:
        model = model_cls.from_pretrained(model_args.model_name_or_path, config=config)
    # print number of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_trainable_params}")

    
    if model_args.model_type == 'deberta_bow_cluster_train':
        assert training_args.do_train
        if not hasattr(model.config, "initialized_clusterer"):
            print('Initializing clustering module')
            cluster_vectors = None
            if model_args.cluster_vectors_path is not None:
                cluster_vectors = torch.from_numpy(np.load(model_args.cluster_vectors_path)).to(model.dtype).to(model.device)
            model.initialize_clusterer(cluster_vectors=cluster_vectors)
            model.config.initialized_clusterer = True
        else:
            print('Using pre-initialized clusterer')

    babel_model = False

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    
    if not data_args.preprocessed:
        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        if data_args.dataset_name is None:
            column_names=data_args.data_schema.split(",")
        elif training_args.do_train:
            column_names = raw_datasets["train"].column_names
        elif training_args.do_eval:
            column_names = raw_datasets["validation"].column_names
        elif training_args.do_predict:
            column_names = raw_datasets["test"].column_names
        else:
            logger.info(
                "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
            )
            return

        # Get the column names for input/target.
        input_column = data_args.input_column
        if input_column not in column_names:
            raise ValueError(
                f"--input_column' value '{data_args.input_column}' needs to be one of: {', '.join(column_names)}"
            )

        target_column = data_args.target_column
        if target_column is not None and target_column not in column_names:
            raise ValueError(
                f"--target_column' value '{data_args.target_column}' needs to be one of: {', '.join(column_names)}"
            )

        score_column = data_args.score_column
        if score_column is not None and score_column not in column_names:
            raise ValueError(
                f"--score_column' value '{data_args.score_column}' needs to be one of: {', '.join(column_names)}"
            )

        extra_score_column = data_args.extra_score_column
        if extra_score_column is not None and extra_score_column not in column_names:
            raise ValueError(
                f"--extra_score_column' value '{data_args.extra_score_column}' needs to be one of: {', '.join(column_names)}"
            )
        
        source_pad_token_id = tokenizer.pad_token_id
        target_pad_token_id = target_tokenizer.pad_token_id
        padding = "max_length" if data_args.pad_to_max_length else False

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    model.config.output_pad_token_id = target_tokenizer.pad_token_id

   
    def string_to_list(data):
        return np.fromstring(data, sep=',', dtype=int)

    # Optimized batch padding
    def pad_batch(batch, max_length, pad_token_id, padding):
        max_len = max_length if padding=='max_length' else max(len(tokens) for tokens in batch)

        # Vectorized padding
        padded_tokens = np.full((len(batch), max_len), pad_token_id, dtype=int)
        attention_masks = np.zeros((len(batch), max_len), dtype=int)

        for i, tokens in enumerate(batch):
            trunc_len = min(len(tokens), max_len)
            padded_tokens[i, :trunc_len] = tokens[:trunc_len]
            attention_masks[i, :trunc_len] = 1

        return {'input_ids': torch.tensor(padded_tokens), 'attention_mask': torch.tensor(attention_masks)}
    
    def preprocess_pretokenized_func(examples):
        inputs, targets = [], []

        for i in range(len(examples[input_column])):
            if target_column is not None and examples[input_column][i] is not None and examples[target_column][i] is not None:
                try:
                    input_data = string_to_list(examples[input_column][i])
                    sep_seq = '|__|'
                    labels = examples[target_column][i].split(sep_seq)
                    labels_data = list(map(string_to_list, labels))
                    inputs.append(input_data)
                    targets.append(labels_data)
                except:
                    print('Error in processing', examples[input_column][i], examples[target_column][i])
            elif target_column is None and examples[input_column][i] is not None:
                input_data = string_to_list(examples[input_column][i])
                inputs.append(input_data)

        model_inputs = pad_batch(
            inputs,
            max_length=data_args.max_source_length,
            pad_token_id=source_pad_token_id,
            padding="max_length",
        )
        if target_column is not None:
            batch_size = len(targets)
            label_data = np.full((batch_size, data_args.num_keywords_per_query, data_args.max_target_length), target_pad_token_id, dtype=int)
            keyword_masks = np.full((batch_size, data_args.num_keywords_per_query), 0, dtype=bool)

            for i, target in enumerate(targets):
                target = target[:data_args.num_keywords_per_query]
                for j, tokens in enumerate(target):
                    trunc_len = min(len(tokens), data_args.max_target_length)
                    label_data[i, j, :trunc_len] = tokens[:trunc_len]
                    keyword_masks[i, j] = True

            model_inputs['keyword_mask'] = keyword_masks
            model_inputs['labels'] = label_data

        return model_inputs

    def preprocess_func(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []

        for i in range(len(examples[input_column])):
            if target_column is not None:  ## Process both inputs and targets
                if (
                    examples[input_column][i] is not None
                    and examples[target_column][i] is not None
                ):
                    inputs.append(examples[input_column][i])
                    sep_seq = '|__|'
                    target_line = examples[target_column][i].split(sep_seq)
                    targets.append(target_line)
            else:
                if examples[input_column][i] is not None:
                    inputs.append(examples[input_column][i])

        model_inputs = tokenizer.tokenize_data(
            inputs,
            max_length=data_args.max_source_length,
            padding="max_length",
        )

        if target_column is not None:
            for key in model_inputs.keys():
                model_inputs[key] = torch.tensor(model_inputs[key])
            
            flattened_keywords = []
            keyword_masks = []
            for i in range(len(targets)):
                keywords = targets[i][:data_args.num_keywords_per_query]
                keyword_masks.append([True] * len(keywords) + [False] * (data_args.num_keywords_per_query - len(keywords)))
                keywords = keywords + [""] * (data_args.num_keywords_per_query - len(keywords))
                flattened_keywords.extend(keywords)
            
            model_inputs["keyword_mask"] = torch.tensor(keyword_masks)

            model_inputs["labels"] = target_tokenizer.tokenize_data(
                flattened_keywords,
                max_length=data_args.max_target_length,
                padding=padding,
            )["input_ids"].reshape(-1, data_args.num_keywords_per_query, data_args.max_target_length)

        return model_inputs
    
    if data_args.pretokenized:
        preprocess_function = preprocess_pretokenized_func
    else:
        preprocess_function = preprocess_func
            
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # print example from train dataset
        #if target_column is not None:
        #    print('input and targets example', train_dataset[0][input_column], train_dataset[0][target_column])
        kwargs = {}
        if not data_args.use_streaming:
            kwargs["num_proc"] = None if babel_model else data_args.preprocessing_num_workers
            kwargs["load_from_cache_file"] = not data_args.overwrite_cache
            kwargs["desc"] = "Running tokenizer on train dataset"
        
        if not data_args.preprocessed:
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    preprocess_function,
                    batched=True,
                    remove_columns=column_names,
                    **kwargs
                )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    # Initialize our Trainer
    new_hf_parser = HfArgumentParser((NARTrainingArgumentsExtended,))
    training_args = new_hf_parser.parse_dict({**training_args.to_dict(), **data_args.to_dict()}, allow_extra_keys=True)[0]
    # training_args.max_target_length = data_args.max_target_length
    # training_args.max_pad_length = data_args.max_pad_length

    trainer = NARTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        train_dataset_size = 0
        if not data_args.use_streaming:
            train_dataset_size = len(train_dataset)

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else train_dataset_size
        )
        metrics["train_samples"] = min(max_train_samples, train_dataset_size)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
