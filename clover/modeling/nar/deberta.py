from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.models.distilbert.modeling_distilbert import Transformer as DistilBertTransformer
from torch import nn, einsum
import torch.nn.functional as F
import torch
import os
import math
import json
import numpy as np
import time
from scipy.sparse import load_npz, save_npz, csr_matrix
# import faiss
# import faiss.contrib.torch_utils
import gc
import copy

class DeBertaNARPretrainedGenerationModel(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor(
            [[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device
        )
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class DeBertaNARGenerationModel(DeBertaNARPretrainedGenerationModel):
    def __init__(self, config):
        super().__init__(config)
        self.deberta = AutoModel.from_config(config)
        no_bias = hasattr(self.config, "no_bias") and self.config.no_bias
        self.proj = nn.Linear(self.config.hidden_size, self.config.target_vocab_size, bias=not no_bias)
        self.post_init()

    def pad_tensor(self, tensor, max_seq_length, pad_value, max_pad_length=3):
        batch_size, seq_length = tensor.size()
        if seq_length >= max_seq_length:
            return tensor[:, :max_seq_length]

        pad_length = min(max_seq_length - seq_length, max_pad_length)
        pad_tensor = tensor.new_zeros(batch_size, pad_length).fill_(pad_value)
        tensor = torch.cat([tensor, pad_tensor], dim=1)
        return tensor

    def pad_to_target_tensor(self, tensor, target_length, pad_value):
        batch_size, seq_length = tensor.size()
        if seq_length >= target_length:
            return tensor[:, :target_length]

        pad_length = target_length - seq_length
        pad_tensor = tensor.new_zeros(batch_size, pad_length).fill_(pad_value)
        tensor = torch.cat([tensor, pad_tensor], dim=1)
        return tensor

class DeBertaNARBoWShortlist(DeBertaNARGenerationModel):
    def __init__(self, config):
        super().__init__(config)
        if hasattr(self.config, "hf_checkpoint_name"):
            self.deberta = AutoModel.from_pretrained(self.config.hf_checkpoint_name)
        else:
            self.deberta = AutoModel.from_config(config)
        self.use_down_proj = False
        self.lm_proj_size = self.config.hidden_size
        if hasattr(self.config, "lm_bottleneck_size"):
            self.use_down_proj = True
            self.lm_proj_size = self.config.lm_bottleneck_size
            self.down_proj = nn.Linear(self.config.hidden_size, self.config.lm_bottleneck_size)
        
        no_bias = hasattr(self.config, "no_bias") and self.config.no_bias
        self.proj = nn.Linear(self.lm_proj_size, self.config.target_vocab_size, bias=not no_bias)
        self.debug_next_step = False
        self.debug_results = {}
        self.post_init()
    
    def forward(
            self,
            input_ids,
            attention_mask,
            labels=None,
            keyword_mask=None,
            label_smoothing=0,
            max_seq_length=16,
            max_pad_length=16,
            do_training=False,
        ):
            last_hidden_state = self.deberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if labels is not None:
                last_hidden_state = last_hidden_state[:, :labels.size(-1) + 1, :]
            if self.use_down_proj:
                last_hidden_state = self.down_proj(last_hidden_state)
            
            if do_training:
                label_token_mask = torch.full(labels.shape, True, dtype=torch.bool, device=labels.device)
                label_token_mask[labels == -100] = False
                label_token_mask[~keyword_mask] = False
                labels[labels == -100] = 0

                bow_hidden_state, last_hidden_state = last_hidden_state[:, 0, :], last_hidden_state[:, 1:, :]
                bow_lprobs = self.proj(bow_hidden_state).log_softmax(dim=-1)
                logits = self.proj(last_hidden_state)

                bow_loss = -bow_lprobs.gather(-1, labels.view(input_ids.size(0), -1))[label_token_mask.view(input_ids.size(0), -1)].mean()
                self_norm_loss = logits.exp().sum(dim=-1, keepdim=True).log().square().expand(-1, -1, labels.size(-2))[label_token_mask.transpose(-1, -2)].mean()

                lprobs = logits.log_softmax(dim=-1)

                nll_loss = -lprobs.gather(-1, labels.transpose(-1, -2)).transpose(-1, -2)[label_token_mask].mean()

                if self.debug_next_step:
                    topk_indices = lprobs.topk(k=self.config.topk, dim=-1).indices.tolist() # shape: batch_size x seq_len x topk .view(-1, self.config.topk).tolist()
                    screened_indices = bow_lprobs.topk(k=10000, dim=-1).indices.tolist() # shape: batch_size x screenk .view(-1, 10000).tolist()

                    mean_coverage_2000, mean_coverage_10000 = 0, 0
                    for i in range(len(topk_indices)):
                        screened_indices_set_2000 = set(screened_indices[i][:2000])
                        screened_indices_set_10000 = set(screened_indices[i][:10000])
                        for j in range(min(8, len(topk_indices[i]))):
                            topk_indices_set = set(topk_indices[i][j])
                            mean_coverage_2000 += len(topk_indices_set.intersection(screened_indices_set_2000)) / len(topk_indices_set)
                            mean_coverage_10000 += len(topk_indices_set.intersection(screened_indices_set_10000)) / len(topk_indices_set)

                    mean_coverage_2000 /= len(topk_indices) * min(8, len(topk_indices[i]))
                    mean_coverage_10000 /= len(topk_indices) * min(8, len(topk_indices[i]))

                    self.debug_results = {
                        'nll_loss': nll_loss.item(),
                        'self_norm_loss': self_norm_loss.item(),
                        'bow_loss': bow_loss.item(),
                        'mean_coverage_2000': mean_coverage_2000,
                        'mean_coverage_10000': mean_coverage_10000,
                    }
                    self.debug_next_step = False

                loss = (nll_loss, self.config.norm_loss_scaling_factor * self_norm_loss, self.config.bow_loss_scaling_factor * bow_loss)
            else:
                if not self.config.use_bow_shortlisting:
                    last_hidden_state = last_hidden_state[:, 1:, :]
                    lprobs = self.proj(last_hidden_state).log_softmax(dim=-1)
                else:
                    bow_hidden_state, last_hidden_state = last_hidden_state[:, 0, :], last_hidden_state[:, 1:, :]
                    bow_logits = self.proj(bow_hidden_state)

                    bow_shortlist = bow_logits.topk(self.config.screen_size, dim=-1).indices
                    mini_lm_heads = self.proj.weight[bow_shortlist]
                    mini_lm_bias = self.proj.bias[bow_shortlist]
                    aprx_topk_distances, aprx_topk_indices = ((last_hidden_state @ mini_lm_heads.transpose(-1, -2)) + mini_lm_bias.unsqueeze(1)).topk(self.config.topk, dim=-1)
                    ridx = torch.arange(aprx_topk_indices.size(0)).repeat_interleave(aprx_topk_indices.size(1) * aprx_topk_indices.size(2))
                    cidx = aprx_topk_indices.view(-1)
                    aprx_topk_indices = bow_shortlist[ridx, cidx].view(aprx_topk_indices.size())
                    lprobs = (aprx_topk_distances, aprx_topk_indices)

                loss = 0

            return lprobs, loss

class DeBertaNARBoWClusterTrain(DeBertaNARGenerationModel):
    def __init__(self, config):
        super().__init__(config)
        if hasattr(self.config, "hf_checkpoint_name"):
            self.deberta = AutoModel.from_pretrained(self.config.hf_checkpoint_name)
        else:
            self.deberta = AutoModel.from_config(config)
        self.deberta.requires_grad_(False)
        self.use_down_proj = False
        self.lm_proj_size = self.config.hidden_size
        if hasattr(self.config, "lm_bottleneck_size"):
            self.use_down_proj = True
            self.lm_proj_size = self.config.lm_bottleneck_size
            self.down_proj = nn.Linear(self.config.hidden_size, self.config.lm_bottleneck_size)
            self.down_proj.requires_grad_(False)
        
        self.cluster_embedding = torch.nn.Embedding(self.config.num_clusters, self.lm_proj_size)
        
        no_bias = hasattr(self.config, "no_bias") and self.config.no_bias
        self.proj = nn.Linear(self.lm_proj_size, self.config.target_vocab_size, bias=not no_bias)
        self.debug_next_step = False
        self.debug_results = {}
        self.post_init()

    def initialize_clusterer(self, cluster_vectors=None):
        with torch.no_grad():
            if cluster_vectors is not None:
                self.cluster_embedding.weight.data.copy_(cluster_vectors)
        self.deberta.requires_grad_(False)
    
    def forward(
            self,
            input_ids,
            attention_mask,
            labels=None,
            keyword_mask=None,
            label_smoothing=0,
            max_seq_length=16,
            max_pad_length=16,
            do_training=False,
        ):
            last_hidden_state = self.deberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if labels is not None:
                last_hidden_state = last_hidden_state[:, :labels.size(-1) + 1, :]
            if self.use_down_proj:
                last_hidden_state = self.down_proj(last_hidden_state)
            
            if do_training:
                label_token_mask = torch.full(labels.shape, True, dtype=torch.bool, device=labels.device)
                label_token_mask[labels == -100] = False
                label_token_mask[~keyword_mask] = False
                labels[labels == -100] = 0

                bow_hidden_state = last_hidden_state[:, 0, :]
                closest_clusters = (bow_hidden_state.matmul(self.cluster_embedding.weight.transpose(0, 1))).argmax(dim=-1)
                vectors_of_closest_clusters = self.cluster_embedding(closest_clusters)
                bow_lprobs = self.proj(vectors_of_closest_clusters).log_softmax(dim=-1)
                bow_cluster_loss = -bow_lprobs.gather(-1, labels.view(input_ids.size(0), -1))[label_token_mask.view(input_ids.size(0), -1)].mean() # batch_size x seq_len
                loss = bow_cluster_loss

            return bow_lprobs, loss

class DeBertaNARBoWClusterInference(DeBertaNARGenerationModel):
    def __init__(self, config):
        super().__init__(config)
        self.deberta = AutoModel.from_config(config)
        self.lm_proj_size = self.config.hidden_size
        self.use_down_proj = False
        if hasattr(self.config, "lm_bottleneck_size"):
            self.use_down_proj = True
            self.lm_proj_size = self.config.lm_bottleneck_size
            self.down_proj = nn.Linear(self.config.hidden_size, self.config.lm_bottleneck_size)
        
        self.cluster_embedding = torch.nn.Embedding(self.config.num_clusters, self.lm_proj_size)
        self.register_buffer('vocab_subsets', torch.zeros(self.config.num_clusters, self.config.vocab_subset_size, dtype=torch.long))

        no_bias = hasattr(self.config, "no_bias") and self.config.no_bias
        self.proj = nn.Linear(self.lm_proj_size, self.config.target_vocab_size, bias=not no_bias)
        self.post_init()

    def forward(
            self,
            input_ids,
            attention_mask,
            labels=None,
            keyword_mask=None,
            label_smoothing=0,
            max_seq_length=16,
            max_pad_length=16,
            do_training=False,
        ):
            last_hidden_state = self.deberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if self.use_down_proj:
                last_hidden_state = self.down_proj(last_hidden_state)
            
            bow_hidden_state, last_hidden_state = last_hidden_state[:, 0, :], last_hidden_state[:, 1:, :]

            closest_clusters = (bow_hidden_state.matmul(self.cluster_embedding.weight.transpose(0, 1))).topk(5, dim=-1).indices    # batch_size
            bow_shortlist = self.vocab_subsets[closest_clusters][:, :self.config.screen_size].unique() # (num_clusters x vocab_subset_size) --> (num_clusters x screen_size)
            mini_lm_head = self.proj.weight[bow_shortlist] # screen_size x lm_bottleneck_size
            mini_lm_bias = self.proj.bias[bow_shortlist] # screen_size

            aprx_topk_distances, aprx_topk_indices =  (last_hidden_state @ mini_lm_head.transpose(-1, -2).unsqueeze(0) + mini_lm_bias.unsqueeze(0).unsqueeze(0)).topk(self.config.topk, dim=-1) # batch_size x seq_len x screen_size
            aprx_topk_indices = bow_shortlist[aprx_topk_indices]

            lprobs = (aprx_topk_distances, aprx_topk_indices)
            loss = 0

            return lprobs, loss

