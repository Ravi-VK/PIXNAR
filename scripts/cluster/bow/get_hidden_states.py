import argparse
import os
from transformers import PreTrainedModel, AutoModel, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import tokenmonster

# To execute the script with the values filled in:
# python scripts/cluster/bow/get_hidden_states.py \
#     --model_path /data_ecstorage/MINDER/experiments/MSMARCO/akash_opt_H100_deberta_v3_base_5M_bow_ms_no_downproj \
#     --tokenizer_name microsoft/deberta-v3-base \
#     --tokenizer_lib hf \
#     --cluster_queries /data_ecstorage/MINDER/data/MSMARCO/cluster_queries.txt \
#     --output_path msmarco_5M_bow_hidden_states.npy

class DeBertaNARBoWShortlist(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
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

def get_bow_hidden_state(
    model,
    input_ids,
    attention_mask,
):
    with torch.no_grad():
        last_hidden_state = model.deberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if model.use_down_proj:
            last_hidden_state = model.down_proj(last_hidden_state)
        
        bow_hidden_state = last_hidden_state[:, 0, :]

        return bow_hidden_state

def get_model(model_dir):
    config = AutoConfig.from_pretrained(model_dir)
    model = DeBertaNARBoWShortlist.from_pretrained(model_dir, config=config).to(torch.float16)
    model.eval()
    model.to("cuda")
    return model

class TokenMonsterWrapper:
    def __init__(self, vocab_file, pad_token='[PAD]', sep_token='[SEP]'):
        if vocab_file.endswith('.yaml'):
            with open(vocab_file, 'rb') as f:
                self.tokenizer = tokenmonster.new(f.read())
        else:
            self.tokenizer = tokenmonster.load(vocab_file)

        self.pad_token = pad_token
        self.pad_token_id = self.tokenizer.token_to_id(pad_token)
        self.sep_token = sep_token
        self.sep_token_id = self.tokenizer.token_to_id(sep_token)
        self.padding_side = 'right'
    
    def pad_and_truncate(self, tokens, max_length):
        tokens = tokens[:max_length]
        attention_mask = np.zeros((max_length,))
        attention_mask[:len(tokens)] = 1
        padded_tokens = np.full((max_length,), self.pad_token_id)
        padded_tokens[:len(tokens)] = tokens
        return padded_tokens, attention_mask
    
    def __call__(self, sentences, padding, truncation, max_length, return_tensors):
        tokens, attention_masks = [], []
        tokenized_batch = self.tokenizer.tokenize(sentences)
        pad_length = max_length if padding == 'max_length' else max(len(tokens) for tokens in tokenized_batch)
        tokenized_batch = [self.pad_and_truncate(tokens, pad_length) for tokens in tokenized_batch]
        for padded_tokens, attention_mask in tokenized_batch:
            tokens.append(padded_tokens)
            attention_masks.append(attention_mask)
        tokenized_data = {
            'input_ids': torch.tensor(tokens),
            'attention_mask': torch.tensor(attention_masks)
        }
        return tokenized_data

def main():
    parser = argparse.ArgumentParser(description="Process model path, tokenizer, and cluster queries.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--tokenizer_name', type=str, required=True, help='Tokenizer name or path')
    parser.add_argument('--tokenizer_lib', type=str, required=True, choices=['hf', 'tokenmonster'], help='Tokenizer library to use')
    parser.add_argument('--cluster_queries', type=str, required=True, help='Path to the cluster queries file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output file')
    
    args = parser.parse_args()
    
    model = get_model(args.model_path)
    
    if args.tokenizer_lib == 'hf':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = TokenMonsterWrapper(args.tokenizer_name)
    
    cluster_queries = []
    with open(args.cluster_queries, 'r') as f:
        for line in f:
            cluster_queries.append(line.strip())
    
    batch_size = 16000
    bow_hidden_states = []
    for i in tqdm(range(0, len(cluster_queries), batch_size)):
        batch_queries = cluster_queries[i:i+batch_size]
        processed_queries = tokenizer(batch_queries, padding="max_length", truncation=True, max_length=16, return_tensors="pt")
        input_ids, attention_mask = processed_queries["input_ids"], processed_queries["attention_mask"]
        
        bow_hidden_state = get_bow_hidden_state(model, input_ids.to(model.device), attention_mask.to(model.device)).detach().cpu()
        bow_hidden_states.append(bow_hidden_state)
    
    bow_hidden_states = torch.cat(bow_hidden_states, dim=0)
    
    np.save(args.output_path, bow_hidden_states.numpy())

if __name__ == "__main__":
    main()
