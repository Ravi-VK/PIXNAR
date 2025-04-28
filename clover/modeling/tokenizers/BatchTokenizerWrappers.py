import tokenmonster
from transformers import AutoTokenizer, PreTrainedTokenizerFast, DebertaV2Tokenizer
import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import os

def can_convert_to_tensor(lst):
    try:
        torch.tensor(lst)
        return True
    except (ValueError, TypeError):
        return False

class DebertaV2TokenizerWrapper:
    def __init__(self, *args, **kwargs):
        self.tokenizer = DebertaV2Tokenizer(*args, **kwargs)
        self.add_special_tokens_to_label = True
    
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)
    
    def tokenize_data(self, sentences, max_length, padding, chunk_size=1000):
        tokenized_data = {'input_ids': [], 'attention_mask': []}
        for i in range(0, len(sentences), chunk_size):
            tokenized_batch = self.tokenizer.batch_encode_plus(
                sentences[i:i+chunk_size],
                max_length=max_length,
                padding=padding,
                return_tensors="np",
                truncation=True,
                return_token_type_ids=False,
                add_special_tokens=self.add_special_tokens_to_label,
            )
            tokenized_data['input_ids'].append(tokenized_batch['input_ids'])
            tokenized_data['attention_mask'].append(tokenized_batch['attention_mask'])
        tokenized_data['input_ids'] = np.concatenate(tokenized_data['input_ids'], axis=0)
        tokenized_data['attention_mask'] = np.concatenate(tokenized_data['attention_mask'], axis=0)
        return tokenized_data

class AutoTokenizerWrapper:
    def __init__(self, *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
        self.add_special_tokens_to_label = True
    
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)
    
    def tokenize_data(self, sentences, max_length, padding, chunk_size=1000):
        tokenized_data = {'input_ids': [], 'attention_mask': []}
        for i in range(0, len(sentences), chunk_size):
            tokenized_batch = self.tokenizer.batch_encode_plus(
                sentences[i:i+chunk_size],
                max_length=max_length,
                padding=padding,
                return_tensors="np",
                truncation=True,
                return_token_type_ids=False,
                add_special_tokens=self.add_special_tokens_to_label,
            )
            tokenized_data['input_ids'].append(tokenized_batch['input_ids'])
            tokenized_data['attention_mask'].append(tokenized_batch['attention_mask'])
        tokenized_data['input_ids'] = np.concatenate(tokenized_data['input_ids'], axis=0)
        tokenized_data['attention_mask'] = np.concatenate(tokenized_data['attention_mask'], axis=0)
        return tokenized_data

class PreTrainedTokenizerFastWrapper:
    def __init__(self, *args, **kwargs):
        self.tokenizer = PreTrainedTokenizerFast(*args, **kwargs)
        self.add_special_tokens_to_label = True
    
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)
    
    def tokenize_data(self, sentences, max_length, padding, chunk_size=1000):
        tokenized_data = {'input_ids': [], 'attention_mask': []}
        for i in range(0, len(sentences), chunk_size):
            tokenized_batch = self.tokenizer.batch_encode_plus(
                sentences[i:i+chunk_size],
                max_length=max_length,
                padding=padding,
                return_tensors="np",
                truncation=True,
                return_token_type_ids=False,
                add_special_tokens=self.add_special_tokens_to_label,
            )
            tokenized_data['input_ids'].append(tokenized_batch['input_ids'])
            tokenized_data['attention_mask'].append(tokenized_batch['attention_mask'])
        tokenized_data['input_ids'] = np.concatenate(tokenized_data['input_ids'], axis=0)
        tokenized_data['attention_mask'] = np.concatenate(tokenized_data['attention_mask'], axis=0)
        return tokenized_data

class EmptyWrapper:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        self.padding_side = 'right'

    def pad(self, features, padding=None, max_length=None, return_tensors='pt', return_attention_mask=None, pad_to_multiple_of=None, remove_labels=False):
       stacked_features = {}
       for key in features[0]:
           stacked_features[key] = torch.stack([torch.tensor(feature[key]) if not isinstance(feature[key], torch.Tensor) else feature[key] for feature in features])
       return stacked_features
    
    def pad_and_truncate(self, tokens, max_length):
        tokens = tokens[:max_length]
        attention_mask = np.zeros((max_length,))
        attention_mask[:len(tokens)] = 1
        padded_tokens = np.full((max_length,), self.pad_token_id)
        padded_tokens[:len(tokens)] = tokens
        return padded_tokens, attention_mask

    def save_pretrained(self, output_dir):
        pass

class TokenMonsterWrapper(EmptyWrapper):
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
    
    def save_pretrained(self, output_dir):
        save_file = os.path.join(output_dir, 'saved-vocab.yaml')
        with open(save_file, 'wb') as f:
            f.write(self.tokenizer.export_yaml()) 
    
    def tokenize_data(self, sentences, max_length, padding, chunk_size=1000):
        tokens, attention_masks = [], []
        for i in range(0, len(sentences), chunk_size):
            tokenized_batch = self.tokenizer.tokenize(sentences[i:i+chunk_size])
            pad_length = max_length if padding=='max_length' else max(len(tokens) for tokens in tokenized_batch)
            tokenized_batch = [self.pad_and_truncate(tokens, pad_length) for tokens in tokenized_batch]
            for padded_tokens, attention_mask in tokenized_batch:
                tokens.append(padded_tokens)
                attention_masks.append(attention_mask)
        tokenized_data = {
            'input_ids': np.array(tokens),
            'attention_mask': np.array(attention_masks)
        }
        return tokenized_data