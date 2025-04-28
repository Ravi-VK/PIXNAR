import tokenmonster
from transformers import AutoTokenizer, PreTrainedTokenizerFast, DebertaV2Tokenizer
import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import os
from treat import treat


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
    
    def pad(self, features, padding=None, max_length=None, return_tensors='pt', return_attention_mask=None, pad_to_multiple_of=None):
        stacked_features = {}
        for key in features[0]:
            stacked_features[key] = torch.stack([torch.tensor(feature[key]) if not isinstance(feature[key], torch.Tensor) else feature[key] for feature in features])
        return stacked_features

    def save_pretrained(self, output_dir):
        pass

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
    
    def save_pretrained(self, output_dir):
        save_file = os.path.join(output_dir, 'saved-vocab.yaml')
        with open(save_file, 'wb') as f:
            f.write(self.tokenizer.export_yaml())
    
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

class TreatWrapper:
    def __init__(self, vocab_folder, ngrams=5, get_topk=1):
        self.tokenizer = CustomTokenizer(ngrams, decoding="viterbi", get_topk=get_topk)
        self.tokenizer.from_file(vocab_folder)    
        
    def save_pretrained(self, output_dir):
        self.tokenizer.save_tokenizer(output_dir)
    
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
    
    def tokenize_data(self, sentences, max_length, padding, chunk_size=1000 **kwargs):
        tokens, attention_masks = [], []
        for i in range(0, len(sentences), chunk_size):
            tokenized_batch = self.tokenizer.encode_batch(sentences[i:i+chunk_size], **kwargs)
            pad_length = max_length if padding=='max_length' else max(len(tokens.ids) for tokens in tokenized_batch)
            tokenized_batch = [self.pad_and_truncate(tokens.ids, pad_length) for tokens in tokenized_batch]
            for padded_tokens, attention_mask in tokenized_batch:
                tokens.append(padded_tokens)
                attention_masks.append(attention_mask)
        tokenized_data = {
            'input_ids': np.array(tokens),
            'attention_mask': np.array(attention_masks)
        }
        return tokenized_data


class MatryoshkaTokenMonsterWrapper:
    def __init__(self, num_vocabs, vocab_dir, pad_token='[PAD]', sep_token='[SEP]'):
        self.tokenizer = []
        self.mapping = []
        self.num_vocabs = num_vocabs
        for i in range(num_vocabs):
            vocab_file = os.path.join(vocab_dir, f'vocab_{i+1}.vocab')
            self.tokenizer.append(tokenmonster.load(vocab_file))
            mapping_file = os.path.join(vocab_dir, f'tokenization_map_{i+1}.npy')
            self.mapping.append(np.load(mapping_file))

        self.pad_token = pad_token
        self.pad_token_id = self.tokenizer[0].token_to_id(pad_token)
        self.sep_token = sep_token
        self.sep_token_id = self.tokenizer[0].token_to_id(sep_token)
        self.padding_side = 'right'
    
    def save_pretrained(self, output_dir):
        for i in range(self.num_vocabs):
            save_file = os.path.join(output_dir, f'saved-vocab-{i+1}.vocab')
            self.tokenizer[i].save(save_file)
    
    def pad(self, features, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors='pt', return_attention_mask=True, remove_labels=False):
        padded_features = {}

        # Determine the maximum length
        if max_length is None:
            max_length = max(len(f['input_ids']) for f in features)

        # Ensure max_length is a multiple of pad_to_multiple_of
        if pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
            max_length += (pad_to_multiple_of - max_length % pad_to_multiple_of)

        for key in features[0]:
            if remove_labels and (key == 'labels' or key == 'keyword_mask'):
                continue
            values = [torch.tensor(feature[key][:max_length]) for feature in features]
            pad_value = self.pad_token_id if key == 'input_ids' else 0
            if self.padding_side == 'left':
                padded_values = [torch.cat([torch.tensor([self.pad_token_id] * (max_length - len(v)), dtype=torch.long), v]) for v in values]
            else:  # padding on the right
                padded_values = [torch.cat([v, torch.tensor([self.pad_token_id] * (max_length - len(v)), dtype=torch.long)]) for v in values]

            padded_features[key] = torch.stack(padded_values)

        return padded_features

    
    def pad_and_truncate(self, tokens, max_length):
        tokens = tokens[:max_length]
        attention_mask = np.zeros((max_length,))
        attention_mask[:len(tokens)] = 1
        padded_tokens = np.full((max_length,), self.pad_token_id)
        padded_tokens[:len(tokens)] = tokens
        return padded_tokens, attention_mask
    
    def tokenize_data(self, sentences, max_length, padding, tokenizer_index, chunk_size=1000):
        tokens, attention_masks = [], []
        for i in range(0, len(sentences), chunk_size):
            tokenized_batch = self.tokenizer[tokenizer_index].tokenize(sentences[i:i+chunk_size])
            tokenized_batch = [np.array(list(map(self.mapping[tokenizer_index].__getitem__, x.tolist()))) for x in tokenized_batch]
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
