# python trie/tokenize_and_convert_to_ids.py --input-file sents.txt --output-file tmoutsents.txt --tokenizer-name ../../../ecstorage/ChatGPTQK/keywordstrict500k-special.yaml --tokenizer-type tokenmonster --start-id 302151
# python trie/tokenize_and_convert_to_ids.py \
#     --input-file sents.txt \
#     --output-file tmoutsents.txt \
#     --tokenizer-name vocab_1.vocab \
#     --tokenizer-type tokenmonster \
#     --apply-mapping True \
#     --tokenization-map tokenization_map_1.npy \
#     --start-id 0

import argparse 
import pickle
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool
import codecs
import sys
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast
)
import tokenmonster
import time

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', help='path to the input file containing sentences that needs to be processed', required=True)
parser.add_argument('--output-file', help='path to save the preprocessed .txt file', required=True)
parser.add_argument('--tokenizer-name', help='huggingface/babel tokenizer name/path', required=True)
parser.add_argument('--tokenizer-type', help='huggingface or tokenizers library to use', choices=["hf", "tokenizers", "tokenmonster"])
parser.add_argument('--apply-mapping', type=lambda x: x.lower()=='true', default=False, help='whether to map token ids to other specified token ids - currently only supported for TokenMonster')
parser.add_argument('--tokenization-map', type=str, default=None, help='a numpy array mapping token ids in the existing tokenizer to new ids')
parser.add_argument('--start-id', type=int, default=0, help='the prefix token to add to the inputs (default: 0), -1 to ignore')
parser.add_argument('--start-column', action='store_true', default=False, help='use the first column as a column with start token ids')
parser.add_argument('--num-lines-block', type=int, default=10000000, help='Number of lines in a block to process')

args = parser.parse_args()

if args.tokenizer_type == "hf":
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
elif args.tokenizer_type == "hf_fast":
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
elif args.tokenizer_type == "tokenizers":
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_name)
    tokenizer.add_special_tokens({"sep_token": "[SEP]", "eos_token": "[SEP]", "bos_token": "[CLS]", "cls_token": "[CLS]", "unk_token": "[UNK]", "pad_token": "[PAD]"})
else:
    if args.tokenizer_name.endswith('.vocab'):
        tokenizer = tokenmonster.load(args.tokenizer_name)
    else:
        with open(args.tokenizer_name, "rb") as f:
            tokenizer = tokenmonster.new(f.read())

class TokenMonsterMappingWrapper:
    def __init__(self, tokenizer, mapping):
        self.tokenizer = tokenizer
        self.mapping = mapping

    def tokenize(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        for i in range(len(tokenized_text)):
            tokenized_text[i] = list(map(self.mapping.__getitem__, tokenized_text[i].tolist()))
        return tokenized_text

if args.apply_mapping:
    assert args.tokenizer_type == "tokenmonster"
    assert os.path.isfile(args.tokenization_map)
    mapping = np.load(args.tokenization_map).tolist()
    tokenizer = TokenMonsterMappingWrapper(tokenizer, mapping)

def preprocess(line):
    token_ids = tokenizer.encode(line, add_special_tokens=False)
    if args.start_id >= 0:
        token_ids = [args.start_id] + token_ids
    return " ".join([str(token_id) for token_id in token_ids])

def preprocess_start_column(line):
    sentence = line.split("\t")[1]
    start_token_id = int(line.split("\t")[0])
    token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    return " ".join([str(token_id) for token_id in [start_token_id] + token_ids])

def hf_process_and_write(lines, fp):
    cpu_count = multiprocessing.cpu_count()
    preprocess_fn = preprocess_start_column if args.start_column else preprocess
    with Pool(cpu_count) as p:
        processed = p.map(preprocess_fn, lines)
    fp.write("\n".join(processed) + "\n")

def tokenmonster_process_and_write(lines, fp):
    if args.start_column:
        start_token_ids, sentences = zip(*[line.split("\t") for line in lines])
    else:
        start_token_ids, sentences = [args.start_id] * len(lines), lines
    token_ids = tokenizer.tokenize(sentences)
    if args.start_column or args.start_id >= 0:
        processed = [" ".join([str(start_token_id)] + [str(token_id) for token_id in token_id_list]) for start_token_id, token_id_list in zip(start_token_ids, token_ids)]
    else:
        processed = [" ".join([str(token_id) for token_id in token_id_list]) for token_id_list in token_ids]
    fp.write("\n".join(processed) + "\n")

process_and_write = hf_process_and_write if args.tokenizer_type == "hf" or args.tokenizer_type == "tokenizers" else tokenmonster_process_and_write

def main():
    f_in = codecs.open(args.input_file,'r', encoding='utf-8')
    f_out = open(args.output_file, 'w')
    lines = []
    idx = 0
    start_time = time.time()
    while True:
        line = f_in.readline()
        if not line:
            break
        line = line.strip()
        if args.start_column and len(line.split("\t")) == 2 and line.split("\t")[0] != "" and line.split("\t")[1] != "":
            lines.append(line)
        elif (not args.start_column) and line != "":
            lines.append(line.partition("\t")[0])

        if (idx + 1) % args.num_lines_block == 0:
            process_and_write(lines, f_out)
            print('Processed {} lines in time {}'.format(idx + 1, time.time() - start_time))
            lines = []

        idx += 1

    if len(lines) != 0:
        process_and_write(lines, f_out)
    
    f_out.truncate(f_out.tell() - 1) #remove last "\n"
    print ('Finished Processing file {}'.format(args.input_file))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
