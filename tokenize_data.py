import argparse
import tokenmonster
import os
from tqdm import tqdm
import pickle
import base64
import numpy as np
from transformers import AutoTokenizer

# python tokenize_matryoshka.py --vocab_file /data_ecstorage/BingGlobal/vocabs/matryoshka/v1/sampled_queries_keywords_1450M_1M_matryoshka_80K_special.vocab --input_file split_dataset/split_0.tsv --output_file split_dataset/tokenized_split_0.tsv
# for i in $(seq 0 7); do
#     python tokenize_data.py \
#         --vocab_file /data_ecstorage/BingGlobal/vocabs/matryoshka/v1/sampled_queries_keywords_1450M_1M_matryoshka_80K_special.vocab \
#         --input_file split_dataset/split_${i}.tsv \
#         --output_file split_dataset/tokenized_split_${i}.tsv &
# done
# wait
# cat split_dataset/tokenized_split_*.tsv > split_dataset/split_full.tsv
# echo "Done!"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_vocab', type=str, required=True)
    parser.add_argument('--output_vocab', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained()
    vocab = tokenmonster.load(args.vocab_file)

    with open(args.input_file) as f:
        for line in tqdm(f):
            lang, query, keywords = line[:-1].split('\t')
            keywords = keywords.split('|__|')
            query = ','.join(map(str, tokenizer(query)['input_ids']))
            keywords = '|__|'.join([','.join(map(str, x.tolist())) for x in vocab.tokenize(keywords)])
            mline = f'{query}\t{keywords}'
            with open(args.output_file, 'a') as g:
                g.write(f'{mline}\n')