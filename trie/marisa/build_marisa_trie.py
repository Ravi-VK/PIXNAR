import argparse
from tqdm import tqdm
import marisa_trie
import numpy as np

# python trie/marisa/build_marisa_trie.py \
#     --input /data_ecstorage/MINDER/mstmp/trie/unique_score0p5_pseudo_queries_msmarco_pq_5M_trie.tokenized62dfe.perm4 \
#     --output_prefix unique_score0p5_pseudo_queries_msmarco_pq_5M_trie

# python trie/marisa/build_marisa_trie.py --input NA_sampled_queries_keywords_1450M_1M_special_tokenized.txt --output NA_sampled_queries_keywords_1450M_1M_special_trie
# python trie/marisa/build_marisa_trie.py --input permtok100m.txt --output pseudo_queries_msmarco_pq_5M_trie

def hex_key(seq):
    return ''.join(hex(token)[2:] + ',' for token in seq)


def build_trie_data(input_path):
    pdict = {}
    pdict[''] = [set(), False]

    with open(input_path, 'r') as f:
        for idx, line in tqdm(enumerate(f), desc="Reading input"):
            line = line.strip()
            if not line:
                continue

            ids = list(map(int, line.split())) # [1:]
            if len(ids) == 0:
                continue
            
            prefix = ''
            pdict[prefix][0].add(ids[0])

            for i in range(1, len(ids) + 1):
                prefix = ''.join((prefix, hex(ids[i - 1])[2:], ','))
                if prefix not in pdict:
                    pdict[prefix] = [set(), False]
                if i == len(ids):
                    pdict[prefix][1] = True
                else:
                    pdict[prefix][0].add(ids[i])

    return pdict


def convert_to_arrays(pdict):
    keys, flat_values, offsets, indexes = [], [], [0], []

    for i, (k, v) in tqdm(enumerate(pdict.items()), desc="Converting to arrays"):
        new_value = [-1] if v[1] else []
        new_value.extend(sorted(v[0]))
        flat_values.extend(new_value)
        offsets.append(len(flat_values))
        keys.append(k)
        indexes.append((i,))

    return keys, flat_values, offsets, indexes


def save_outputs(keys, flat_values, offsets, indexes, output_prefix):
    trie = marisa_trie.RecordTrie('<Q', zip(keys, indexes))
    trie.save(f'{output_prefix}.marisa')

    flat_values = np.array(flat_values, dtype=np.int64)
    offsets = np.array(offsets, dtype=np.int64)
    np.save(f'{output_prefix}.marisa.values.npy', flat_values)
    np.save(f'{output_prefix}.marisa.offsets.npy', offsets)


def main():
    parser = argparse.ArgumentParser(description="Build a MARISA trie from token prefix data.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output_prefix', type=str, required=True, help='Output prefix for trie and arrays')

    args = parser.parse_args()

    pdict = build_trie_data(args.input)
    keys, flat_values, offsets, indexes = convert_to_arrays(pdict)
    save_outputs(keys, flat_values, offsets, indexes, args.output_prefix)


if __name__ == "__main__":
    main()
