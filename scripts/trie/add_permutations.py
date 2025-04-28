import argparse
from tqdm import tqdm

# python add_permutations.py --tokenized_sentences_file nq_unique_pseudo_queries_pseudo_query_500k_strict_special_trie.tokenized1e805 --output_file nq_unique_pseudo_queries_pseudo_query_500k_strict_special_trie.tokenized1e805.perm8 --max_permutations 8

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenized_sentences_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--max_permutations', type=int, default=8)
    args = parser.parse_args()

    with open(args.tokenized_sentences_file) as f, open(args.output_file, 'w') as g:
        for line in tqdm(f):
            tokens = line[:-1].split()
            for i in range(min(args.max_permutations, len(tokens) - 1)):
                g.write(' '.join([str(i)] + tokens[1:2+i][::-1] + tokens[2+i:]) + '\n')