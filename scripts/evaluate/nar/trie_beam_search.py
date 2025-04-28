import argparse
import json
import heapq
import numpy as np
from tqdm import tqdm
import torch

from trie.marisa import TrieWithContinuations
from clover.modeling.tokenizers.BatchTokenizerWrappers import TokenMonsterWrapper, AutoTokenizerWrapper


def hex_key(seq):
    return ''.join(hex(token)[2:] + ',' for token in seq)


def parse_args():
    parser = argparse.ArgumentParser(description="Trie Beam Search Evaluation Script")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input TSV file with query and probability data.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output TSV file where predictions will be saved.")
    parser.add_argument("--trie_path", type=str, required=True,
                        help="Path to the trie file.")
    parser.add_argument("--vocab_file", type=str, required=True,
                        help="Path to the vocabulary file for TokenMonsterWrapper.")
    parser.add_argument("--beam_size", type=int, default=100,
                        help="Beam size for beam search.")
    parser.add_argument("--num_candidates", type=int, default=100,
                        help="Number of candidate predictions per query.")
    parser.add_argument("--length_norm", type=float, default=2.5,
                        help="Length normalization constant.")
    parser.add_argument("--max_permutations", type=int, default=4,
                        help="Maximum number of permutations to consider.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize vocabulary and trie
    vocab = TokenMonsterWrapper(args.vocab_file)
    trie = TrieWithContinuations(args.trie_path, vocab.pad_token_id)

    # Load input data
    queries, topk_probs, topk_indices = [], [], []
    with open(args.input_file, 'r') as f:
        for line in tqdm(f, desc="Loading input"):
            query, prdict = line.rstrip('\n').split('\t')
            prdict = json.loads(prdict)
            queries.append(query)
            topk_probs.append(np.array(prdict['probs']))
            topk_indices.append(np.array(prdict['indices']).astype(np.int64))

    D, I = topk_probs, topk_indices
    D, I = np.stack(D, axis=0), np.stack(I, axis=0)

    all_predictions = []
    for i in tqdm(range(D.shape[0]), desc="Processing queries"):
        completed_sequences = []
        for permutation in range(args.max_permutations):
            beams = [([permutation], 0)]
            completed_permutations = []
            permutation_indexes = list(range(D.shape[1]))
            permutation_indexes = permutation_indexes[:permutation+1][::-1] + permutation_indexes[permutation+1:]
            for j in permutation_indexes:
                new_beams = []
                for beam, score in beams:
                    key = hex_key(beam)
                    valid_continuations = torch.from_numpy(trie.valid_continuations_direct(key))
                    is_complete_sequence = len(valid_continuations) > 0 and valid_continuations[0].item() == -1

                    if is_complete_sequence:
                        normalized_score = score * ((6 / (5 + len(beam))) ** args.length_norm)
                        unscrambled_beam = tuple(beam[1:permutation+2][::-1] + beam[permutation+2:])
                        heapq.heappush(completed_permutations, (normalized_score, unscrambled_beam))
                        if len(completed_permutations) > args.num_candidates:
                            heapq.heappop(completed_permutations)
                        valid_continuations = valid_continuations[1:]

                    if len(valid_continuations) == 0:
                        continue

                    top_valid_continuations = set(valid_continuations.tolist()).intersection(set(I[i, j].tolist()))
                    for k in range(D.shape[2]):
                        tok_lprob, topk_idx = D[i, j, k].item(), I[i, j, k].item()
                        if topk_idx in top_valid_continuations:
                            new_beam = beam + [topk_idx]
                            new_score = score + tok_lprob
                            new_beams.append((new_beam, new_score))
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:args.num_candidates]
            completed_sequences.extend(completed_permutations)
        all_predictions.append(list(set(completed_sequences)))

    for i in tqdm(range(len(all_predictions)), desc="Decoding predictions"):
        all_predictions[i] = sorted(all_predictions[i], key=lambda x: x[0], reverse=True)
        decoded = []
        for score, seq in all_predictions[i]:
            decoded.append((score, vocab.tokenizer.decode(seq)))
        all_predictions[i] = decoded

    with open(args.output_file, 'w') as f:
        for i in tqdm(range(len(all_predictions)), desc="Writing output"):
            for score, seq in all_predictions[i]:
                f.write(f'{queries[i]}\t{seq}\t{score}\n')


if __name__ == "__main__":
    main()

# python scripts/evaluate/nar/trie_beam_search.py --trie_path pseudo_queries_msmarco_pq_5M_trie --vocab_file /data_ecstorage/MINDER/mstmp/vocabs/msmarco_pseudo_queries_5M_special.vocab --input_file msmarco_topk_probs.tsv --output_file msmarco_topk_predictions.tsv