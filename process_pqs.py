import pickle
from tqdm import tqdm
import argparse
import os

# python process_pqs.py --pseudo_query_dict_path /data_ecstorage/MINDER/data/pseudo_queries/pid2query_Wikipedia.pkl --output_dir tmp_nq

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pseudo_query_dict_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_pseudo_queries', type=int, default=40)
    args = parser.parse_args()

    with open(args.pseudo_query_dict_path, 'rb') as f:
        pq_dict = pickle.load(f)

output_file = os.path.join(args.output_dir, 'train_using_pqs.tsv')
unique_pqs = set()

with open(output_file, 'w') as f:
    for pqs in tqdm(pq_dict.values()):
        pqs = list(set(pqs))
        unique_pqs.update(pqs)
        join_pqs = '|__|'.join(pqs)
        for i in range(min(len(pqs), args.max_pseudo_queries)):
            f.write(f'{pqs[i]}\t{join_pqs}\n')

with open(os.path.join(args.output_dir, 'unique_pqs.txt'), 'w') as f:
    for pq in tqdm(unique_pqs):
        f.write(f'{pq}\n')