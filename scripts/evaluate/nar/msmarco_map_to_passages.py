from tqdm import tqdm
import json
import pickle
import argparse
import numpy as np

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trie_predictions_file', type=str, required=True,
                        help="Path to the trie predictions file.")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Path to the output file that will contain the mapped passages.")
    parser.add_argument('--pseudo_query_to_passage_file', type=str, required=True,
                        help="Path to the pseudo query to passage id pickle file.")
    parser.add_argument('--test_answers_file', type=str, required=True,
                        help="Path to the test answers file (qrels).")
    parser.add_argument('--test_queries_file', type=str, required=True,
                        help="Path to the test queries TSV file.")
    args = parser.parse_args()

    pseudo_query_to_passage_id = pickle.load(open(args.pseudo_query_to_passage_file, 'rb'))

    model_predictions = {}
    with open(args.trie_predictions_file, 'r') as f:
        for line in f:
            query, keyword, score = line[:-1].split('\t')
            if query not in model_predictions:
                model_predictions[query] = ([], [])
            model_predictions[query][0].append(keyword)
            model_predictions[query][1].append(float(score))

    test_answers_dict = {}
    with open(args.test_answers_file, 'r') as f:
        for i, line in enumerate(f):
            qid, _, pid, _ = line[:-1].split()
            qid, pid = int(qid), int(pid)
            if qid not in test_answers_dict:
                test_answers_dict[qid] = []
            test_answers_dict[qid].append(pid)

    test_queries = {}
    test_qids_only = []
    with open(args.test_queries_file, 'r') as f:
        for line in f:
            qid, query = line[:-1].split('\t')
            qid = int(qid)
            test_queries[qid] = query
            test_qids_only.append(qid)

    test_query_to_qid = {}
    for qid, query in test_queries.items():
        test_query_to_qid[query] = qid

    model_predictions_dict = {}
    for i in range(len(test_queries)):
        if test_queries[test_qids_only[i]].strip() in model_predictions:
            model_predictions_dict[test_qids_only[i]] = model_predictions[test_queries[test_qids_only[i]].strip()]
        else:
            model_predictions_dict[test_qids_only[i]] = ([], [])

    model_passage_predictions_dict = {}
    for q, p in tqdm(model_predictions_dict.items()):
        if len(p[0]) == 0:
            model_passage_predictions_dict[q] = []
            continue
        passage_set = {}
        for i in range(min(2000, len(p[0]))):
            pq, model_score = p[0][i], p[1][i]
            if pq in pseudo_query_to_passage_id:
                pids = sorted(pseudo_query_to_passage_id[pq], key=lambda x: x[1], reverse=True)
                for pid, score in pids[:10]:
                    if pid not in passage_set:
                        passage_set[pid] = 0
                    passage_set[pid] += np.exp(model_score) / len(pseudo_query_to_passage_id[pq])
        passage_set = sorted(passage_set.items(), key=lambda x: x[1], reverse=True)
        model_passage_predictions_dict[q] = [x[0] for x in passage_set]

    with open(args.output_file, 'w') as f:
        for qid, pids in model_passage_predictions_dict.items():
            for i, pid in enumerate(pids):
                if i == 6000:
                    break
                f.write(f'{int(qid)}\t{int(pid)}\t{i+1}\n')