import argparse
import json
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoConfig
from clover.modeling.nar import DeBertaNARBoWClusterInference
from clover.modeling.tokenizers.BatchTokenizerWrappers import AutoTokenizerWrapper

def main():
    parser = argparse.ArgumentParser(description="Get top-k tokens for queries.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="HuggingFace tokenizer name.")
    parser.add_argument("--queries_file", type=str, required=True, help="Path to the queries file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file.")
    args = parser.parse_args()

    # Load model and tokenizer
    config = AutoConfig.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizerWrapper(args.tokenizer_name)
    model = DeBertaNARBoWClusterInference.from_pretrained(args.model_dir, config=config).to(torch.bfloat16).cuda()
    model.eval()

    # Load queries
    queries = []
    with open(args.queries_file, 'r') as f:
        for line in f:
            queries.append(line.strip())

    # Tokenize queries
    tokenized_queries = tokenizer.tokenize_data(queries, max_length=16, padding='max_length')

    # Process queries in batches
    batch_size = 16
    topk_probs, topk_indices = [], []
    for i in tqdm(range(0, len(tokenized_queries['input_ids']), batch_size)):
        input_ids = tokenized_queries['input_ids'][i:i + batch_size]
        attention_mask = tokenized_queries['attention_mask'][i:i + batch_size]

        input_ids = torch.tensor(input_ids).cuda()
        attention_mask = torch.tensor(attention_mask).cuda()

        with torch.no_grad():
            D, I = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            topk_probs.append(D.detach().cpu().float().numpy())
            topk_indices.append(I.detach().cpu().float().numpy())

    # Concatenate results
    topk_probs = np.concatenate(topk_probs, axis=0)
    topk_indices = np.concatenate(topk_indices, axis=0)

    # Write results to output file
    with open(args.output_file, 'w') as f:
        for i in tqdm(range(len(queries))):
            query = queries[i]
            probs = topk_probs[i].tolist()
            indices = topk_indices[i].tolist()
            prdict = {'probs': probs, 'indices': indices}
            f.write(f'{query}\t{json.dumps(prdict)}\n')

if __name__ == "__main__":
    main()
    
# python get_topk_tokens.py --model_dir /path/to/model \
#                           --tokenizer_name microsoft/deberta-v3-base \
#                           --queries_file /path/to/queries.tsv \
#                           --output_file /path/to/output.tsv