# PIXNAR

## OVERVIEW

This repository contains code that implements "Scaling the Vocabulary of Non-autoregressive Models for Fast Generative Retrieval." Use this code to build very large phrase based vocabularies and train NAR generative retrieval models with efficient inference pipelines.

Currently being cleaned up! Will add trained models, vocabularies, and end-to-end scripts soon.

## Installation 

Install the dependencies:

Create a conda environment with Python 3.8
```
conda create -y -n pixnar python=3.8
conda init
source ~/.bashrc
conda activate pixnar
```

NVIDIA:
```
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124; pip install -e .
```

AMD:
```
pip3 install torch==2.4.1 --index-url https://download.pytorch.org/whl/rocm6.1; pip install -e .
```

For GPU accelerated clustering on NVIDIA GPUs, create the following environment:
```
conda create -y -n rapids python=3.9
conda init
source ~/.bashrc
conda activate rapids

pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12==24.4.*
pip install scikit-learn tqdm
```

## Training an NAR model

Prior to training the model, login to Wandb. This is done with the `wandb login` command; enter your key when prompted.

The first stage involves training an NAR model to (1) predict the most likely tokens at each target position and (2) predict a subset of the vocabulary containing all likely tokens that map to a relevant document identifier.

```
chmod +x ./scripts/trainer/nar/examples/finetune_extreme_nar_bow_shortlist.sh
./scripts/trainer/nar/examples/finetune_extreme_nar_bow_shortlist.sh
```

If you're training from scratch, fill `model_name_or_path` with a path to your model config - this will allow you specify the target vocab size and initialize the transformer layers with a pretrained model with the same body from Huggingface. An example is present in `model_configs`.

For the second stage of training, initialize the vectors $c_1, \dots, c_m \in \mathbb{R}^d$ using k-means clustering on the shortlist embeddings from a small set of queries.
```
python scripts/cluster/bow/get_hidden_states.py \
    --model_path model_checkpoint_dir \
    --tokenizer_name microsoft/deberta-v3-base \
    --tokenizer_lib hf \
    --cluster_queries /data_ecstorage/MINDER/data/MSMARCO/cluster_queries.txt \
    --output_path hidden_states.npy
```

Perform k-means clustering to get the cluster vectors from these hidden states. This particular command needs to be run in the `rapids` environment.
```
python scripts/cluster/bow/cluster_hidden_states.py \
    --hidden_states_path hidden_states.npy \
    --num_clusters 4096 \
    --output_path bow_clusters_kmeans4096.npy
```


Fill in the `cluster_vectors_path` field with the output path from the previous command, and run this script:
```
chmod +x ./scripts/trainer/nar/examples/finetune_extreme_nar_bow_cluster_train.sh
./scripts/trainer/nar/examples/finetune_extreme_nar_bow_cluster_train.sh
```

Combine the outputs from `finetune_extreme_nar_bow_shortlist.sh` and `finetune_extreme_nar_bow_cluster_train.sh` into a single model state dict using a script like below:
```
python scripts/cluster/bow/process_bow_clustering_model.py \
    --bow_model_path /data_ecstorage/MINDER/experiments/MSMARCO/akash_opt_H100_deberta_v3_base_5M_bow_ms_no_downproj \
    --bow_cluster_model_path /data_ecstorage/MINDER/experiments/rebuttal/cluster/deberta_v3_base_msmarco_nc2_5M_bow_cluster4096/ \
    --output_path bow_cluster_5M_4096
```

Run the script on a small subset of the data to ensure that the batch size doesn't cause memory issues. Initially, the TokenMonster server might not have been created. Fixes for tokenmonster related issues:
- Delete server if it already exists: ```rm -r ~/_tokenmonster```
- Give execute permission to the server if file is busy: ```chmod +x ~/_tokenmonster/tokenmonsterserver```


## Evaluating the NAR model

Prior to evaluating the NAR model, ensure that you have the files to the (1) pickled dictionary mapping pseudo-queries to passage ids, (2) the test answers file (qrels), and the test queries file. The test answers and test queries file are downloadable for both MSMARCO and NQ, and are linked in the MINDER repository. For example, for MSMARCO the test answers file looks like `MSMARCO/qrels.msmarco-passage.dev-subset.txt`. You can set up the test queries file (TSV file with query id and query columns) from the dev set in `Tevatron/msmarco-passage`.

Execute the following scripts:
1. `scripts/evaluate/nar/get_topk_tokens.py`
2. `scripts/evaluate/nar/trie_beam_search.py`

MSMARCO:
1. `scripts/evaluate/nar/msmarco_map_to_passages.py`
2. `cripts/evaluate/nar/evaluate_output_msmarco.py`

## Build Vocabulary

Build the vocabulary in a machine with many CPU cores and high RAM. Training generally takes a while.

Set up training environment:
```
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install golang
git clone https://github.com/alasdairforsythe/tokenmonster
cd tokenmonster/training
go mod init tokenmonster
go mod tidy
go build getalltokens.go
go build trainvocab.go
go build exportvocab.go
```

Alternatively you can directly download the pre-built libraries for linux using:
```
for filename in {exportvocab,getalltokens,tokenmonsterserver,trainvocab}; do wget -O $filename "https://huggingface.co/alasdairforsythe/tokenmonster/resolve/main/binaries/linux_x86_64/${filename}?download=true"; chmod u+x $filename; done
```

The data file should a text file with each line being an individual keyword or query.

Extract all possible tokens.
```
./getalltokens \
-chunk-size 1000000000 \
-dataset document_identifiers.txt \
-min-occur 50 \
-mode 4 \
-output all_tokens \
-workers 96
```

Distil vocabulary.
```
./trainvocab \
-dataset document_identifiers.txt \
-dictionary all_tokens \
-dir trained_vocab_checkpoints \
-fast \
-vocab-size 5000000 \
-workers 96
```

Export Vocabulary to .vocab format
```
./exportvocab -input trained_vocab_checkpoints -output trained_vocab_5M.vocab
```

Add special tokens
```
./exportvocab -input-vocab trained_vocab_5M.vocab -add-special-token "[PAD]" -resize 5000000 -output trained_vocab_5M_special.vocab -reset-token-ids
```

## Build the Trie

1. Tokenize the identifiers and add permutations: `bash scripts/trie/tokenize_docids.sh`
2. Construct the trie: `trie/marisa/build_marisa_trie.py`

## Data Processing

1. Constructing the dataset: 

2. Creating the corpus for vocabulary and trie construction:

3. Pseudo-query dictionaries:

