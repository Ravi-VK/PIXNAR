import argparse
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoConfig
from tqdm import tqdm

class DeBertaNARBoWShortlist(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.deberta = AutoModel.from_config(config)
        self.lm_proj_size = self.config.hidden_size
        if hasattr(self.config, "lm_bottleneck_size"):
            self.lm_proj_size = self.config.lm_bottleneck_size
            self.down_proj = nn.Linear(self.config.hidden_size, self.config.lm_bottleneck_size)
        
        no_bias = hasattr(self.config, "no_bias") and self.config.no_bias
        self.proj = nn.Linear(self.lm_proj_size, self.config.target_vocab_size, bias=not no_bias)
        self.post_init()

class DeBertaNARBoWClustering(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.deberta = AutoModel.from_config(config)
        self.lm_proj_size = self.config.hidden_size
        if hasattr(self.config, "lm_bottleneck_size"):
            self.lm_proj_size = self.config.lm_bottleneck_size
            self.down_proj = nn.Linear(self.config.hidden_size, self.config.lm_bottleneck_size)
        
        self.cluster_embedding = torch.nn.Embedding(self.config.num_clusters, self.lm_proj_size)
        
        no_bias = hasattr(self.config, "no_bias") and self.config.no_bias
        self.proj = nn.Linear(self.lm_proj_size, self.config.target_vocab_size, bias=not no_bias)
        self.post_init()

class DeBertaNARBoWClusterInference(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.deberta = AutoModel.from_config(config)
        self.lm_proj_size = self.config.hidden_size
        if hasattr(self.config, "lm_bottleneck_size"):
            self.lm_proj_size = self.config.lm_bottleneck_size
            self.down_proj = nn.Linear(self.config.hidden_size, self.config.lm_bottleneck_size)
        
        self.cluster_embedding = torch.nn.Embedding(self.config.num_clusters, self.lm_proj_size)
        self.register_buffer('vocab_subsets', torch.zeros(self.config.num_clusters, self.config.vocab_subset_size, dtype=torch.long))
        
        no_bias = hasattr(self.config, "no_bias") and self.config.no_bias
        self.proj = nn.Linear(self.lm_proj_size, self.config.target_vocab_size, bias=not no_bias)
        self.post_init()

def main():
    parser = argparse.ArgumentParser(description="Combine BoW model and BoW cluster model into a new model.")
    parser.add_argument('--bow_model_path', type=str, required=True, help='Path to the BoW model')
    parser.add_argument('--bow_cluster_model_path', type=str, required=True, help='Path to the BoW cluster model')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the combined model')
    
    args = parser.parse_args()
    
    config_bow = AutoConfig.from_pretrained(args.bow_model_path)
    model_bow = DeBertaNARBoWShortlist.from_pretrained(args.bow_model_path, config=config_bow).to('cpu')
    
    config_bow_cluster = AutoConfig.from_pretrained(args.bow_cluster_model_path)
    model_bow_cluster = DeBertaNARBoWClustering.from_pretrained(args.bow_cluster_model_path, config=config_bow_cluster).to('cpu')
    
    batch_size = 128
    vocab_subset_size = 20000
    vocab_subsets = []
    for i in tqdm(range(0, model_bow_cluster.cluster_embedding.weight.data.shape[0], batch_size)):
        batch_vocab_subsets = model_bow_cluster.proj(model_bow_cluster.cluster_embedding.weight[i:i+batch_size]).topk(vocab_subset_size, dim=-1).indices
        vocab_subsets.append(batch_vocab_subsets)
    vocab_subsets = torch.cat(vocab_subsets, dim=0)
    
    config_bow_cluster_inference = config_bow_cluster
    config_bow_cluster_inference.vocab_subset_size = vocab_subset_size
    model_bow_cluster_inference = DeBertaNARBoWClusterInference(config_bow_cluster_inference).to('cpu')
    model_bow_cluster_inference.vocab_subsets = vocab_subsets
    model_bow_cluster_inference.deberta.load_state_dict(model_bow.deberta.state_dict())
    model_bow_cluster_inference.proj.load_state_dict(model_bow.proj.state_dict())
    if hasattr(model_bow, 'down_proj'):
        model_bow_cluster_inference.down_proj.load_state_dict(model_bow.down_proj.state_dict())
    model_bow_cluster_inference.cluster_embedding.load_state_dict(model_bow_cluster.cluster_embedding.state_dict())
    
    model_bow_cluster_inference.save_pretrained(args.output_path)

if __name__ == "__main__":
    main()

# To execute the script with the values filled in:
# python scripts/cluster/bow/process_bow_clustering_model.py --bow_model_path /data_ecstorage/MINDER/experiments/MSMARCO/akash_opt_H100_deberta_v3_base_5M_bow_ms_no_downproj --bow_cluster_model_path /data_ecstorage/MINDER/experiments/MSMARCO/akash_opt_H100_deberta_v3_base_5M_bow_ms_no_downproj_emv1_epochs2 --output_path /data_ecstorage/MINDER/experiments/MSMARCO/combined_model_output
# python scripts/cluster/bow/process_bow_clustering_model.py \
#     --bow_model_path /data_ecstorage/MINDER/experiments/MSMARCO/akash_opt_H100_deberta_v3_base_5M_bow_ms_no_downproj \
#     --bow_cluster_model_path /data_ecstorage/MINDER/experiments/rebuttal/cluster/deberta_v3_base_msmarco_nc2_5M_bow_cluster4096/checkpoint-27632 \
#     --output_path bow_cluster_5M_4096