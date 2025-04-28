import argparse
import numpy as np
from cuml.cluster import KMeans
from tqdm import tqdm
from sklearn.preprocessing import normalize

def main():
    parser = argparse.ArgumentParser(description="Cluster context vectors using KMeans.")
    parser.add_argument('--hidden_states_path', type=str, required=True, help='Path to the hidden states file')
    parser.add_argument('--num_clusters', type=int, required=True, help='Number of clusters for KMeans')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the cluster vectors')
    
    args = parser.parse_args()
    
    context_vectors = np.load(args.hidden_states_path).astype(np.float32)
    
    average_magnitude = np.mean(np.linalg.norm(context_vectors, axis=1))
    print('Average magnitude of context vectors: {}'.format(average_magnitude))
    
    context_vectors = normalize(context_vectors, axis=1, norm='l2')
    
    kmeans_labels_ = KMeans(n_clusters=args.num_clusters).fit_predict(context_vectors)
    
    kmeans_clusters = np.zeros((args.num_clusters, context_vectors.shape[1]))
    kmeans_cluster_sizes = np.zeros(args.num_clusters)
    
    for i in tqdm(range(context_vectors.shape[0])):
        kmeans_clusters[kmeans_labels_[i]] += context_vectors[i]
        kmeans_cluster_sizes[kmeans_labels_[i]] += 1
    
    kmeans_clusters = kmeans_clusters / kmeans_cluster_sizes[:, None]
    kmeans_clusters = normalize(kmeans_clusters, axis=1, norm='l2')
    kmeans_clusters = kmeans_clusters * average_magnitude
    
    np.save(args.output_path, kmeans_clusters)

if __name__ == "__main__":
    main()

# To execute the script with the values filled in:
# python scripts/cluster/bow/cluster_hidden_states.py --hidden_states_path msmarco_5M_bow_hidden_states.npy --num_clusters 4096 --output_path msmarco_5M_bow_clusters_kmeans4096.npy
# python scripts/cluster/bow/cluster_hidden_states.py --hidden_states_path msmarco_5M_bow_hidden_states.npy --num_clusters 1024 --output_path msmarco_5M_bow_clusters_kmeans1024.npy
# python scripts/cluster/bow/cluster_hidden_states.py --hidden_states_path msmarco_5M_bow_hidden_states.npy --num_clusters 256 --output_path msmarco_5M_bow_clusters_kmeans256.npy