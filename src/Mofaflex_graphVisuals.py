
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import scipy.sparse as sp
import warnings

# Try to import PyTorch Geometric, handles if not installed
try:
    from torch_geometric.data import Data
    from torch_geometric.utils import from_scipy_sparse_matrix
    print("✅ PyTorch Geometric is available.")
except ImportError:
    print("⚠️ PyTorch Geometric not found. GNN objects will be simulated dictionaries.")
    Data = None

warnings.filterwarnings('ignore')

# ====================================================================
# CONFIGURATION
# ====================================================================
RESULTS_DIR = "Results_Mofaflex"
GRAPH_OUTPUT_DIR = os.path.join(RESULTS_DIR, "Graphs")
os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)

FEATURE_COUNTS = [2000, 4000, 6000, 10000]
K_NEIGHBORS = 10  # k for KNN
RANDOM_STATE = 42

print(f"📁 Graphs will be saved to: {os.path.abspath(GRAPH_OUTPUT_DIR)}")

# ====================================================================
# GRAPH CONSTRUCTION FUNCTIONS
# ====================================================================

def build_knn_graph(X, k=K_NEIGHBORS):
    """
    Construct a k-Nearest Neighbors graph.
    Returns a scipy sparse adjacency matrix.
    """
    print(f"   🔨 Building KNN Graph (k={k})...")
    # mode='connectivity' gives 1s and 0s
    adj = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    return adj

def build_adaptive_graph(X, k=K_NEIGHBORS, metric='euclidean'):
    """
    Construct an Adaptive Graph using Zelnik-Manor local scaling.
    Ref: Zelnik-Manor & Perona, "Self-Tuning Spectral Clustering", NIPS 2004
    """
    print(f"   🔨 Building Adaptive Graph (Local Scaling, k={k})...")
    
    # Compute pairwise distances
    dists = pairwise_distances(X, metric=metric)
    
    # Find distance to k-th neighbor for each point (local scale sigma)
    # sort distances row-wise
    sorted_dists = np.sort(dists, axis=1)
    # sigma_i is the distance to the k-th neighbor
    sigmas = sorted_dists[:, k]
    
    # Compute affinity matrix: A_ij = exp(-d^2(x_i, x_j) / (sigma_i * sigma_j))
    n = X.shape[0]
    adj = np.zeros((n, n))
    
    # Efficient computation using broadcasting
    # sigma_i * sigma_j
    sigmas_prod = np.outer(sigmas, sigmas)
    
    # Avoid division by zero
    sigmas_prod[sigmas_prod == 0] = 1e-10
    
    adj = np.exp(- (dists ** 2) / sigmas_prod)
    
    # Zero out diagonal
    np.fill_diagonal(adj, 0)
    
    # Pruning: Keep only top k connections or threshold to make it sparse like KNN
    # Here we'll combine it with KNN structure but use adaptive weights
    knn_mask = kneighbors_graph(X, k, mode='connectivity', include_self=False).toarray()
    
    # Symmetric knn mask (OR logic)
    knn_mask = ((knn_mask + knn_mask.T) > 0).astype(float)
    
    # Apply mask
    adj_sparse = adj * knn_mask
    
    return sp.csr_matrix(adj_sparse)

def prepare_gnn_data(X, y, adj_matrix, train_mask, test_mask):
    """
    Convert embeddings and adjacency to PyTorch Geometric Data object.
    Suitable for GCN, GAT, etc.
    """
    print("   📦 Preparing GNN Data Object...")
    
    # Node features
    x = torch.tensor(X, dtype=torch.float)
    
    # Labels
    y = torch.tensor(y, dtype=torch.long)
    
    # Edge Index (from sparse matrix)
    if Data is not None:
        edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        
        # Add masks
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        
        return data, edge_index
    else:
        # Fallback dictionary for simulation
        coo = adj_matrix.tocoo()
        edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        return {"x": x, "y": y, "edge_index": edge_index}, edge_index

# ====================================================================
# VISUALIZATION
# ====================================================================

def visualize_graph(X, edge_index, labels, title, filename):
    """
    Visualize the graph structure overlaid on t-SNE embedding.
    """
    print(f"   🎨 distinct visualising {title}...")
    
    # 1. Compute t-SNE for node positions (if not already low dim)
    if X.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
        pos = tsne.fit_transform(X)
    else:
        pos = X
        
    plt.figure(figsize=(10, 8))
    
    # Draw edges first (so they are behind nodes)
    # Convert edge_index to numpy
    if isinstance(edge_index, torch.Tensor):
        edges = edge_index.cpu().numpy()
    else:
        edges = edge_index
        
    # Valid edges only (don't draw too many if dense)
    num_edges = edges.shape[1]
    if num_edges > 10000:
        print(f"   ⚠️ Too many edges ({num_edges}), plotting a subset for visualization...")
        indices = np.random.choice(num_edges, 10000, replace=False)
        edges = edges[:, indices]
    
    # Plot edges
    for i in range(edges.shape[1]):
        start = pos[edges[0, i]]
        end = pos[edges[1, i]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'k-', alpha=0.05, linewidth=0.5)
        
    # Plot nodes
    scatter = plt.scatter(pos[:, 0], pos[:, 1], c=labels, cmap='tab10', s=20, zorder=10)
    plt.colorbar(scatter, label='Class')
    
    plt.title(title)
    plt.axis('off')
    
    save_path = os.path.join(GRAPH_OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved graph plot: {filename}")

# ====================================================================
# MAIN LOOP
# ====================================================================

def main():
    print("=" * 80)
    print("GRAPH GENERATION FROM MOFA EMBEDDINGS")
    print("=" * 80)
    
    for n_features in FEATURE_COUNTS:
        print(f"\n📊 Processing Top {n_features} Features...")
        
        # 1. Load Embeddings
        file_path = os.path.join(RESULTS_DIR, f"mofa_embeddings_{n_features}_features.csv")
        if not os.path.exists(file_path):
            print(f"   ❌ File not found: {file_path}")
            continue
            
        print(f"   📖 Loading {file_path}...")
        df = pd.read_csv(file_path, index_col=0)
        
        # Extract Features (Factors), Labels, and Split info
        factor_cols = [c for c in df.columns if c.startswith('Factor_')]
        X = df[factor_cols].values
        labels = df['Label'].values
        split = df['Split'].values
        
        train_mask = (split == 'train')
        test_mask = (split == 'test')
        
        print(f"   Nodes: {X.shape[0]}, Factors: {X.shape[1]}")
        
        # 2. Build KNN Graph
        knn_adj = build_knn_graph(X, k=K_NEIGHBORS)
        
        # 3. Build Adaptive Graph
        adaptive_adj = build_adaptive_graph(X, k=K_NEIGHBORS)
        
        # 4. Prepare GNN Data Objects
        # For KNN
        gnn_data_knn, edge_index_knn = prepare_gnn_data(X, labels, knn_adj, train_mask, test_mask)
        # For Adaptive
        gnn_data_adaptive, edge_index_adaptive = prepare_gnn_data(X, labels, adaptive_adj, train_mask, test_mask)
        
        # 5. Save Graph Structures (Adjacency Lists)
        # Save edge indices as CSV
        knn_edges_df = pd.DataFrame(edge_index_knn.numpy().T, columns=['Source', 'Target'])
        knn_edges_df.to_csv(os.path.join(GRAPH_OUTPUT_DIR, f"knn_edges_{n_features}_features.csv"), index=False)
        
        adaptive_edges_df = pd.DataFrame(edge_index_adaptive.numpy().T, columns=['Source', 'Target'])
        adaptive_edges_df.to_csv(os.path.join(GRAPH_OUTPUT_DIR, f"adaptive_edges_{n_features}_features.csv"), index=False)
        
        # 6. Visualize
        visualize_graph(X, edge_index_knn, labels, 
                        f"KNN Graph (k={K_NEIGHBORS}) - {n_features} Feats", 
                        f"knn_graph_{n_features}.png")
        
        visualize_graph(X, edge_index_adaptive, labels, 
                        f"Adaptive Graph (Loc. Scale) - {n_features} Feats", 
                        f"adaptive_graph_{n_features}.png")

        # 7. (Optional) Save PyTorch Pickle if desired
        if Data is not None:
            torch.save(gnn_data_knn, os.path.join(GRAPH_OUTPUT_DIR, f"gnn_data_knn_{n_features}.pt"))
            torch.save(gnn_data_adaptive, os.path.join(GRAPH_OUTPUT_DIR, f"gnn_data_adaptive_{n_features}.pt"))
            print(f"   💾 Saved PyTorch Geometric Data objects (.pt)")

    print("\n" + "=" * 80)
    print("✅ GRAPH GENERATION COMPLETED!")
    print(f"Results saved in: {GRAPH_OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
