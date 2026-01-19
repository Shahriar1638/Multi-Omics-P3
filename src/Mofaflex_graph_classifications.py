import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import scipy.sparse as sp
import warnings

from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import GCNConv, GATConv

warnings.filterwarnings('ignore')

# CONFIGURATION
RESULTS_DIR = "Results_Mofaflex"
GRAPH_OUTPUT_DIR = os.path.join(RESULTS_DIR, "Graphs")
os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)

FEATURE_COUNTS = [2000, 4000, 6000]
K_NEIGHBORS = 10
RANDOM_STATE = 42

# MODELS
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, out_channels)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train_and_eval(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    pred = model(data).argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

# GRAPH FUNCTIONS
def build_knn_graph(X, k=K_NEIGHBORS):
    adj = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    return adj

def build_adaptive_graph(X, k=K_NEIGHBORS, metric='euclidean'):
    dists = pairwise_distances(X, metric=metric)
    sorted_dists = np.sort(dists, axis=1)
    sigmas = sorted_dists[:, k]
    sigmas_prod = np.outer(sigmas, sigmas)
    sigmas_prod[sigmas_prod == 0] = 1e-10
    adj = np.exp(- (dists ** 2) / sigmas_prod)
    np.fill_diagonal(adj, 0)
    knn_mask = kneighbors_graph(X, k, mode='connectivity', include_self=False).toarray()
    knn_mask = ((knn_mask + knn_mask.T) > 0).astype(float)
    return sp.csr_matrix(adj * knn_mask)

def prepare_gnn_data(X, y, adj_matrix, train_mask, test_mask):
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
    data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, 
                edge_attr=edge_weight, y=torch.tensor(y, dtype=torch.long))
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    return data

def main():
    results = []
    for n_features in FEATURE_COUNTS:
        file_path = os.path.join(RESULTS_DIR, f"mofa_embeddings_{n_features}_features.csv")
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path, index_col=0)
        X = df[[c for c in df.columns if c.startswith('Factor_')]].values
        labels, split = df['Label'].values, df['Split'].values
        train_mask, test_mask = (split == 'train'), (split == 'test')
        num_classes = len(np.unique(labels))

        graphs = {'KNN': build_knn_graph(X), 'Adaptive': build_adaptive_graph(X)}
        for g_name, adj in graphs.items():
            data = prepare_gnn_data(X, labels, adj, train_mask, test_mask)
            for m_name, m_cls in [('GCN', GCN), ('GAT', GAT)]:
                acc = train_and_eval(m_cls(X.shape[1], num_classes), data)
                results.append({'Features': n_features, 'Graph': g_name, 'Model': m_name, 'Accuracy': acc})
                print(f"Feats: {n_features} | Graph: {g_name} | Model: {m_name} | Acc: {acc:.4f}")

    pd.DataFrame(results).to_csv(os.path.join(GRAPH_OUTPUT_DIR, "classification_results.csv"), index=False)

if __name__ == "__main__":
    main()
