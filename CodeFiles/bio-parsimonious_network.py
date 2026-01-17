import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import random
import sys
import os

# ==============================================================================
# 1. REALISTIC "SOLVABLE HARD MODE" BIOLOGICAL SIMULATOR
#    (Balanced Signal vs. Batch Effects to allow recovery)
# ==============================================================================

def load_data():
    print("Loading datasets (this may take a while)...")
    gene_df = pd.read_csv("NewDatasets/processed_expression_4O.csv", index_col=0).T
    meth_df = pd.read_csv("NewDatasets/processed_methylation_4O.csv", index_col=0).T
    cnv_df  = pd.read_csv("NewDatasets/processed_cnv_4O.csv", index_col=0).T
    labels_df = pd.read_csv("NewDatasets/processed_labels_3Omics_FXS_OG.csv", index_col=0)
    phenotype_df = pd.read_csv("NewDatasets/phenotype_data_clean_FXS_MOFA_3Omics.csv", index_col=0)
    return gene_df, meth_df, cnv_df, labels_df, phenotype_df

class RealDataset(Dataset):
    def __init__(self, gene_df, meth_df, cnv_df, labels_df, phenotype_df, var_percent=1.0):
        # --- Align Indices ---
        common_indices = gene_df.index.intersection(meth_df.index).intersection(cnv_df.index).intersection(labels_df.index).intersection(phenotype_df.index)
        
        # print(f"Samples aligned: {len(common_indices)}")
        self.gene_df = gene_df.loc[common_indices]
        self.meth_df = meth_df.loc[common_indices]
        self.cnv_df = cnv_df.loc[common_indices]
        self.labels_df = labels_df.loc[common_indices]
        self.phenotype_df = phenotype_df.loc[common_indices]

        # --- Variance Filtering ---
        if var_percent < 1.0:
            # print(f"Filtering top {var_percent*100}% high variance features...")
            self.gene_df = self.filter_variance(self.gene_df, var_percent)
            self.meth_df = self.filter_variance(self.meth_df, var_percent)
            self.cnv_df = self.filter_variance(self.cnv_df, var_percent)

        # --- Standard Scaling ---
        scaler_g = StandardScaler()
        scaler_m = StandardScaler()
        scaler_c = StandardScaler()
        
        gene_vals = scaler_g.fit_transform(self.gene_df.values)
        meth_vals = scaler_m.fit_transform(self.meth_df.values)
        cnv_vals = scaler_c.fit_transform(self.cnv_df.values)
        
        # --- Convert to Tensors ---
        self.rna = torch.tensor(gene_vals, dtype=torch.float32)
        self.meth = torch.tensor(meth_vals, dtype=torch.float32)
        self.cnv = torch.tensor(cnv_vals, dtype=torch.float32)
        
        # Labels
        self.labels = torch.tensor(self.labels_df.iloc[:,0].values, dtype=torch.long)
        self.num_classes = len(torch.unique(self.labels))

        # Mask (All ones for real data)
        self.mask = torch.ones(len(self.labels), 3)

    def filter_variance(self, df, percentile):
        variances = df.var()
        threshold = variances.quantile(1.0 - percentile)
        return df.loc[:, variances > threshold]

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.rna[idx], self.meth[idx], self.cnv[idx], self.mask[idx], self.labels[idx]

# ==============================================================================
# 2. BIO-PARSIMONIOUS NETWORK ARCHITECTURE
# ==============================================================================

class ExplainableEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.4):
        super().__init__()
        # The Gate: Learnable parameter deciding if a gene is useful
        self.feature_importance = nn.Parameter(torch.randn(input_dim) * 0.01)
       
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x):
        # Apply the gate (Feature Selection)
        gate = torch.sigmoid(self.feature_importance)
        x_selected = x * gate
        return F.normalize(self.net(x_selected), dim=1)

class BioFMPN(nn.Module):
    def __init__(self, dims, num_classes, latent_dim=32):
        super().__init__()
        self.enc_rna = ExplainableEncoder(dims['rna'], latent_dim)
        self.enc_meth = ExplainableEncoder(dims['meth'], latent_dim)
        self.enc_cnv = ExplainableEncoder(dims['cnv'], latent_dim)
       
        # Attention Gate for fusing modalities
        self.attn_gate = nn.Sequential(
            nn.Linear(latent_dim*3, 32),
            nn.Tanh(),
            nn.Linear(32, 3)
        )
        self.head = nn.Linear(latent_dim, num_classes, bias=False)

    def forward(self, rna, meth, cnv, mask):
        z_r = self.enc_rna(rna)
        z_m = self.enc_meth(meth)
        z_c = self.enc_cnv(cnv)
       
        # Zero out missing modalities for attention calculation
        z_cat = torch.cat([z_r*mask[:,0:1], z_m*mask[:,1:2], z_c*mask[:,2:3]], dim=1)
       
        # Calculate Attention Weights
        scores = self.attn_gate(z_cat)
        # Mask penalty (make missing modalities have -inf score)
        scores = scores + ((1.0 - mask) * -1e9)
        w = F.softmax(scores, dim=1)
       
        # Weighted Fusion
        z_fused = (w[:,0:1]*z_r) + (w[:,1:2]*z_m) + (w[:,2:3]*z_c)
        z_fused = F.normalize(z_fused, dim=1)
       
        return self.head(z_fused), w

# ==============================================================================
# 3. METRICS AND UTILITIES
# ==============================================================================

def calculate_metrics(y_true, logits):
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
   
    acc = accuracy_score(y_true, preds)
    bacc = balanced_accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='macro')
    try:
        auroc = roc_auc_score(y_true, probs, multi_class='ovr')
    except:
        auroc = 0.5
       
    return {'Acc': acc, 'BalAcc': bacc, 'F1': f1, 'AUROC': auroc}

def jaccard_index(set_a, set_b):
    if len(set_a) == 0 and len(set_b) == 0: return 1.0
    intersection = len(set(set_a).intersection(set_b))
    union = len(set(set_a).union(set_b))
    return intersection / union

# ==============================================================================
# 4. STABILITY SELECTION ENGINE (Optimized for Recovery)
# ==============================================================================

def train_single_run(run_id, dataset, device):
    """Trains one instance of the model and returns best metrics + selected features"""
   
    # Stratified Split (New random split per run)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.3, stratify=dataset.labels)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=64, shuffle=False)
   
    dims = {'rna': dataset.rna.shape[1], 'meth': dataset.meth.shape[1], 'cnv': dataset.cnv.shape[1]}
    model = BioFMPN(dims, num_classes=dataset.num_classes).to(device)
   
    # OPTIMIZATION: Lower Learning Rate & Gentler Sparsity
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
   
    best_metrics = None
    best_bacc = 0.0
    
    # Early Stopping Support
    patience = 15
    counter = 0
   
    # Train Loop
    for epoch in range(300): # Increased max epochs slightly
        model.train()
        for rna, meth, cnv, mask, y in train_loader:
            rna, meth, cnv, mask, y = rna.to(device), meth.to(device), cnv.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            logits, w = model(rna, meth, cnv, mask)
           
            # L1 Sparsity Loss (Tuned to 0.0002)
            # This is "gentle" enough to not kill the weak signal, but strong enough to reduce noise.
            l1_reg = torch.sum(torch.sigmoid(model.enc_rna.feature_importance)) + \
                     torch.sum(torch.sigmoid(model.enc_meth.feature_importance))
           
            loss = criterion(logits, y) + 0.0002 * l1_reg
            loss.backward()
            optimizer.step()
   
        # Validation
        model.eval()
        val_logits, val_y = [], []
        with torch.no_grad():
            for rna, meth, cnv, mask, y in val_loader:
                rna, meth, cnv, mask, y = rna.to(device), meth.to(device), cnv.to(device), mask.to(device), y.to(device)
                val_logits.append(model(rna, meth, cnv, mask)[0].cpu())
                val_y.append(y.cpu())
       
        val_logits = torch.cat(val_logits).numpy()
        val_y = torch.cat(val_y).numpy()
        metrics = calculate_metrics(val_y, val_logits)
        
        print(f"Epoch {epoch}: BalAcc={metrics['BalAcc']:.3f}")
       
        if metrics['BalAcc'] > best_bacc:
            best_bacc = metrics['BalAcc']
            best_metrics = metrics
            counter = 0 # Reset counter
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stop at epoch {epoch} (Patient {patience})")
                break
           
    # Extract Top Features
    imp_rna = torch.sigmoid(model.enc_rna.feature_importance).detach().cpu().numpy()
    # Select Top 40 genes (Targeting our 30 drivers + some buffer)
    selected_genes = np.argsort(imp_rna)[::-1][:40]
   
    return best_metrics, selected_genes

def run_stability_selection(iterations=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    print(f"Running on: {device}")
    
    # 1. Load Data ONCE
    # dataset = RealDataset() # Old way
    
    # New way: Load raw dfs first
    g_df, m_df, c_df, l_df, p_df = load_data()
    
    var_thresholds = [0.4, 0.5, 0.6, 0.7] # Checking these variances
    
    for var_p in var_thresholds:
        print(f"\n{'='*40}")
        print(f"Processing Top {var_p*100}% High Variance Features")
        print(f"{'='*40}")
        
        dataset = RealDataset(g_df, m_df, c_df, l_df, p_df, var_percent=var_p)
        print(f"Features after filtering :: RNA: {dataset.rna.shape[1]}, Meth: {dataset.meth.shape[1]}, CNV: {dataset.cnv.shape[1]}")

        all_metrics = []
        selected_features_history = []
       
        print("-" * 80)
        print(f"{'Run':<5} | {'BalAcc':<8} {'F1':<8} {'AUROC':<8} | {'Jaccard (vs prev)':<15}")
        print("-" * 80)

        for i in range(iterations):
            # Seed ensures different train/val splits, but SAME dataset
            torch.manual_seed(i + 3407)
            np.random.seed(i + 3407)
           
            metrics, feats = train_single_run(i, dataset, device)
            all_metrics.append(metrics)
            selected_features_history.append(set(feats))
           
            # Calculate Stability
            j_score = 0.0
            if i > 0:
                j_score = jaccard_index(selected_features_history[-1], selected_features_history[-2])
               
            print(f"{i+1:<5} | {metrics['BalAcc']:.3f}    {metrics['F1']:.3f}    {metrics['AUROC']:.3f}    | {j_score:.3f}")

        # --- FINAL REPORT FOR THIS THRESHOLD ---
        print("\n" + "-"*35 + f" EVALUATION (Top {var_p*100}%) " + "-"*35)
       
        # 1. Performance Stability
        avg_bacc = np.mean([m['BalAcc'] for m in all_metrics])
        std_bacc = np.std([m['BalAcc'] for m in all_metrics])
        print(f"Average Balanced Acc:     {avg_bacc:.3f} +/- {std_bacc:.3f}")
       
        # 2. Consensus Features (The "Stable" Set)
        all_indices = [item for sublist in selected_features_history for item in sublist]
        counts = {}
        for idx in all_indices:
            counts[idx] = counts.get(idx, 0) + 1
       
        # Filter: Gene must appear in >= 60% of runs (3 out of 5)
        stable_genes = [k for k, v in counts.items() if v >= (iterations * 0.6)]
        stable_genes.sort()
       
        print(f"Stable Biomarkers Found:  {len(stable_genes)}")
        
        # 4. Global Stability Score
        j_scores = []
        if len(selected_features_history) > 1:
            for i in range(len(selected_features_history)):
                for j in range(i+1, len(selected_features_history)):
                    j_scores.append(jaccard_index(selected_features_history[i], selected_features_history[j]))
            print(f"Global Stability (Avg J): {np.mean(j_scores):.3f}")
        else:
            print(f"Global Stability (Avg J): N/A (1 run)")

if __name__ == "__main__":
    run_stability_selection(iterations=5)