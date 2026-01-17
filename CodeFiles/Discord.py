#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-05T14:01:32.061Z

Updated: Added early stopping with patience 15 for overfitting detection
         Adjusted architecture for high-dimensional, low-sample multi-omics data
         (205 samples with ~50k, ~400k, ~60k features)
"""

import sys
import subprocess

# 1. INSTALL OPTUNA IF MISSING
try:
    import optuna
except ImportError:
    print("Installing optuna...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import optuna
import pandas as pd

# ==========================================
# 0. CONFIGURATION & DEVICE
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f">>> Running on: {DEVICE}")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Early Stopping Configuration
EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for 15 epochs

# ==========================================
# 1. LOAD REAL DATA FROM CSV FILES
# ==========================================
print(">>> Loading real data from CSV files...")

# Load the three omics datasets (transposed so samples are rows)
gene_df = pd.read_csv("NewDatasets/processed_expression_4O.csv", index_col=0).T
meth_df = pd.read_csv("NewDatasets/processed_methylation_4O.csv", index_col=0).T
cnv_df  = pd.read_csv("NewDatasets/processed_cnv_4O.csv", index_col=0).T

# Load labels
labels_df = pd.read_csv("NewDatasets/processed_labels_3Omics_FXS_OG.csv", index_col=0)

print(f"    Gene Expression: {gene_df.shape}")
print(f"    Methylation: {meth_df.shape}")
print(f"    CNV: {cnv_df.shape}")
print(f"    Labels: {labels_df.shape}")

# Find common samples across all datasets
common_samples = gene_df.index.intersection(meth_df.index).intersection(cnv_df.index).intersection(labels_df.index)
print(f">>> Common samples across all datasets: {len(common_samples)}")

# Align all datasets to common samples
gene_df = gene_df.loc[common_samples]
meth_df = meth_df.loc[common_samples]
cnv_df = cnv_df.loc[common_samples]
labels_df = labels_df.loc[common_samples]

# Convert to numpy arrays
X_rna = gene_df.values.astype(np.float32)
X_meth = meth_df.values.astype(np.float32)
X_clin = cnv_df.values.astype(np.float32)  # Using CNV as the third modality

# Handle NaN values in features
print(f">>> NaN check - RNA: {np.isnan(X_rna).sum()}, Meth: {np.isnan(X_meth).sum()}, CNV: {np.isnan(X_clin).sum()}")
X_rna = np.nan_to_num(X_rna, nan=0.0)
X_meth = np.nan_to_num(X_meth, nan=0.0)
X_clin = np.nan_to_num(X_clin, nan=0.0)

# Store original (unscaled, full-feature) data for variance experiments
X_rna_original = X_rna.copy()
X_meth_original = X_meth.copy()
X_clin_original = X_clin.copy()

print(f">>> Original feature dimensions: RNA={X_rna_original.shape[1]}, Meth={X_meth_original.shape[1]}, CNV={X_clin_original.shape[1]}")

# Extract labels - handle different possible column structures
if labels_df.shape[1] == 1:
    # Single column of labels
    raw_labels = labels_df.iloc[:, 0].values
else:
    # Multiple columns - use the first one or look for common label column names
    label_cols = [col for col in labels_df.columns if col.lower() in ['label', 'class', 'target', 'y', 'subtype']]
    if label_cols:
        raw_labels = labels_df[label_cols[0]].values
    else:
        raw_labels = labels_df.iloc[:, 0].values

# Always use LabelEncoder to ensure labels are 0-indexed integers
from sklearn.preprocessing import LabelEncoder
import pandas as pd
le = LabelEncoder()
raw_labels_clean = pd.Series(raw_labels).fillna('UNKNOWN').astype(str).values
Y_labels = le.fit_transform(raw_labels_clean)
print(f">>> Label classes: {le.classes_}")
print(f">>> Label distribution: {dict(zip(le.classes_, np.bincount(Y_labels)))}")

# Validate labels
n_samples = len(Y_labels)
n_classes = len(np.unique(Y_labels))
assert Y_labels.min() >= 0, f"Labels must be >= 0, got min: {Y_labels.min()}"
assert Y_labels.max() < n_classes, f"Labels must be < n_classes ({n_classes}), got max: {Y_labels.max()}"
print(f">>> Loaded {n_samples} samples with {n_classes} classes (labels range: {Y_labels.min()}-{Y_labels.max()})")


# ==========================================
# VARIANCE-BASED FEATURE SELECTION
# ==========================================
def select_high_variance_features(X, top_percent):
    """
    Select top X% features with highest variance.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        top_percent: float between 0 and 1, e.g., 0.3 for top 30%
    
    Returns:
        X_selected: numpy array with selected features
        selected_indices: indices of selected features
    """
    # Calculate variance for each feature
    variances = np.var(X, axis=0)
    
    # Number of features to keep
    n_keep = int(np.ceil(X.shape[1] * top_percent))
    
    # Get indices of top variance features
    selected_indices = np.argsort(variances)[-n_keep:]
    
    # Sort indices to maintain original order
    selected_indices = np.sort(selected_indices)
    
    return X[:, selected_indices], selected_indices


def prepare_data_with_variance_selection(X_rna_orig, X_meth_orig, X_clin_orig, top_percent):
    """
    Prepare data with variance-based feature selection and scaling.
    
    Args:
        X_rna_orig, X_meth_orig, X_clin_orig: Original unscaled data
        top_percent: Percentage of high-variance features to keep (0.3 = 30%)
    
    Returns:
        Scaled data arrays and dimensions
    """
    # Select high variance features
    X_rna_sel, rna_idx = select_high_variance_features(X_rna_orig, top_percent)
    X_meth_sel, meth_idx = select_high_variance_features(X_meth_orig, top_percent)
    X_clin_sel, clin_idx = select_high_variance_features(X_clin_orig, top_percent)
    
    # Scale the selected features
    scaler_rna = StandardScaler()
    scaler_meth = StandardScaler()
    scaler_clin = StandardScaler()
    
    X_rna_scaled = scaler_rna.fit_transform(X_rna_sel)
    X_meth_scaled = scaler_meth.fit_transform(X_meth_sel)
    X_clin_scaled = scaler_clin.fit_transform(X_clin_sel)
    
    dims = (X_rna_scaled.shape[1], X_meth_scaled.shape[1], X_clin_scaled.shape[1])
    
    return X_rna_scaled, X_meth_scaled, X_clin_scaled, dims


# For backward compatibility, prepare default full-feature scaled data
scaler_rna = StandardScaler()
scaler_meth = StandardScaler()
scaler_clin = StandardScaler()

X_rna = scaler_rna.fit_transform(X_rna_original)
X_meth = scaler_meth.fit_transform(X_meth_original)
X_clin = scaler_clin.fit_transform(X_clin_original)

# Get dimensions for model initialization
DIMS = (X_rna.shape[1], X_meth.shape[1], X_clin.shape[1])
print(f">>> Data loaded successfully! Dimensions: RNA={DIMS[0]}, Meth={DIMS[1]}, CNV={DIMS[2]}")


# ==========================================
# EARLY STOPPING HELPER CLASS
# ==========================================
class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    Triggers when val_loss increases for 'patience' consecutive epochs (overfitting detection).
    """
    def __init__(self, patience=15, min_delta=0.0, mode='min'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            mode (str): 'min' for loss (lower is better), 'max' for accuracy (higher is better).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.mode == 'min':
            score = -score  # Convert to maximization problem
            
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
        return self.early_stop
    
    def reset(self):
        """Reset the early stopping state for a new fold."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

# ==========================================
# 2. MODEL CLASSES
# ==========================================
# Optimized for high-dimensional, low-sample multi-omics data
# (205 samples with ~50k, ~400k, ~60k features)
# Using deeper bottleneck architecture with heavy regularization
# ==========================================

class PerOmicCMAE(nn.Module):
    """
    Per-Omic Contrastive Masked Autoencoder.
    
    Architecture optimized for high-dimensional omics data (50k-400k features) 
    with only 205 samples. Uses aggressive dimensionality reduction with 
    multiple bottleneck layers and strong regularization.
    """
    def __init__(self, input_dim, latent_dim, dropout_rate=0.3):
        super().__init__()
        
        # Determine intermediate dimensions based on input size
        # For very high-dimensional inputs, use more aggressive reduction
        if input_dim > 200000:  # ~400k methylation features
            hidden1 = 2048
            hidden2 = 512
        elif input_dim > 40000:  # ~50k/60k gene/CNV features
            hidden1 = 1024
            hidden2 = 256
        else:
            hidden1 = 512
            hidden2 = 128
        
        self.encoder = nn.Sequential(
            # First layer: aggressive reduction
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # Second layer: further compression
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # Final layer to latent space
            nn.Linear(hidden2, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),  # Less dropout in decoder
            nn.Linear(hidden2, hidden1),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden1, input_dim)
        )
        
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x, mask_ratio=0.0):
        if mask_ratio > 0 and self.training:
            mask = (torch.rand_like(x) > mask_ratio).float()
            x_masked = x * mask
        else:
            mask = torch.ones_like(x)
            x_masked = x
        z = self.encoder(x_masked)
        return self.decoder(z), self.projector(z), z, mask


class GatedAttentionFusion(nn.Module):
    """
    Gated Attention Fusion for multi-omics integration.
    Enhanced with regularization for small sample sizes.
    """
    def __init__(self, latent_dim, n_classes=3, dropout_rate=0.3):
        super().__init__()
        self.gate_rna = nn.Linear(latent_dim, 1)
        self.gate_meth = nn.Linear(latent_dim, 1)
        self.gate_clin = nn.Linear(latent_dim, 1)
        
        # Add hidden layer before classifier for better representation
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim // 2, n_classes)
        )
        self.drop_rate = dropout_rate

    def forward(self, z_rna, z_meth, z_clin, apply_dropout=False):
        if apply_dropout and self.training:
            if torch.rand(1).item() < self.drop_rate: z_rna = torch.zeros_like(z_rna)
            if torch.rand(1).item() < self.drop_rate: z_meth = torch.zeros_like(z_meth)
            if torch.rand(1).item() < self.drop_rate: z_clin = torch.zeros_like(z_clin)

        w_rna = torch.sigmoid(self.gate_rna(z_rna))
        w_meth = torch.sigmoid(self.gate_meth(z_meth))
        w_clin = torch.sigmoid(self.gate_clin(z_clin))

        z_fused = (w_rna * z_rna + w_meth * z_meth + w_clin * z_clin) / (w_rna + w_meth + w_clin + 1e-8)
        return self.classifier(z_fused), torch.cat([w_rna, w_meth, w_clin], dim=1)


class StabilizedUncertaintyLoss(nn.Module):
    def __init__(self, num_losses):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            prec = torch.clamp(0.5 * torch.exp(-self.log_vars[i]), 0.2, 3.0)
            total += prec * loss + 0.5 * self.log_vars[i]
        return total


def contrastive_loss(q, k, queue, temp=0.1):
    q = F.normalize(q, dim=1); k = F.normalize(k, dim=1); queue = queue.detach()
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    l_neg = torch.einsum('nc,ck->nk', [q, queue])
    logits = torch.cat([l_pos, l_neg], dim=1) / temp
    return F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long).to(q.device))


# ==========================================
# 3. OPTUNA OBJECTIVE FUNCTION
# ==========================================
# Optimized hyperparameter ranges for 205 samples with high-dimensional features
# Key adjustments:
#   - Smaller batch sizes (16) to get more gradient updates per epoch
#   - Smaller latent dimensions (32-64) to prevent overfitting
#   - Higher dropout and weight decay for regularization
#   - Early stopping with patience 15
# ==========================================

def create_objective(X_rna_data, X_meth_data, X_clin_data, dims, Y_labels_data, n_classes_data, verbose=True):
    """
    Factory function to create an objective function with specific data.
    This allows running experiments with different variance-selected features.
    
    Args:
        X_rna_data, X_meth_data, X_clin_data: Scaled omics data arrays
        dims: Tuple of (rna_dim, meth_dim, clin_dim)
        Y_labels_data: Label array
        n_classes_data: Number of classes
        verbose: Whether to print progress
    
    Returns:
        objective function for Optuna
    """
    def objective(trial):
        # --- A. Suggest Hyperparameters ---
        # Pre-training hyperparameters
        lr_pre = trial.suggest_float("lr_pre", 5e-5, 2e-3, log=True)
        mask_ratio = trial.suggest_float("mask_ratio", 0.15, 0.50)
        latent_dim = trial.suggest_categorical("latent_dim", [32, 48, 64])
        weight_decay_pre = trial.suggest_float("weight_decay_pre", 1e-4, 1e-2, log=True)
        
        # Fine-tuning hyperparameters
        lr_fine = trial.suggest_float("lr_fine", 5e-5, 5e-3, log=True)
        lambda_ent = trial.suggest_float("lambda_ent", 0.0, 0.15)
        dropout_rate = trial.suggest_float("dropout_rate", 0.25, 0.50)
        weight_decay_fine = trial.suggest_float("weight_decay_fine", 1e-4, 1e-2, log=True)
        
        # Batch size - smaller for more gradient updates with limited samples
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 24])

        # --- B. Cross-Validation Loop ---
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_rna_data, Y_labels_data)):

            # 1. Prepare Data
            tr_ds = TensorDataset(
                torch.FloatTensor(X_rna_data[train_idx]), torch.FloatTensor(X_meth_data[train_idx]),
                torch.FloatTensor(X_clin_data[train_idx]), torch.LongTensor(Y_labels_data[train_idx])
            )
            val_ds = TensorDataset(
                torch.FloatTensor(X_rna_data[val_idx]), torch.FloatTensor(X_meth_data[val_idx]),
                torch.FloatTensor(X_clin_data[val_idx]), torch.LongTensor(Y_labels_data[val_idx])
            )

            # Use smaller batch size for more gradient updates
            tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True)
            val_rna, val_meth, val_clin, val_y = val_ds[:]
            val_rna, val_meth, val_clin, val_y = val_rna.to(DEVICE), val_meth.to(DEVICE), val_clin.to(DEVICE), val_y.to(DEVICE)

            # 2. Init Models with dropout
            cmae_r = PerOmicCMAE(dims[0], latent_dim, dropout_rate=dropout_rate).to(DEVICE)
            cmae_m = PerOmicCMAE(dims[1], latent_dim, dropout_rate=dropout_rate).to(DEVICE)
            cmae_c = PerOmicCMAE(dims[2], latent_dim, dropout_rate=dropout_rate).to(DEVICE)
            mem_bank = nn.Parameter(F.normalize(torch.randn(latent_dim, 64), dim=0), requires_grad=False).to(DEVICE)
            loss_fn = StabilizedUncertaintyLoss(4).to(DEVICE)

            # AdamW with weight decay for L2 regularization
            opt_pre = optim.AdamW(
                list(cmae_r.parameters()) + list(cmae_m.parameters()) + 
                list(cmae_c.parameters()) + list(loss_fn.parameters()), 
                lr=lr_pre, 
                weight_decay=weight_decay_pre
            )

            # 3. Phase 1: Pre-training with Early Stopping
            early_stop_pre = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode='min')
            cmae_r.train(); cmae_m.train(); cmae_c.train()
            
            max_pretrain_epochs = 50  # Increased max but with early stopping
            
            for epoch in range(max_pretrain_epochs):
                epoch_losses = []
                for r, m, c, _ in tr_loader:
                    r, m, c = r.to(DEVICE), m.to(DEVICE), c.to(DEVICE)

                    # Forward View 1
                    rec_r1, proj_r1, _, _ = cmae_r(r, mask_ratio)
                    rec_m1, proj_m1, _, _ = cmae_m(m, mask_ratio)
                    rec_c1, proj_c1, _, _ = cmae_c(c, mask_ratio)

                    # Forward View 2
                    with torch.no_grad():
                        _, proj_r2, _, _ = cmae_r(r, mask_ratio)
                        _, proj_m2, _, _ = cmae_m(m, mask_ratio)
                        _, proj_c2, _, _ = cmae_c(c, mask_ratio)

                    loss = loss_fn([
                        F.mse_loss(rec_r1, r), F.mse_loss(rec_m1, m), F.mse_loss(rec_c1, c),
                        (contrastive_loss(proj_r1, proj_r2, mem_bank) +
                         contrastive_loss(proj_m1, proj_m2, mem_bank) +
                         contrastive_loss(proj_c1, proj_c2, mem_bank)) / 3
                    ])

                    opt_pre.zero_grad()
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(
                        list(cmae_r.parameters()) + list(cmae_m.parameters()) + list(cmae_c.parameters()), 
                        max_norm=1.0
                    )
                    opt_pre.step()
                    
                    epoch_losses.append(loss.item())
                    
                    with torch.no_grad():
                        avg_proj = (proj_r1 + proj_m1 + proj_c1) / 3
                        mem_bank.data = torch.cat([mem_bank[:, avg_proj.shape[0]:], avg_proj.T], dim=1)
                
                # Compute validation loss for early stopping
                cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
                with torch.no_grad():
                    rec_r_val, _, _, _ = cmae_r(val_rna)
                    rec_m_val, _, _, _ = cmae_m(val_meth)
                    rec_c_val, _, _, _ = cmae_c(val_clin)
                    val_loss = (F.mse_loss(rec_r_val, val_rna) + 
                               F.mse_loss(rec_m_val, val_meth) + 
                               F.mse_loss(rec_c_val, val_clin)).item()
                cmae_r.train(); cmae_m.train(); cmae_c.train()
                
                # Check early stopping
                if early_stop_pre(val_loss, epoch):
                    if verbose:
                        print(f"    Pre-training early stopped at epoch {epoch} (best: {early_stop_pre.best_epoch})")
                    break

            # 4. Phase 2: Fine-tuning with Early Stopping
            cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
            fusion = GatedAttentionFusion(latent_dim, n_classes=n_classes_data, dropout_rate=dropout_rate).to(DEVICE)
            opt_fine = optim.AdamW(fusion.parameters(), lr=lr_fine, weight_decay=weight_decay_fine)

            best_fold_acc = 0
            best_model_state = None
            early_stop_fine = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode='max')  # Max for accuracy
            
            max_finetune_epochs = 300

            for epoch in range(max_finetune_epochs):
                fusion.train()
                epoch_train_loss = 0
                num_batches = 0
                
                for r, m, c, y in tr_loader:
                    r, m, c, y = r.to(DEVICE), m.to(DEVICE), c.to(DEVICE), y.to(DEVICE)
                    with torch.no_grad():
                        _, _, zr, _ = cmae_r(r)
                        _, _, zm, _ = cmae_m(m)
                        _, _, zc, _ = cmae_c(c)

                    logits, weights = fusion(zr, zm, zc, apply_dropout=True)
                    cls_loss = F.cross_entropy(logits, y)
                    entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean()
                    loss = cls_loss + lambda_ent * entropy

                    opt_fine.zero_grad()
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_norm=1.0)
                    opt_fine.step()
                    
                    epoch_train_loss += loss.item()
                    num_batches += 1

                # Validation
                fusion.eval()
                with torch.no_grad():
                    _, _, zr, _ = cmae_r(val_rna)
                    _, _, zm, _ = cmae_m(val_meth)
                    _, _, zc, _ = cmae_c(val_clin)

                    logits, _ = fusion(zr, zm, zc, apply_dropout=False)
                    val_loss = F.cross_entropy(logits, val_y).item()
                    preds = logits.argmax(dim=1)
                    acc = accuracy_score(val_y.cpu(), preds.cpu())
                    
                    if acc > best_fold_acc:
                        best_fold_acc = acc
                        # Save best model state
                        best_model_state = {
                            'fusion': fusion.state_dict(),
                            'epoch': epoch
                        }
                
                # Early stopping based on validation accuracy (mode='max')
                if early_stop_fine(acc, epoch):
                    if verbose:
                        print(f"    Fine-tuning early stopped at epoch {epoch} (best: {early_stop_fine.best_epoch}, acc: {best_fold_acc:.4f})")
                    break

            fold_accuracies.append(best_fold_acc)
            if verbose:
                print(f"  Fold {fold + 1}/3: Best Accuracy = {best_fold_acc:.4f}")

            trial.report(np.mean(fold_accuracies), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        mean_acc = np.mean(fold_accuracies)
        if verbose:
            print(f"  Trial Mean Accuracy: {mean_acc:.4f}")
        return mean_acc
    
    return objective


# ==========================================
# 4. VARIANCE EXPERIMENT: Compare Different Feature Percentages
# ==========================================
# Test top 30%, 40%, 50%, 60%, 70% high-variance features
# and compare results in a side-by-side table
# ==========================================

VARIANCE_PERCENTAGES = [0.30, 0.40, 0.50, 0.60, 0.70]
N_TRIALS_PER_EXPERIMENT = 15  # Reduced trials per experiment for faster comparison

# Store results for comparison table
experiment_results = []

print("\n" + "=" * 80)
print("VARIANCE-BASED FEATURE SELECTION EXPERIMENT")
print("=" * 80)
print(f"Testing variance thresholds: {[f'{p*100:.0f}%' for p in VARIANCE_PERCENTAGES]}")
print(f"Trials per experiment: {N_TRIALS_PER_EXPERIMENT}")
print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
print("=" * 80)

for var_percent in VARIANCE_PERCENTAGES:
    print(f"\n{'='*60}")
    print(f">>> EXPERIMENT: Top {var_percent*100:.0f}% High-Variance Features")
    print(f"{'='*60}")
    
    # Prepare data with variance selection
    X_rna_var, X_meth_var, X_clin_var, dims_var = prepare_data_with_variance_selection(
        X_rna_original, X_meth_original, X_clin_original, var_percent
    )
    
    print(f">>> Selected features: RNA={dims_var[0]}, Meth={dims_var[1]}, CNV={dims_var[2]}")
    print(f">>> Total features: {sum(dims_var)} (from original {sum([X_rna_original.shape[1], X_meth_original.shape[1], X_clin_original.shape[1]])})")
    
    # Create objective function for this data
    objective_fn = create_objective(
        X_rna_var, X_meth_var, X_clin_var, 
        dims_var, Y_labels, n_classes, 
        verbose=True
    )
    
    # Run Optuna optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce Optuna verbosity
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    )
    study.optimize(objective_fn, n_trials=N_TRIALS_PER_EXPERIMENT, show_progress_bar=False)
    
    # Store results
    result = {
        'variance_percent': f"{var_percent*100:.0f}%",
        'rna_features': dims_var[0],
        'meth_features': dims_var[1],
        'cnv_features': dims_var[2],
        'total_features': sum(dims_var),
        'best_accuracy': study.best_value,
        'best_params': study.best_params
    }
    experiment_results.append(result)
    
    print(f"\n>>> Best Accuracy for {var_percent*100:.0f}%: {study.best_value:.4f}")


# ==========================================
# 5. RESULTS COMPARISON TABLE
# ==========================================
print("\n\n")
print("=" * 100)
print("VARIANCE EXPERIMENT RESULTS - COMPARISON TABLE")
print("=" * 100)

# Create DataFrame for results
results_df = pd.DataFrame([{
    'Variance %': r['variance_percent'],
    'RNA Features': r['rna_features'],
    'Meth Features': r['meth_features'],
    'CNV Features': r['cnv_features'],
    'Total Features': r['total_features'],
    'Best Accuracy': f"{r['best_accuracy']:.4f}",
    'Latent Dim': r['best_params'].get('latent_dim', 'N/A'),
    'Dropout': f"{r['best_params'].get('dropout_rate', 0):.3f}",
    'Batch Size': r['best_params'].get('batch_size', 'N/A')
} for r in experiment_results])

# Print formatted table
print("\n" + results_df.to_string(index=False))

# Also print as markdown table for easy copy-paste
print("\n\n### Markdown Table Format:")
print(results_df.to_markdown(index=False))

# Find best performing variance threshold
best_result = max(experiment_results, key=lambda x: x['best_accuracy'])
print(f"\n>>> BEST PERFORMING CONFIGURATION:")
print(f"    Variance Threshold: {best_result['variance_percent']}")
print(f"    Best Accuracy: {best_result['best_accuracy']:.4f}")
print(f"    Total Features: {best_result['total_features']}")
print(f"    Best Hyperparameters:")
for k, v in best_result['best_params'].items():
    print(f"      {k}: {v}")

# Save results to CSV
results_df.to_csv("variance_experiment_results.csv", index=False)
print(f"\n>>> Results saved to: variance_experiment_results.csv")

print("\n" + "=" * 100)
print("EXPERIMENT COMPLETE!")
print("=" * 100)