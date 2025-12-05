#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-05T14:01:32.061Z
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

# ==========================================
# 1. LOAD REAL DATA FROM CSV FILES
# ==========================================
print(">>> Loading real data from CSV files...")

# Load the three omics datasets (transposed so samples are rows)
gene_df = pd.read_csv("../NewDatasets/processed_expression_4O.csv", index_col=0).T
meth_df = pd.read_csv("../NewDatasets/processed_methylation_4O.csv", index_col=0).T
cnv_df  = pd.read_csv("../NewDatasets/processed_cnv_4O.csv", index_col=0).T

# Load labels
labels_df = pd.read_csv("../NewDatasets/processed_labels_3Omics_FXS_OG.csv", index_col=0)

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

# Scale the data
scaler_rna = StandardScaler()
scaler_meth = StandardScaler()
scaler_clin = StandardScaler()

X_rna = scaler_rna.fit_transform(X_rna)
X_meth = scaler_meth.fit_transform(X_meth)
X_clin = scaler_clin.fit_transform(X_clin)

# Get dimensions for model initialization
DIMS = (X_rna.shape[1], X_meth.shape[1], X_clin.shape[1])
print(f">>> Data loaded successfully! Dimensions: RNA={DIMS[0]}, Meth={DIMS[1]}, CNV={DIMS[2]}")

# ==========================================
# 2. MODEL CLASSES
# ==========================================
# ==========================================
# 2. MODEL CLASSES (FIXED)
# ==========================================
class PerOmicCMAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.GELU(),
            nn.Linear(512, input_dim)
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
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

        # --- FIX: RETURN 4 VALUES (Added 'mask') ---
        return self.decoder(z), self.projector(z), z, mask

class GatedAttentionFusion(nn.Module):
    def __init__(self, latent_dim, n_classes=3, dropout_rate=0.2):
        super().__init__()
        self.gate_rna = nn.Linear(latent_dim, 1)
        self.gate_meth = nn.Linear(latent_dim, 1)
        self.gate_clin = nn.Linear(latent_dim, 1)
        self.classifier = nn.Linear(latent_dim, n_classes)
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
# 3. OPTUNA OBJECTIVE FUNCTION (FIXED)
# ==========================================
def objective(trial):
    # --- A. Suggest Hyperparameters ---
    lr_pre = trial.suggest_float("lr_pre", 1e-4, 5e-3, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.25, 0.75)
    latent_dim = trial.suggest_categorical("latent_dim", [64, 128])

    lr_fine = trial.suggest_float("lr_fine", 1e-4, 1e-2, log=True)
    lambda_ent = trial.suggest_float("lambda_ent", 0.0, 0.2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)

    # --- B. Cross-Validation Loop ---
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_rna, Y_labels)):

        # 1. Prepare Data
        tr_ds = TensorDataset(
            torch.FloatTensor(X_rna[train_idx]), torch.FloatTensor(X_meth[train_idx]),
            torch.FloatTensor(X_clin[train_idx]), torch.LongTensor(Y_labels[train_idx])
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_rna[val_idx]), torch.FloatTensor(X_meth[val_idx]),
            torch.FloatTensor(X_clin[val_idx]), torch.LongTensor(Y_labels[val_idx])
        )

        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, drop_last=True)
        val_rna, val_meth, val_clin, val_y = val_ds[:]
        val_rna, val_meth, val_clin, val_y = val_rna.to(DEVICE), val_meth.to(DEVICE), val_clin.to(DEVICE), val_y.to(DEVICE)

        # 2. Init Models
        cmae_r = PerOmicCMAE(DIMS[0], latent_dim).to(DEVICE)
        cmae_m = PerOmicCMAE(DIMS[1], latent_dim).to(DEVICE)
        cmae_c = PerOmicCMAE(DIMS[2], latent_dim).to(DEVICE)
        mem_bank = nn.Parameter(F.normalize(torch.randn(latent_dim, 128), dim=0), requires_grad=False).to(DEVICE)
        loss_fn = StabilizedUncertaintyLoss(4).to(DEVICE)

        opt_pre = optim.AdamW(list(cmae_r.parameters())+list(cmae_m.parameters())+list(cmae_c.parameters())+list(loss_fn.parameters()), lr=lr_pre)

        # 3. Phase 1: Pre-training
        cmae_r.train(); cmae_m.train(); cmae_c.train()
        for epoch in range(15): # Reduced epochs for speed
            for r, m, c, _ in tr_loader:
                r, m, c = r.to(DEVICE), m.to(DEVICE), c.to(DEVICE)

                # Forward View 1 (Now unpacking 4 values works)
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
                     contrastive_loss(proj_c1, proj_c2, mem_bank))/3
                ])

                opt_pre.zero_grad(); loss.backward(); opt_pre.step()
                with torch.no_grad():
                    avg_proj = (proj_r1 + proj_m1 + proj_c1) / 3
                    mem_bank.data = torch.cat([mem_bank[:, avg_proj.shape[0]:], avg_proj.T], dim=1)

        # 4. Phase 2: Fine-tuning
        cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
        fusion = GatedAttentionFusion(latent_dim, n_classes=n_classes, dropout_rate=dropout_rate).to(DEVICE)
        opt_fine = optim.AdamW(fusion.parameters(), lr=lr_fine)

        best_fold_acc = 0

        for epoch in range(20):
            fusion.train()
            for r, m, c, y in tr_loader:
                r, m, c, y = r.to(DEVICE), m.to(DEVICE), c.to(DEVICE), y.to(DEVICE)
                with torch.no_grad():
                    # --- FIX: UNPACK 4 VALUES HERE TOO (using _) ---
                    _, _, zr, _ = cmae_r(r)
                    _, _, zm, _ = cmae_m(m)
                    _, _, zc, _ = cmae_c(c)

                logits, weights = fusion(zr, zm, zc, apply_dropout=True)
                cls_loss = F.cross_entropy(logits, y)
                entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean()
                loss = cls_loss + lambda_ent * entropy

                opt_fine.zero_grad(); loss.backward(); opt_fine.step()

            fusion.eval()
            with torch.no_grad():
                # --- FIX: UNPACK 4 VALUES HERE TOO ---
                _, _, zr, _ = cmae_r(val_rna)
                _, _, zm, _ = cmae_m(val_meth)
                _, _, zc, _ = cmae_c(val_clin)

                logits, _ = fusion(zr, zm, zc, apply_dropout=False)
                preds = logits.argmax(dim=1)
                acc = accuracy_score(val_y.cpu(), preds.cpu())
                if acc > best_fold_acc: best_fold_acc = acc

        fold_accuracies.append(best_fold_acc)

        trial.report(np.mean(fold_accuracies), step=fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(fold_accuracies)


# ==========================================
# 4. RUN OPTIMIZATION
# ==========================================
print("\n>>> STARTING OPTUNA OPTIMIZATION")
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)

print("\n" + "="*50)
print(f"BEST RESULT: {study.best_value:.4f}")
print("BEST PARAMS:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print("="*50)